import copy
import csv
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from .detection import DetectionList, Edge
from .floor import Floor, segment_floors
from .pointcloud import pcd_denoise_dbscan
from .rooms import Room, assign_frames_to_rooms, assign_tracks_to_rooms, segment_rooms
from .track import get_trackid_by_name, load_tracks, object_to_track
from .visualizer import Visualizer2D, Visualizer3D


class SemanticTree:
    def __init__(self, visual_memory_size, device="cuda"):
        self.geometry_map = o3d.geometry.PointCloud()
        self.device = device
        self.hierarchy_matrix = None
        self.hierarchy_type_matrix = None
        self.track_ids = []
        self.temp_refine_dir = "temp-refine"

        self.tracks = {}
        self.navigation_log = []
        self.visual_memory = []
        self.visual_memory_size = visual_memory_size

        # variables for consolidate_edges
        self.imgs_for_consolidation = []
        self.edges_for_consolidation = []
        self.detections_for_consolidation = []

        # floors and rooms
        self.rooms = {}
        self.floors = []
        self.room_types = [
            "kitchen",
            "living room",
            "bedroom",
            "dinning room",
            "living and dining area",
            "bathroom",
            "washroom",
            "hallway",
            "garage",
            "corridor",
            "office",
        ]
        self.clip_processor = AutoProcessor.from_pretrained(
            "google/siglip-base-patch16-224"
        )
        self.clip_model = AutoModel.from_pretrained(
            "google/siglip-base-patch16-224"
        ).to(device)
        self.clip_tokenizer = AutoTokenizer.from_pretrained(
            "google/siglip-base-patch16-224"
        )

    def load(self, folder, load_floors=True, load_rooms=True):
        folder = Path(folder)
        navigation_log_path = os.path.join(folder, "semantic_tree/navigation_log.json")
        with open(navigation_log_path) as f:
            self.navigation_log = json.load(f)
        shutil.rmtree(folder / self.temp_refine_dir, ignore_errors=True)
        self.extract_visual_memory(self.visual_memory_size)

        with open(folder / "semantic_tree/hierarchy_matrix.json") as f:
            json_data = json.load(f)
            hierarchy_matrix = {}
            for i in json_data.keys():
                hierarchy_matrix[int(i)] = {}
                for j in json_data[i].keys():
                    hierarchy_matrix[int(i)][int(j)] = json_data[i][j]
            self.hierarchy_matrix = hierarchy_matrix
        with open(folder / "semantic_tree/hierarchy_type_matrix.json") as f:
            json_data = json.load(f)
            hierarchy_type_matrix = {}
            for i in json_data.keys():
                hierarchy_type_matrix[int(i)] = {}
                for j in json_data[i].keys():
                    hierarchy_type_matrix[int(i)][int(j)] = json_data[i][j]
            self.hierarchy_type_matrix = hierarchy_type_matrix
        # self.neighbour_matrix = np.loadtxt(
        #     save_dir / "semantic_tree/neighbours.txt", np.float32
        # )
        self.track_ids = np.loadtxt(folder / "semantic_tree/track_ids.txt", np.float32)
        self.tracks = load_tracks(folder / "tracks")
        self.load_pcd(folder=folder)

        if load_floors:
            self.floors = []
            floor_files = os.listdir(os.path.join(folder, "semantic_tree/floors"))
            floor_files.sort()
            floor_files = sorted([f for f in floor_files if f.endswith(".ply")])
            for floor_file in floor_files:
                floor_file = floor_file.split(".")[0]
                floor = Floor(str(floor_file), name="floor_" + str(floor_file))
                floor.load(os.path.join(folder, "semantic_tree/floors"))
                self.floors.append(floor)

        if load_rooms:
            self.rooms = {}
            room_files = os.listdir(os.path.join(folder, "semantic_tree/rooms"))
            room_files.sort()
            room_files = [f for f in room_files if f.endswith(".ply")]
            for room_file in room_files:
                room_file = room_file.split(".")[0]
                room = Room(str(room_file), room_file.split("_")[0])
                room.load(os.path.join(folder, "semantic_tree/rooms"))

                floor = None
                for i in range(len(self.floors)):
                    if self.floors[i].floor_id == room.floor_id:
                        floor = self.floors[i]
                        break
                floor.rooms.append(room)
                self.rooms[room.room_id] = room

    def extract_visual_memory(self, visual_memory_size):
        self.visual_memory = []
        for i, log in enumerate(self.navigation_log):
            if log.get("Generic Mapping") != None:
                self.visual_memory.append(i)
        k_indices = np.linspace(
            0, len(self.visual_memory) - 1, num=visual_memory_size, dtype=np.int32
        )
        k_indices = np.unique(k_indices)
        self.visual_memory = [
            self.visual_memory[i] for i in k_indices
        ]  # only keep k frames in the memory

    def add_pcd(self, pcd):
        self.geometry_map += pcd

    def denoise_geometrymap(self, downsample_voxel_size, dbscan_eps, min_points):
        self.geometry_map = self.geometry_map.voxel_down_sample(
            voxel_size=downsample_voxel_size
        )
        self.geometry_map = pcd_denoise_dbscan(
            self.geometry_map, eps=dbscan_eps, min_points=min_points
        )

    def save_pcd(self, path=None, folder=None, file="full_scene.pcd"):
        if not path:
            path = os.path.join(folder, file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            os.remove(path)
        o3d.io.write_point_cloud(path, self.geometry_map)

    def load_pcd(self, path=None, folder=None, file="full_scene.pcd"):
        if not path:
            path = os.path.join(folder, file)
        self.geometry_map = o3d.io.read_point_cloud(path)

    def get_tracks(self):
        return list(self.tracks.values())

    def extend_navigation_log(self, frame_idx):
        frames = [b["Frame Index"] for b in self.navigation_log]
        if frame_idx in frames:
            return
        for i in range(len(self.navigation_log), frame_idx + 1, 1):
            log = {
                "Frame Index": i,
                "Generic Mapping": None,
                "Focused Analyses and Search": [],
            }
            self.navigation_log.append(log)

    def get_navigation_log_idx(self, frame_idx):
        self.extend_navigation_log(frame_idx)
        for i in range(len(self.navigation_log)):
            if frame_idx != self.navigation_log[i]["Frame Index"]:
                continue
            else:
                return i

    def integrate_generic_log(self, llm_response, detections, frame_idx):
        i = self.get_navigation_log_idx(frame_idx)

        # edges_in_frame = []
        # for det in detections:
        #     edges_in_frame += [edge for edge in det.edges]

        self.navigation_log[i]["Generic Mapping"] = {
            "Relative Motion": llm_response["Relative Motion"],
            "Estimated Current Location": llm_response["Current Location"],
            "View": llm_response["View"],
            "Summary": llm_response["Summary"],
            "Detections": [d.matched_track_name for d in detections],
            # "Edges": [edge.json() for edge in edges_in_frame],
        }

    def integrate_refinement_log(
        self, request, llm_response, keyframe_id, detections: DetectionList = None
    ):
        refinement_log = llm_response
        i = self.get_navigation_log_idx(keyframe_id)
        self.navigation_log[i]["Focused Analyses and Search"].append(refinement_log)
        self.navigation_log[i]["Generic Mapping"]["Detections"] = list(
            set(
                self.navigation_log[i]["Generic Mapping"]["Detections"]
                + detections.get_field("matched_track_name")
            )
        )

    def integrate_detections(
        self,
        detections: DetectionList,
        is_matched,
        matched_trackidx,
        frame_idx,
        img,
        consolidate=True,
    ):
        # Either merge each detection with its matched track, or initialize the detection as a new track
        track_ids = list(self.tracks.keys())
        self.track_ids = track_ids
        detlabel2trkname = {}
        for i in range(len(detections)):
            if is_matched[i]:
                self.tracks[track_ids[matched_trackidx[i]]].merge_detection(
                    detections[i], frame_idx
                )
                matched_track_name = self.tracks[track_ids[matched_trackidx[i]]].name
                detections[i].matched_track_name = matched_track_name
                detlabel2trkname[detections[i].label] = matched_track_name
            else:
                new_track = object_to_track(detections[i], frame_idx, edges=[])
                assert not new_track.id in self.tracks.keys()
                self.tracks[new_track.id] = new_track
                self.track_ids = list(self.tracks.keys())
                detections[i].matched_track_name = new_track.name
                detlabel2trkname[detections[i].label] = detections[i].matched_track_name

        # Replace the names of the edges from the detection names to the track names. Then add the edges into the track list.
        edges = []
        for i in range(len(detections)):
            det = detections[i]
            for edge in det.edges:
                subject_trkname = detlabel2trkname[edge.subject]
                object_trkname = detlabel2trkname[edge.related_object]
                if subject_trkname == object_trkname:
                    continue
                edges.append(
                    Edge(
                        edge.type,
                        subject_trkname,
                        object_trkname,
                        frame_id=frame_idx,
                        description=edge.description,
                    )
                )
        if consolidate:
            self.imgs_for_consolidation.append(img)
            self.detections_for_consolidation.append(detections)
            self.edges_for_consolidation.append(edges)
        return detections, edges

    def merge_edge_buffer(self, edges, clear_buffers=True):
        # Replace the names of the edges from the detection names to the track names. Then add the edges into the track list.
        for edge in edges:
            subject_trkid = get_trackid_by_name(self.tracks, edge.subject_object)
            self.tracks[subject_trkid].edges.append(edge)
        if clear_buffers:
            self.edges_for_consolidation = []
            self.imgs_for_consolidation = []
            self.detections_for_consolidation = []

    def process_rooms(
        self,
        room_grid_resolution,
        img_list,
        pose_list,
        frame_list,
        debug_folder,
        device="cpu",
    ):
        self.clip_model = self.clip_model.to(device)
        self.floors = segment_floors(
            self.geometry_map,
            graph_tmp_folder=debug_folder,
            save_intermediate_results=True,
            # flip_zy=True,
        )
        for floor in self.floors:
            detected_rooms = segment_rooms(
                floor=floor,
                clip_processor=self.clip_processor,
                clip_model=self.clip_model,
                grid_resolution=room_grid_resolution,
                graph_tmp_folder=debug_folder,
                rgb_list=img_list,
                pose_list=pose_list,
                frameidx_list=frame_list,
                save_intermediate_results=True,
                device=device,
            )
            floor.rooms = detected_rooms

        frame_id_to_rooms = {}
        for floor in self.floors:
            for i, room in enumerate(floor.rooms):
                room_type = room.infer_room_type_from_view_embedding(
                    self.room_types, self.clip_tokenizer, self.clip_model
                )
                room.room_type = room_type
                print("room id", room.room_id, "is", room_type)
                self.rooms[room.room_id] = room

        poses_xyz = [p.detach().numpy()[:3, 3] for p in pose_list]
        room_ids = assign_frames_to_rooms(poses_xyz, self.geometry_map, self.floors)
        for i, frame_id in enumerate(frame_list):
            frame_id_to_rooms[frame_id] = room_ids[i]

        for frame_id in frame_list:
            log = self.navigation_log[frame_id]
            assert int(log["Frame Index"]) == frame_id
            if frame_id_to_rooms.get(frame_id):
                log["Generic Mapping"]["Present Room"] = frame_id_to_rooms[frame_id]
            else:
                log["Generic Mapping"]["Present Room"] = None

        self.tracks = assign_tracks_to_rooms(
            self.tracks, self.geometry_map, self.floors
        )

    def get_children_ids(self, id, hierarchy_matrix):
        children_ids = [
            key for key, value in hierarchy_matrix[id].items() if value >= 1
        ]
        return children_ids

    def compute_node_levels(self, hierarchy_matrix=None, hierarchy_type_matrix=None):
        if not (hierarchy_matrix is None):
            self.hierarchy_matrix = hierarchy_matrix
        if not (hierarchy_type_matrix is None):
            self.hierarchy_type_matrix = hierarchy_type_matrix
        # Re-compute all levels
        for id in self.tracks:
            self.tracks[id].level = None
        unknown_levels = set(self.track_ids)
        known_levels = set()
        level = 0
        any_update = False

        while len(unknown_levels) > 0:
            if level == 0:
                for id in unknown_levels:
                    children_ids = self.get_children_ids(id, self.hierarchy_matrix)
                    if len(children_ids) == 0:
                        self.tracks[id].level = level
                        known_levels.add(id)
                        any_update = True
            else:
                any_update = False
                for id in unknown_levels:
                    children_ids = self.get_children_ids(id, self.hierarchy_matrix)
                    children_levels = [self.tracks[id].level for id in children_ids]
                    if None in children_levels:
                        continue
                    else:
                        self.tracks[id].level = level
                        known_levels.add(id)
                        any_update = True

            if any_update:
                unknown_levels = unknown_levels - known_levels
                level += 1
            else:
                # There is a circular dependence
                for id in unknown_levels:
                    self.tracks[id].level = level
                    known_levels.add(id)
                unknown_levels = unknown_levels - known_levels

    def save(self, save_dir, hierarchy_matrix=None, hierarchy_type_matrix=None):
        if not (hierarchy_matrix is None):
            self.hierarchy_matrix = hierarchy_matrix
        if not (hierarchy_type_matrix is None):
            self.hierarchy_type_matrix = hierarchy_type_matrix
        os.makedirs(save_dir, exist_ok=True)
        # Save Navigation Log
        navigation_log_path = os.path.join(save_dir, "navigation_log.json")
        with open(navigation_log_path, "w") as f:
            json.dump(self.navigation_log, f, indent=2)

        # Save Semantic Tree
        with open(save_dir / "hierarchy_matrix.json", "w") as f:
            json.dump(self.hierarchy_matrix, f)
        with open(save_dir / "hierarchy_type_matrix.json", "w") as f:
            json.dump(self.hierarchy_type_matrix, f)
        # np.savetxt(save_dir / "neighbours.txt", neighbour_matrix, fmt="%.4f")
        np.savetxt(save_dir / "track_ids.txt", np.array(self.track_ids, dtype=np.int32))

        rooms_folder = os.path.join(save_dir, "rooms")
        floors_folder = os.path.join(save_dir, "floors")
        shutil.rmtree(rooms_folder, ignore_errors=True)
        shutil.rmtree(floors_folder, ignore_errors=True)
        os.makedirs(rooms_folder, exist_ok=True)
        os.makedirs(floors_folder, exist_ok=True)
        for floor in self.floors:
            floor.save(floors_folder)
        for room in self.rooms.values():
            room.save(rooms_folder)

    def visualize_2d(
        self,
        visualizer2d: Visualizer2D,
        folder,
        hierarchy_matrix,
        hierarchy_type_matrix,
    ):
        labels = [self.tracks[id].label for id in self.track_ids]
        levels = [self.tracks[id].level for id in self.track_ids]
        ids = [self.tracks[id].id for id in self.track_ids]
        visualizer2d.visualize(
            folder,
            ids,
            labels,
            levels,
            hierarchy_matrix,
            hierarchy_type_matrix=hierarchy_type_matrix,
        )

    def visualize_3d(self, visualizer3d: Visualizer3D):

        visualizer3d.add_geometry_map(
            self.geometry_map.points, self.geometry_map.colors
        )

        # Compute visualization fields
        for idx in self.tracks.keys():
            self.tracks[idx].compute_vis_centroid(self.tracks[idx].level)

        # Visualize tracks
        for id in self.tracks.keys():
            children_ids = self.get_children_ids(id, self.hierarchy_matrix)
            visualizer3d.add_semanticnode(
                id=self.tracks[id].id,
                label=self.tracks[id].label,
                caption=self.tracks[id].captions[-1],
                location=self.tracks[id].vis_centroid,
                level=self.tracks[id].level,
                children_locs=[self.tracks[c].vis_centroid for c in children_ids],
                children_labels=[self.tracks[c].label for c in children_ids],
                points_xyz=self.tracks[id].compute_local_pcd().points,
                points_colors=self.tracks[id].compute_local_pcd().colors,
            )
