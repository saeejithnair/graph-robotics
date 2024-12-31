import copy
import csv
import json
import os
import pickle

import numpy as np
import open3d as o3d
import torch

from .detection import DetectionList, Edge
from .pointcloud import pcd_denoise_dbscan
from .track import get_trackid_by_name, load_tracks, object_to_track
from .visualizer import Visualizer2D, Visualizer3D


class SemanticTree:
    def __init__(self, device="cuda"):
        self.geometry_map = o3d.geometry.PointCloud()
        self.device = device
        self.hierarchy_matrix = None
        self.hierarchy_type_matrix = None
        self.track_ids = []

        self.tracks = {}
        self.navigation_log = []

    def load(self, save_dir):
        navigation_log_path = os.path.join(
            save_dir, "semantic_tree/navigation_log.json"
        )
        with open(navigation_log_path) as f:
            self.navigation_log = json.load(f)

        with open(save_dir / "semantic_tree/hierarchy_matrix.json") as f:
            json_data = json.load(f)
            hierarchy_matrix = {}
            for i in json_data.keys():
                hierarchy_matrix[int(i)] = {}
                for j in json_data[i].keys():
                    hierarchy_matrix[int(i)][int(j)] = json_data[i][j]
            self.hierarchy_matrix = hierarchy_matrix
        with open(save_dir / "semantic_tree/hierarchy_type_matrix.json") as f:
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
        self.track_ids = np.loadtxt(
            save_dir / "semantic_tree/track_ids.txt", np.float32
        )
        self.tracks = load_tracks(save_dir / "tracks")
        self.load_pcd(folder=save_dir)

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
        o3d.io.write_point_cloud(path, self.geometry_map)

    def load_pcd(self, path=None, folder=None, file="full_scene.pcd"):
        if not path:
            path = os.path.join(folder, file)
        self.geometry_map = o3d.io.read_point_cloud(path)

    def get_tracks(self):
        return list(self.tracks.values())

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
            self.tracks[idx].compute_vis_centroid(
                self.geometry_map, self.tracks[idx].level
            )

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
                points=self.tracks[id].get_points_at_idxs(self.geometry_map),
            )

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

        edges_in_frame = []
        for det in detections:
            edges_in_frame += [edge for edge in det.edges]

        self.navigation_log[i]["Generic Mapping"] = {
            "Relative Motion": llm_response["Relative Motion"],
            "Current Location": llm_response["Current Location"],
            "View": llm_response["View"],
            "Novelty": llm_response["Novelty"],
            "Detections": [d.matched_track_name for d in detections],
            "Edges": [edge.json() for edge in edges_in_frame],
        }

    def integrate_refinement_log(self, request, llm_response, keyframe_id):
        refinement_log = llm_response
        i = self.get_navigation_log_idx(keyframe_id)
        self.navigation_log[i]["Focused Analyses and Search"].append(refinement_log)

    def integrate_detections(
        self, detections: DetectionList, is_matched, matched_trackidx, frame_idx
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
        for i in range(len(detections)):
            det = detections[i]
            for edge in det.edges:
                subject_trkname = detlabel2trkname[edge.subject]
                object_trkname = detlabel2trkname[edge.related_object]
                subject_trkid = get_trackid_by_name(self.tracks, subject_trkname)

                self.tracks[subject_trkid].edges.append(
                    Edge(edge.type, subject_trkname, object_trkname, keyframe=frame_idx)
                )

        return detections

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

    def save(self, save_dir, hierarchy_matrix, hierarchy_type_matrix):
        os.makedirs(save_dir, exist_ok=True)
        # Save Navigation Log
        navigation_log_path = os.path.join(save_dir, "navigation_log.json")
        with open(navigation_log_path, "w") as f:
            json.dump(self.navigation_log, f, indent=2)

        # Save Semantic Tree
        with open(save_dir / "hierarchy_matrix.json", "w") as f:
            json.dump(hierarchy_matrix, f)
        with open(save_dir / "hierarchy_type_matrix.json", "w") as f:
            json.dump(hierarchy_type_matrix, f)
        # np.savetxt(save_dir / "neighbours.txt", neighbour_matrix, fmt="%.4f")
        np.savetxt(save_dir / "track_ids.txt", np.array(self.track_ids, dtype=np.int32))
