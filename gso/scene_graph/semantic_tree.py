import copy
import json
import os

import networkx as nx
import numpy as np
import open3d as o3d
import torch

from .object import ObjectList
from .pointcloud import dynamic_downsample, find_nearest_points, pcd_denoise_dbscan
from .track import load_tracks, object_to_track
from .visualizer import Visualizer2D, Visualizer3D


class SemanticTree:
    def __init__(self, device="cuda"):
        self.geometry_map = o3d.geometry.PointCloud()
        self.device = device

        self.in_matrix = None
        self.on_matrix = None
        self.hierarchy_matrix = None
        self.track_ids = []

        self.tracks = {}
        self.navigation_log = []

    def load(self, save_dir):
        navigation_log_path = os.path.join(
            save_dir, "semantic_tree/navigation_log.json"
        )
        with open(navigation_log_path) as f:
            self.navigation_log = json.load(f)

        self.hierarchy_matrix = np.loadtxt(
            save_dir / "semantic_tree/hierarchy.txt", np.float32
        )
        self.neighbour_matrix = np.loadtxt(
            save_dir / "semantic_tree/neighbours.txt", np.float32
        )
        self.in_matrix = np.loadtxt(save_dir / "semantic_tree/in.txt", np.float32)
        self.on_matrix = np.loadtxt(save_dir / "semantic_tree/on.txt", np.float32)
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
        in_matrix,
        on_matrix,
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
            in_matrix,
            on_matrix,
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
            children_ids = self._get_children_ids(id)
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
        self.navigation_log[i]["Generic Mapping"] = {
            "Relative Motion": llm_response["Relative Motion"],
            "Current Location": llm_response["Current Location"],
            "View": llm_response["View"],
            "Novelty": llm_response["Novelty"],
            "Detections": [d.matched_track_name for d in detections],
        }

    def integrate_refinement_log(self, request, llm_response, keyframe_id):
        refinement_log = llm_response
        i = self.get_navigation_log_idx(keyframe_id)
        self.navigation_log[i]["Focused Analyses and Search"].append(refinement_log)

    def integrate_detections(
        self, detections: ObjectList, is_matched, matched_trackidx, frame_idx
    ):
        # Either merge each detection with its matched track, or initialize the detection as a new track
        track_ids = list(self.tracks.keys())
        self.track_ids = track_ids
        for i in range(len(detections)):
            if is_matched[i]:
                self.tracks[track_ids[matched_trackidx[i]]].merge_detection(
                    detections[i], frame_idx
                )
                matched_track_name = self.tracks[track_ids[matched_trackidx[i]]].name
                detections[i].matched_track_name = matched_track_name
            else:
                new_track = object_to_track(detections[i], frame_idx, self.geometry_map)
                assert not new_track.id in self.tracks.keys()
                self.tracks[new_track.id] = new_track
                self.track_ids = list(self.tracks.keys())
                detections[i].matched_track_name = new_track.name

    def _get_children_ids(self, id):
        index = self.track_ids.index(id)
        children = np.where(self.hierarchy_matrix[:, index])[0]
        children_ids = [self.track_ids[i] for i in children]
        return children_ids

    def process_hierarchy(
        self, neighbour_matrix, in_matrix, on_matrix, hierarchy_matrix
    ):
        self.neighbour_matrix = neighbour_matrix
        self.on_matrix = on_matrix
        self.in_matrix = in_matrix
        self.hierarchy_matrix = hierarchy_matrix

        # Re-compute all levels
        unknown_levels = set(self.track_ids)
        known_levels = set()
        level = 0
        any_update = False

        while len(unknown_levels) > 0:
            if level == 0:
                for id in unknown_levels:
                    children_ids = self._get_children_ids(id)
                    if len(children_ids) == 0:
                        self.tracks[id].level = level
                        known_levels.add(id)
            else:
                any_update = False
                for id in unknown_levels:
                    children_ids = self._get_children_ids(id)
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

    def save(self, save_dir, neighbour_matrix, in_matrix, on_matrix, hierarchy_matrix):
        os.makedirs(save_dir, exist_ok=True)
        # Save Navigation Log
        navigation_log_path = os.path.join(save_dir, "navigation_log.json")
        with open(navigation_log_path, "w") as f:
            json.dump(self.navigation_log, f, indent=4)

        # Save Semantic Tree
        np.savetxt(save_dir / "hierarchy.txt", hierarchy_matrix, fmt="%.4f")
        np.savetxt(save_dir / "neighbours.txt", neighbour_matrix, fmt="%.4f")
        np.savetxt(save_dir / "in.txt", in_matrix, fmt="%.4f")
        np.savetxt(save_dir / "on.txt", on_matrix, fmt="%.4f")
        np.savetxt(save_dir / "track_ids.txt", np.array(self.track_ids, dtype=np.int32))
