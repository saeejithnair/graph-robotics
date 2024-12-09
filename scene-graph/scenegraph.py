import copy
import json
import os

import networkx as nx
import numpy as np
import open3d as o3d
import torch

from pointcloud import dynamic_downsample, find_nearest_points, pcd_denoise_dbscan
from track import object_to_track
from visualizer import Visualizer2D, Visualizer3D


class SceneGraph:
    def __init__(self, device="cuda"):
        self.geometry_map = o3d.geometry.PointCloud()
        self.device = device

        self.in_matrix = None
        self.on_matrix = None
        self.hierarchy_matrix = None
        self.track_ids = []

        self.tracks = {}
        self.navigation_log = []

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

    def integrate_navigation_log(self, llm_response, frame_idx):
        log = {"Frame Index": frame_idx}
        log.update(copy.deepcopy(llm_response))
        for i in range(len(log["DetectionList"])):
            log["DetectionList"][i].pop("bbox")
        log["Detections"] = log["DetectionList"]
        del log["DetectionList"]
        self.navigation_log.append(log)

    def integrate_detections(self, detections, det_matched, matched_tracks, frame_idx):
        # Either merge each detection with its matched track, or initialize the detection as a new track
        track_ids = list(self.tracks.keys())
        self.track_ids = track_ids
        for i in range(len(detections)):
            if det_matched[i]:
                self.tracks[track_ids[matched_tracks[i]]].merge_detection(
                    detections[i], frame_idx
                )
            else:
                new_track = object_to_track(detections[i], frame_idx, self.geometry_map)
                assert not new_track.id in self.tracks.keys()
                self.tracks[new_track.id] = new_track
                self.track_ids = list(self.tracks.keys())

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

    # def process_hierarchy(self, track_to_parent):
    #     # Clear all the children relationships
    #     for id in self.children.keys():
    #         self.children[id] = []

    #     # Update the children relationships using track_to_parent
    #     for id in track_to_parent.keys():
    #         if track_to_parent[id]:
    #             self.children[track_to_parent[id]].append(id)

    #     # Clear all track levels
    #     for track in self.tracks.values():
    #         track.level = 0
    #     # Re-compute all levels
    #     unknown_levels = set(self.tracks.keys())
    #     known_levels = set()
    #     level = 0
    #     while len(unknown_levels) > 0:
    #         if level == 0:
    #             for id in unknown_levels:
    #                 if len(self.children[id]) == 0:
    #                     self.tracks[id].level = level
    #                     known_levels.add(id)
    #         else:
    #             for id in unknown_levels:
    #                 children = self.children[id]
    #                 children_levels = [self.tracks[i].level for i in children]
    #                 if None in children_levels:
    #                     continue
    #                 else:
    #                     self.tracks[id].level = level
    #                     known_levels.add(id)
    #         unknown_levels = unknown_levels - known_levels
    #         level += 1


class Graph:
    def __init__(self, node_ids=[]) -> None:
        self.node_ids = node_ids
        self.n_nodes = len(node_ids)
        self.adjacency_matrix = np.zeros((self.n_nodes, self.n_nodes), np.int32)
        self.edge_weights = np.zeros((self.n_nodes, self.n_nodes), np.float32)

    def add_node(self, node_id):
        self.node_ids.append(node_id)
        self.adjacency_matrix = np.concatenate(
            (
                np.concatenate(
                    (
                        self.adjacency_matrix,
                        np.zeros((self.adjacency_matrix.shape[0], 1)),
                    ),
                    axis=1,
                    dtype=np.int32,
                ),
                np.zeros((1, self.adjacency_matrix.shape[1] + 1)),
            ),
            axis=0,
        )
        self.edge_weights = np.concatenate(
            (
                np.concatenate(
                    (self.edge_weights, np.zeros((self.edge_weights.shape[0], 1))),
                    axis=1,
                ),
                np.zeros((1, self.edge_weights.shape[1] + 1)),
            ),
            axis=0,
            dtype=np.float32,
        )
        # self.adjacency_matrix = np.concatenate((self.adjacency_matrix, np.zeros((self.n_nodes, 1))), 0)
        # self.adjacency_matrix = np.concatenate((self.adjacency_matrix, np.zeros((self.n_nodes+1,))), 1)
        # self.edge_weights = np.concatenate((self.edge_weights, np.zeros((self.n_nodes, 1))), 0)
        # self.edge_weights = np.concatenate((self.edge_weights, np.zeros((self.n_nodes+1,))), 1)
        self.n_nodes += 1

    def remove_node(self, node_id):
        # Check if i is within valid range
        assert node_id in self.node_ids
        i = self.node_ids.index(node_id)
        self.adjacency_matrix = np.delete(self.adjacency_matrix, i, axis=0)
        self.adjacency_matrix = np.delete(self.adjacency_matrix, i, axis=1)
        self.edge_weights = np.delete(self.edge_weights, i, axis=0)
        self.edge_weights = np.delete(self.edge_weights, i, axis=1)
        self.node_ids.remove(node_id)
        self.n_nodes -= 1

    def set_edge_weight(self, node_id1, node_id2, weight):
        assert node_id1 in self.node_ids
        assert node_id2 in self.node_ids
        i1 = self.node_ids.index(node_id1)
        i2 = self.node_ids.index(node_id2)
        self.edge_weights[i1][i2] = weight

    def set_adjacency(self, node_id1, node_id2, val):
        assert node_id1 in self.node_ids
        assert node_id2 in self.node_ids
        i1 = self.node_ids.index(node_id1)
        i2 = self.node_ids.index(node_id2)
        self.adjacency_matrix[i1][i2] = val
