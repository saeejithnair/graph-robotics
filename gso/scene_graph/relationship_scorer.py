import os

import faiss
import numpy as np
import supervision as sv
import torch
from PIL import Image
from scipy.spatial import ConvexHull
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from . import pointcloud
from .detection import DetectionList, Features
from .track import get_trackid_by_name
from .utils import get_crop


class RelationshipScorer:
    def __init__(self, downsample_voxel_size) -> None:
        self.downsample_voxel_size = downsample_voxel_size

    # def compute_contained_within(self, child_points: np.ndarray, parent_points: np.ndarray) -> bool:
    #     # Create convex hull of parent
    #     try:
    #         parent_hull = ConvexHull(parent_points)
    #     except Exception:
    #         return False

    #     # Check what fraction of child points lie within parent's convex hull
    #     points_inside = 0
    #     for point in child_points:
    #         # Use point-in-hull test
    #         if self._point_in_hull(point, parent_points[parent_hull.vertices]):
    #             points_inside += 1

    #     containment_ratio = points_inside / len(child_points)
    #     return containment_ratio > self.threshold_containment

    # def compute_contained_within(self, child_points: np.ndarray, parent_points: np.ndarray) -> bool:
    #     # Create convex hull of parent
    #     try:
    #         parent_hull = ConvexHull(parent_points)
    #     except Exception:
    #         return False

    #     # Check what fraction of child points lie within parent's convex hull
    #     points_inside = 0
    #     for point in child_points:
    #         # Use point-in-hull test
    #         if self._point_in_hull(point, parent_points[parent_hull.vertices]):
    #             points_inside += 1

    #     containment_ratio = points_inside / len(child_points)
    #     return containment_ratio > self.threshold_containment

    # def compute_supported_by(self, child_points: np.ndarray, parent_points: np.ndarray) -> bool:
    #     """
    #     Check if child point cloud is supported by parent point cloud

    #     Args:
    #         child_points: Nx3 array of potential child point cloud
    #         parent_points: Mx3 array of potential parent point cloud

    #     Returns:
    #         bool: True if child is likely supported by parent
    #     """
    #     # Find the bottom points of the child (lowest 10% in z-axis)
    #     child_bottom = pointcloud.get_bottom_points(child_points)

    #     # Find the top points of the parent (highest 10% in z-axis)
    #     parent_top = pointcloud.get_top_points(parent_points)

    #     # Count contact points
    #     contact_points = 0
    #     for child_point in child_bottom:
    #         for parent_point in parent_top:
    #             if np.linalg.norm(child_point - parent_point) < self.contact_threshold:
    #                 contact_points += 1
    #                 break

    #     contact_ratio = contact_points / len(child_bottom)
    #     return contact_ratio > self.threshold_support
    # def _point_in_hull(self, point, hull_points):
    #     """Helper method to check if a point lies within a convex hull"""
    #     hull = ConvexHull(hull_points)
    #     new_points = np.vstack((hull_points, point))
    #     new_hull = ConvexHull(new_points)
    #     return np.array_equal(hull.vertices, new_hull.vertices)

    def find_neighbours(self, current_tracks, distance_threshold=1):
        track_ids = list(current_tracks.keys())
        track_values = list(current_tracks.values())
        n_tracks = len(track_ids)
        neighbours = np.zeros((n_tracks, n_tracks))
        for i in range(n_tracks):
            for j in range(i + 1, n_tracks):
                box_i = track_values[i].features.bbox_3d
                box_j = track_values[j].features.bbox_3d

                # iou = pointcloud.compute_3d_iou(box_i, box_j)
                distance = pointcloud.compute_3d_bbox_distance(box_i, box_j)

                neighbours[i][j] = neighbours[j][i] = int(distance < distance_threshold)
        return neighbours

    def infer_hierarchy_vlm(self, current_tracks):
        track_ids = list(current_tracks.keys())
        n_tracks = len(track_ids)
        hierarchy_matrix = {}
        hierarchy_type_matrix = {}
        for i in current_tracks.keys():
            hierarchy_matrix[i] = {}
            hierarchy_type_matrix[i] = {}
            for j in current_tracks.keys():
                hierarchy_matrix[i][j] = 0
                hierarchy_type_matrix[i][j] = ""
        for subject_id in track_ids:
            for edge in current_tracks[subject_id].edges:
                related_object = edge.related_object
                related_object_id = get_trackid_by_name(current_tracks, related_object)
                print(
                    "Debug: subject_id",
                    subject_id,
                    "related_object_id",
                    related_object_id,
                    "edge.type",
                    edge.type,
                )
                assert not (related_object_id is None)
                assert not (subject_id == edge.subject)
                # Parent object = related_object_id, Child object = subject_id
                hierarchy_matrix[related_object_id][subject_id] += 1
                hierarchy_type_matrix[related_object_id][subject_id] = edge.type
        return hierarchy_matrix, hierarchy_type_matrix

    def infer_hierarchy_heuristics(
        self,
        current_tracks,
        full_pcd,
        neighbour_thresh=1,
        downsample_voxel_size=0.02,
        in_threshold=0.8,
        on_threshold=0.8,
    ):
        track_ids = list(current_tracks.keys())
        track_values = list(current_tracks.values())
        n_tracks = len(track_ids)

        neighbour_matrix = self.find_neighbours(current_tracks, neighbour_thresh)
        in_matrix = np.zeros((n_tracks, n_tracks))
        on_matrix = np.zeros((n_tracks, n_tracks))
        hierarchy_type_matrix = [["" for i in range(n_tracks)] for i in range(n_tracks)]

        # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
        track_pcds = [
            np.asarray(trk.compute_local_pcd(full_pcd).points, dtype=np.float32)
            for trk in track_values
        ]
        indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in track_pcds]
        for index, arr in zip(indices, track_pcds):
            index.add(arr)

        track_top_pcds = [
            pointcloud.get_top_points(
                np.asarray(trk.compute_local_pcd(full_pcd).points, dtype=np.float32)
            )
            for trk in track_values
        ]
        track_bottom_pcds = [
            pointcloud.get_bottom_points(
                np.asarray(trk.compute_local_pcd(full_pcd).points, dtype=np.float32)
            )
            for trk in track_values
        ]
        top_indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in track_top_pcds]
        for index, arr in zip(top_indices, track_top_pcds):
            index.add(arr)

        for i in range(n_tracks):
            for j in range(n_tracks):
                if not neighbour_matrix[i][j]:
                    continue

                # Probability track_j is in track_i
                D, I = indices[i].search(track_pcds[j], 1)
                overlap = (
                    D < downsample_voxel_size**2
                ).sum()  # D is the squared distance
                in_matrix[i, j] = overlap / len(track_pcds[j])

                # Probability track_j is on track_i
                D, I = top_indices[i].search(track_bottom_pcds[j], 1)
                overlap = (
                    D < downsample_voxel_size**2
                ).sum()  # D is the squared distance
                on_matrix[i, j] = overlap / len(track_bottom_pcds[j])

        in_matrix = np.where(in_matrix > in_threshold, in_matrix, 0)
        on_matrix = np.where(on_matrix > on_threshold, on_matrix, 0)
        for i in range(n_tracks):
            for j in range(n_tracks):
                if in_matrix[i][j]:
                    hierarchy_type_matrix[i][j] = "inside"
                if on_matrix[i][j]:
                    hierarchy_type_matrix[i][j] = "on"
        hierarchy_matrix = np.logical_or(on_matrix, in_matrix)
        return hierarchy_matrix, hierarchy_type_matrix

    # def infer_hierarchy_relationships(self, current_tracks, full_pcd):
    #     ontop_matrix = self.compute_ontop_matrix(current_tracks, full_pcd)
    #     containedin_matrix = self.compute_containedin_matrix(current_tracks, full_pcd)

    #     ontop_matrix = np.where(ontop_matrix > self.threshold_support, 1, 0)
    #     containedin_matrix = np.where(containedin_matrix > self.threshold_containment, 1, 0)

    #     parent_matrix = np.logical_or(ontop_matrix, containedin_matrix)

    #     parent_prob = np.max(parent_matrix, 1)
    #     parent_indxs = np.argmax(parent_matrix, 1)
    #     parent_ids = [current_tracks[idx].id for idx in parent_indxs]

    #     track_to_parent = {}
    #     i = 0
    #     for id in current_tracks.keys():
    #         if parent_prob[i] < 0.5:
    #             track_to_parent[id] = None
    #         else:
    #             track_to_parent[id] = parent_ids[i]
    #         i += 1
    #     return track_to_parent

    def compute_containedin_matrix(self, tracks, full_pcd):
        downsample_voxel_size = self.downsample_voxel_size
        track_values = list(tracks.values())

        # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
        track_pcds = [
            np.asarray(trk.compute_local_pcd(full_pcd).points, dtype=np.float32)
            for trk in track_values
        ]
        indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in track_pcds]

        # Add the points from the numpy arrays to the corresponding FAISS indices
        for index, arr in zip(indices, track_pcds):
            index.add(arr)

        overlap_matrix = np.zeros((len(track_values), len(track_values)))
        # Compute the pairwise overlaps
        for i in range(len(track_values)):
            for j in range(len(track_values)):
                if i == j:
                    continue
                box_i = track_values[i].features.bbox_3d
                box_j = track_values[j].features.bbox_3d

                # Skip if the boxes do not overlap at all (saves computation)
                iou = pointcloud.compute_3d_iou(box_i, box_j)
                if iou == 0:
                    continue

                # # Use range_search to find points within the threshold
                # _, I = indices[j].range_search(point_arrays[i], threshold ** 2)
                D, I = indices[j].search(track_pcds[i], 1)

                # # If any points are found within the threshold, increase overlap count
                # overlap += sum([len(i) for i in I])

                overlap = (
                    D < downsample_voxel_size**2
                ).sum()  # D is the squared distance

                # Calculate the ratio of points within the threshold
                overlap_matrix[i, j] = overlap / len(track_pcds[i])

        return overlap_matrix

    def compute_ontop_matrix(self, tracks, full_pcd):
        downsample_voxel_size = self.downsample_voxel_size
        track_values = list(tracks.values())

        # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
        track_top_pcds = [
            pointcloud.get_top_points(
                np.asarray(trk.compute_local_pcd(full_pcd).points)
            )
            for trk in track_values
        ]
        track_bottom_pcds = [
            pointcloud.get_bottom_points(
                np.asarray(trk.compute_local_pcd(full_pcd).points)
            )
            for trk in track_values
        ]

        indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in track_top_pcds]

        # Add the points from the numpy arrays to the corresponding FAISS indices
        for index, arr in zip(indices, track_top_pcds):
            index.add(arr)

        overlap_matrix = np.zeros((len(track_bottom_pcds), len(track_top_pcds)))
        # Compute the pairwise overlaps
        for i in range(len(track_values)):
            for j in range(len(track_values)):
                if i == j:
                    continue
                box_i = track_values[i].features.bbox_3d
                box_j = track_values[j].features.bbox_3d

                # Skip if the boxes do not overlap at all (saves computation)
                iou = pointcloud.compute_3d_iou(box_i, box_j)
                if iou == 0:
                    continue

                # # Use range_search to find points within the threshold
                # _, I = indices[j].range_search(point_arrays[i], threshold ** 2)
                D, I = indices[j].search(track_bottom_pcds[i], 1)

                # # If any points are found within the threshold, increase overlap count
                # overlap += sum([len(i) for i in I])

                overlap = (
                    D < downsample_voxel_size**2
                ).sum()  # D is the squared distance

                # Calculate the ratio of points within the threshold
                overlap_matrix[i, j] = overlap / len(track_bottom_pcds[i])

        return overlap_matrix


if __name__ == "__main__":
    from perception import GenericMapper

    from scene_graph.semantic_tree import SemanticTree

    from .detection import load_objects

    scene_graph = SemanticTree()
    scene_graph.load_pcd(folder="/pub3/qasim/hm3d/data/ham-sg/000-hm3d-BFRyYbPCCPE")

    perceptor = GenericMapper()
    similarity = FeatureComputer()

    img1, objects1 = load_objects(
        "/pub3/qasim/hm3d/data/ham-sg/000-hm3d-BFRyYbPCCPE/detections", 0
    )
    # _, objects1_clip = similarity.compute_clip_features(
    #     perceptor.img, image_crops=objects1['crops']
    #     )
    # objects1_clip = similarity.compute_clip_features_avg(
    #     img1, masks=objects1.get_field("mask")
    # )
    similarity.compute_features(img1, scene_graph.geometry_map, objects1)
    img2, objects2 = load_objects(
        "/pub3/qasim/hm3d/data/ham-sg/000-hm3d-BFRyYbPCCPE/detections", 1
    )
    # _, objects2_clip = similarity.compute_clip_features(
    #     perceptor.img, image_crops=objects2['crops']
    #     )
    # objects2_clip = similarity.compute_clip_features_avg(
    #     img2, masks=objects2.get_field("mask")
    # )
    similarity.compute_features(img2, scene_graph.geometry_map, objects2)

    print("Clip Similarities:")
    for i in range(len(objects1)):
        for j in range(len(objects2)):
            print(
                objects1[i].label,
                "-",
                objects2[j].label,
                np.dot(
                    objects1[i].features.visual_emb, objects2[j].features.visual_emb
                ),
                np.dot(
                    objects1[i].features.caption_emb, objects2[j].features.caption_emb
                ),
                np.linalg.norm(
                    objects1[i].features.centroid - objects2[j].features.centroid
                ),
            )
