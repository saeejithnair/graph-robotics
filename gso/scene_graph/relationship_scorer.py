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
from .scene_graph import get_nodeid_by_name
from .utils import get_crop


class HierarchyExtractor:
    '''
    Extracts hierarchy from a set of scene graph nodes and edges.
    Parent child relationships are computed based on the direction of the edges. 
    E.g. 
        if 'cup' is 'on' the 'table', then 'cup' is a child of 'table'  
        if 'apple' is 'in' the 'fridge', then 'apple' is a child of 'fridge' 
        etc. 
    '''
    def __init__(self, downsample_voxel_size) -> None:
        self.downsample_voxel_size = downsample_voxel_size

    def infer_hierarchy(self, scene_graph):
        track_ids = scene_graph.get_node_ids()
        hierarchy_matrix = {}
        hierarchy_type_matrix = {}
        for i in scene_graph.get_node_ids():
            hierarchy_matrix[i] = {}
            hierarchy_type_matrix[i] = {}
            for j in scene_graph.get_node_ids():
                hierarchy_matrix[i][j] = 0
                hierarchy_type_matrix[i][j] = ""
        for subject_id in track_ids:
            for edge in scene_graph[subject_id].edges:
                related_object = edge.related_object
                related_object_id = get_nodeid_by_name(scene_graph, related_object)
                assert not (related_object_id is None)
                assert not (subject_id is None)
                # Parent object = related_object_id, Child object = subject_id
                hierarchy_matrix[related_object_id][subject_id] += 1
                hierarchy_type_matrix[related_object_id][subject_id] = edge.type
        return hierarchy_matrix, hierarchy_type_matrix

class HierarchyExtractorHeuristics(HierarchyExtractor):
    '''
    This class is not used anywhere.
    It also calculates the hierarchy, but extracting them from heuristics on the point clouds of the objects.
    '''
    def infer_hierarchy_heuristics(
        self,
        current_tracks,
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
            np.asarray(trk.compute_local_pcd().points, dtype=np.float32)
            for trk in track_values
        ]
        indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in track_pcds]
        for index, arr in zip(indices, track_pcds):
            index.add(arr)

        track_top_pcds = [
            pointcloud.get_top_points(
                np.asarray(trk.compute_local_pcd().points, dtype=np.float32)
            )
            for trk in track_values
        ]
        track_bottom_pcds = [
            pointcloud.get_bottom_points(
                np.asarray(trk.compute_local_pcd().points, dtype=np.float32)
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

    def compute_containedin_matrix(self, tracks):
        downsample_voxel_size = self.downsample_voxel_size
        track_values = list(tracks.values())

        # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
        track_pcds = [
            np.asarray(trk.compute_local_pcd().points, dtype=np.float32)
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

    def compute_ontop_matrix(self, tracks):
        downsample_voxel_size = self.downsample_voxel_size
        track_values = list(tracks.values())

        # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
        track_top_pcds = [
            pointcloud.get_top_points(
                np.asarray(trk.compute_local_pcd().points)
            )
            for trk in track_values
        ]
        track_bottom_pcds = [
            pointcloud.get_bottom_points(
                np.asarray(trk.compute_local_pcd().points)
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

