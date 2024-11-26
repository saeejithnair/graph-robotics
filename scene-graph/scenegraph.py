import torch
import numpy as np
import open3d as o3d
from pointcloud import pcd_denoise_dbscan, dynamic_downsample, find_nearest_points
import os
from track import object_to_track

class SceneGraph:
    def __init__(self, device='cuda'):
        self.geometry_map = o3d.geometry.PointCloud()
        self.device = device 
        self.tracks = {}
        self.children = {}
    def add_pcd(self, pcd):
        self.geometry_map += pcd
    def denoise_geometrymap(self, downsample_voxel_size, dbscan_eps, min_points):
        self.geometry_map = self.geometry_map.voxel_down_sample(
            voxel_size=downsample_voxel_size
        )
        self.geometry_map = pcd_denoise_dbscan(self.geometry_map, eps=dbscan_eps, min_points=min_points)
    def save_pcd(self, path=None, folder=None, file='full_scene.pcd'):
        if not path:
            path = os.path.join(folder, file)
        o3d.io.write_point_cloud(path, self.geometry_map)
    def load_pcd(self, path=None, folder=None, file='full_scene.pcd'):
        if not path:
            path = os.path.join(folder, file)
        self.geometry_map = o3d.io.read_point_cloud(path)
    def get_tracks(self):
        return list(self.tracks.values())
    def visualize_graph(self, visualizer):
        visualizer.add_geometry_map(self.geometry_map.points, self.geometry_map.colors)
        
        # Compute visualization fields
        for idx in self.tracks.keys():
            self.tracks[idx].compute_vis_centroid(
                self.geometry_map,
                self.tracks[idx].level
            )
            
        # Visualize tracks
        for idx in self.tracks.keys():
            visualizer.add_semanticnode(
                id=self.tracks[idx].id,
                label=self.tracks[idx].label,
                caption=self.tracks[idx].captions[-1],
                location=self.tracks[idx].vis_centroid,
                level = self.tracks[idx].level,
                children_locs=[self.tracks[c].vis_centroid for c in self.children[idx]],
                children_labels=[self.tracks[c].label for c in self.children[idx]],
                points=self.tracks[idx].get_points_at_idxs(self.geometry_map),
                )
    def process_detections(self, detections, det_matched, matched_tracks, frame_idx):
        # Merge the matches, or add them to the list of matches
        track_ids = list(self.tracks.keys())
        for i in range(len(detections)):
            if det_matched[i]:
                self.tracks[track_ids[matched_tracks[i]]].merge_detection(detections[i], frame_idx)
            else:
                new_track = object_to_track(detections[i], frame_idx, self.geometry_map)
                assert not new_track.id in self.tracks.keys()
                self.tracks[new_track.id] = new_track
                self.children[new_track.id] = []
    
    def process_hierarchy(self, track_to_parent):
        # Clear all the children relationships
        for id in self.children.keys():
            self.children[id] = []
        
        # Update the children relationships using track_to_parent
        for id in track_to_parent.keys():
            if track_to_parent[id]:
                self.children[track_to_parent[id]].append(id)
            
        # Clear all track levels
        for track in self.tracks.values():
            track.level = 0
        # Re-compute all levels
        unknown_levels = set(self.tracks.keys())
        known_levels = set()
        level = 0
        while len(unknown_levels) > 0:
            if level == 0:
                for id in unknown_levels:
                    if len(self.children[id]) == 0:
                        self.tracks[id].level = level
                        known_levels.add(id)
            else:
                for id in unknown_levels:
                    children = self.children[id]
                    children_levels = [self.tracks[i].level for i in children]
                    if None in children_levels:
                        continue
                    else:
                        self.tracks[id].level = level
                        known_levels.add(id)
            unknown_levels = unknown_levels - known_levels
            level += 1