import torch
import numpy as np
import open3d as o3d
from pointcloud import pcd_denoise_dbscan, dynamic_downsample, find_nearest_points

class SceneGraph:
    def __init__(self, device='cuda'):
        self.geometry_map = o3d.geometry.PointCloud()
        self.device = device 
        self.nodes = {}
        self.children = {}
    def add_pcd(self, pcd):
        self.geometry_map += pcd
    def denoise_geometrymap(self, downsample_voxel_size, dbscan_eps, min_points):
        self.geometry_map = self.geometry_map.voxel_down_sample(
            voxel_size=downsample_voxel_size
        )
        self.geometry_map = pcd_denoise_dbscan(self.geometry_map, eps=dbscan_eps, min_points=min_points)
    def load_pcd(self, path):
        self.geometry_map = o3d.io.read_point_cloud(path)
    def process_detections(self):
        pass
    def add_node(self, node):
        self.children[node.id] = []
        self.nodes[node.id] = node

    def visualize_graph(self, visualizer):
        visualizer.add_geometry_map(self.geometry_map.points, self.geometry_map.colors)
        # Visualize Nodes
        for idx in self.nodes.keys():
            visualizer.add_semanticnode(
                id=self.nodes[idx].id,
                label=self.nodes[idx].label,
                caption=self.nodes[idx].caption,
                location=self.nodes[idx].vis_centroid,
                level = self.nodes[idx].level,
                children_locs=[self.nodes[c].location for c in self.children[idx]],
                children_labels=[self.nodes[c].label for c in self.children[idx]],
                points=self.nodes[idx].get_points_at_idxs(self.geometry_map),
                )
node_counter = 0
