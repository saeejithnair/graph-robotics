import torch
import numpy as np
import open3d as o3d
from pointcloud import pcd_denoise_dbscan, dynamic_downsample, find_nearest_points
from perception import Perceptor

node_counter = 0

class Node:
    def __init__(self, id, point_idxs, 
                crop, label, 
                caption, level=0, 
                global_pcd=None,
                ) -> None:
        self.id = node_counter
        self.level = level
        self.point_idxs = point_idxs
        self.label = f'node_{str(id)}' if not label else label
        self.caption = 'None' if not caption else caption
        self.crop = [crop]
        if global_pcd:
            self.compute_vis_centroid(global_pcd)
        else:
            self.vis_centroid = None
    
    def compute_vis_centroid(self, global_pcd, height_by_level=False):
        mean_pcd = np.mean(np.asarray(global_pcd.points)[self.point_idxs], 0)
        height = None
        if height_by_level:
            height = self.level*0.5 + 4 # heursitic
        else:
            max_pcd_height = np.max(- np.asarray(global_pcd.points)[self.point_idxs], 0)[1]
            height = max_pcd_height + self.level*0.3 + 0.5
        location = [mean_pcd[0], -1 * height, mean_pcd[2]]
        self.vis_centroid = location
    
    def add_point_idxs(self, point_idxs):
        self.point_idxs = point_idxs
    
    def add_crop(self, crop):
        self.crop.append(crop)
        
    def get_points_at_idxs(self, pcd):
        return np.asarray(pcd.points)[self.point_idxs]
        
class Detection:
    def __init__(self,                  
                object,
                depth_cloud,
                img_rgb,
                global_pcd, 
                trans_pose,
                obj_pcd_max_points=5000
                ) -> None:
        self.object = object
        # Create point cloud
        downsampled_points, downsampled_colors = dynamic_downsample(
            depth_cloud[object['mask']],
            colors=img_rgb[object['mask']], 
            target=obj_pcd_max_points)
        local_pcd = o3d.geometry.PointCloud()
        local_pcd.points = o3d.utility.Vector3dVector(downsampled_points)
        if downsampled_colors is not None:
            local_pcd.colors = o3d.utility.Vector3dVector(downsampled_colors)

        if trans_pose is not None:
            local_pcd.transform(trans_pose)  # Apply transformation directly to the point cloud
        
        self.local_points = local_pcd
        _, self.global_pcd_idxs = find_nearest_points(
            np.array(self.local_points.points),
            np.array(global_pcd.points)
            )
        
    def to_node(self, global_pcd):
        global node_counter
        node = Node(
            id=node_counter,
            point_idxs=self.global_pcd_idxs[:,0], 
            crop=self.object['crop'], 
            label=self.object['label'], 
            caption=self.object['caption'], 
            global_pcd=global_pcd,
        )
        node_counter += 1
        return node
    