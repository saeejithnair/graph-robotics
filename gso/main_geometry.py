import os
import sys
from pathlib import Path

import hydra
import numpy as np
import open3d as o3d
import torch
import torchvision
from omegaconf import DictConfig
from PIL import Image
from tqdm import trange

import scene_graph.pointcloud as pointcloud
import scene_graph.utils as utils
from scene_graph.datasets import get_dataset
from scene_graph.semantic_tree import SemanticTree
from scene_graph.utils import MappingTracker, cfg_to_dict, process_cfg


# A logger for this file
@hydra.main(
    version_base=None,
    config_path="scene_graph/configs/mapping",
    config_name="hm3d_mapping",
)
# @profile
def main_geometry(cfg: DictConfig):
    tracker = MappingTracker()

    cfg = process_cfg(cfg)

    # Initialize the dataset
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
        # relative_pose=True
    )
    # cam_K = dataset.get_cam_K()
    scene_graph = SemanticTree()

    exit_early_flag = False
    counter = 0
    for frame_idx in trange(len(dataset)):
        tracker.curr_frame_idx = frame_idx
        counter += 1

        # Check if we should exit early only if the flag hasn't been set yet
        if not exit_early_flag and utils.should_exit_early(cfg.exit_early_file):
            print("Exit early signal detected. Skipping to the final frame...")
            exit_early_flag = True
        # If exit early flag is set and we're not at the last frame, skip this iteration
        if exit_early_flag and frame_idx < len(dataset) - 1:
            continue

        # color and depth tensors, and camera instrinsics matrix
        color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]

        # Covert to numpy and do some sanity checks
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        color_np = color_tensor.cpu().numpy()  # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8)  # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        # get pose, this is the untrasformed pose.
        unt_pose = dataset.poses[frame_idx]
        # unt_pose = np.eye(4) # dataset.poses[frame_idx]
        # unt_pose = dataset.transformed_poses[frame_idx]
        unt_pose = unt_pose.cpu().numpy()

        # Don't apply any transformation otherwise
        adjusted_pose = unt_pose

        new_pcd = pointcloud.create_depth_pcdo3d(
            image_rgb, depth_array, adjusted_pose, dataset.get_cam_K()
        )
        scene_graph.add_pcd(new_pcd)

        # I added this manually, to resolve shape issues
        image_rgb = (color_np).astype(np.uint8)  # (H, W, 3)

    scene_graph.denoise_geometrymap(
        downsample_voxel_size=0.02,  # cfg["downsample_voxel_size"],
        dbscan_eps=0.01,  # cfg["dbscan_eps"],
        min_points=100,  # cfg['dbscan_min_points']
        # dbscan_min_points=cfg["dbscan_min_points"],
    )

    result_dir = Path(cfg.result_root) / cfg.scene_id

    scene_graph.save_pcd(folder=result_dir)


if __name__ == "__main__":
    scene_ids = os.listdir("/pub3/qasim/hm3d/data/concept-graphs/with_edges")
    sorted(scene_ids)
    for id in scene_ids:
        sys.argv.append("scene_id=" + id)
        print("PROCESSSING SCENE:", id)
        main_geometry()
