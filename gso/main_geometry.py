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
    scene_graph = SemanticTree(cfg.visual_memory_size)

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
    scene_ids = [
        # "000-hm3d-BFRyYbPCCPE",
        "001-hm3d-TPhiubUHKcP",
        # "002-hm3d-wcojb4TFT35",
        # "003-hm3d-c5eTyR3Rxyh",
        # "004-hm3d-66seV3BWPoX",
        # "005-hm3d-yZME6UR9dUN",
        # "006-hm3d-q3hn1WQ12rz",
        # "007-hm3d-bxsVRursffK",
        "008-hm3d-SiKqEZx7Ejt",
        "010-hm3d-5cdEh9F2hJL",
        # "011-hm3d-bzCsHPLDztK",
        # "012-hm3d-XB4GS9ShBRE",
        # "013-hm3d-svBbv1Pavdk",
        # "014-hm3d-rsggHU7g7dh",
        # "015-hm3d-5jp3fCRSRjc",
        # "016-hm3d-nrA1tAA17Yp",
        # "017-hm3d-Dd4bFSTQ8gi",
        # "018-hm3d-dHwjuKfkRUR",
        # "019-hm3d-y9hTuugGdiq",
        # "021-hm3d-LT9Jq6dN3Ea",
        # "023-hm3d-VBzV5z6i1WS",
        # "024-hm3d-c3WKCnkEdha",
        # "025-hm3d-ziup5kvtCCR",
        # "026-hm3d-tQ5s4ShP627",
        # "028-hm3d-rXXL6twQiWc",
        # "029-hm3d-mv2HUxq3B53",
        # "030-hm3d-RJaJt8UjXav",
        # "031-hm3d-Nfvxx8J5NCo",
        # "032-hm3d-6s7QHgap2fW",
        # "033-hm3d-vd3HHTEpmyA",
        # "035-hm3d-BAbdmeyTvMZ",
        # "036-hm3d-rJhMRvNn4DS",
        # "037-hm3d-FnSn2KSrALj",
        # "038-hm3d-b28CWbpQvor",
        # "039-hm3d-uSKXQ5fFg6u",
        # "040-hm3d-HaxA7YrQdEC",
        # "041-hm3d-GLAQ4DNUx5U",
        # "042-hm3d-hkr2MGpHD6B",
        # "046-hm3d-X4qjx5vquwH",
        # "048-hm3d-kJJyRFXVpx2",
        # "049-hm3d-SUHsP6z2gcJ",
        # "050-hm3d-cvZr5TUy5C5",
        # "055-hm3d-W7k2QWzBrFY",
        # "056-hm3d-7UrtFsADwob",
        # "057-hm3d-q3zU7Yy5E5s",
        # "058-hm3d-7MXmsvcQjpJ",
        # "059-hm3d-T6nG3E2Uui9",
        # "068-hm3d-p53SfW6mjZe",
        # "069-hm3d-HMkoS756sz6",
        # "072-hm3d-a8BtkwhxdRV",
        # "073-hm3d-LEFTm3JecaC",
        # "083-hm3d-LNg5mXe1BDj",
        # "084-hm3d-zt1RVoi7PcG",
        # "086-hm3d-cYkrGrCg2kB",
        # "087-hm3d-mma8eWq3nNQ",
        # "088-hm3d-hDBqLgydy1n",
        # "089-hm3d-AYpsNQsWncn",
        # "092-hm3d-eF36g7L6Z9M",
        # "094-hm3d-Qpor2mEya8F",
        # "096-hm3d-uLz9jNga3kC",
        # "097-hm3d-QHhQZWdMpGJ",
        "098-hm3d-bCPU9suPUw9",
        "099-hm3d-q5QZSEeHe5g",
    ]
    sorted(scene_ids)
    for id in scene_ids:
        sys.argv.append("scene_id=" + id)
        print("PROCESSSING SCENE:", id)
        main_geometry()
