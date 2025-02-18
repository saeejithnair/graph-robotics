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

import embodied_memory.pointcloud as pointcloud
import embodied_memory.utils as utils
from embodied_memory.datasets import get_dataset
from embodied_memory.embodied_memory import EmbodiedMemory
from embodied_memory.utils import MappingTracker, cfg_to_dict, process_cfg


# A logger for this file
@hydra.main(
    version_base=None,
    config_path="embodied_memory/configs/mapping",
    # config_name="hm3d_mapping",
    config_name="scannet_mapping",
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
        device="cuda:1",
        dtype=torch.float,
        # relative_pose=True
    )
    # cam_K = dataset.get_cam_K()
    scene_graph = EmbodiedMemory(cfg.visual_memory_size, cfg.room_types)

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
        scene_graph.merge_scene_pcd(new_pcd)

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
        # "002-scannet-scene0709_00",
        # "003-scannet-scene0762_00",
        # "012-scannet-scene0785_00",
        # "013-scannet-scene0720_00",
        # "014-scannet-scene0714_00",
        # "031-scannet-scene0787_00",
        # "037-scannet-scene0763_00",
        # "046-scannet-scene0724_00",
        # "047-scannet-scene0747_00",
        # "048-scannet-scene0745_00",
        # "100-scannet-scene0598_00",
        # "101-scannet-scene0256_00",
        # "102-scannet-scene0222_00",
        # "103-scannet-scene0527_00",
        # "104-scannet-scene0616_01",
        # "105-scannet-scene0207_01",
        # "106-scannet-scene0633_00",
        # "108-scannet-scene0354_00",
        # "109-scannet-scene0648_01",
        "110-scannet-scene0050_00",
        "111-scannet-scene0550_00",
        "112-scannet-scene0591_02",
        "113-scannet-scene0207_00",
        "114-scannet-scene0084_00",
        "115-scannet-scene0046_02",
        "116-scannet-scene0077_01",
        "120-scannet-scene0684_01",
        "121-scannet-scene0462_00",
        "122-scannet-scene0647_01",
        "123-scannet-scene0412_01",
        "124-scannet-scene0131_02",
        "125-scannet-scene0426_00",
        "126-scannet-scene0574_02",
        "127-scannet-scene0578_00",
        "128-scannet-scene0678_02",
        "129-scannet-scene0575_00",
        "130-scannet-scene0696_00",
        "132-scannet-scene0645_01",
        "133-scannet-scene0704_00",
        "134-scannet-scene0695_03",
        "135-scannet-scene0131_00",
        "137-scannet-scene0598_01",
        "138-scannet-scene0500_01",
        "139-scannet-scene0647_00",
        "140-scannet-scene0077_00",
        "141-scannet-scene0651_01",
        "142-scannet-scene0653_01",
        "144-scannet-scene0700_01",
        "145-scannet-scene0193_00",
        "146-scannet-scene0518_00",
        "147-scannet-scene0699_00",
        "148-scannet-scene0203_01",
        "149-scannet-scene0426_02",
        "150-scannet-scene0648_00",
        "151-scannet-scene0217_00",
        "152-scannet-scene0494_00",
        "154-scannet-scene0193_01",
        "155-scannet-scene0164_02",
        "156-scannet-scene0461_00",
        "157-scannet-scene0015_00",
        "158-scannet-scene0356_00",
        "160-scannet-scene0488_01",
        "161-scannet-scene0583_00",
        "162-scannet-scene0535_00",
        "163-scannet-scene0164_03",
        "165-scannet-scene0406_00",
        "166-scannet-scene0435_03",
        "167-scannet-scene0307_01",
        "168-scannet-scene0655_01",
        "169-scannet-scene0695_01",
        "170-scannet-scene0378_01",
        "171-scannet-scene0222_01",
        "172-scannet-scene0496_00",
        "173-scannet-scene0278_01",
        "174-scannet-scene0086_01",
        "175-scannet-scene0329_01",
        "176-scannet-scene0643_00",
        "177-scannet-scene0608_02",
        "178-scannet-scene0685_02",
        "179-scannet-scene0300_01",
        "180-scannet-scene0100_02",
        "181-scannet-scene0314_00",
        "182-scannet-scene0645_00",
        "183-scannet-scene0231_01",
        "185-scannet-scene0435_00",
        "186-scannet-scene0549_01",
        "187-scannet-scene0655_02",
        "188-scannet-scene0593_00",
        "189-scannet-scene0500_00",
    ]

    # sorted(scene_ids)
    for id in scene_ids:
        sys.argv.append("scene_id=" + id)
        print("PROCESSSING SCENE:", id)
        main_geometry()
