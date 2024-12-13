import os
import shutil
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
from scene_graph.perception import Perceptor
from scene_graph.relationship_scorer import FeatureComputer, RelationshipScorer
from scene_graph.semantic_tree import SemanticTree
from scene_graph.track import save_tracks
from scene_graph.visualizer import Visualizer2D, Visualizer3D


# A logger for this file
@hydra.main(
    version_base=None,
    config_path="scene_graph/configs/mapping",
    config_name="hm3d_mapping",
)
# @profile
def main_semantictree(cfg: DictConfig):
    tracker = utils.MappingTracker()

    cfg = utils.process_cfg(cfg)

    # Initialize the dataset
    device = "cpu"
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device=device,
        dtype=torch.float,
        # relative_pose=True
    )
    # cam_K = dataset.get_cam_K()

    result_dir = Path(cfg.result_root) / cfg.scene_id
    perception_result_dir = result_dir / f"{cfg.detections_exp_suffix}"
    semantic_tree_result_dir = result_dir / f"semantic_tree"
    tracks_result_dir = result_dir / f"tracks"
    shutil.rmtree(perception_result_dir, ignore_errors=True)
    shutil.rmtree(semantic_tree_result_dir, ignore_errors=True)
    shutil.rmtree(tracks_result_dir, ignore_errors=True)

    semantic_tree = SemanticTree()
    semantic_tree.load_pcd(folder=result_dir)
    perceptor = Perceptor(device=device)
    feature_computer = FeatureComputer(device)
    relationship_scorer = RelationshipScorer(downsample_voxel_size=0.02)
    perceptor.init()
    feature_computer.init()

    exit_early_flag = False
    counter = 0
    for frame_idx in range(0, len(dataset), 7):  # len(dataset)
        # Check if we should exit early only if the flag hasn't been set yet
        if not exit_early_flag and utils.should_exit_early(cfg.exit_early_file):
            print("Exit early signal detected. Skipping to the final frame...")
            exit_early_flag = True
        # If exit early flag is set and we're not at the last frame, skip this iteration
        if exit_early_flag and frame_idx < len(dataset) - 1:
            continue

        tracker.curr_frame_idx = frame_idx
        counter += 1

        # color and depth tensors, and camera instrinsics matrix
        color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]

        # Covert to numpy and do some sanity checks
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        color_np = color_tensor.cpu().numpy()  # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8)  # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        unt_pose = dataset.poses[frame_idx]
        trans_pose = unt_pose.cpu().numpy()
        depth_cloud = pointcloud.create_depth_cloud(depth_array, dataset.get_cam_K())

        detections, llm_response = perceptor.perceive(color_np, pose=trans_pose)
        detections.extract_local_pcd(
            depth_cloud,
            color_np,
            semantic_tree.geometry_map,
            trans_pose,
            cfg.obj_pcd_max_points,
        )

        feature_computer.compute_features(
            image_rgb, semantic_tree.geometry_map, detections
        )

        det_matched, matched_tracks = relationship_scorer.associate_dets_to_tracks(
            detections, semantic_tree.get_tracks()
        )

        matched_track_names = semantic_tree.integrate_detections(
            detections, det_matched, matched_tracks, frame_idx
        )
        semantic_tree.integrate_generic_log(
            llm_response, matched_track_names, frame_idx
        )

        perceptor.save_results(
            llm_response, detections, perception_result_dir, frame_idx
        )
        detections.save(perception_result_dir / str(frame_idx), perceptor.img)

    neighbour_matrix, in_matrix, on_matrix, hierarchy_matrix = (
        relationship_scorer.infer_hierarchy_relationships(
            semantic_tree.tracks, semantic_tree.geometry_map
        )
    )

    semantic_tree.process_hierarchy(
        neighbour_matrix, in_matrix, on_matrix, hierarchy_matrix
    )

    semantic_tree.save(
        semantic_tree_result_dir,
        neighbour_matrix,
        in_matrix,
        on_matrix,
        hierarchy_matrix,
    )
    save_tracks(semantic_tree.tracks, tracks_result_dir)

    level2color = {
        0: "yellow",
        1: "green",
        2: "red",
        3: "purple",
        4: "orange",
        5: "brown",
        6: "turquoise",
        7: "blue",
    }

    if cfg.use_rerun:
        visualizer3d = Visualizer3D(level2color)
        visualizer3d.init()
        semantic_tree.visualize_3d(visualizer3d)

    visualizer2d = Visualizer2D(level2color)
    visualizer2d.init()
    semantic_tree.visualize_2d(
        visualizer2d, result_dir, hierarchy_matrix, in_matrix, on_matrix
    )

    print("Complete")


if __name__ == "__main__":
    # main_semantictree()

    # scene_ids = os.listdir("/pub3/qasim/hm3d/data/concept-graphs/with_edges")
    scene_ids = [
        "000-hm3d-BFRyYbPCCPE",
        "002-hm3d-wcojb4TFT35",
        "003-hm3d-c5eTyR3Rxyh",
        "004-hm3d-66seV3BWPoX",
        "005-hm3d-yZME6UR9dUN",
        "006-hm3d-q3hn1WQ12rz",
        "007-hm3d-bxsVRursffK",
        "011-hm3d-bzCsHPLDztK",
        "012-hm3d-XB4GS9ShBRE",
        "013-hm3d-svBbv1Pavdk",
        "014-hm3d-rsggHU7g7dh",
        "015-hm3d-5jp3fCRSRjc",
    ]
    sorted(scene_ids)
    for id in scene_ids:
        sys.argv.append("scene_id=" + id)
        print("PROCESSSING SCENE:", id)
        main_semantictree()
