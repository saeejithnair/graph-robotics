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
from tqdm import tqdm

import scene_graph.pointcloud as pointcloud
import scene_graph.utils as utils
from scene_graph.datasets import get_dataset
from scene_graph.features import FeatureComputer
from scene_graph.perception import CaptionConsolidator, EdgeConsolidator, GenericMapper
from scene_graph.relationship_scorer import RelationshipScorer
from scene_graph.semantic_tree import SemanticTree
from scene_graph.track import associate_dets_to_tracks, save_tracks
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
    perceptor = GenericMapper(device=device, with_edges=False)
    edge_consolidator = EdgeConsolidator(
        prompt_file="scene_graph/perception/prompts/edge_consolidation.txt",
    )
    caption_consolidator = CaptionConsolidator(
        json_detection_key=None,
        json_other_keys=["label", "caption"],
        prompt_file="scene_graph/perception/prompts/caption_consolidation.txt",
    )
    feature_computer = FeatureComputer(device)
    relationship_scorer = RelationshipScorer(downsample_voxel_size=0.02)
    perceptor.init()
    feature_computer.init()

    exit_early_flag = False
    counter = 0
    img_list = []
    pose_list = []
    frame_list = []
    for frame_idx in tqdm(range(0, len(dataset), 7)):  # len(dataset)
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

        img_list.append(image_rgb)
        pose_list.append(unt_pose)
        frame_list.append(frame_idx)

        detections, llm_response = perceptor.perceive(color_np, pose=trans_pose)
        detections.extract_pcd(
            depth_cloud,
            color_np,
            downsample_voxel_size=cfg["downsample_voxel_size"],
            dbscan_remove_noise=cfg["dbscan_remove_noise"],
            dbscan_eps=cfg["dbscan_eps"],
            dbscan_min_points=cfg["dbscan_min_points"],
            trans_pose=trans_pose,
            obj_pcd_max_points=cfg["obj_pcd_max_points"],
        )

        feature_computer.compute_features(image_rgb, detections)

        is_matched, matched_trackidx = associate_dets_to_tracks(
            detections,
            semantic_tree.get_tracks(),
            downsample_voxel_size=cfg["downsample_voxel_size"],
        )

        detections, edges = semantic_tree.integrate_detections(
            detections,
            is_matched,
            matched_trackidx,
            frame_idx,
            color_np,
            consolidate=True,
        )

        if counter > 0 and counter % 3 == 0:
            semantic_tree.tracks = caption_consolidator.perceive(
                semantic_tree.tracks, color_np
            )
            consolidated_edges = edge_consolidator.perceive(
                semantic_tree.tracks,
                semantic_tree.detections_for_consolidation,
                semantic_tree.imgs_for_consolidation,
                edges_buffer=semantic_tree.edges_for_consolidation,
            )
            semantic_tree.merge_edge_buffer(consolidated_edges)

        semantic_tree.integrate_generic_log(llm_response, detections, frame_idx)

        perceptor.save_results(
            detections, perception_result_dir, frame_idx, llm_response=llm_response
        )

        # Periodically denoise the point clouds
        if cfg["denoise_interval"] > 0 and (counter + 1) % cfg["denoise_interval"] == 0:
            for id in semantic_tree.track_ids:
                semantic_tree.tracks[id].denoise(
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                )

    if cfg["run_filter_final_frame"]:
        for id in semantic_tree.track_ids:
            semantic_tree.tracks[id].denoise(
                downsample_voxel_size=cfg["downsample_voxel_size"],
                dbscan_remove_noise=cfg["dbscan_remove_noise"],
                dbscan_eps=cfg["dbscan_eps"],
                dbscan_min_points=cfg["dbscan_min_points"],
            )

    # hierarchy_matrix, hierarchy_type_matrix = relationship_scorer.infer_hierarchy_heuristics(
    #         semantic_tree.tracks
    #     )
    if len(semantic_tree.imgs_for_consolidation) > 0:
        consolidated_edges = edge_consolidator.perceive(
            semantic_tree.tracks,
            semantic_tree.detections_for_consolidation,
            semantic_tree.imgs_for_consolidation,
            edges_buffer=semantic_tree.edges_for_consolidation,
        )
        semantic_tree.merge_edge_buffer(consolidated_edges)

    hierarchy_matrix, hierarchy_type_matrix = relationship_scorer.infer_hierarchy_vlm(
        semantic_tree.tracks
    )

    semantic_tree.compute_node_levels(hierarchy_matrix, hierarchy_type_matrix)

    semantic_tree.process_rooms(
        room_grid_resolution=cfg.room_grid_resolution,
        img_list=img_list,
        pose_list=pose_list,
        frame_list=frame_list,
        debug_folder=cfg.debug_folder,
        device=device,
    )

    semantic_tree.save(
        semantic_tree_result_dir, hierarchy_matrix, hierarchy_type_matrix
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
        visualizer2d, result_dir, hierarchy_matrix, hierarchy_type_matrix
    )

    print("Complete")


if __name__ == "__main__":
    # main_semantictree()

    # scene_ids = os.listdir("/pub3/qasim/hm3d/data/concept-graphs/with_edges")
    scene_ids = [
        "000-hm3d-BFRyYbPCCPE",
        "001-hm3d-TPhiubUHKcP",
        "002-hm3d-wcojb4TFT35",
        "003-hm3d-c5eTyR3Rxyh",
        # "004-hm3d-66seV3BWPoX",
        # "005-hm3d-yZME6UR9dUN",
        # "006-hm3d-q3hn1WQ12rz",
        # "007-hm3d-bxsVRursffK",
        # "008-hm3d-SiKqEZx7Ejt",
        # "010-hm3d-5cdEh9F2hJL",
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
        # "086-hm3d-cYkrGrCg2kB",
        # "087-hm3d-mma8eWq3nNQ",
        # "088-hm3d-hDBqLgydy1n",
        # "089-hm3d-AYpsNQsWncn",
        # "092-hm3d-eF36g7L6Z9M",
        # "094-hm3d-Qpor2mEya8F",
        # "096-hm3d-uLz9jNga3kC",
        # "097-hm3d-QHhQZWdMpGJ",
        # "098-hm3d-bCPU9suPUw9",
        # "099-hm3d-q5QZSEeHe5g",
        # "084-hm3d-zt1RVoi7PcG",
    ]
    sorted(scene_ids)
    for id in scene_ids:
        sys.argv.append("scene_id=" + id)
        print("PROCESSSING SCENE:", id)
        main_semantictree()
        # import read_graphs

        # semantic_tree = read_graphs.read_hamsg_flatgraph(
        #     "/pub3/qasim/hm3d/data/ham-sg/015-hm3d-5jp3fCRSRjc"
        # )
        # semantic_tree.compute_node_levels()
