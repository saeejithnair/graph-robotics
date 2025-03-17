import os
import shutil
import sys
from pathlib import Path

import hydra
import numpy as np
import open3d as o3d
import torch
# import torchvision
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

import embodied_memory.pointcloud as pointcloud
import embodied_memory.utils as utils
from embodied_memory.datasets import get_dataset
from embodied_memory.detection_feature_extractor import DetectionFeatureExtractor
from embodied_memory.embodied_memory import EmbodiedMemory
from embodied_memory.relationship_scorer import HierarchyExtractor
from embodied_memory.scene_graph import associate_dets_to_nodes
from embodied_memory.visualizer import Visualizer2D, Visualizer3D
from embodied_memory.vlm_detectors import (
    CaptionConsolidator,
    EdgeConsolidator,
    VLMObjectDetector,
    VLMObjectDetectorYOLO,
)

import google.generativeai as genai


# A logger for this file
@hydra.main(
    version_base=None,
    config_path="embodied_memory/configs/mapping",
    config_name="hm3d_mapping",
    # config_name="scannet_mapping",
)
# @profile
def main(cfg: DictConfig):
    cfg = utils.process_cfg(cfg)
        # check for GOOGLE_API_KEY api key
    with open("../api_keys/gemini_key.txt") as f:
        GOOGLE_API_KEY = f.read().strip()
        # genai.set_api_key(GOOGLE_API_KEY)

    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        "/home/qasim/Projects/graph-robotics/api_keys/total-byte-432318-q3-78e6d4aa6497.json"
    )

    # Initialize the dataset
    device = cfg.device
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
    )

    # Initialize result paths
    result_dir = Path(cfg.result_root) / cfg.scene_id
    result_dir_detections = result_dir / "detections"
    result_dir_embodied_memory = result_dir / "embodied_memory"

    # Delete previous results
    shutil.rmtree(result_dir_detections, ignore_errors=True)
    shutil.rmtree(result_dir_embodied_memory, ignore_errors=True)

    # Initialize modules
    embodied_memory = EmbodiedMemory(cfg.visual_memory_size, cfg.room_types, cfg.device)
    perceptor = VLMObjectDetectorYOLO(
        gemini_model=cfg.detections_model, device=device, with_edges=False
    )
    # perceptor = VLMObjectDetector(
    #     gemini_model=cfg.detections_model, device=device, with_edges=False
    # )
    perceptor.init()
    edge_consolidator = EdgeConsolidator(
        gemini_model=cfg.detections_model,
        device=device,
        prompt_file="embodied_memory/vlm_detectors/prompts/edge_consolidation.txt",
    )
    caption_consolidator = CaptionConsolidator(
        gemini_model=cfg.detections_model,
        device=device,
        response_detection_key=None,
        response_other_keys=["label", "caption"],
        prompt_file="embodied_memory/vlm_detectors/prompts/caption_consolidation.txt",
    )
    feature_extractor = DetectionFeatureExtractor(device)
    feature_extractor.init()
    hierarchy_extractor = HierarchyExtractor(downsample_voxel_size=0.02)

    counter = 0
    img_list = []
    pose_list = []
    frame_list = []

    for frame_idx in tqdm(range(0, len(dataset), cfg.skip_frame)):  # len(dataset)
        if np.any(np.isinf(dataset.poses[frame_idx].numpy())):
            print("Infitinity poses detected!")
            print("skipping frame: ", frame_idx, "in scene_id: ", cfg.scene_id)
            continue

        counter += 1

        # color and depth tensors, and camera instrinsics matrix
        color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]

        # Covert to numpy and do some sanity checks
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().detach().numpy()
        color_np = color_tensor.cpu().detach().numpy()  # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8)  # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        # extract camera poses
        unt_pose = dataset.poses[frame_idx]
        trans_pose = unt_pose.cpu().detach().numpy()

        # compute depth cloud
        depth_cloud = pointcloud.create_depth_cloud(depth_array, dataset.get_cam_K())

        # merge depth cloud with full point cloud
        embodied_memory.full_scene_pcd.merge_scene_pcd(
            pointcloud.create_depth_pcdo3d(
                image_rgb, depth_array, trans_pose, dataset.get_cam_K()
            )
        )

        img_list.append(image_rgb)
        pose_list.append(unt_pose)
        frame_list.append(frame_idx)

        # detect items in the scene
        detections, llm_response = perceptor.perceive(color_np, pose=trans_pose)

        # process detections by extracting point cloud, clip embeddings, caption embeddings, etc.
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
        feature_extractor.compute_features(image_rgb, detections)

        # associate detections to nodes in the scene graph
        is_matched, matched_nodeidx = associate_dets_to_nodes(
            detections,
            embodied_memory.scene_graph.get_nodes(),
            downsample_voxel_size=cfg["downsample_voxel_size"],
        )

        #
        detections, edges = embodied_memory.update_scene_graph(
            detections,
            is_matched,
            matched_nodeidx,
            frame_idx,
            color_np,
            consolidate=True,
        )

        # Every few keyframes, ask the VLM to verify and enhance perceived edges
        if counter > 0 and counter % 3 == 0:
            consolidated_edges = edge_consolidator.perceive(
                embodied_memory.scene_graph,
                embodied_memory.detections_for_consolidation,
                embodied_memory.imgs_for_consolidation,
                edges_buffer=embodied_memory.edges_for_consolidation,
            )
            embodied_memory.merge_edge_buffer(consolidated_edges)

        # Every few keyframes, ask the VLM to consolidate the multiple captions (collected across multiple frames) into a single caption
        if counter > 0 and counter % 5 == 0:
            embodied_memory.scene_graph = caption_consolidator.perceive(
                embodied_memory.scene_graph, color_np
            )

        embodied_memory.navigation_log.add_general_log(
            llm_response, detections, frame_idx
        )

        perceptor.save_detection_results(
            detections, result_dir_detections, frame_idx, llm_response=llm_response
        )

        # Periodically denoise the point clouds
        if cfg["denoise_interval"] > 0 and (counter + 1) % cfg["denoise_interval"] == 0:
            for id in embodied_memory.scene_graph.get_node_ids():
                embodied_memory.scene_graph[id].denoise(
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                )
        break
    if cfg["run_filter_final_frame"]:
        for id in embodied_memory.scene_graph.get_node_ids():
            embodied_memory.scene_graph[id].denoise(
                downsample_voxel_size=cfg["downsample_voxel_size"],
                dbscan_remove_noise=cfg["dbscan_remove_noise"],
                dbscan_eps=cfg["dbscan_eps"],
                dbscan_min_points=cfg["dbscan_min_points"],
            )

    # Conslidate any remaining edges in the buffer
    if len(embodied_memory.imgs_for_consolidation) > 0:
        consolidated_edges = edge_consolidator.perceive(
            embodied_memory.scene_graph,
            embodied_memory.detections_for_consolidation,
            embodied_memory.imgs_for_consolidation,
            edges_buffer=embodied_memory.edges_for_consolidation,
        )
        embodied_memory.merge_edge_buffer(consolidated_edges)

    # Compute the hierarchy and node levels
    hierarchy_matrix, hierarchy_type_matrix = hierarchy_extractor.infer_hierarchy(
        embodied_memory.scene_graph
    )
    embodied_memory.compute_node_levels(hierarchy_matrix, hierarchy_type_matrix)

    # Segment rooms. Also assigns each frame in the navigation log and each node in the scene graph to a room.
    embodied_memory.process_rooms(
        room_grid_resolution=cfg.room_grid_resolution,
        img_list=img_list,
        pose_list=pose_list,
        frame_list=frame_list,
        debug_folder=cfg.debug_folder,
        device=device,
    )

    # Denoise the full scene point cloud
    embodied_memory.full_scene_pcd.denoise_geometrymap(
        downsample_voxel_size=0.02,  # cfg["downsample_voxel_size"],
        dbscan_eps=0.01,  # cfg["dbscan_eps"],
        min_points=100,  # cfg['dbscan_min_points']
        # dbscan_min_points=cfg["dbscan_min_points"],
    )

    # Save the embodied memory
    embodied_memory.save(
        save_dir=result_dir_embodied_memory,
        hierarchy_matrix=hierarchy_matrix,
        hierarchy_type_matrix=hierarchy_type_matrix,
    )

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
        embodied_memory.visualize_3d(visualizer3d)

    visualizer2d = Visualizer2D(level2color)
    visualizer2d.init()
    embodied_memory.visualize_2d(
        visualizer2d, result_dir, hierarchy_matrix, hierarchy_type_matrix
    )

    print("Completed generation of", cfg.scene_id)


if __name__ == "__main__":
    scene_ids = []
    with open("subset_scene_ids.txt", "r") as f:
        for line in f.readlines():
            if "#" in line or len(line.strip()) == 0:
                continue
            scene_ids.append(line.strip())
    # sorted(scene_ids)
    for id in scene_ids:
        sys.argv.append("scene_id=" + id)
        print("PROCESSSING SCENE:", id)
        main()
