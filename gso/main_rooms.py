import os
import shutil
import sys
from pathlib import Path

import hydra
import numpy as np
import open3d as o3d
import open_clip
import torch
import torchvision
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer

import scene_graph.pointcloud as pointcloud
import scene_graph.utils as utils
from scene_graph.datasets import get_dataset
from scene_graph.features import FeatureComputer
from scene_graph.floor import segment_floors
from scene_graph.perception import EdgeConsolidator, GenericMapper
from scene_graph.relationship_scorer import RelationshipScorer
from scene_graph.rooms import assign_tracks_to_rooms, segment_rooms
from scene_graph.semantic_tree import SemanticTree
from scene_graph.track import associate_dets_to_tracks, save_tracks
from scene_graph.visualizer import Visualizer2D, Visualizer3D


# A logger for this file
@hydra.main(
    version_base=None,
    config_path="scene_graph/configs/mapping",
    config_name="hm3d_mapping",
)
def main(cfg: DictConfig):
    cfg = utils.process_cfg(cfg)
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

    semantic_tree = SemanticTree()
    semantic_tree.load_pcd(folder=result_dir)
    semantic_tree.load(result_dir, load_floors=False, load_rooms=False)

    default_room_types = [
        "kitchen",
        "living room",
        "bedroom",
        "dinning room",
        "living and dining area",
        "bathroom",
        "washroom",
        "hallway",
        "garage",
        "corridor",
        "office",
    ]
    clip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    clip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device)
    clip_tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
    # clip_model, _, preprocess = open_clip.create_model_and_transforms(
    #     "ViT-L-14",
    #     device=device,
    # )

    img_list = []
    pose_list = []
    frame_list = []
    for frame_idx in tqdm(range(0, len(dataset), 7)):  # len(dataset)

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

    semantic_tree.process_rooms(
        room_grid_resolution=cfg.room_grid_resolution,
        img_list=img_list,
        pose_list=pose_list,
        frame_list=frame_list,
        debug_folder=cfg.debug_folder,
        device=device,
    )

    # debug_folder = cfg.debug_folder
    # floors = segment_floors(
    #     semantic_tree.geometry_map,
    #     graph_tmp_folder=debug_folder,
    #     save_intermediate_results=True,
    #     # flip_zy=True,
    # )
    # rooms = []
    # for floor in floors:
    #     detected_rooms = segment_rooms(
    #         floor=floor,
    #         clip_processor=clip_processor,
    #         clip_model=clip_model,
    #         grid_resolution=cfg.room_grid_resolution,
    #         rgb_list=img_list,
    #         pose_list=pose_list,
    #         frameidx_list=frame_list,
    #         graph_tmp_folder=debug_folder,
    #         save_intermediate_results=True,
    #         device=device,
    #     )
    #     rooms += detected_rooms
    #     floor.rooms = detected_rooms
    # print("Number of room:", len(rooms))
    # frame_id_to_rooms = {}
    # for floor in floors:
    #     for i, room in enumerate(floor.rooms):
    #         room_type = room.infer_room_type_from_view_embedding(
    #             default_room_types, clip_tokenizer, clip_model
    #         )
    #         room.room_type = room_type
    #         print("room id", room.room_id, "is", room_type)
    #         for id in room.represent_images:
    #             if id in frame_id_to_rooms:
    #                 frame_id_to_rooms[id].append(room.room_id)
    #             else:
    #                 frame_id_to_rooms[id] = [room.room_id]

    # assign_tracks_to_rooms(semantic_tree.tracks, semantic_tree.geometry_map, floors)

    semantic_tree.save(semantic_tree_result_dir)
    save_tracks(semantic_tree.tracks, tracks_result_dir)


if __name__ == "__main__":
    # main_semantictree()

    # scene_ids = os.listdir("/pub3/qasim/hm3d/data/concept-graphs/with_edges")
    scene_ids = [
        # "000-hm3d-BFRyYbPCCPE",
        # "002-hm3d-wcojb4TFT35",
        # "003-hm3d-c5eTyR3Rxyh",
        # "004-hm3d-66seV3BWPoX",
        # "005-hm3d-yZME6UR9dUN",
        "006-hm3d-q3hn1WQ12rz",
        # "007-hm3d-bxsVRursffK",
        # "011-hm3d-bzCsHPLDztK",
        # "012-hm3d-XB4GS9ShBRE",
        # "013-hm3d-svBbv1Pavdk",
        # "014-hm3d-rsggHU7g7dh",
        # "015-hm3d-5jp3fCRSRjc",
    ]
    sorted(scene_ids)
    for id in scene_ids:
        sys.argv.append("scene_id=" + id)
        print("PROCESSSING SCENE:", id)
        main()
        # import read_graphs

# from scene_graph.perception import GenericMapper

# perceptor = GenericMapper()
# perceptor.init()
# perceptor.load_results(
#     "/pub3/qasim/hm3d/data/ham-sg/000-hm3d-BFRyYbPCCPE/detections", 1
# )
# print(perceptor)
