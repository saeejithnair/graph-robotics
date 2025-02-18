import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer

import embodied_memory.pointcloud as pointcloud
import embodied_memory.utils as utils
from embodied_memory.datasets import get_dataset
from embodied_memory.embodied_memory import EmbodiedMemory


# A logger for this file
@hydra.main(
    version_base=None,
    config_path="embodied_memory/configs/mapping",
    # config_name="hm3d_mapping",
    config_name="scannet_mapping",
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
    embodied_memory_result_dir = result_dir / f"embodied_memory"
    tracks_result_dir = result_dir / f"tracks"

    embodied_memory = EmbodiedMemory(cfg.visual_memory_size, cfg.room_types)
    embodied_memory.load(result_dir, load_floors=False, load_rooms=False)

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
    for frame_idx in tqdm(range(0, len(dataset), cfg.skip_frame)):  # len(dataset)
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

    embodied_memory.process_rooms(
        room_grid_resolution=cfg.room_grid_resolution,
        img_list=img_list,
        pose_list=pose_list,
        frame_list=frame_list,
        debug_folder=cfg.debug_folder,
        device=device,
    )

    # debug_folder = cfg.debug_folder
    # floors = segment_floors(
    #     embodied_memory.full_scene_pcd,
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

    # assign_tracks_to_rooms(embodied_memory.tracks, embodied_memory.full_scene_pcd, floors)

    embodied_memory.save(embodied_memory_result_dir)


if __name__ == "__main__":

    # scene_ids = os.listdir("/pub3/qasim/hm3d/data/concept-graphs/with_edges")
    # scene_ids = [
    #     # "000-hm3d-BFRyYbPCCPE",
    #     # "001-hm3d-TPhiubUHKcP",
    #     # "002-hm3d-wcojb4TFT35",
    #     # "003-hm3d-c5eTyR3Rxyh",
    #     # "004-hm3d-66seV3BWPoX",
    #     # "005-hm3d-yZME6UR9dUN",
    #     # "006-hm3d-q3hn1WQ12rz",
    #     # "007-hm3d-bxsVRursffK",
    #     # "008-hm3d-SiKqEZx7Ejt",
    #     # "010-hm3d-5cdEh9F2hJL",
    #     # "011-hm3d-bzCsHPLDztK",
    #     # "012-hm3d-XB4GS9ShBRE",
    #     # "013-hm3d-svBbv1Pavdk",
    #     # "014-hm3d-rsggHU7g7dh",
    #     # "015-hm3d-5jp3fCRSRjc",
    #     # "016-hm3d-nrA1tAA17Yp",
    #     # "017-hm3d-Dd4bFSTQ8gi",
    #     # "018-hm3d-dHwjuKfkRUR",
    #     # "019-hm3d-y9hTuugGdiq",
    #     # "021-hm3d-LT9Jq6dN3Ea",
    #     # "023-hm3d-VBzV5z6i1WS",
    #     # "024-hm3d-c3WKCnkEdha",
    #     # "025-hm3d-ziup5kvtCCR",
    #     # "026-hm3d-tQ5s4ShP627",
    #     # "028-hm3d-rXXL6twQiWc",
    #     # "029-hm3d-mv2HUxq3B53",
    #     # "030-hm3d-RJaJt8UjXav",
    #     # "031-hm3d-Nfvxx8J5NCo",
    #     # "032-hm3d-6s7QHgap2fW",
    #     # "033-hm3d-vd3HHTEpmyA",
    #     # "035-hm3d-BAbdmeyTvMZ",
    #     # "036-hm3d-rJhMRvNn4DS",
    #     # "037-hm3d-FnSn2KSrALj",
    #     # "038-hm3d-b28CWbpQvor",
    #     # "039-hm3d-uSKXQ5fFg6u",
    #     # "040-hm3d-HaxA7YrQdEC",
    #     # "041-hm3d-GLAQ4DNUx5U",
    #     # "042-hm3d-hkr2MGpHD6B",
    #     # "046-hm3d-X4qjx5vquwH",
    #     # "048-hm3d-kJJyRFXVpx2",
    #     # "049-hm3d-SUHsP6z2gcJ",
    #     # "050-hm3d-cvZr5TUy5C5",
    #     # "055-hm3d-W7k2QWzBrFY",
    #     # "056-hm3d-7UrtFsADwob",
    #     # "057-hm3d-q3zU7Yy5E5s",
    #     # "058-hm3d-7MXmsvcQjpJ",
    #     # "059-hm3d-T6nG3E2Uui9",
    #     # "068-hm3d-p53SfW6mjZe",
    #     # "069-hm3d-HMkoS756sz6",
    #     # "072-hm3d-a8BtkwhxdRV",
    #     # "073-hm3d-LEFTm3JecaC",
    #     # "083-hm3d-LNg5mXe1BDj",
    #     # "086-hm3d-cYkrGrCg2kB",
    #     # "087-hm3d-mma8eWq3nNQ",
    #     # "088-hm3d-hDBqLgydy1n",
    #     "089-hm3d-AYpsNQsWncn",
    #     "092-hm3d-eF36g7L6Z9M",
    #     "094-hm3d-Qpor2mEya8F",
    #     "096-hm3d-uLz9jNga3kC",
    #     "097-hm3d-QHhQZWdMpGJ",
    #     "098-hm3d-bCPU9suPUw9",
    #     "099-hm3d-q5QZSEeHe5g",
    #     "084-hm3d-zt1RVoi7PcG",
    # ]
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
        # "110-scannet-scene0050_00",
        # "111-scannet-scene0550_00",
        # "112-scannet-scene0591_02",
        # "113-scannet-scene0207_00",
        # "114-scannet-scene0084_00",
        # "115-scannet-scene0046_02",
        # "116-scannet-scene0077_01",
        # "121-scannet-scene0462_00",
        # "122-scannet-scene0647_01",
        # "123-scannet-scene0412_01",
        # "124-scannet-scene0131_02",
        # "125-scannet-scene0426_00",
        # "126-scannet-scene0574_02",
        # "127-scannet-scene0578_00",
        # "128-scannet-scene0678_02",
        # "129-scannet-scene0575_00",
        # "130-scannet-scene0696_00",
        # "132-scannet-scene0645_01",
        # "133-scannet-scene0704_00",
        # "134-scannet-scene0695_03",
        # "135-scannet-scene0131_00",
        # "137-scannet-scene0598_01",
        # "139-scannet-scene0647_00",
        # "141-scannet-scene0651_01",
        # "142-scannet-scene0653_01",
        # "144-scannet-scene0700_01",
        # "145-scannet-scene0193_00",
        # "146-scannet-scene0518_00",
        # "147-scannet-scene0699_00",
        # "148-scannet-scene0203_01",
        # "149-scannet-scene0426_02",
        # "150-scannet-scene0648_00",
        # "151-scannet-scene0217_00",
        # "152-scannet-scene0494_00",
        # "155-scannet-scene0164_02",
        # "156-scannet-scene0461_00",
        # "157-scannet-scene0015_00",
        # "158-scannet-scene0356_00",
        # "160-scannet-scene0488_01",
        # "161-scannet-scene0583_00",
        # "162-scannet-scene0535_00",
        # "163-scannet-scene0164_03",
        # "165-scannet-scene0406_00",
        # "166-scannet-scene0435_03",
        # "167-scannet-scene0307_01",
        # "168-scannet-scene0655_01",
        # "170-scannet-scene0378_01",
        # "171-scannet-scene0222_01",
        # "173-scannet-scene0278_01",
        # "174-scannet-scene0086_01",
        # "175-scannet-scene0329_01",
        # "176-scannet-scene0643_00",
        # "177-scannet-scene0608_02",
        # "178-scannet-scene0685_02",
        # "179-scannet-scene0300_01",
        # "180-scannet-scene0100_02",
        # "181-scannet-scene0314_00",
        # "182-scannet-scene0645_00",
        # "183-scannet-scene0231_01",
        "185-scannet-scene0435_00",
        "186-scannet-scene0549_01",
        "188-scannet-scene0593_00",
        "189-scannet-scene0500_00",
        # Not done:
        "138-scannet-scene0500_01",
        "120-scannet-scene0684_01",
        "140-scannet-scene0077_00",
        "154-scannet-scene0193_01",
        "169-scannet-scene0695_01",
        "187-scannet-scene0655_02",
        "172-scannet-scene0496_00",
    ]
    sorted(scene_ids)
    for id in scene_ids:
        sys.argv.append("scene_id=" + id)
        print("PROCESSSING SCENE:", id)
        main()
        # import read_graphs

# from scene_graph.perception import VLMObjectDetector

# perceptor = VLMObjectDetector()
# perceptor.init()
# perceptor.load_results(
#     "/pub3/qasim/hm3d/data/ham-sg/000-hm3d-BFRyYbPCCPE/detections", 1
# )
# print(perceptor)
