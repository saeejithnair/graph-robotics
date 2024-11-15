import hydra
from omegaconf import DictConfig
from utils import MappingTracker, process_cfg, cfg_to_dict
import utils
from datasets import get_dataset
import torch 
import torchvision 
from tqdm import trange
from pathlib import Path
from PIL import Image
from scenegraph import SceneGraph
import numpy as np
import os
import sys
import open3d as o3d
import pointcloud

# A logger for this file
@hydra.main(
    version_base=None,
    config_path="/home/qasim/Projects/graph-robotics/scene-graph/configs/mapping",
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

    # output folder for this mapping experiment
    exp_out_path = utils.get_exp_out_path(cfg.result_root, cfg.scene_id, cfg.exp_suffix)

    # output folder of the detections experiment to use
    det_exp_path = utils.get_exp_out_path(
        cfg.result_root, cfg.scene_id, cfg.detections_exp_suffix, make_dir=False
    )

    # we need to make sure to use the same classes as the ones used in the detections
    detections_exp_cfg = cfg_to_dict(cfg)

    # if we need to do detections
    det_exp_pkl_path = utils.get_det_out_path(det_exp_path)
    det_exp_vis_path = utils.get_vis_out_path(det_exp_path)

    prev_adjusted_pose = None
    resize_transform = torchvision.transforms.Resize((cfg.image_height, cfg.image_width))
    scene_graph = SceneGraph()

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
        
        new_pcd = pointcloud.create_depth_pcdo3d(image_rgb, depth_array, adjusted_pose, dataset.get_cam_K())
        scene_graph.add_pcd(new_pcd)
        
        # prev_adjusted_pose = orr_log_camera(
        #     intrinsics,
        #     adjusted_pose,
        #     prev_adjusted_pose,
        #     cfg.image_width,
        #     cfg.image_height,
        #     frame_idx,
        # )
        # orr_log_rgb_image(color_path)
        # orr_log_annotated_image(color_path, det_exp_vis_path)
        # orr_log_depth_image(depth_tensor)
        # orr_log_vlm_image(vis_save_path_for_vlm)
        # orr_log_vlm_image(vis_save_path_for_vlm_edges, label="w_edges")
        # orr_log_objs_pcd_and_bbox(objects, obj_classes)
        # orr_log_edges(objects, map_edges, obj_classes)
        
        # I added this manually, to resolve shape issues
        image_rgb = (color_np).astype(np.uint8)  # (H, W, 3)

        # resize the observation if needed
        # resized_gobs = resize_gobs(raw_gobs, image_rgb)
        # filtered_gobs = filter_gobs(
        #     resized_gobs,
        #     image_rgb,
        #     skip_bg=cfg.skip_bg,
        #     BG_CLASSES=obj_classes.get_bg_classes_arr(),
        #     mask_area_threshold=cfg.mask_area_threshold,
        #     max_bbox_area_ratio=cfg.max_bbox_area_ratio,
        #     mask_conf_threshold=cfg.mask_conf_threshold,
        # )
        # gobs = filtered_gobs
        # if len(gobs["mask"]) == 0:  # no detections in this frame
        #     continue
        
    scene_graph.denoise_geometrymap(
        downsample_voxel_size= 0.02, # cfg["downsample_voxel_size"],
        dbscan_eps= 0.01, # cfg["dbscan_eps"],
        min_points= 100 # cfg['dbscan_min_points']
        # dbscan_min_points=cfg["dbscan_min_points"],
    )
    
    o3d.io.write_point_cloud("outputs/pcd/" + cfg.scene_id + '.pcd', scene_graph.full_pcd)

        
if __name__ == '__main__':
    main_geometry()
    
    # scene_ids = os.listdir('/pub0/qasim/scenegraph/hm3d/data/frames/')
    # sorted(scene_ids)
    # for id in scene_ids:
    #     sys.argv.append("scene_id="+ id )
    #     print('PROCESSSING SCENE:' , id)
    #     main()