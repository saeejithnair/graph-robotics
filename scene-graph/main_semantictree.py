import hydra
from omegaconf import DictConfig
import utils
from datasets import get_dataset
import torch 
import torchvision 
from tqdm import trange
from pathlib import Path
from PIL import Image
from scenegraph import SceneGraph
import numpy as np
import os, sys, shutil
import open3d as o3d
from perception import Perceptor
from visualizer import Visualizer3D
from relationship_scorer import RelationshipScorer, FeatureComputer
import pointcloud
from track import object_to_track

# A logger for this file
@hydra.main(
    version_base=None,
    config_path="/home/qasim/Projects/graph-robotics/scene-graph/configs/mapping",
    config_name="hm3d_mapping",
)
# @profile
def main_semantictree(cfg: DictConfig):
    tracker = utils.MappingTracker()

    cfg = utils.process_cfg(cfg)
    
    # Initialize the dataset
    device = 'cuda'
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
    shutil.rmtree(perception_result_dir, ignore_errors=True)
    
    scene_graph = SceneGraph()
    scene_graph.load_pcd(folder=result_dir)
    perceptor = Perceptor()
    feature_computer = FeatureComputer(device)
    relationship_scorer = RelationshipScorer(downsample_voxel_size=0.02)
    perceptor.init()
    feature_computer.init()

    exit_early_flag = False
    counter = 0
    for frame_idx in range(19,29): # len(dataset)
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
        
        detections, _ = perceptor.process(color_np)
        detections.extract_local_pcd(depth_cloud,
                                     color_np,
                                     scene_graph.geometry_map,
                                     trans_pose,
                                     cfg.obj_pcd_max_points)
        
        feature_computer.compute_features(image_rgb, scene_graph.geometry_map, detections)
        
        det_matched, matched_tracks = relationship_scorer.associate_dets_to_tracks(detections, scene_graph.get_tracks())
        
        scene_graph.process_detections(
            detections,
            det_matched,
            matched_tracks,
            frame_idx
        )
                
        perceptor.save_results(perception_result_dir, frame_idx)
        detections.save(perception_result_dir / str(frame_idx), perceptor.img)
    
    track_to_parent = relationship_scorer.infer_hierarchy_relationships(scene_graph.tracks, scene_graph.geometry_map)
    
    scene_graph.process_hierarchy(track_to_parent)
    
    visualizer = Visualizer3D()
    visualizer.init()
    scene_graph.visualize_graph(visualizer)
    
    print('Complete')
        
    
if __name__ == '__main__':
    main_semantictree()
    
    # scene_ids = os.listdir('/pub0/qasim/scenegraph/hm3d/data/frames/')
    # sorted(scene_ids)
    # for id in scene_ids:
    #     sys.argv.append("scene_id="+ id )
    #     print('PROCESSSING SCENE:' , id)
    #     main()