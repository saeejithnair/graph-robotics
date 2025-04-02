import os
import torch
import numpy as np
import sys
from embodied_memory.datasets.masterslam_dataset import MasterSLAMDataset

def test_masterslam_dataset():
    print("Testing MasterSLAM dataset directly...")
    
    # Create a minimal config dict
    config_dict = {
        "dataset_name": "masterslam",
        "camera_params": {
            "image_height": 1440,
            "image_width": 1920,
            "fx": -1,
            "fy": -1,
            "cx": -1,
            "cy": -1,
            "png_depth_scale": 1.0
        }
    }
    
    # Initialize the dataset directly
    dataset = MasterSLAMDataset(
        config_dict=config_dict,
        basedir="/pub0/smnair/robotics/MASt3R-SLAM/logs",
        sequence="IMG_8114_v3_3",
        stride=1,
        start=0,
        end=10,  # Only test first 10 frames
        desired_height=480,
        desired_width=640,
        device="cpu",
        dtype=torch.float,
        normalize_color=False,  # Important: Disable normalization
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test accessing the first item
    if len(dataset) > 0:
        color, depth, intrinsics, pose = dataset[0]
        
        print(f"Color shape: {color.shape}")
        print(f"Depth shape: {depth.shape}")
        print(f"Intrinsics:\n{intrinsics}")
        print(f"Pose:\n{pose}")
        
        # Test get_cam_K
        K = dataset.get_cam_K()
        print(f"Camera K:\n{K}")
        
        # Print pose information
        print(f"Number of poses: {len(dataset.poses)}")
        
    else:
        print("Dataset is empty!")

if __name__ == "__main__":
    test_masterslam_dataset() 