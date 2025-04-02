import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from embodied_memory.datasets import get_dataset

def test_masterslam_dataset():
    print("Testing MasterSLAM dataset...")
    
    # Initialize the dataset
    dataset = get_dataset(
        dataconfig="/pub0/smnair/robotics/scene_rep/graph-robotics/graph-pad/embodied_memory/configs/dataconfigs/masterslam.yaml",
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
        
        # Visualize the color and depth
        plt.figure(figsize=(10, 5))
        
        # Convert color to uint8 for proper display
        plt.subplot(1, 2, 1)
        color_np = color.cpu().numpy()
        plt.imshow(np.clip(color_np, 0, 255).astype(np.uint8))
        plt.title("Color")
        
        # Remove channel dimension from depth for proper display
        plt.subplot(1, 2, 2)
        depth_np = depth.cpu().numpy().squeeze()
        plt.imshow(depth_np, cmap='viridis')
        plt.colorbar(label='Depth (m)')
        plt.title(f"Depth (min={depth_np.min():.2f}, max={depth_np.max():.2f})")
        
        plt.tight_layout()
        plt.savefig("masterslam_test.png")
        print("Saved visualization to masterslam_test.png")
    else:
        print("Dataset is empty!")

if __name__ == "__main__":
    test_masterslam_dataset() 