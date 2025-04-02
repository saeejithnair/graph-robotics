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
        normalize_color=False,  # Explicitly set to False
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test accessing the first item
    if len(dataset) > 0:
        # Get processed data from dataset
        color, depth, intrinsics, pose = dataset[0]
        
        print(f"Color shape: {color.shape}, dtype: {color.dtype}")
        print(f"Color min: {color.min().item()}, max: {color.max().item()}")
        print(f"Depth shape: {depth.shape}")
        print(f"Intrinsics:\n{intrinsics}")
        print(f"Pose:\n{pose}")
        
        # Test get_cam_K
        K = dataset.get_cam_K()
        print(f"Camera K:\n{K}")
        
        # Load original image for comparison
        original_path = dataset.color_paths[0]
        original_img = imageio.imread(original_path)
        print(f"Original image shape: {original_img.shape}, dtype: {original_img.dtype}")
        print(f"Original min: {original_img.min()}, max: {original_img.max()}")
        
        # Create a 2x2 visualization
        plt.figure(figsize=(15, 10))
        
        # Original RGB
        plt.subplot(2, 2, 1)
        plt.imshow(original_img)
        plt.title("Original RGB")
        
        # Processed RGB - convert back to uint8 for proper display
        plt.subplot(2, 2, 2)
        color_np = color.cpu().numpy()
        plt.imshow(np.clip(color_np, 0, 255).astype(np.uint8))
        plt.title("Processed RGB")
        
        # Depth map
        plt.subplot(2, 2, 3)
        depth_np = depth.cpu().numpy().squeeze()  # Remove channel dimension
        plt.imshow(depth_np, cmap='viridis')
        plt.colorbar(label='Depth (m)')
        plt.title(f"Depth (min={depth_np.min():.2f}, max={depth_np.max():.2f})")
        
        # RGB pixels comparison 
        plt.subplot(2, 2, 4)
        
        # Plot a scatter plot of original vs processed pixel values for a sample
        sample_size = 1000
        r_orig = original_img[:,:,0].flatten()[::100][:sample_size]
        g_orig = original_img[:,:,1].flatten()[::100][:sample_size]
        b_orig = original_img[:,:,2].flatten()[::100][:sample_size]
        
        r_proc = color_np[:,:,0].flatten()[::100][:sample_size]
        g_proc = color_np[:,:,1].flatten()[::100][:sample_size]
        b_proc = color_np[:,:,2].flatten()[::100][:sample_size]
        
        plt.scatter(r_orig, r_proc, color='red', alpha=0.5, label='R')
        plt.scatter(g_orig, g_proc, color='green', alpha=0.5, label='G')
        plt.scatter(b_orig, b_proc, color='blue', alpha=0.5, label='B')
        plt.plot([0, 255], [0, 255], 'k--', alpha=0.3)  # Identity line
        plt.xlabel('Original pixel values')
        plt.ylabel('Processed pixel values')
        plt.title('Original vs Processed RGB Values')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("masterslam_detailed_test.png")
        print("Saved visualization to masterslam_detailed_test.png")
    else:
        print("Dataset is empty!")

if __name__ == "__main__":
    test_masterslam_dataset() 