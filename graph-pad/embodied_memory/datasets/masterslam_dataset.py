import os
import glob
import numpy as np
import torch
import imageio
from natsort import natsorted
from scipy.spatial.transform import Rotation as R

from .datasets import GradSLAMDataset, as_intrinsics_matrix


class MasterSLAMDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: int = None,
        start: int = 0,
        end: int = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        load_embeddings: bool = False,
        embedding_dir: str = "embeddings",
        embedding_dim: int = 512,
        **kwargs,
    ):
        # Store paths for the dataset
        self.sequence = sequence
        self.base_path = os.path.join(basedir, sequence)
        
        # Use the correct paths directly instead of adding sequence name twice
        self.keyframes_path = os.path.join(self.base_path, "keyframes")
        self.depth_path = os.path.join(self.base_path, "depth")
        self.confidence_path = os.path.join(self.base_path, "confidence")
        self.pose_file = os.path.join(self.base_path, f"{sequence}.txt")
        
        # Print paths for debugging
        print(f"Dataset paths:")
        print(f"  - Base path: {self.base_path}")
        print(f"  - Keyframes path: {self.keyframes_path}")
        print(f"  - Depth path: {self.depth_path}")
        print(f"  - Pose file: {self.pose_file}")
        
        # Check if directories exist
        if not os.path.exists(self.keyframes_path):
            print(f"Error: Keyframes path does not exist: {self.keyframes_path}")
        
        if not os.path.exists(self.depth_path):
            print(f"Error: Depth path does not exist: {self.depth_path}")
        
        if not os.path.exists(self.pose_file):
            print(f"Warning: Pose file does not exist: {self.pose_file}")
        
        # Ensure normalization is off - iPhone images are already in the right format
        kwargs['normalize_color'] = False
        
        # Call parent constructor
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        """Get filepaths for RGB images, depth maps, and embeddings."""
        # Get RGB image paths
        color_paths = natsorted(glob.glob(f"{self.keyframes_path}/*.png"))
        if len(color_paths) == 0:
            print(f"No PNG files found in {self.keyframes_path}")
            return [], [], None
        
        print(f"Found {len(color_paths)} color images")
        
        # Get corresponding depth paths
        depth_paths = []
        for color_path in color_paths:
            # Extract timestamp from filename
            timestamp = os.path.basename(color_path).split('.png')[0]
            depth_file = os.path.join(self.depth_path, f"{timestamp}.npy")
            
            # Check if depth file exists
            if os.path.exists(depth_file):
                depth_paths.append(depth_file)
            else:
                print(f"Warning: Depth file not found for timestamp {timestamp}")
        
        # If lengths don't match, there's a mismatch
        if len(color_paths) != len(depth_paths):
            # Keep only the color paths that have corresponding depth paths
            valid_color_paths = []
            for color_path in color_paths:
                timestamp = os.path.basename(color_path).split('.png')[0]
                depth_file = os.path.join(self.depth_path, f"{timestamp}.npy")
                if os.path.exists(depth_file):
                    valid_color_paths.append(color_path)
            
            color_paths = valid_color_paths
            print(f"Warning: Found {len(color_paths)} valid RGB-D pairs out of {len(color_paths)} RGB images")
        
        # For embeddings, return None since we're not using them initially
        embedding_paths = None
        if self.load_embeddings:
            # If embeddings are needed, implement loading logic here
            embedding_paths = []
            
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        """Load camera poses from the pose file."""
        poses = []
        
        if not os.path.exists(self.pose_file):
            print(f"Error: Pose file not found - {self.pose_file}")
            # Return identity poses as a fallback
            return [torch.eye(4).float() for _ in range(len(self.color_paths))]
            
        # Load pose data from file
        try:
            pose_data = np.loadtxt(self.pose_file)
        except Exception as e:
            print(f"Error loading pose file: {e}")
            return [torch.eye(4).float() for _ in range(len(self.color_paths))]
        
        # Create a dictionary mapping timestamps to poses
        timestamp_to_pose = {}
        for entry in pose_data:
            # Format: timestamp tx ty tz qx qy qz qw
            timestamp = entry[0]
            tx, ty, tz = entry[1:4]
            qx, qy, qz, qw = entry[4:8]
            timestamp_to_pose[timestamp] = (tx, ty, tz, qx, qy, qz, qw)
        
        # Tolerance for timestamp matching (to handle floating point precision)
        timestamp_tolerance = 1e-6
        
        # For each RGB image, find the corresponding pose
        for color_path in self.color_paths:
            # Extract timestamp from filename
            try:
                timestamp = float(os.path.basename(color_path).split('.png')[0])
            except ValueError:
                print(f"Warning: Could not parse timestamp from {color_path}")
                # Add identity pose as fallback
                poses.append(torch.eye(4).float())
                continue
            
            # Find the closest timestamp in the pose data
            closest_timestamp = None
            min_diff = float('inf')
            
            for pose_timestamp in timestamp_to_pose.keys():
                diff = abs(pose_timestamp - timestamp)
                if diff < min_diff:
                    min_diff = diff
                    closest_timestamp = pose_timestamp
            
            # Check if we found a pose within tolerance
            if min_diff > timestamp_tolerance and closest_timestamp is not None:
                print(f"Warning: Using approximate timestamp match for {timestamp} (diff: {min_diff})")
            
            if closest_timestamp is not None:
                # Get pose data for the matched timestamp
                tx, ty, tz, qx, qy, qz, qw = timestamp_to_pose[closest_timestamp]
                
                # Create 4x4 transformation matrix
                T = np.eye(4)
                # Convert quaternion to rotation matrix
                T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
                # Set translation
                T[:3, 3] = [tx, ty, tz]
                
                # Convert to torch tensor and add to poses list
                pose = torch.from_numpy(T).float()
                poses.append(pose)
            else:
                print(f"Error: No pose found for timestamp {timestamp}")
                # Return identity pose as fallback
                pose = torch.eye(4).float()
                poses.append(pose)
        
        return poses
    
    def get_cam_K(self):
        """
        Return camera intrinsics matrix K calculated from image dimensions.
        This is used by the main_embodied_memory.py script for point cloud creation.

        Returns:
            K (torch.Tensor): Camera intrinsics matrix, of shape (3, 3)
        """
        # If we have valid images, use the first one to get dimensions
        if len(self.color_paths) > 0:
            try:
                # Get dimensions from the first image
                img = imageio.imread(self.color_paths[0])
                height, width = img.shape[:2]
                
                # Use image dimensions to estimate intrinsics
                cx, cy = width / 2, height / 2
                fx, fy = width, width  # Approximation based on typical FOV
                
                # Create and return the intrinsics matrix
                K = as_intrinsics_matrix([fx, fy, cx, cy])
                return torch.from_numpy(K)
            except Exception as e:
                print(f"Error calculating camera matrix from image: {e}")
                # Fall back to using the values from the config
                pass
        
        # If we couldn't calculate from image, use the values from the config
        # (even though they might be invalid placeholders)
        return super().get_cam_K()
    
    def __getitem__(self, index):
        """Override __getitem__ to handle dataset-specific needs."""
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        
        # Load RGB image - load as uint8 to preserve pixel values
        orig_img = imageio.imread(color_path)
        
        # Handle RGB color correctly - convert to float after copying
        color = np.asarray(orig_img).astype(np.float32)
        
        # Apply standard preprocessing
        color = self._preprocess_color(color)
        
        # Convert to torch tensor
        color = torch.from_numpy(color)
        
        # Load depth map
        depth = np.load(depth_path)
        depth = self._preprocess_depth(depth, scale=False)  # No scaling needed as depth is in meters
        depth = torch.from_numpy(depth)
        
        # Get image dimensions from the original color image
        height, width = orig_img.shape[:2]
        
        # Create a reasonable default intrinsics matrix
        cx, cy = width / 2, height / 2
        # Use image width as focal length (approximation)
        fx, fy = width, width
        
        # Create intrinsics matrix
        K = as_intrinsics_matrix([fx, fy, cx, cy])
        K = torch.from_numpy(K)
        
        # Scale intrinsics to match desired dimensions
        K = self._scale_intrinsics(K, height, width)
        
        # Create full 4x4 intrinsics matrix
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K
        
        # Get the pose
        pose = self.transformed_poses[index]
        
        # Return the data
        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
        )
    
    def _scale_intrinsics(self, K, orig_height, orig_width):
        """Scale intrinsics matrix to match desired dimensions."""
        height_ratio = float(self.desired_height) / orig_height
        width_ratio = float(self.desired_width) / orig_width
        
        K = K.clone()
        K[0, 0] *= width_ratio   # fx
        K[1, 1] *= height_ratio  # fy
        K[0, 2] *= width_ratio   # cx
        K[1, 2] *= height_ratio  # cy
        
        return K
    
    def read_embedding_from_file(self, embedding_path: str):
        """
        Read embedding from file and process it.
        This method is required by the GradSLAMDataset parent class.
        """
        # Not used in this implementation
        raise NotImplementedError("Embeddings are not supported for MasterSLAMDataset") 