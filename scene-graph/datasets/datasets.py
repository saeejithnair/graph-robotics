import abc
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from natsort import natsorted
from scipy.spatial.transform import Rotation as R
import open3d as o3d 

from utils import measure_time
import utils
import images
import geometry

def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.
    Input: [fx, fy, cx, cy]
    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K



def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header["dataWindow"]
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header["channels"]:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if "Y" not in header["channels"] else channelData["Y"]

    return Y

def read_txt_to_numpy_array(filename):
  with open(filename, 'r') as f:
    data = []
    for line in f:
      row = [float(x) for x in line.split()]
      data.append(row)
  return np.array(data)

class GradSLAMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_dict,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        channels_first: bool = False,
        normalize_color: bool = False,
        device="cuda:0",
        dtype=torch.float,
        load_embeddings: bool = False,
        embedding_dir: str = "feat_lseg_240_320",
        embedding_dim: int = 512,
        relative_pose: bool = True, # If True, the pose is relative to the first frame
        **kwargs,
    ):
        super().__init__()
        self.name = config_dict["dataset_name"]
        self.device = device
        self.png_depth_scale = config_dict["camera_params"]["png_depth_scale"]

        self.orig_height = config_dict["camera_params"]["image_height"]
        self.orig_width = config_dict["camera_params"]["image_width"]
        self.fx = config_dict["camera_params"]["fx"]
        self.fy = config_dict["camera_params"]["fy"]
        self.cx = config_dict["camera_params"]["cx"]
        self.cy = config_dict["camera_params"]["cy"]

        self.dtype = dtype

        self.desired_height = desired_height
        self.desired_width = desired_width
        self.height_downsample_ratio = float(self.desired_height) / self.orig_height
        self.width_downsample_ratio = float(self.desired_width) / self.orig_width
        self.channels_first = channels_first
        self.normalize_color = normalize_color

        self.load_embeddings = load_embeddings
        self.embedding_dir = embedding_dir
        self.embedding_dim = embedding_dim
        self.relative_pose = relative_pose

        self.start = start
        self.end = end
        if start < 0:
            raise ValueError("start must be positive. Got {0}.".format(stride))
        if not (end == -1 or end > start):
            raise ValueError(
                "end ({0}) must be -1 (use all images) or greater than start ({1})".format(end, start)
            )

        self.distortion = (
            np.array(config_dict["camera_params"]["distortion"])
            if "distortion" in config_dict["camera_params"]
            else None
        )
        self.crop_size = (
            config_dict["camera_params"]["crop_size"]
            if "crop_size" in config_dict["camera_params"]
            else None
        )

        self.crop_edge = None
        if "crop_edge" in config_dict["camera_params"].keys():
            self.crop_edge = config_dict["camera_params"]["crop_edge"]

        self.color_paths, self.depth_paths, self.embedding_paths = self.get_filepaths()
        if len(self.color_paths) != len(self.depth_paths):
            raise ValueError("Number of color and depth images must be the same.")
        if self.load_embeddings:
            if len(self.color_paths) != len(self.embedding_paths):
                raise ValueError(
                    "Mismatch between number of color images and number of embedding files."
                )
        self.num_imgs = len(self.color_paths)
        self.poses = self.load_poses()
        
        if self.end == -1:
            self.end = self.num_imgs

        self.color_paths = self.color_paths[self.start : self.end : stride]
        self.depth_paths = self.depth_paths[self.start : self.end : stride]
        if self.load_embeddings:
            self.embedding_paths = self.embedding_paths[self.start : self.end : stride]
        self.poses = self.poses[self.start : self.end : stride]
        # Tensor of retained indices (indices of frames and poses that were retained)
        self.retained_inds = torch.arange(self.num_imgs)[self.start : self.end : stride]
        # Update self.num_images after subsampling the dataset
        self.num_imgs = len(self.color_paths)

        # self.transformed_poses = datautils.poses_to_transforms(self.poses)
        self.poses = torch.stack(self.poses)
        if self.relative_pose:
            self.transformed_poses = self._preprocess_poses(self.poses)
        else:
            self.transformed_poses = self.poses

    def __len__(self):
        return self.num_imgs

    def get_filepaths(self):
        """Return paths to color images, depth images. Implement in subclass."""
        raise NotImplementedError

    def load_poses(self):
        """Load camera poses. Implement in subclass."""
        raise NotImplementedError

    def _preprocess_color(self, color: np.ndarray):
        r"""Preprocesses the color image by resizing to :math:`(H, W, C)`, (optionally) normalizing values to
        :math:`[0, 1]`, and (optionally) using channels first :math:`(C, H, W)` representation.

        Args:
            color (np.ndarray): Raw input rgb image

        Retruns:
            np.ndarray: Preprocessed rgb image

        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
        """
        color = cv2.resize(
            color,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if self.normalize_color:
            color = images.normalize_image(color)
        if self.channels_first:
            color = images.channels_first(color)
        return color

    def _preprocess_depth(self, depth: np.ndarray, scale=True):
        r"""Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        depth = cv2.resize(
            depth.astype(float),
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth, -1)
        if self.channels_first:
            depth = images.channels_first(depth)
        if scale:
            depth = depth / self.png_depth_scale
        return depth
    
    def _preprocess_poses(self, poses: torch.Tensor):
        r"""Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
        homogenous transformation for all other poses.

        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed

        Returns:
            Output (torch.Tensor): Preprocessed poses

        Shape:
            - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
            - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        """
        return geometry.relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )
        
    def get_cam_K(self):
        '''
        Return camera intrinsics matrix K
        
        Returns:
            K (torch.Tensor): Camera intrinsics matrix, of shape (3, 3)
        '''
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        return K
    
    def read_embedding_from_file(self, embedding_path: str):
        '''
        Read embedding from file and process it. To be implemented in subclass for each dataset separately.
        '''
        raise NotImplementedError

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color = np.asarray(imageio.imread(color_path), dtype=float)
        color = self._preprocess_color(color)
        color = torch.from_numpy(color)
        if ".png" in depth_path:
            # depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = np.asarray(imageio.imread(depth_path), dtype=np.int64)
        elif ".exr" in depth_path:
            depth = readEXR_onlydepth(depth_path)
        elif ".npy" in depth_path:
            depth = np.load(depth_path)
        else:
            raise NotImplementedError

        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color = cv2.undistort(color, K, self.distortion)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        K = geometry.scale_intrinsics(
            K, self.height_downsample_ratio, self.width_downsample_ratio
        )
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        pose = self.transformed_poses[index]

        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index])
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                embedding.to(self.device),  # Allow embedding to be another dtype
                # self.retained_inds[index].item(),
            )

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
            # self.retained_inds[index].item(),
        )

    def create__pcd(self, rgb, depth, camera_pose=None):
        """
        Create a point cloud from RGB-D images.

        Args:
            rgb: RGB image as a numpy array.
            depth: Depth image as a numpy array.
            camera_pose: Camera pose as a numpy array (4x4 matrix).

        Returns:
            Point cloud as an Open3D object.
        """
        raise NotImplementedError()
        
class Hm3dDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = None
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
        color_paths = natsorted(glob.glob(f"{self.input_folder}/*-rgb.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/*-depth.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths
    
    def load_poses(self):
        poses = []
        posefiles = natsorted(glob.glob(f"{self.input_folder}/0*.txt"))
        
        P = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ]
        ).float()
        # P = torch.eye(4).float()
        
        for posefile in posefiles:
            pose = read_txt_to_numpy_array(posefile)
            # pose = np.asarray(pose_raw['pose'])
            
            pose = torch.from_numpy(pose).float()
            # pose[:3,3] /= (65535/10)
            pose = P @ pose @ P.T
            
            poses.append(pose)
            
        return poses

    # def get_cam_K(self):
    #     '''
    #     Return camera intrinsics matrix K
        
    #     Returns:
    #         K (torch.Tensor): Camera intrinsics matrix, of shape (3, 3)
    #     '''
    #     # instrinsic_file = os.path.join(self.input_folder, 'intrinsic_color.txt')
    #     # K = read_txt_to_numpy_array(instrinsic_file)
    #     # # K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
    #     # K = torch.from_numpy(K)
    #     # return K
    #     K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
    #     K = torch.from_numpy(K)
    #     return K

    # def __getitem__(self, index):
    #     color_path = self.color_paths[index]
    #     depth_path = self.depth_paths[index]
    #     color = np.asarray(imageio.imread(color_path), dtype=float)
    #     color = self._preprocess_color(color)
    #     color = torch.from_numpy(color)
    #     if ".png" in depth_path:
    #         # depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    #         depth = np.asarray(imageio.imread(depth_path), dtype=np.int64)
    #     elif ".exr" in depth_path:
    #         depth = readEXR_onlydepth(depth_path)
    #     elif ".npy" in depth_path:
    #         depth = np.load(depth_path)
    #     else:
    #         raise NotImplementedError

    #     K = self.get_cam_K()
    #     if self.distortion is not None:
    #         # undistortion is only applied on color image, not depth!
    #         color = cv2.undistort(color, K, self.distortion)
        
    #     depth = self._preprocess_depth(depth, scale=False)
        
    #     # I added this depth rescaling myself, undoing the effect of the extraction
    #     depth = depth*10/65535
        
    #     depth = torch.from_numpy(depth)

    #     K = conceptgraphs_datautils.scale_intrinsics(
    #         K, self.height_downsample_ratio, self.width_downsample_ratio
    #     )
    #     intrinsics = torch.eye(4).to(K)
    #     intrinsics[:3, :3] = K[:3, :3]

    #     pose = self.transformed_poses[index]

    #     if self.load_embeddings:
    #         embedding = self.read_embedding_from_file(self.embedding_paths[index])
    #         return (
    #             color.to(self.device).type(self.dtype),
    #             depth.to(self.device).type(self.dtype),
    #             intrinsics.to(self.device).type(self.dtype),
    #             pose.to(self.device).type(self.dtype),
    #             embedding.to(self.device),  # Allow embedding to be another dtype
    #             # self.retained_inds[index].item(),
    #         )

    #     return (
    #         color.to(self.device).type(self.dtype),
    #         depth.to(self.device).type(self.dtype),
    #         intrinsics.to(self.device).type(self.dtype),
    #         pose.to(self.device).type(self.dtype),
    #         # self.retained_inds[index].item(),
    #     )