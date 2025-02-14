import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import omegaconf
import open3d as o3d
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        # print(f"Starting {func.__name__}...")
        result = func(
            *args, **kwargs
        )  # Call the function with any arguments it was called with
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"Done! Execution time of {func.__name__} function: {elapsed_time:.2f} seconds"
        )
        return result  # Return the result of the function call

    return wrapper


def process_cfg(cfg: DictConfig):
    cfg.dataset_root = Path(cfg.dataset_root)
    cfg.dataset_config = Path(cfg.dataset_config)

    if cfg.dataset_config.name != "multiscan.yaml":
        # For datasets whose depth and RGB have the same resolution
        # Set the desired image heights and width from the dataset config
        dataset_cfg = omegaconf.OmegaConf.load(cfg.dataset_config)
        if cfg.image_height is None:
            cfg.image_height = dataset_cfg.camera_params.image_height
        if cfg.image_width is None:
            cfg.image_width = dataset_cfg.camera_params.image_width
        print(
            f"Setting image height and width to {cfg.image_height} x {cfg.image_width}"
        )
    else:
        # For dataset whose depth and RGB have different resolutions
        assert (
            cfg.image_height is not None and cfg.image_width is not None
        ), "For multiscan dataset, image height and width must be specified"

    return cfg


def make_dir(dataset_root, scene_id, exp_suffix, make_dir=True):
    exp_out_path = Path(dataset_root) / scene_id / "exps" / f"{exp_suffix}"
    if make_dir:
        exp_out_path.mkdir(exist_ok=True, parents=True)
    return exp_out_path


def get_exp_out_path(dataset_root, scene_id, exp_suffix, make_dir=True):
    exp_out_path = Path(dataset_root) / scene_id / "exps" / f"{exp_suffix}"
    if make_dir:
        exp_out_path.mkdir(exist_ok=True, parents=True)
    return exp_out_path


def get_vis_out_path(exp_out_path):
    vis_folder_path = exp_out_path / "vis"
    vis_folder_path.mkdir(exist_ok=True, parents=True)
    return vis_folder_path


def get_det_out_path(exp_out_path, make_dir=True):
    detections_folder_path = exp_out_path / "detections"
    if make_dir:
        detections_folder_path.mkdir(exist_ok=True, parents=True)
    return detections_folder_path


def cfg_to_dict(input_cfg):
    """Convert a Hydra configuration object to a native Python dictionary,
    ensuring all special types (e.g., ListConfig, DictConfig, PosixPath) are
    converted to serializable types for JSON. Checks for non-serializable objects."""

    def convert_to_serializable(obj):
        """Recursively convert non-serializable objects to serializable types."""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    def check_serializability(obj, context=""):
        """Attempt to serialize the object, raising an error if not possible."""
        try:
            json.dumps(obj)
        except TypeError as e:
            raise TypeError(f"Non-serializable object encountered in {context}: {e}")

        if isinstance(obj, dict):
            for k, v in obj.items():
                check_serializability(
                    v, context=f"{context}.{k}" if context else str(k)
                )
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                check_serializability(item, context=f"{context}[{idx}]")

    # Convert Hydra configs to native Python types
    # check if its already a dictionary, in which case we don't need to convert it
    if not isinstance(input_cfg, dict):
        native_cfg = OmegaConf.to_container(input_cfg, resolve=True)
    else:
        native_cfg = input_cfg
    # Convert all elements to serializable types
    serializable_cfg = convert_to_serializable(native_cfg)
    # Check for serializability of the entire config
    check_serializability(serializable_cfg)

    return serializable_cfg


def should_exit_early(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)

        # Check if we should exit early
        if data.get("exit_early", False):
            # Reset the exit_early flag to False
            data["exit_early"] = False
            # Write the updated data back to the file
            with open(file_path, "w") as file:
                json.dump(data, file)
            return True
        else:
            return False
    except Exception as e:
        # If there's an error reading the file or the key doesn't exist,
        # log the error and return False
        print(f"Error reading {file_path}: {e}")
        logging.info(f"Error reading {file_path}: {e}")
        return False


def from_intrinsics_matrix(K: torch.Tensor) -> tuple[float, float, float, float]:
    """
    Get fx, fy, cx, cy from the intrinsics matrix

    return 4 scalars
    """
    fx = to_scalar(K[0, 0])
    fy = to_scalar(K[1, 1])
    cx = to_scalar(K[0, 2])
    cy = to_scalar(K[1, 2])
    return fx, fy, cx, cy


def to_scalar(d):
    """
    Convert the d to a scalar
    """
    if isinstance(d, float):
        return d

    elif "numpy" in str(type(d)):
        assert d.size == 1
        return d.item()

    elif isinstance(d, torch.Tensor):
        assert d.numel() == 1
        return d.item()

    else:
        raise TypeError(f"Invalid type for conversion: {type(d)}")


def save_oriented_bounding_boxes(bbox_list, file_path):
    """
    Save a list of OrientedBoundingBox objects to a JSON file.
    """
    data = []
    for bbox in bbox_list:
        try:
            # open3d.cuda.pybind.geometry.OrientedBoundingBox
            bbox_data = {
                "center": bbox.center.tolist(),
                "extent": bbox.extent.tolist(),
                "R": bbox.R.tolist(),  # Rotation matrix
            }
            data.append(bbox_data)
        except:
            # open3d.cuda.pybind.geometry.AxisAlignedBoundingBox
            bbox_data = {
                "center": bbox.get_center().tolist(),
                "extent": bbox.get_extent().tolist(),
                "R": np.identity(3).tolist(),  # Rotation matrix
            }
            data.append(bbox_data)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def load_oriented_bounding_boxes(file_path):
    """
    Load a list of OrientedBoundingBox objects from a JSON file.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    bbox_list = []
    for bbox_data in data:
        bbox = o3d.geometry.OrientedBoundingBox()
        bbox.center = bbox_data["center"]
        bbox.extent = bbox_data["extent"]
        bbox.R = bbox_data["R"]
        bbox_list.append(bbox)

    return bbox_list

