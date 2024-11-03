"""
The script is used to model Grounded SAM detections in 3D, it assumes the tag2text classes are avaialable. It also assumes the dataset has Clip features saved for each object/mask.
"""

import copy
import gzip

# Standard library imports
import os
import pickle
import uuid
from collections import Counter
from pathlib import Path

# Third-party imports
import cv2
import hydra
import numpy as np
import open_clip
import scipy.ndimage as ndi
import supervision as sv
import torch
import torchvision
from omegaconf import DictConfig
from open3d.io import read_pinhole_camera_parameters
from PIL import Image
from tqdm import trange
from ultralytics import SAM, YOLO

from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.slam.mapping import (
    aggregate_similarities,
    compute_spatial_similarities,
    compute_visual_similarities,
    match_detections_to_objects,
    merge_obj_matches,
)
from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList
from conceptgraph.slam.utils import (
    denoise_objects,
    detections_to_obj_pcd_and_bbox,
    filter_gobs,
    filter_objects,
    get_bounding_box,
    init_process_pcd,
    make_detection_list_from_pcd_and_gobs,
    merge_objects,
    prepare_objects_save_vis,
    process_cfg,
    process_edges,
    process_pcd,
    processing_needed,
    resize_gobs,
)
from conceptgraph.utils.general_utils import (
    ObjectClasses,
    cfg_to_dict,
    check_run_detections,
    find_existing_image_path,
    get_det_out_path,
    get_exp_out_path,
    get_vis_out_path,
    get_vlm_annotated_image_path,
    handle_rerun_saving,
    load_saved_detections,
    load_saved_hydra_json_config,
    make_vlm_edges_and_captions,
    measure_time,
    save_detection_results,
    save_edge_json,
    save_hydra_config,
    save_obj_json,
    save_objects_for_frame,
    save_pointcloud,
    should_exit_early,
    vis_render_image,
)
from conceptgraph.utils.geometry import rotation_matrix_to_quaternion
from conceptgraph.utils.ious import mask_subtract_contained
from conceptgraph.utils.logging_metrics import DenoisingTracker, MappingTracker
from conceptgraph.utils.model_utils import compute_clip_features_batched

# Local application/library specific imports
from conceptgraph.utils.optional_rerun_wrapper import (
    OptionalReRun,
    orr_log_annotated_image,
    orr_log_camera,
    orr_log_depth_image,
    orr_log_edges,
    orr_log_objs_pcd_and_bbox,
    orr_log_rgb_image,
    orr_log_vlm_image,
)
from conceptgraph.utils.optional_wandb_wrapper import OptionalWandB
from conceptgraph.utils.vis import (
    OnlineObjectRenderer,
    save_video_detections,
    save_video_from_frames,
    vis_result_fast,
    vis_result_fast_on_depth,
    vis_result_for_vlm,
)
from conceptgraph.utils.vlm import (
    consolidate_captions,
    get_obj_rel_from_image_gpt4v,
    get_openai_client,
)

# Disable torch gradient computation
torch.set_grad_enabled(False)

from hm3d_mapping import main
import sys 

if __name__ == "__main__":
    scene_ids = os.listdir('/pub3/qasim/hm3d/data/frames/')
    for id in scene_ids:
        sys.argv.append("scene_id="+ id )
        print('PROCESSSING SCENE:' , id)
        main()
