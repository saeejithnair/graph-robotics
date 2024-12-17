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
from conceptgraph.dataset import conceptgraphs_datautils
from conceptgraph.dataset.conceptgraphs_rgbd_images import RGBDImages
from conceptgraph.utils.general_utils import measure_time
from conceptgraph.utils.geometry import relative_transformation
from natsort import natsorted
from scipy.spatial.transform import Rotation as R

from scene_graph.semantic_tree import SemanticTree


def read_conceptgraph(graph_path, image_paths):
    f = open(graph_path / "exps/r_mapping/obj_json_r_mapping.json")
    graph = json.load(f)
    f.close()
    graph_keyframes = {}
    for obj in graph:
        keyframes = graph[obj]["image_idxs"]
        keyframes = list(set(keyframes))
        keyframes = [image_paths[i] for i in keyframes]
        graph_keyframes[obj] = keyframes
        graph[obj] = {
            "id": obj,  # graph[obj]['id'],
            "caption": graph[obj]["object_tag"],
            "bbox_center": graph[obj]["bbox_center"],
            "bbox_volume": graph[obj]["bbox_volume"],
        }
    return graph, graph_keyframes


def read_hamsg_flatgraph(graph_path, device="cuda") -> SemanticTree:
    semantic_tree = SemanticTree(device)
    semantic_tree.load(graph_path)
    return semantic_tree

    f = open(graph_path / "tracks/tracks.json")
    graph = json.load(f)
    f.close()

    f = open(graph_path / "semantic_tree/navigation_log.json")
    navigation_log = json.load(f)
    f.close()

    valid_keyframes = set()

    for i in range(len(graph)):
        valid_keyframes.update(graph[i]["keyframe_ids"])
        graph[i] = {
            "id": graph[i]["id"],
            "label": graph[i]["label"],
            "caption": graph[i]["caption"][0],
            "times_scene": graph[i]["times_scene"],
            "centroid": graph[i]["centroid"],
        }
    return graph, navigation_log
