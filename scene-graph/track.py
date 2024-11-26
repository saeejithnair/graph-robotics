import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import PIL
import torch

import utils
from perception import Perceptor
from pointcloud import dynamic_downsample, find_nearest_points, pcd_denoise_dbscan
from object import Object

track_counter = 0

class Track:
    def __init__(
        self,
        id,
        label,
        caption,
        features,
        keyframe_id,
        mask,
        bbox,
        local_pcd_idxs,
        crop,
        level=None,
        global_pcd=None,
    ) -> None:
        self.id = id
        self.label = label
        self.captions = [caption]
        self.keyframe_ids = [keyframe_id]
        self.masks = [mask]
        self.bboxes = [bbox]
        self.crops = [crop]
        self.features = features
        self.times_scene = 0
        self.local_pcd_idxs = local_pcd_idxs
        self.level = level

    def compute_vis_centroid(self, global_pcd, level=None, height_by_level=False):
        mean_pcd = np.mean(np.asarray(global_pcd.points)[self.local_pcd_idxs], 0)
        height = None
        if height_by_level:
            height = self.level * 0.5 + 4  # heursitic
        else:
            max_pcd_height = np.max(-np.asarray(global_pcd.points)[self.local_pcd_idxs], 0)[
                1
            ]
            height = max_pcd_height + level * 0.3 + 0.5
        location = [mean_pcd[0], -1 * height, mean_pcd[2]]
        self.vis_centroid = location

    def add_local_pcd_idxs(self, local_pcd_idxs):
        self.local_pcd_idxs = np.concatenate((self.local_pcd_idxs, local_pcd_idxs))

    def get_points_at_idxs(self, full_pcd):
        return np.asarray(full_pcd.points)[self.local_pcd_idxs]
    
    def compute_local_pcd(self, full_pcd):
        local_pcd = o3d.geometry.PointCloud()
        local_pcd.points = o3d.utility.Vector3dVector(np.array(full_pcd.points)[self.local_pcd_idxs] )
        local_pcd.colors = o3d.utility.Vector3dVector(np.array(full_pcd.colors)[self.local_pcd_idxs] )
        return local_pcd
    
    def merge_detection(self, object: Object, keyframe_id):
        self.captions.append(object.caption)
        self.keyframe_ids.append(keyframe_id)
        self.masks.append(object.mask)
        self.bboxes.append(object.bbox)
        self.crops.append(object.crop)
        
        self.features.visual_emb = (self.features.visual_emb * self.times_scene + object.features.visual_emb) / (self.times_scene + 1)
        self.features.caption_emb = (self.features.caption_emb * self.times_scene + object.features.caption_emb) / (self.times_scene + 1)
        self.features.centroid = (self.features.centroid * self.times_scene + object.features.centroid) / (self.times_scene + 1)
        
        self.add_local_pcd_idxs(object.local_pcd_idxs)
        
        self.times_scene += 1

def object_to_track(object: Object, keyframe_id, global_pcd):
    global track_counter
    node = Track(
        id=track_counter,
        label=object.label,
        caption=object.caption,
        features=object.features,
        mask=object.mask,
        bbox=object.bbox,
        keyframe_id=keyframe_id,
        local_pcd_idxs=object.local_pcd_idxs,
        crop = object.crop,
        global_pcd=global_pcd
    )
    track_counter += 1
    return node


# class Detection:
#     def __init__(
#         self,
#         object,
#         depth_cloud,
#         img_rgb,
#         global_pcd,
#         trans_pose,
#         obj_pcd_max_points=5000,
#     ) -> None:
#         self.object = object
#         # Create point cloud
#         downsampled_points, downsampled_colors = dynamic_downsample(
#             depth_cloud[object["mask"]],
#             colors=img_rgb[object["mask"]],
#             target=obj_pcd_max_points,
#         )
#         local_pcd = o3d.geometry.PointCloud()
#         local_pcd.points = o3d.utility.Vector3dVector(downsampled_points)
#         if downsampled_colors is not None:
#             local_pcd.colors = o3d.utility.Vector3dVector(downsampled_colors)

#         if trans_pose is not None:
#             local_pcd.transform(
#                 trans_pose
#             )  # Apply transformation directly to the point cloud

#         self.local_points = local_pcd
#         _, self.global_pcd_idxs = find_nearest_points(
#             np.array(self.local_points.points), np.array(global_pcd.points)
#         )

#     def to_node(self, global_pcd):
#         global node_counter
#         node = Node(
#             id=node_counter,
#             local_pcd_idxs=self.global_pcd_idxs[:, 0],
#             crop=self.object["crop"],
#             label=self.object["label"],
#             caption=self.object["caption"],
#             global_pcd=global_pcd,
#         )
#         node_counter += 1
#         return node


def save_results(dir, detections, img):

    os.makedirs(dir, exist_ok=True)
    json_path = dir / "llm_detections.txt"
    annotated_img_path = dir / "annotated_llm_detections.png"
    img_path = dir / "input_img.png"
    mask_path = dir / "masks.npz"

    json_data = []

    for i in range(len(detections)):
        obj = detections[i]

        mask = np.array(obj["mask"])
        crop = utils.annotate_img(
            np.copy(obj["crop"]), utils.get_crop(mask, obj["bbox"]), obj["labels"]
        )
        plt.imsave(
            dir / ("annotated_bbox_crop-" + str(i) + "-" + obj["label"] + ".png"),
            np.uint8(crop),
        )

        data = {
            "label": obj["label"],
            "caption": obj["caption"],
            "bbox": obj["bbox"],
            "confidence": obj["confidence"],
        }
        json_data.append(data)

    masks = [d["masks"] for d in detections]
    np.savez(mask_path, masks=masks)
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)
    annotated_image = utils.annotate_img(img, masks)
    plt.imsave(annotated_img_path, np.uint8(annotated_image))
    plt.imsave(img_path, np.uint8(img))


def load_results(result_dir, frame):
    frame_dir = Path(result_dir) / str(frame)
    json_path = frame_dir / "llm_detections.txt"
    masks_path = frame_dir / "masks.npz"
    img_path = frame_dir / "input_img.png"

    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory {frame_dir} does not exist.")
    if not json_path.exists() or not masks_path.exists():
        raise FileNotFoundError(f"Detection files not found in {frame_dir}.")

    img = PIL.Image.open(img_path).convert("RGB")
    with open(json_path, "r") as f:
        json_data = json.load(f)
    num_detections = len(json_data)

    crops = []
    img_np = np.asarray(img)
    for i in range(num_detections):
        box = json_data[i]["bbox"]
        crops.append(utils.get_crop(img_np, box))

    detections = dict(
        masks=np.load(masks_path)["masks"],
        crops=crops,
        labels=[o["label"] for o in json_data],
        captions=[o["caption"] for o in json_data],
        bboxes=[o["bbox"] for o in json_data],
        confidences=[o["confidence"] for o in json_data],
    )

    return img, num_detections, detections
