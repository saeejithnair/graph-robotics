import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import PIL

from . import utils
from .pointcloud import dynamic_downsample, find_nearest_points


class Features:
    def __init__(
        self, visual_emb=None, caption_emb=None, centroid=None, bbox_3d=None
    ) -> None:
        self.visual_emb = np.array(visual_emb)
        self.caption_emb = np.array(caption_emb)
        self.centroid = np.array(centroid)
        self.bbox_3d = bbox_3d


class Object:
    def __init__(
        self,
        mask=None,
        crop=None,
        label=None,
        caption=None,
        bbox=None,
        confidence=None,
        local_points=None,
        local_pcd_idxs=None,
        features: Features = None,
    ):
        self.mask = mask
        self.crop = crop
        self.label = label
        self.caption = caption
        self.bbox = bbox
        self.confidence = confidence
        self.local_points = local_points
        self.local_pcd_idxs = local_pcd_idxs
        self.features = features

    def extract_local_pcd(
        self, depth_cloud, img_rgb, global_pcd, trans_pose=None, obj_pcd_max_points=5000
    ):
        downsampled_points, downsampled_colors = dynamic_downsample(
            depth_cloud[self.mask],
            colors=img_rgb[self.mask],
            target=obj_pcd_max_points,
        )
        local_pcd = o3d.geometry.PointCloud()
        local_pcd.points = o3d.utility.Vector3dVector(downsampled_points)
        if downsampled_colors is not None:
            local_pcd.colors = o3d.utility.Vector3dVector(downsampled_colors)

        if trans_pose is not None:
            local_pcd.transform(
                trans_pose
            )  # Apply transformation directly to the point cloud

        self.local_points = local_pcd
        _, self.local_pcd_idxs = find_nearest_points(
            np.array(self.local_points.points), np.array(global_pcd.points)
        )
        self.local_pcd_idxs = self.local_pcd_idxs.squeeze(-1)

    def compute_local_pcd(self, full_pcd):
        local_pcd = o3d.geometry.PointCloud()
        local_pcd.points = o3d.utility.Vector3dVector(
            np.array(full_pcd.points)[self.local_pcd_idxs]
        )
        local_pcd.colors = o3d.utility.Vector3dVector(
            np.array(full_pcd.colors)[self.local_pcd_idxs]
        )
        return local_pcd


class ObjectList:
    def __init__(self, objects: Object = None):
        if objects == None:
            objects = []
        self.objects = objects

    def add_object(self, object):
        self.objects.append(object)

    def get_field(self, field: str):
        return [getattr(o, field) for o in self.objects]

    def extract_local_pcd(
        self, depth_cloud, img_rgb, global_pcd, trans_pose=None, obj_pcd_max_points=5000
    ):
        for object in self.objects:
            object.extract_local_pcd(
                depth_cloud, img_rgb, global_pcd, trans_pose, obj_pcd_max_points
            )

    def __iter__(self):
        self.c = 0
        return self

    def __next__(self):
        if self.c < len(self.objects):
            x = self.objects[self.c]
            self.c += 1
            return x
        else:
            raise StopIteration

    def __getitem__(self, item):
        return self.objects[item]

    def save(self, dir, img):
        os.makedirs(dir, exist_ok=True)
        json_path = dir / "object_list.txt"
        annotated_img_path = dir / "annotated_masks.png"
        mask_path = dir / "masks.npz"
        pcd_path = dir / "local_pcd_idxs.npz"
        json_data = []

        for i in range(len(self.objects)):
            obj = self.objects[i]

            mask = np.array(obj.mask)
            crop = utils.annotate_img_masks(
                np.copy(obj.crop), utils.get_crop(mask, obj.bbox), obj.label
            )
            plt.imsave(
                dir / ("bbox_crop-" + str(i) + "-" + obj.label + ".png"),
                np.uint8(crop),
            )

            data = {
                "label": obj.label,
                "caption": obj.caption,
                "bbox": obj.bbox,
                "confidence": obj.confidence,
            }
            # if obj.local_pcd_idxs:
            #     data['local_pcd_idxs'] = obj.local_pcd_idxs

            json_data.append(data)

        local_pcd_idxs = self.get_field("local_pcd_idxs")
        np.savez(pcd_path, *local_pcd_idxs)
        masks = self.get_field("mask")
        np.savez(mask_path, masks=masks)

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)
        annotated_image = utils.annotate_img_masks(
            img, self.get_field("mask"), self.get_field("label")
        )
        plt.imsave(annotated_img_path, np.uint8(annotated_image))

    def __len__(self):
        return len(self.objects)


def load_objects(result_dir, frame):
    frame_dir = Path(result_dir) / str(frame)
    json_path = frame_dir / "object_list.txt"
    masks_path = frame_dir / "masks.npz"
    pcd_path = frame_dir / "local_pcd_idxs.npz"
    img_path = frame_dir / "input_image.png"

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

    masks = np.load(masks_path)["masks"]
    local_pcd_idxs = np.load(pcd_path)
    num_detections = masks.shape[0]
    object_list = ObjectList()
    for i in range(num_detections):
        object_list.add_object(
            Object(
                mask=masks[i],
                crop=crops[i],
                label=json_data[i]["label"],
                caption=json_data[i]["caption"],
                bbox=json_data[i]["bbox"],
                confidence=json_data[i]["confidence"],
                local_pcd_idxs=local_pcd_idxs["arr_" + str(i)],
            )
        )

    return img, object_list
