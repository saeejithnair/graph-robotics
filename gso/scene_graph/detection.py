import json
import os
import pickle
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import PIL
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from . import utils
from .edges import Edge, load_edge
from .features import Features
from .pointcloud import denoise_pcd, dynamic_downsample, find_nearest_points
from .utils import get_crop


class Detection:
    def __init__(
        self,
        mask=None,
        crop=None,
        label=None,
        visual_caption=None,
        spatial_caption=None,
        bbox=None,
        confidence=None,
        local_pcd=None,
        matched_track_name=None,
        features: Features = None,
        notes="",
        edges: List[Edge] = [],
    ):
        self.mask = mask
        self.crop = crop
        self.label = label
        self.visual_caption = visual_caption
        self.spatial_caption = spatial_caption
        self.bbox = bbox
        self.confidence = confidence
        self.local_pcd = local_pcd
        self.features = features
        self.matched_track_name = matched_track_name
        self.edges = edges
        self.notes = notes

    def extract_pcd(
        self,
        depth_cloud,
        img_rgb,
        downsample_voxel_size,
        dbscan_remove_noise,
        dbscan_eps,
        dbscan_min_points,
        run_dbscan=True,
        trans_pose=None,
        obj_pcd_max_points=5000,
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

        local_pcd = denoise_pcd(
            local_pcd,
            downsample_voxel_size=downsample_voxel_size,
            dbscan_remove_noise=dbscan_remove_noise,
            dbscan_eps=dbscan_eps,
            dbscan_min_points=dbscan_min_points,
            run_dbscan=run_dbscan,
        )

        if trans_pose is not None:
            local_pcd.transform(
                trans_pose
            )  # Apply transformation directly to the point cloud

        self.local_pcd = local_pcd

    def compute_local_pcd(self):
        return self.local_pcd 


class DetectionList:
    def __init__(self, objects: List[Detection] = None):
        if objects == None:
            objects = []
        self.objects = objects

    def add_object(self, object):
        self.objects.append(object)

    def get_field(self, field: str):
        return [getattr(o, field) for o in self.objects]

    def filter_edges(self, valid_types, valid_object_names):
        for obj in self.objects:
            valid_edges = []
            for i in range(len(obj.edges)):
                if obj.edges[i].verify(valid_types, valid_object_names):
                    valid_edges.append(obj.edges[i])
            obj.edges = valid_edges

    def extract_pcd(
        self,
        depth_cloud,
        img_rgb,
        downsample_voxel_size,
        dbscan_remove_noise,
        dbscan_eps,
        dbscan_min_points,
        trans_pose=None,
        obj_pcd_max_points=5000,
    ):

        for object in self.objects:
            object.extract_pcd(
                depth_cloud,
                img_rgb,
                downsample_voxel_size=downsample_voxel_size,
                dbscan_remove_noise=dbscan_remove_noise,
                dbscan_eps=dbscan_eps,
                dbscan_min_points=dbscan_min_points,
                run_dbscan=True,
                trans_pose=trans_pose,
                obj_pcd_max_points=obj_pcd_max_points,
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
        json_path = dir / "detections.json"
        annotated_img_path = dir / "annotated_masks.png"
        mask_path = dir / "masks.npz"
        pcd_path = dir / "local_pcd.npz"
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
            
            o3d.io.write_point_cloud(str(dir / Path('pcd-'+str(i)+'.pcd')), obj.local_pcd)
            
            data = {
                "label": obj.label,
                "visual caption": obj.visual_caption,
                "spatial caption": obj.spatial_caption,
                "bbox": obj.bbox,
                "confidence": obj.confidence,
                "track name": obj.matched_track_name,
                "edges": [edge.json() for edge in obj.edges],
            }
            # if obj.local_pcd_idxs:
            #     data['local_pcd_idxs'] = obj.local_pcd_idxs

            json_data.append(data)

        masks = self.get_field("mask")
        np.savez(mask_path, masks=masks)

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        annotated_image = utils.annotate_img_masks(
            img, self.get_field("mask"), self.get_field("label")
        )
        plt.imsave(annotated_img_path, np.uint8(annotated_image))

    def __len__(self):
        return len(self.objects)


def load_objects(result_dir, frame, temp_refine="temp-refine"):
    dir = Path(result_dir) / temp_refine / str(frame)
    if not os.path.exists(dir):
        dir = Path(result_dir) / "detections" / str(frame)
    json_path = dir / "detections.json"
    masks_path = dir / "masks.npz"
    img_path = dir / "input_image.png"

    if not dir.exists():
        raise FileNotFoundError(f"Frame directory {dir} does not exist.")
    if not json_path.exists() or not masks_path.exists():
        raise FileNotFoundError(f"Detection files not found in {dir}.")

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
    num_detections = masks.shape[0]
    object_list = DetectionList()
    for i in range(num_detections):
        edges_i = []
        for edge in json_data[i]["edges"]:
            edges_i.append(load_edge(edge))
            
        local_pcd = o3d.io.read_point_cloud(str(dir / Path('pcd-'+str(i)+'.pcd')))
        object_list.add_object(
            Detection(
                mask=masks[i],
                crop=crops[i],
                label=json_data[i]["label"],
                visual_caption=json_data[i]["visual caption"],
                spatial_caption=json_data[i]["visual caption"],
                bbox=json_data[i]["bbox"],
                confidence=json_data[i]["confidence"],
                matched_track_name=json_data[i]["track name"],
                local_pcd=local_pcd,
                edges=edges_i,
            )
        )

    return img, object_list
