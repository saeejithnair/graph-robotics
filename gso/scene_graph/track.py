import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from . import utils
from .object import Features, Object
from .pointcloud import union_bounding_boxes

track_counter = 0


class Track:
    def __init__(
        self,
        id,
        label: str,
        captions: list,
        features: Features,
        keyframe_ids,
        masks: list,
        bboxes: list,
        local_pcd_idxs,
        crops: list,
        level=None,
        times_scene=0,
    ) -> None:
        self.id = id
        self.label = label
        self.name = str(id) + " " + label

        self.captions = list(captions)
        self.masks = list(masks)
        self.bboxes = list(bboxes)
        self.crops = list(crops)
        self.features = features
        self.times_scene = times_scene
        self.local_pcd_idxs = local_pcd_idxs
        self.level = level
        self.keyframe_ids = list(keyframe_ids)
        self.notes = ""

    def compute_vis_centroid(self, global_pcd, level=None, height_by_level=False):
        mean_pcd = np.mean(np.asarray(global_pcd.points)[self.local_pcd_idxs], 0)
        height = None
        if height_by_level:
            height = self.level * 0.5 + 4  # heursitic
        else:
            max_pcd_height = np.max(
                -np.asarray(global_pcd.points)[self.local_pcd_idxs], 0
            )[1]
            height = max_pcd_height + level * 0.3 + 0.5
        location = [mean_pcd[0], -1 * height, mean_pcd[2]]
        self.vis_centroid = location

    def add_local_pcd_idxs(self, local_pcd_idxs):
        self.local_pcd_idxs = np.concatenate((self.local_pcd_idxs, local_pcd_idxs))

    def get_points_at_idxs(self, full_pcd):
        return np.asarray(full_pcd.points)[self.local_pcd_idxs]

    def compute_local_pcd(self, full_pcd):
        local_pcd = o3d.geometry.PointCloud()
        local_pcd.points = o3d.utility.Vector3dVector(
            np.array(full_pcd.points)[self.local_pcd_idxs]
        )
        local_pcd.colors = o3d.utility.Vector3dVector(
            np.array(full_pcd.colors)[self.local_pcd_idxs]
        )
        return local_pcd

    def merge_detection(self, object: Object, keyframe_id):
        self.captions.append(object.visual_caption)
        self.keyframe_ids.append(keyframe_id)
        self.masks.append(object.mask)
        self.bboxes.append(object.bbox)
        self.crops.append(object.crop)

        self.features.visual_emb = (
            self.features.visual_emb * self.times_scene + object.features.visual_emb
        ) / (self.times_scene + 1)
        self.features.caption_emb = (
            self.features.caption_emb * self.times_scene + object.features.caption_emb
        ) / (self.times_scene + 1)
        self.features.centroid = (
            self.features.centroid * self.times_scene + object.features.centroid
        ) / (self.times_scene + 1)

        self.add_local_pcd_idxs(object.local_pcd_idxs)

        self.features.bbox_3d = union_bounding_boxes(
            self.features.bbox_3d, object.features.bbox_3d
        )

        self.times_scene += 1


def save_tracks(track_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_dir = Path(save_dir)
    json_path = save_dir / "tracks.json"
    masks_path = save_dir / "masks.npz"
    bboxes_path = save_dir / "bboxes.npz"
    visual_embs_path = save_dir / "visual_embs.npz"
    caption_embs_path = save_dir / "caption_embs.npz"
    pcd_path = save_dir / "local_pcd_idxs.npz"
    bboxes3d_path = save_dir / "bboxes_3d.txt"
    crops_path = save_dir / "crops.pkl"

    json_data, local_pcd_idxs = [], []
    masks, bboxes, bboxes_3d, crops = [], [], [], []
    visual_embs, caption_embs = [], []
    for i in track_list:
        trk = track_list[i]
        masks.append(np.array(trk.masks))
        bboxes.append(np.array(trk.bboxes))

        # Bug: Centroid is a moving average of centroid, not actual centroid
        json_data.append(
            {
                "id": trk.id,
                "label": trk.label,
                "caption": trk.captions,
                "level": trk.level,
                "times_scene": trk.times_scene,
                "keyframe_ids": trk.keyframe_ids,
                "centroid": trk.features.centroid.tolist(),
            }
        )
        local_pcd_idxs.append(trk.local_pcd_idxs)

        sample_crop = trk.crops[0]
        plt.imsave(
            save_dir / ("track-" + str(trk.id) + "-" + str(trk.label) + ".png"),
            sample_crop,
        )
        crops.append(trk.crops)
        bboxes_3d.append(trk.features.bbox_3d)
        visual_embs.append(trk.features.visual_emb)
        caption_embs.append(trk.features.caption_emb)

    np.savez(pcd_path, *local_pcd_idxs)
    np.savez(masks_path, masks=masks)
    np.savez(bboxes_path, bboxes=bboxes)
    np.savez(visual_embs_path, visual_embs=visual_embs)
    np.savez(caption_embs_path, caption_embs=caption_embs)

    utils.save_oriented_bounding_boxes(bboxes_3d, bboxes3d_path)
    with open(crops_path, "wb") as f:
        pickle.dump(crops, f)

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)
    # Reset track counter
    reset_track_counter()


def reset_track_counter():
    global track_counter
    track_counter = 0


def load_tracks(save_dir):
    tracks = {}
    save_dir = Path(save_dir)
    masks = np.load(save_dir / "masks.npz", allow_pickle=True)["masks"]
    bboxes = np.load(save_dir / "bboxes.npz", allow_pickle=True)["bboxes"]
    visual_embs = np.load(save_dir / "visual_embs.npz", allow_pickle=True)[
        "visual_embs"
    ]
    caption_embs = np.load(save_dir / "caption_embs.npz", allow_pickle=True)[
        "caption_embs"
    ]
    local_pcd_idxs = np.load(save_dir / "local_pcd_idxs.npz", allow_pickle=True)

    with open(save_dir / "tracks.json") as f:
        text_data = json.load(f)
    with open(save_dir / "crops.pkl", "rb") as f:
        crops = pickle.load(f)

    bboxes_3d = utils.load_oriented_bounding_boxes(save_dir / "bboxes_3d.txt")

    max_id = -1
    for i in range(len(text_data)):
        data = text_data[i]
        features = Features(
            visual_embs[i], caption_embs[i], data["centroid"], bboxes_3d[i]
        )
        trk = Track(
            id=data["id"],
            label=data["label"],
            captions=data["caption"],
            features=features,
            keyframe_ids=data["keyframe_ids"],
            masks=masks[i],
            bboxes=bboxes[i],
            local_pcd_idxs=local_pcd_idxs["arr_" + str(i)],
            crops=crops[i],
            level=data["level"],
            times_scene=data["times_scene"],
        )
        tracks[data["id"]] = trk
        max_id = max(max_id, data["id"])
    global track_counter
    track_counter = max_id + 1
    return tracks


def get_trackid_by_name(tracks, name):
    for id in tracks.keys():
        if tracks[id].name == name:
            return id
    return None


def object_to_track(object: Object, keyframe_id, global_pcd):
    global track_counter
    node = Track(
        id=track_counter,
        label=object.label,
        captions=[object.visual_caption],
        features=object.features,
        masks=[object.mask],
        bboxes=[object.bbox],
        keyframe_ids=[keyframe_id],
        local_pcd_idxs=object.local_pcd_idxs,
        crops=[object.crop],
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


# def save_results(dir, detections, img):

#     os.makedirs(dir, exist_ok=True)
#     json_path = dir / "llm_detections.txt"
#     annotated_img_path = dir / "annotated_llm_detections.png"
#     img_path = dir / "input_img.png"
#     mask_path = dir / "masks.npz"

#     json_data = []

#     for i in range(len(detections)):
#         obj = detections[i]

#         mask = np.array(obj["mask"])
#         crop = utils.annotate_img(
#             np.copy(obj["crop"]), utils.get_crop(mask, obj["bbox"]), obj["labels"]
#         )
#         plt.imsave(
#             dir / ("annotated_bbox_crop-" + str(i) + "-" + obj["label"] + ".png"),
#             np.uint8(crop),
#         )

#         data = {
#             "label": obj["label"],
#             "caption": obj["caption"],
#             "bbox": obj["bbox"],
#             "confidence": obj["confidence"],
#         }
#         json_data.append(data)

#     masks = [d["masks"] for d in detections]
#     np.savez(mask_path, masks=masks)
#     with open(json_path, "w") as f:
#         json.dump(json_data, f, indent=4)
#     annotated_image = utils.annotate_img(img, masks)
#     plt.imsave(annotated_img_path, np.uint8(annotated_image))
#     plt.imsave(img_path, np.uint8(img))


# def load_results(result_dir, frame):
#     frame_dir = Path(result_dir) / str(frame)
#     json_path = frame_dir / "llm_detections.txt"
#     masks_path = frame_dir / "masks.npz"
#     img_path = frame_dir / "input_img.png"

#     if not frame_dir.exists():
#         raise FileNotFoundError(f"Frame directory {frame_dir} does not exist.")
#     if not json_path.exists() or not masks_path.exists():
#         raise FileNotFoundError(f"Detection files not found in {frame_dir}.")

#     img = PIL.Image.open(img_path).convert("RGB")
#     with open(json_path, "r") as f:
#         json_data = json.load(f)
#     num_detections = len(json_data)

#     crops = []
#     img_np = np.asarray(img)
#     for i in range(num_detections):
#         box = json_data[i]["bbox"]
#         crops.append(utils.get_crop(img_np, box))

#     detections = dict(
#         masks=np.load(masks_path)["masks"],
#         crops=crops,
#         labels=[o["label"] for o in json_data],
#         captions=[o["caption"] for o in json_data],
#         bboxes=[o["bbox"] for o in json_data],
#         confidences=[o["confidence"] for o in json_data],
#     )

#     return img, num_detections, detections
