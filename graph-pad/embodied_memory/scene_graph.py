import json
import os
import pickle
from pathlib import Path
from typing import List

import faiss
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from . import utils
from .detection import Detection, DetectionList, Edge, Features
from .edges import load_edge
from .pointcloud import denoise_pcd, find_nearest_points, union_bounding_boxes


class Node:
    def __init__(
        self,
        id,
        label: str,
        captions: List[str],
        features: Features,
        keyframe_ids: List[int],
        masks: list,
        bboxes: list,
        local_pcd,
        crops: list,
        level=None,
        times_scene=0,
        notes="",
        room_id=None,
        edges: List[Edge] = [],
    ) -> None:
        self.id = id
        self.label = label
        self.name = str(id) + " " + label
        self.captions = list(captions)
        self.masks = list(masks)
        self.bboxes = list(bboxes)
        self.crops = list(crops)
        self.edges = edges
        self.features = features
        self.times_scene = times_scene
        self.local_pcd = local_pcd
        self.level = level
        self.keyframe_ids = list(keyframe_ids)
        self.notes = notes
        self.room_id = room_id

    def compute_vis_centroid(self, level=None, height_by_level=False):
        local_points = np.asarray(self.local_pcd.points)
        mean_pcd = np.mean(local_points, 0)
        height = None
        if height_by_level:
            height = self.level * 0.5 + 4  # heursitic
        else:
            max_pcd_height = np.max(-local_points, 0)[1]
            height = max_pcd_height + level * 0.3 + 0.5
        location = [mean_pcd[0], -1 * height, mean_pcd[2]]
        self.vis_centroid = location

    def add_local_pcd(self, local_pcd):
        self.local_pcd += local_pcd

    def compute_local_pcd(self):
        return self.local_pcd

    def merge_detection(self, object: Detection, keyframe_id):
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

        self.add_local_pcd(object.local_pcd)

        if not (object.notes == ""):
            self.notes = object.notes

        self.features.bbox_3d = union_bounding_boxes(
            self.features.bbox_3d, object.features.bbox_3d
        )

        self.times_scene += 1

    def denoise(
        self,
        downsample_voxel_size,
        dbscan_remove_noise,
        dbscan_eps,
        dbscan_min_points,
        run_dbscan=True,
    ):
        local_pcd = self.compute_local_pcd()

        self.local_pcd = denoise_pcd(
            local_pcd,
            downsample_voxel_size=downsample_voxel_size,
            dbscan_remove_noise=dbscan_remove_noise,
            dbscan_eps=dbscan_eps,
            dbscan_min_points=dbscan_min_points,
            run_dbscan=run_dbscan,
        )


node_counter = 0


def reset_node_counter():
    global node_counter
    node_counter = 0


class SceneGraph:
    def __init__(self, nodes={}):
        # nodes is a dictionary mapping from id to the actual node object
        self.nodes = nodes

    def get_node_ids(self):
        return list(self.nodes.keys())

    def get_nodes(self):
        return list(self.nodes.values())

    def __iter__(self):
        self.c = 0
        return self

    def __next__(self):
        if self.c < len(self.nodes):
            x = self.nodes[self.get_node_ids()[self.c]]
            self.c += 1
            return x
        else:
            raise StopIteration

    def __getitem__(self, item):
        return self.nodes[item]

    def __len__(self):
        return len(self.objects)

    def add_node(self, node_id, node):
        self.nodes[node_id] = node

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        save_dir = Path(save_dir)
        json_path = save_dir / "scene_graph.json"
        visual_embs_path = save_dir / "visual_embs.npz"
        caption_embs_path = save_dir / "caption_embs.npz"
        bboxes3d_path = save_dir / "bboxes_3d.txt"
        crops_path = save_dir / "crops.pkl"

        json_data = []
        masks, bboxes, bboxes_3d, crops = [], [], [], []
        visual_embs, caption_embs = [], []
        for id in self.get_node_ids():
            node = self.nodes[id]

            # Bug: Centroid is a moving average of centroid, not actual centroid
            json_data.append(
                {
                    "id": node.id,
                    "label": node.label,
                    "caption": node.captions,
                    "level": node.level,
                    "times_scene": node.times_scene,
                    "keyframe_ids": node.keyframe_ids,
                    "centroid": node.features.centroid.tolist(),
                    "edges": [edge.json() for edge in node.edges],
                    "notes": node.notes,
                    "room_id": node.room_id,
                }
            )

            sample_crop = node.crops[0]
            plt.imsave(
                save_dir / ("node-" + str(node.id) + "-" + str(node.label) + ".png"),
                sample_crop,
            )

            o3d.io.write_point_cloud(
                str(save_dir / Path("pcd-" + str(node.id) + ".pcd")), node.local_pcd
            )

            np.savez(
                save_dir / ("masks-" + str(node.id) + "-" + str(node.label) + ".npz"),
                masks=np.array(node.masks),
            )
            np.savez(
                save_dir / ("bboxes-" + str(node.id) + "-" + str(node.label) + ".npz"),
                bboxes=np.array(node.bboxes),
            )

            crops.append(node.crops)
            bboxes_3d.append(node.features.bbox_3d)
            visual_embs.append(node.features.visual_emb)
            caption_embs.append(node.features.caption_emb)

        np.savez(visual_embs_path, visual_embs=visual_embs)
        np.savez(caption_embs_path, caption_embs=caption_embs)

        utils.save_oriented_bounding_boxes(bboxes_3d, bboxes3d_path)
        with open(crops_path, "wb") as f:
            pickle.dump(crops, f)

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        # Reset node counter
        reset_node_counter()


def load_scenegraph(save_dir):
    nodes = {}
    save_dir = Path(save_dir)
    visual_embs = np.load(save_dir / "visual_embs.npz", allow_pickle=True)[
        "visual_embs"
    ]
    caption_embs = np.load(save_dir / "caption_embs.npz", allow_pickle=True)[
        "caption_embs"
    ]

    with open(save_dir / "scene_graph.json") as f:
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
        local_pcd = o3d.io.read_point_cloud(
            str(save_dir / Path("pcd-" + str(data["id"]) + ".pcd"))
        )

        masks = np.load(
            save_dir / ("masks-" + str(data["id"]) + "-" + str(data["label"]) + ".npz"),
            allow_pickle=True,
        )["masks"]
        bboxes = np.load(
            save_dir
            / ("bboxes-" + str(data["id"]) + "-" + str(data["label"]) + ".npz"),
            allow_pickle=True,
        )["bboxes"]

        trk = Node(
            id=data["id"],
            label=data["label"],
            captions=data["caption"],
            features=features,
            keyframe_ids=data["keyframe_ids"],
            masks=masks,
            bboxes=bboxes,
            local_pcd=local_pcd,
            crops=crops[i],
            level=data["level"],
            times_scene=data["times_scene"],
            notes=data.get("notes", ""),
            room_id=data.get("room_id"),
            edges=[load_edge(e) for e in data["edges"]],
        )
        nodes[data["id"]] = trk
        max_id = max(max_id, data["id"])
    global node_counter
    node_counter = max_id + 1
    return SceneGraph(nodes)


def get_nodeid_by_name(scene_graph: SceneGraph, name):
    for id in scene_graph.get_node_ids():
        if scene_graph.nodes[id].name == name:
            return id
    return None


def object_to_node(object: Detection, keyframe_id, edges=[]):
    global node_counter
    node = Node(
        id=node_counter,
        label=object.label,
        captions=[object.visual_caption],
        features=object.features,
        masks=[object.mask],
        bboxes=[object.bbox],
        keyframe_ids=[keyframe_id],
        local_pcd=object.local_pcd,
        crops=[object.crop],
        notes=object.notes,
        edges=edges,
    )
    node_counter += 1
    return node


def associate_dets_to_nodes(
    new_objects: DetectionList, current_scene_graph, downsample_voxel_size
):
    if len(current_scene_graph) == 0:
        return np.zeros((len(new_objects))), np.zeros(len(new_objects))
    elif len(new_objects) == 0:
        return [], []

    node_pcds = [
        np.asarray(node.compute_local_pcd().points, dtype=np.float32)
        for node in current_scene_graph
    ]
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in node_pcds]
    for index, arr in zip(indices, node_pcds):
        index.add(arr)

    # Compute the score matrix
    visual_scores = np.zeros((len(new_objects), len(current_scene_graph)))
    caption_scores = np.zeros((len(new_objects), len(current_scene_graph)))
    geometry_scores = np.zeros((len(new_objects), len(current_scene_graph)))
    centroid_distances = np.zeros((len(new_objects), len(current_scene_graph)))
    for i in range(len(new_objects)):
        new_obj_pcd = np.array(
            new_objects[i].compute_local_pcd().points, dtype=np.float32
        )
        for j in range(len(current_scene_graph)):
            visual_scores[i][j] = np.dot(
                new_objects[i].features.visual_emb,
                current_scene_graph[j].features.visual_emb,
            )
            caption_scores[i][j] = np.dot(
                new_objects[i].features.caption_emb,
                current_scene_graph[j].features.caption_emb,
            )

            D, I = indices[j].search(new_obj_pcd, 1)
            overlap = (D < downsample_voxel_size**2).sum()  # D is the squared distance

            geometry_scores[i][j] = overlap / len(new_obj_pcd)
            centroid_distances[i][j] = np.linalg.norm(
                new_objects[i].features.centroid
                - current_scene_graph[j].features.centroid
            )

    score_matrix = np.mean(
        np.stack(
            [
                np.where(visual_scores > 0.7, 1, 0),
                np.where(caption_scores > 0.8, 1, 0),
                np.where(geometry_scores > 0.4, 1, 0),
            ]
        ),
        0,
    )
    # score_matrix = np.where(caption_scores > 0.98, 1, score_matrix)
    score_matrix = np.where(centroid_distances > 6, 0, score_matrix)
    score_matrix = np.where(caption_scores < 0.7, 0, score_matrix)

    # Compute the matches
    is_matched = np.max(score_matrix, 1) > 0.5
    matched_nodeidx = np.argmax(score_matrix, 1)

    return is_matched, matched_nodeidx


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
#         json.dump(json_data, f, indent=2)
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
