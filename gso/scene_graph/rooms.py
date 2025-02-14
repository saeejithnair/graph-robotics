"""
Room class to represent a room in a HOV-SGraph.
"""

import json
import os
from collections import defaultdict
from typing import Any, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open_clip
import tqdm
from hovsg.utils.clip_utils import (
    custom_siglip_textfeats,
    get_img_feats,
    get_siglip_img_feats,
    get_text_feats_multiple_templates,
)
from hovsg.utils.constants import CLIP_DIM
from hovsg.utils.graph_utils import (
    compute_room_embeddings,
    distance_transform,
    feats_denoise_dbscan,
    find_containment,
    find_intersection_share,
    map_grid_to_point_cloud,
)
from omegaconf import DictConfig
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from scene_graph.floor import Floor
from scene_graph.scene_graph import SceneGraph


class Room:
    """
    Class to represent a room in a building.
    :param room_id: Unique identifier for the room
    :param floor_id: Identifier of the floor this room belongs to
    :param name: Name of the room (e.g., "Living Room", "Bedroom")
    """

    def __init__(self, room_id, floor_id, name=None):
        self.room_id = room_id  # Unique identifier for the room
        self.name = name  # Name of the room (e.g., "room_0")
        self.category = None  # placeholder for a GT category
        self.room_type = None  # (e.g., "Living Room", "Bedroom")
        self.floor_id = floor_id  # Identifier of the floor this room belongs to
        self.objects = []  # List of objects inside the room
        self.vertices = []  # indices of the room in the point cloud 8 vertices
        self.embeddings = []  # List of tensors of embeddings of the room
        self.pcd = None  # Point cloud of the room
        self.room_height = None  # Height of the room
        self.room_zero_level = None  # Zero level of the room
        self.represent_images = []  # 5 images that represent the appearance of the room
        self.object_counter = 0

    def get_semantic_name(self):
        return str(self.room_id) + " " + self.room_type

    def infer_room_type_from_view_embedding(
        self, default_room_types: List[str], clip_tokenizer, clip_model, device
    ) -> str:
        """Use the embeddings stored inside the room to infer room type. We should already
           save k views CLIP embeddings for each room. We match the k embeddings with room
           types' textual CLIP embeddings to get a room label for each of the k views. Then
           we count which room type has the most votes and return that.

        Args:
            default_room_types (List[str]): the output room type should only be a room type from the list.
            clip_model (Any): when the generate_method is set to "embedding", a clip model needs to be
                              provided to the method.
            clip_feat_dim (int): when the generate_method is set to "embedding", the clip features dimension
                                 needs to be provided to this method

        Returns:
            str: a room type from the default_room_types list
        """
        if len(self.embeddings) == 0:
            print("empty embeddings")
            return "unknown room type"
        text_feats = custom_siglip_textfeats(
            default_room_types, clip_tokenizer, clip_model, device
        )
        embeddings = np.array(self.embeddings)
        sim_mat = np.dot(embeddings, text_feats.cpu().T)
        # sim_mat = compute_similarity(embeddings, text_feats)
        print(sim_mat)
        col_ids = np.argmax(sim_mat, axis=1)
        votes = [default_room_types[i] for i in col_ids]
        print(f"the votes are: {votes}")
        unique, counts = np.unique(col_ids, return_counts=True)
        unique_id = np.argmax(counts)
        type_id = unique[unique_id]
        self.name = default_room_types[type_id] + " " + str(self.room_id)
        print(f"The room view ids are {self.represent_images}")
        print(f"The room type is {default_room_types[type_id]}")
        return default_room_types[type_id]

    def save(self, path):
        """
        Save the room in folder as ply for the point cloud
        and json for the metadata
        """
        # save the point cloud
        o3d.io.write_point_cloud(
            os.path.join(path, str(self.room_id) + ".ply"), self.pcd
        )
        # save the metadata
        metadata = {
            "room_id": self.room_id,
            "name": self.name,
            "floor_id": self.floor_id,
            "objects": self.objects,
            "vertices": self.vertices.tolist(),
            "room_height": self.room_height,
            "room_zero_level": self.room_zero_level,
            "embeddings": [i.tolist() for i in self.embeddings],
        }
        with open(os.path.join(path, str(self.room_id) + ".json"), "w") as outfile:
            json.dump(metadata, outfile)

    def load(self, path):
        """
        Load the room from folder as ply for the point cloud
        and json for the metadata
        """
        # load the point cloud
        self.pcd = o3d.io.read_point_cloud(
            os.path.join(path, str(self.room_id) + ".ply")
        )
        # load the metadata
        with open(path + "/" + str(self.room_id) + ".json") as json_file:
            metadata = json.load(json_file)
            self.name = metadata["name"]
            self.floor_id = metadata["floor_id"]
            self.vertices = np.asarray(metadata["vertices"])
            self.room_height = metadata["room_height"]
            self.room_zero_level = metadata["room_zero_level"]
            self.embeddings = [np.asarray(i) for i in metadata["embeddings"]]
            # self.represent_images = metadata["represent_images"]

    def __str__(self):
        return f"Room ID: {self.room_id}, Name: {self.name}, Floor ID: {self.floor_id}, Objects: {len(self.objects)}"


def assign_nodes_to_rooms(scene_graph: SceneGraph, global_pcd, floors):
    """
    Assigns each track to the room with the highest overlap and populates the room_id attribute.
    Uses both overlap-based and distance-based assignment strategies.

    Args:
    tracks (List[Track]): List of Track objects to process.
    global_pcd (o3d.geometry.PointCloud): The global point cloud for the scene.
    floors (List[Floor]): List of Floor objects containing rooms.
    """
    margin = 0.2  # Height margin for associating tracks to floors

    # Iterate through all nodes
    node_ids = scene_graph.get_node_ids()
    for id in node_ids:
        node = scene_graph[id]
        try:
            # Compute the local point cloud for the node
            local_pcd = node.compute_local_pcd()
            local_pcd_points = np.asarray(local_pcd.points)

            # Initialize variables to track the best match
            best_room_name = "unknown"
            best_association = float("-inf")  # Changed to handle negative distances

            # Iterate through all floors to find the best room
            for floor in floors:
                min_z = np.min(local_pcd_points[:, 1])
                max_z = np.max(local_pcd_points[:, 1])

                # Check if the node lies within the current floor's height range
                if min_z > floor.floor_zero_level - margin:
                    # Calculate room associations for all rooms in the floor
                    room_assoc = []
                    for room in floor.rooms:
                        # Calculate overlap between room and node's local point cloud
                        overlap = find_intersection_share(
                            room.vertices, local_pcd_points[:, [0, 2]], 0.2
                        )
                        room_assoc.append(overlap)

                    # If no overlap found with any room, use distance-based assignment
                    if np.sum(room_assoc) == 0:
                        node_center = np.mean(local_pcd_points[:, [0, 2]], axis=0)
                        for r_idx, room in enumerate(floor.rooms):
                            # Calculate negative distance (higher is better)
                            room_center = np.mean(room.vertices, axis=0)
                            distance = -1 * np.linalg.norm(room_center - node_center)
                            room_assoc[r_idx] = distance

                    # Find the best room match for this floor
                    max_assoc = max(room_assoc)
                    if max_assoc > best_association:
                        best_association = max_assoc
                        best_room_idx = np.argmax(room_assoc)
                        best_room_name = floor.rooms[best_room_idx].name

            # Assign the best matching room ID to the track
            node.room_id = best_room_name

            # Log assignment information
            assignment_type = "overlap" if best_association >= 0 else "distance"
            print(
                f"Node {node.id} ('{node.label}') assigned to Room {best_room_name} "
                f"with {assignment_type}-based score {best_association:.2f}"
            )

        except Exception as e:
            print(f"Error processing track {node.name}: {str(e)}")
            continue

    return scene_graph


def assign_frames_to_rooms(poses, global_pcd, floors):
    """
    Assigns each camera pose to a room using both overlap and distance-based methods.

    Args:
        poses (List[np.ndarray]): List of camera poses, each as [x, y, z] coordinates
        global_pcd (o3d.geometry.PointCloud): The global point cloud for the scene
        floors (List[Floor]): List of Floor objects containing rooms

    Returns:
        List[str]: List of room names corresponding to each pose
    """
    margin = 0.2  # Height margin for associating poses to floors
    room_ids = []

    # Iterate through all poses
    for i, pose in enumerate(poses):
        try:
            # Extract pose coordinates
            x, y, z = pose[0], pose[1], pose[2]
            pose_point = np.array([[x, z]])  # Using x,z for 2D plane

            # Initialize variables to track the best match
            best_room_name = "unknown"
            best_association = float("-inf")  # Handle both overlap and distance scores

            # Iterate through all floors to find the best room
            for floor in floors:
                # Check if the pose lies within the current floor's height range
                if y > floor.floor_zero_level - margin:
                    # Calculate room associations for all rooms in the floor
                    room_assoc = []

                    for room in floor.rooms:
                        # Check if point is inside room polygon
                        room_vertices = (
                            room.vertices
                        )  # Assuming vertices are in (x,z) format
                        overlap = find_intersection_share(
                            room_vertices, pose_point, 0.2
                        )
                        room_assoc.append(overlap)

                    # If no overlap found with any room, use distance-based assignment
                    if np.sum(room_assoc) == 0:
                        for r_idx, room in enumerate(floor.rooms):
                            # Calculate negative distance to room center (higher is better)
                            room_center = np.mean(room.vertices, axis=0)
                            distance = -1 * np.linalg.norm(room_center - pose_point[0])
                            room_assoc[r_idx] = distance

                    # Find the best room match for this floor
                    max_assoc = max(room_assoc)
                    if max_assoc > best_association:
                        best_association = max_assoc
                        best_room_idx = np.argmax(room_assoc)
                        best_room_name = floor.rooms[best_room_idx].name

            # Add the best matching room ID to the list
            room_ids.append(best_room_name)

            # Log assignment information
            assignment_type = "overlap" if best_association >= 0 else "distance"
            print(
                f"Frame {i} assigned to Room {best_room_name} "
                f"with {assignment_type}-based score {best_association:.2f}"
            )

        except Exception as e:
            print(f"Error processing pose {i}: {str(e)}")
            room_ids.append("unknown")  # Add default value for failed assignments
            continue

    return room_ids


def segment_rooms(
    floor: Floor,
    clip_processor,
    clip_model,
    grid_resolution,
    rgb_list,
    pose_list,
    frameidx_list,
    room_height_thresh=1.5,
    graph_tmp_folder="outputs/",
    save_intermediate_results=False,
    device="cuda:1",
):
    """
    Segment the rooms from the floor point cloud
    :param floor: Floor, The floor object
    :param path: str, The path to save the intermediate results
    """
    preprocess = clip_processor
    # clip_model, _, preprocess = open_clip.create_model_and_transforms(
    #     "ViT-L-14",
    #     # pretrained=str(self.cfg.models.clip.checkpoint),
    #     device=device,
    # )
    clip_feat_dim = CLIP_DIM["ViT-L-14"]

    tmp_floor_path = os.path.join(graph_tmp_folder, floor.floor_id)
    if not os.path.exists(tmp_floor_path):
        os.makedirs(tmp_floor_path, exist_ok=True)

    floor_pcd = floor.pcd
    xyz = np.asarray(floor_pcd.points)
    xyz_full = xyz.copy()
    floor_zero_level = floor.floor_zero_level
    floor_height = floor.floor_height
    ## Slice below the ceiling ##
    xyz = xyz[xyz[:, 1] < floor_zero_level + floor_height - 0.3]
    xyz = xyz[xyz[:, 1] >= floor_zero_level + room_height_thresh]
    xyz_full = xyz_full[xyz_full[:, 1] < floor_zero_level + floor_height - 0.2]
    ## Slice above the floor and below the ceiling ##
    # xyz = xyz[xyz[:, 1] < floor_zero_level + 1.8]
    # xyz = xyz[xyz[:, 1] > floor_zero_level + 0.8]
    # xyz_full = xyz_full[xyz_full[:, 1] < floor_zero_level + 1.8]

    # project the point cloud to 2d
    pcd_2d = xyz[:, [0, 2]]
    xyz_full = xyz_full[:, [0, 2]]

    # define the grid size and resolution based on the 2d point cloud
    grid_size = (
        int(np.max(pcd_2d[:, 0]) - np.min(pcd_2d[:, 0])),
        int(np.max(pcd_2d[:, 1]) - np.min(pcd_2d[:, 1])),
    )
    grid_size = (grid_size[0] + 1, grid_size[1] + 1)
    print("grid_size: ", grid_resolution)

    # calc 2d histogram of the floor using the xyz point cloud to extract the walls skeleton
    num_bins = (
        int(grid_size[0] // grid_resolution),
        int(grid_size[1] // grid_resolution),
    )
    num_bins = (num_bins[1] + 1, num_bins[0] + 1)
    hist, _, _ = np.histogram2d(pcd_2d[:, 1], pcd_2d[:, 0], bins=num_bins)

    # applythresholding
    hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hist = cv2.GaussianBlur(hist, (5, 5), 1)
    hist_threshold = 0.25 * np.max(hist)
    _, walls_skeleton = cv2.threshold(hist, hist_threshold, 255, cv2.THRESH_BINARY)

    # create a bigger image to avoid losing the walls
    walls_skeleton = cv2.copyMakeBorder(
        walls_skeleton, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0
    )

    # apply closing to the walls skeleton
    kernal = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    walls_skeleton = cv2.morphologyEx(
        walls_skeleton, cv2.MORPH_CLOSE, kernal, iterations=1
    )

    # extract outside boundary from histogram of xyz_full
    hist_full, _, _ = np.histogram2d(xyz_full[:, 1], xyz_full[:, 0], bins=num_bins)
    hist_full = cv2.normalize(hist_full, hist_full, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    hist_full = cv2.GaussianBlur(hist_full, (21, 21), 2)
    _, outside_boundary = cv2.threshold(hist_full, 0, 255, cv2.THRESH_BINARY)

    # create a bigger image to avoid losing the walls
    outside_boundary = cv2.copyMakeBorder(
        outside_boundary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0
    )

    # apply closing to the outside boundary
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    outside_boundary = cv2.morphologyEx(
        outside_boundary, cv2.MORPH_CLOSE, kernal, iterations=3
    )

    # extract the outside contour from the outside boundary
    contours, _ = cv2.findContours(
        outside_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    outside_boundary = np.zeros_like(outside_boundary)
    cv2.drawContours(outside_boundary, contours, -1, (255, 255, 255), -1)
    outside_boundary = outside_boundary.astype(np.uint8)

    if save_intermediate_results:
        plt.figure()
        plt.imshow(walls_skeleton, cmap="gray", origin="lower")
        plt.savefig(os.path.join(tmp_floor_path, "walls_skeleton.png"))

        plt.figure()
        plt.imshow(outside_boundary, cmap="gray", origin="lower")
        plt.savefig(os.path.join(tmp_floor_path, "outside_boundary.png"))

    # combine the walls skelton and outside boundary
    full_map = cv2.bitwise_or(walls_skeleton, cv2.bitwise_not(outside_boundary))

    # apply closing to the full map
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    full_map = cv2.morphologyEx(full_map, cv2.MORPH_CLOSE, kernal, iterations=2)

    if save_intermediate_results:
        # plot the full map
        plt.figure()
        plt.imshow(full_map, cmap="gray", origin="lower")
        plt.savefig(os.path.join(tmp_floor_path, "full_map.png"))
    # apply distance transform to the full map
    room_vertices = distance_transform(full_map, grid_resolution, tmp_floor_path)

    # using the 2D room vertices, map the room back to the original point cloud using KDTree
    room_pcds = []
    room_masks = []
    room_2d_points = []
    floor_tree = cKDTree(np.array(floor_pcd.points))
    for i in tqdm.tqdm(range(len(room_vertices)), desc="Assign floor points to rooms"):
        room = np.zeros_like(full_map)
        room[room_vertices[i][0], room_vertices[i][1]] = 255
        room_masks.append(room)
        room_m = map_grid_to_point_cloud(room, grid_resolution, pcd_2d)
        room_2d_points.append(room_m)
        # extrude the 2D room to 3D room by adding z value from floor zero level to floor zero level + floor height, step by 0.1m
        z_levels = np.arange(floor_zero_level, floor_zero_level + floor_height, 0.05)
        z_levels = z_levels.reshape(-1, 1)
        z_levels *= -1
        room_m3dd = []
        for z in z_levels:
            room_m3d = np.hstack((room_m, np.ones((room_m.shape[0], 1)) * z))
            room_m3dd.append(room_m3d)
        room_m3d = np.concatenate(room_m3dd, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(room_m3d)
        # rotate floor pcd to align with the original point cloud
        T1 = np.eye(4)
        T1[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        pcd.transform(T1)
        # find the nearest point in the original point cloud
        _, idx = floor_tree.query(np.array(pcd.points), workers=-1)
        pcd = floor_pcd.select_by_index(idx)
        room_pcds.append(pcd)

    # CODE I MODIFIED
    # compute the features of room: input a list of poses and images, output a list of embeddings list
    F_g_list = []
    all_global_clip_feats = dict()

    # for i in range(len(rgb_list)):
    #     rgb_image = rgb_list[i]
    #     F_g = get_siglip_img_feats(
    #         np.array(rgb_image), preprocess, clip_model, device=device
    #     )
    #     F_g_list.append(F_g)
    batch = get_siglip_img_feats(rgb_list, clip_processor, clip_model.to(device), device=device)
    for i in range(batch.shape[0]):
        F_g_list.append(batch[i].cpu().detach().numpy())

    np.savez(
        os.path.join(graph_tmp_folder, "room_views.npz"),
        **all_global_clip_feats,
    )

    pcd_min = np.min(np.array(floor_pcd.points), axis=0)
    pcd_max = np.max(np.array(floor_pcd.points), axis=0)
    assert pcd_min.shape[0] == 3

    repr_embs_list, repr_img_ids_list = compute_room_embeddings(
        room_pcds, pose_list, F_g_list, pcd_min, pcd_max, 50, tmp_floor_path
    )
    assert len(repr_embs_list) == len(room_2d_points)
    assert len(repr_img_ids_list) == len(room_2d_points)

    room_index = 0
    rooms = []
    for i in range(len(room_2d_points)):
        room = Room(
            str(floor.floor_id) + "_" + str(room_index),
            floor.floor_id,
            name="room_" + str(room_index),
        )
        room.pcd = room_pcds[i]
        room.vertices = room_2d_points[i]
        room.room_height = floor_height
        room.room_zero_level = floor.floor_zero_level
        room.embeddings = repr_embs_list[i]
        room.represent_images = [frameidx_list[k] for k in repr_img_ids_list[i]]
        rooms.append(room)
        room_index += 1
    return rooms

    # CODE I REMOVED
    # self.room_masks[floor.floor_id] = room_masks
    # self.floors[int(floor.floor_id)].add_room(room)
    # print(
    #     "number of rooms in floor {} is {}".format(
    #         floor.floor_id, len(self.floors[int(floor.floor_id)].rooms)
    #     )
    # )
