import json
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN


class Floor:
    """
    Class to represent a floor in a building.
    :param floor_id: Unique identifier for the floor
    :param name: Name of the floor (e.g., "First", "Second")
    """

    def __init__(self, floor_id, name=None):
        self.floor_id = floor_id  # Unique identifier for the floor
        self.name = name  # Name of the floor (e.g., "First", "Second")
        self.rooms = []  # List of rooms in the floor
        self.txt_embeddings = []  # List of tensors of text embeddings of the floor
        self.pcd = None  # Point cloud of the floor
        self.vertices = []  # indices of the floor in the point cloud 8 vertices
        self.floor_height = None  # Height of the floor
        self.floor_zero_level = None  # Zero level of the floor

    def add_room(self, room):
        """
        Method to add rooms to the floor
        :param room: Room object to be added to the floor
        """
        self.rooms.append(room)  # Method to add rooms to the floor

    def save(self, path):
        """
        Save the floor in folder as ply for the point cloud
        and json for the metadata
        """
        # save the point cloud
        o3d.io.write_point_cloud(
            os.path.join(path, str(self.floor_id) + ".ply"), self.pcd
        )
        # save the metadata
        metadata = {
            "floor_id": self.floor_id,
            "name": self.name,
            "rooms": [room.room_id for room in self.rooms],
            "vertices": self.vertices.tolist(),
            "floor_height": self.floor_height,
            "floor_zero_level": self.floor_zero_level,
        }
        with open(os.path.join(path, str(self.floor_id) + ".json"), "w") as outfile:
            json.dump(metadata, outfile)

    def load(self, path):
        """
        Load the floor from folder as ply for the point cloud
        and json for the metadata
        """
        # load the point cloud
        self.pcd = o3d.io.read_point_cloud(path + "/" + str(self.floor_id) + ".ply")
        # load the metadata
        with open(path + "/" + str(self.floor_id) + ".json") as json_file:
            metadata = json.load(json_file)
            self.name = metadata["name"]
            self.rooms = metadata["rooms"]
            self.vertices = np.asarray(metadata["vertices"])
            self.floor_height = metadata["floor_height"]
            self.floor_zero_level = metadata["floor_zero_level"]

    def __str__(self):
        return f"Floor ID: {self.floor_id}, Name: {self.name}, Rooms: {len(self.rooms)}"


def segment_floors(
    full_pcd,
    graph_tmp_folder,
    save_intermediate_results=False,
    flip_zy=False,
):
    """
    Segment the floors from the full point cloud
    :param path: str, The path to save the intermediate results
    """
    # downsample the point cloud
    downpcd = full_pcd.voxel_down_sample(voxel_size=0.05)
    # flip the z and y axis
    if flip_zy:
        downpcd.points = o3d.utility.Vector3dVector(
            np.array(downpcd.points)[:, [0, 2, 1]]
        )
        downpcd.transform(np.eye(4) * np.array([1, 1, -1, 1]))
    # rotate the point cloud to align floor with the y axis
    T1 = np.eye(4)
    T1[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    downpcd = np.asarray(downpcd.points)
    print("downpcd", downpcd.shape)

    # divide z axis range into 0.01m bin
    reselotion = 0.01
    bins = np.abs(np.max(downpcd[:, 1]) - np.min(downpcd[:, 1])) / reselotion
    print("bins", bins)
    z_hist = np.histogram(downpcd[:, 1], bins=int(bins))
    # smooth the histogram
    z_hist_smooth = gaussian_filter1d(z_hist[0], sigma=2)
    # Find the peaks in this histogram.
    distance = 0.2 / reselotion
    print("distance", distance)
    # set the min peak height based on the histogram
    print(np.mean(z_hist_smooth))
    min_peak_height = np.percentile(z_hist_smooth, 90)
    print("min_peak_height", min_peak_height)
    peaks, _ = find_peaks(z_hist_smooth, distance=distance, height=min_peak_height)

    # plot the histogram
    if save_intermediate_results:
        plt.figure()
        plt.plot(z_hist[1][:-1], z_hist_smooth)
        plt.plot(z_hist[1][peaks], z_hist_smooth[peaks], "x")
        plt.hlines(min_peak_height, np.min(z_hist[1]), np.max(z_hist[1]), colors="r")
        plt.savefig(os.path.join(graph_tmp_folder, "floor_histogram.png"))

    # cluster the peaks using DBSCAN
    peaks_locations = z_hist[1][peaks]
    clustering = DBSCAN(eps=1, min_samples=1).fit(peaks_locations.reshape(-1, 1))
    labels = clustering.labels_

    # plot the histogram
    if save_intermediate_results:
        plt.figure()
        plt.plot(z_hist[1][:-1], z_hist_smooth)
        plt.plot(z_hist[1][peaks], z_hist_smooth[peaks], "x")
        plt.hlines(min_peak_height, np.min(z_hist[1]), np.max(z_hist[1]), colors="r")
        # plot the clusters
        for i in range(len(np.unique(labels))):
            plt.plot(
                z_hist[1][peaks[labels == i]],
                z_hist_smooth[peaks[labels == i]],
                "o",
            )
        plt.savefig(os.path.join(graph_tmp_folder, "floor_histogram_cluster.png"))

    # for each cluster find the top 2 peaks
    clustred_peaks = []
    for i in range(len(np.unique(labels))):
        # for first and last cluster, find the top 1 peak
        if i == 0 or i == len(np.unique(labels)) - 1:
            p = peaks[labels == i]
            top_p = p[np.argsort(z_hist_smooth[p])[-1:]].tolist()
            top_p = [z_hist[1][p] for p in top_p]
            clustred_peaks.append(top_p)
            continue
        p = peaks[labels == i]
        top_p = p[np.argsort(z_hist_smooth[p])[-2:]].tolist()
        top_p = [z_hist[1][p] for p in top_p]
        clustred_peaks.append(top_p)
    clustred_peaks = [item for sublist in clustred_peaks for item in sublist]
    clustred_peaks = np.sort(clustred_peaks)
    print("clustred_peaks", clustred_peaks)

    floors = []
    if len(np.unique(labels)) < 2:
        print("Only one peak detected; treating entire range as a single floor.")
        floors = [[np.min(downpcd[:, 1]), np.max(downpcd[:, 1])]]
    else:
        # for every two consecutive peaks with 2m distance, assign floor level
        for i in range(0, len(clustred_peaks) - 1, 2):
            floors.append([clustred_peaks[i], clustred_peaks[i + 1]])
        print("floors", floors)
        # for the first floor extend the floor to the ground
    floors[0][0] = (floors[0][0] + np.min(downpcd[:, 1])) / 2
    # for the last floor extend the floor to the ceiling
    floors[-1][1] = (floors[-1][1] + np.max(downpcd[:, 1])) / 2
    print("number of floors: ", len(floors))

    final_floors = []
    floors_pcd = []
    for i, floor in enumerate(floors):
        floor_obj = Floor(str(i), name="floor_" + str(i))
        floor_pcd = full_pcd.crop(
            o3d.geometry.AxisAlignedBoundingBox(
                min_bound=(-np.inf, floor[0], -np.inf),
                max_bound=(np.inf, floor[1], np.inf),
            )
        )
        bbox = floor_pcd.get_axis_aligned_bounding_box()
        floor_obj.vertices = np.asarray(bbox.get_box_points())
        floor_obj.pcd = floor_pcd
        floor_obj.floor_zero_level = np.min(np.array(floor_pcd.points)[:, 1])
        floor_obj.floor_height = floor[1] - floor_obj.floor_zero_level
        final_floors.append(floor_obj)
        floors_pcd.append(floor_pcd)
    return final_floors
