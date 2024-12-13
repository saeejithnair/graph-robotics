from collections import Counter

import faiss
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

from . import utils


def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10):
    """
    Denoise the point cloud using DBSCAN.
    :param pcd: Point cloud to denoise.
    :param eps: Maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_points: The number of samples in a neighborhood for a point to be considered as a core point.
    :return: Denoised point cloud.
    """
    ### Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )

    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]

        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]

        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)

        pcd = largest_cluster_pcd

    return pcd


def find_nearest_points(
    local_points: np.ndarray, global_points: np.ndarray, k: int = 1
):
    """
    k : Number of nearest neighbors to find (default: 1)
    --------
    distances : Squared distances to nearest neighbors, shape (N, k)
    indices : Indices of nearest neighbors in global_points, shape (N, k)
    """
    # Ensure points are float32 (required by FAISS)
    local_points = np.ascontiguousarray(local_points.astype("float32"))
    global_points = np.ascontiguousarray(global_points.astype("float32"))

    # Create FAISS index
    dimension = global_points.shape[1]  # Should be 3 for x,y,z
    index = faiss.IndexFlatL2(dimension)

    # Add global points to the index
    index.add(global_points)

    # Search for nearest neighbors
    distances, indices = index.search(local_points, k)

    return distances, indices


# @profile
def dynamic_downsample(points, colors=None, target=5000):
    """
    Simplified and configurable downsampling function that dynamically adjusts the
    downsampling rate based on the number of input points. If a target of -1 is provided,
    downsampling is bypassed, returning the original points and colors.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) for N points.
        target (int): Target number of points to aim for in the downsampled output,
                      or -1 to bypass downsampling.
        colors (torch.Tensor, optional): Corresponding colors tensor of shape (N, 3).
                                         Defaults to None.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: Downsampled points and optionally
                                                     downsampled colors, or the original
                                                     points and colors if target is -1.
    """
    # Check if downsampling is bypassed
    if target == -1:
        return points, colors

    num_points = points.shape[0]

    # If the number of points is less than or equal to the target, return the original points and colors
    if num_points <= target:
        return points, colors

    # Calculate downsampling factor to aim for the target number of points
    downsample_factor = max(1, num_points // target)

    # Select points based on the calculated downsampling factor
    downsampled_points = points[::downsample_factor]

    # If colors are provided, downsample them with the same factor
    downsampled_colors = colors[::downsample_factor] if colors is not None else None

    return downsampled_points, downsampled_colors


def create_depth_pcdo3d(rgb, depth, camera_pose, cam_K):
    """
    Create a point cloud from RGB-D images.

    Args:
        rgb: RGB image as a numpy array.
        depth: Depth image as a numpy array.
        camera_pose: Camera pose as a numpy array (4x4 matrix).

    Returns:
        Point cloud as an Open3D object.
    """
    # convert rgb and depth images to numpy arrays
    rgb = np.array(rgb)
    depth = np.array(depth)
    # load depth camera intrinsics
    H = depth.shape[0]
    W = depth.shape[1]
    fx, fy, cx, cy = utils.from_intrinsics_matrix(cam_K)
    # create point cloud
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    depth = depth.astype(np.float32)  # / 1000.0
    mask = depth > 0
    x = x[mask]
    y = y[mask]
    depth = depth[mask]
    # convert to 3D
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    # convert to open3d point cloud
    points = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)))

    # Perturb the points a bit to avoid colinearity
    # points += np.random.normal(0, 4e-3, points.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = rgb[mask]
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    pcd.transform(camera_pose)
    return pcd


def create_depth_cloud(depth, cam_K):
    """
    Create a point cloud from RGB-D images.

    Args:
        rgb: RGB image as a numpy array.
        depth: Depth image as a numpy array.
        camera_pose: Camera pose as a numpy array (4x4 matrix).

    Returns:
        Point cloud as an Open3D object.
    """
    # convert rgb and depth images to numpy arrays
    depth = np.array(depth)
    # load depth camera intrinsics
    H = depth.shape[0]
    W = depth.shape[1]
    fx, fy, cx, cy = utils.from_intrinsics_matrix(cam_K)
    # create point cloud
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    depth = depth.astype(np.float32)  # / 1000.0

    # mask = depth > 0
    # x = x[mask]
    # y = y[mask]
    # depth = depth[mask]

    # convert to 3D
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth

    points = np.stack((X, Y, Z), -1)
    return points


def compute_3d_iou(bbox1, bbox2, padding=0, use_iou=True):
    # Get the coordinates of the first bounding box
    bbox1_min = np.asarray(bbox1.get_min_bound()) - padding
    bbox1_max = np.asarray(bbox1.get_max_bound()) + padding

    # Get the coordinates of the second bounding box
    bbox2_min = np.asarray(bbox2.get_min_bound()) - padding
    bbox2_max = np.asarray(bbox2.get_max_bound()) + padding

    # Compute the overlap between the two bounding boxes
    overlap_min = np.maximum(bbox1_min, bbox2_min)
    overlap_max = np.minimum(bbox1_max, bbox2_max)
    overlap_size = np.maximum(overlap_max - overlap_min, 0.0)

    overlap_volume = np.prod(overlap_size)
    bbox1_volume = np.prod(bbox1_max - bbox1_min)
    bbox2_volume = np.prod(bbox2_max - bbox2_min)

    obj_1_overlap = overlap_volume / bbox1_volume
    obj_2_overlap = overlap_volume / bbox2_volume
    max_overlap = max(obj_1_overlap, obj_2_overlap)

    iou = overlap_volume / (bbox1_volume + bbox2_volume - overlap_volume)

    if use_iou:
        return iou
    else:
        return max_overlap


def compute_3d_bbox_distance(bbox1, bbox2, distance_type="surface"):
    """
    Compute the distance between two 3D bounding boxes.

    Parameters:
    - bbox1, bbox2: Bounding box objects with `get_min_bound()` and `get_max_bound()` methods.
    - distance_type: 'center' for center-to-center distance or 'surface' for surface-to-surface distance.

    Returns:
    - Distance between the two bounding boxes.
    """
    # Get the coordinates of the bounding boxes
    bbox1_min = np.asarray(bbox1.get_min_bound())
    bbox1_max = np.asarray(bbox1.get_max_bound())
    bbox2_min = np.asarray(bbox2.get_min_bound())
    bbox2_max = np.asarray(bbox2.get_max_bound())

    if distance_type == "center":
        # Compute the centers of the bounding boxes
        bbox1_center = (bbox1_min + bbox1_max) / 2.0
        bbox2_center = (bbox2_min + bbox2_max) / 2.0

        # Compute Euclidean distance between the centers
        distance = np.linalg.norm(bbox1_center - bbox2_center)

    elif distance_type == "surface":
        # Compute the closest distance between the surfaces
        distance_x = max(
            0, max(bbox1_min[0] - bbox2_max[0], bbox2_min[0] - bbox1_max[0])
        )
        distance_y = max(
            0, max(bbox1_min[1] - bbox2_max[1], bbox2_min[1] - bbox1_max[1])
        )
        distance_z = max(
            0, max(bbox1_min[2] - bbox2_max[2], bbox2_min[2] - bbox1_max[2])
        )

        # Compute the overall closest distance
        distance = np.sqrt(distance_x**2 + distance_y**2 + distance_z**2)
    else:
        raise ValueError("Invalid distance_type. Use 'center' or 'surface'.")

    return distance


def get_bottom_points(points, percentile=10):
    """Helper method to get bottom points of a point cloud"""
    z_threshold = np.percentile(points[:, 2], percentile)
    return points[points[:, 2] <= z_threshold]


def get_top_points(points, percentile=90):
    """Helper method to get top points of a point cloud"""
    z_threshold = np.percentile(points[:, 2], percentile)
    return points[points[:, 2] >= z_threshold]


def union_bounding_boxes(bb1, bb2):
    """
    Computes the union of two bounding boxes (oriented or axis-aligned).

    Parameters:
        bb1, bb2: Bounding boxes (either OrientedBoundingBox or AxisAlignedBoundingBox)

    Returns:
        union_bb: An OrientedBoundingBox or AxisAlignedBoundingBox representing the union.
    """
    if isinstance(bb1, o3d.geometry.AxisAlignedBoundingBox) and isinstance(
        bb2, o3d.geometry.AxisAlignedBoundingBox
    ):
        # Union of Axis-Aligned Bounding Boxes (AABBs)
        min_bound = np.minimum(bb1.min_bound, bb2.min_bound)
        max_bound = np.maximum(bb1.max_bound, bb2.max_bound)
        return o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    elif isinstance(bb1, o3d.geometry.OrientedBoundingBox) or isinstance(
        bb2, o3d.geometry.OrientedBoundingBox
    ):
        # Union of Oriented Bounding Boxes (OBBs)

        # Step 1: Get the corners of both OBBs
        corners_bb1 = np.asarray(bb1.get_box_points())
        corners_bb2 = np.asarray(bb2.get_box_points())

        # Step 2: Combine all corners into a single set
        # Add small random noise to break exact coplanarity
        all_corners = np.vstack((corners_bb1, corners_bb2))
        noise = np.random.normal(0, 1e-10, all_corners.shape)
        all_corners += noise

        # Step 3: Compute the new oriented bounding box that encloses all corners
        union_bb = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(all_corners)
        )

        return union_bb
    else:
        raise ValueError("Unsupported bounding box types")
