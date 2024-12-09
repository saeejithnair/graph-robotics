import os

import faiss
import numpy as np
import supervision as sv
import torch
from PIL import Image
from scipy.spatial import ConvexHull
from transformers import AutoModel, AutoProcessor, AutoTokenizer

import pointcloud
from object import Features, ObjectList, load_objects
from utils import get_crop


class FeatureComputer:
    def __init__(self, device="cuda") -> None:
        os.environ["HF_HOME"] = os.path.join(os.getcwd(), "checkpoints")
        self.device = device

    def init(self):
        self.clip_processor = AutoProcessor.from_pretrained(
            "google/siglip-base-patch16-224"
        )
        self.clip_model = AutoModel.from_pretrained(
            "google/siglip-base-patch16-224"
        ).to(self.device)
        self.sentence_tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/bge-small-en-v1.5"
        )
        self.sentence_model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")

    def compute_features(self, img, full_pcd, objects: ObjectList):
        if len(objects) == 0:
            return

        visual_features = self.compute_clip_features_avg(img, objects.get_field("mask"))
        caption_features = self.compute_sentence_features(objects.get_field("caption"))
        centroids = []
        bboxes_3d = []
        for i in range(len(objects)):
            local_pcd = objects[i].compute_local_pcd(full_pcd)
            centroids.append(np.mean(local_pcd.points, 0))
            bboxes_3d.append(self.get_bounding_box(local_pcd))

        for i in range(len(objects)):
            objects[i].features = Features(
                visual_features[i], caption_features[i], centroids[i], bboxes_3d[i]
            )

    def get_bounding_box(self, pcd):
        try:
            return pcd.get_oriented_bounding_box(robust=True)
        except RuntimeError as e:
            print(f"Met {e}, use axis aligned bounding box instead")
            return pcd.get_axis_aligned_bounding_box()

    def compute_sentence_features(self, captions):
        encoded_input = self.sentence_tokenizer(
            captions, padding=True, truncation=True, return_tensors="pt"
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.sentence_model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1
        )
        return sentence_embeddings

    def compute_clip_features(self, image, image_crops=None, bboxes=None, padding=5):
        if image_crops == None:
            image_crops = []
            for idx in range(len(bboxes)):
                x_min, y_min, x_max, y_max = bboxes[idx]
                image_width, image_height = image.size
                left_padding = min(padding, x_min)
                top_padding = min(padding, y_min)
                right_padding = min(padding, image_width - x_max)
                bottom_padding = min(padding, image_height - y_max)

                x_min -= left_padding
                y_min -= top_padding
                x_max += right_padding
                y_max += bottom_padding

                cropped_image = get_crop(image, (x_min, y_min, x_max, y_max))

                image_crops.append(cropped_image)

        # Convert lists to batches
        preprocessed_images_batch = self.clip_processor(
            images=image_crops, return_tensors="pt"
        ).to(self.device)

        # Batch inference
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(
                **preprocessed_images_batch
            )
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # text_features = clip_model.encode_text(text_tokens_batch)
            # text_features /= text_features.norm(dim=-1, keepdim=True)

        # Convert to numpy
        image_feats = image_features.detach().cpu().numpy()
        # text_feats = text_features.cpu().numpy()
        # image_feats = []

        return image_crops, image_feats

    def compute_clip_features_avg(self, image, masks, bbox_crops=None):
        bbox_masks = sv.mask_to_xyxy(np.array(masks))
        masks_crops = []
        masks_crops_nocontext = []
        for i in range(len(masks)):
            mask_box = bbox_masks[i]
            mask_crop = get_crop(image, mask_box)
            mask = get_crop(masks[i], mask_box)
            # mask = sv.crop_image(masks[i], mask_box)

            mask_crop_nocontext = np.where(mask[:, :, np.newaxis], mask_crop, 0)

            masks_crops.append(mask_crop)
            masks_crops_nocontext.append(mask_crop_nocontext)

        feats = [
            self.compute_clip_features(image, image_crops=masks_crops)[1],
            self.compute_clip_features(image, image_crops=masks_crops_nocontext)[1],
        ]

        if bbox_crops:
            feats.append(self.compute_clip_features(image, image_crops=bbox_crops))

        feat_vector = np.mean(np.stack(feats, 0), 0)
        return feat_vector


class RelationshipScorer:
    def __init__(self, downsample_voxel_size) -> None:
        self.downsample_voxel_size = downsample_voxel_size

    # def compute_contained_within(self, child_points: np.ndarray, parent_points: np.ndarray) -> bool:
    #     # Create convex hull of parent
    #     try:
    #         parent_hull = ConvexHull(parent_points)
    #     except Exception:
    #         return False

    #     # Check what fraction of child points lie within parent's convex hull
    #     points_inside = 0
    #     for point in child_points:
    #         # Use point-in-hull test
    #         if self._point_in_hull(point, parent_points[parent_hull.vertices]):
    #             points_inside += 1

    #     containment_ratio = points_inside / len(child_points)
    #     return containment_ratio > self.threshold_containment

    # def compute_contained_within(self, child_points: np.ndarray, parent_points: np.ndarray) -> bool:
    #     # Create convex hull of parent
    #     try:
    #         parent_hull = ConvexHull(parent_points)
    #     except Exception:
    #         return False

    #     # Check what fraction of child points lie within parent's convex hull
    #     points_inside = 0
    #     for point in child_points:
    #         # Use point-in-hull test
    #         if self._point_in_hull(point, parent_points[parent_hull.vertices]):
    #             points_inside += 1

    #     containment_ratio = points_inside / len(child_points)
    #     return containment_ratio > self.threshold_containment

    # def compute_supported_by(self, child_points: np.ndarray, parent_points: np.ndarray) -> bool:
    #     """
    #     Check if child point cloud is supported by parent point cloud

    #     Args:
    #         child_points: Nx3 array of potential child point cloud
    #         parent_points: Mx3 array of potential parent point cloud

    #     Returns:
    #         bool: True if child is likely supported by parent
    #     """
    #     # Find the bottom points of the child (lowest 10% in z-axis)
    #     child_bottom = pointcloud.get_bottom_points(child_points)

    #     # Find the top points of the parent (highest 10% in z-axis)
    #     parent_top = pointcloud.get_top_points(parent_points)

    #     # Count contact points
    #     contact_points = 0
    #     for child_point in child_bottom:
    #         for parent_point in parent_top:
    #             if np.linalg.norm(child_point - parent_point) < self.contact_threshold:
    #                 contact_points += 1
    #                 break

    #     contact_ratio = contact_points / len(child_bottom)
    #     return contact_ratio > self.threshold_support
    # def _point_in_hull(self, point, hull_points):
    #     """Helper method to check if a point lies within a convex hull"""
    #     hull = ConvexHull(hull_points)
    #     new_points = np.vstack((hull_points, point))
    #     new_hull = ConvexHull(new_points)
    #     return np.array_equal(hull.vertices, new_hull.vertices)

    def associate_dets_to_tracks(self, new_objects: ObjectList, current_tracks):
        if len(current_tracks) == 0:
            return np.zeros((len(new_objects))), np.zeros(len(new_objects))
        elif len(new_objects) == 0:
            return [], []
        # Compute the score matrix

        visual_scores = np.zeros((len(new_objects), len(current_tracks)))
        caption_scores = np.zeros((len(new_objects), len(current_tracks)))
        centroid_scores = np.zeros((len(new_objects), len(current_tracks)))
        for i in range(len(new_objects)):
            for j in range(len(current_tracks)):
                visual_scores[i][j] = np.dot(
                    new_objects[i].features.visual_emb,
                    current_tracks[j].features.visual_emb,
                )
                caption_scores[i][j] = np.dot(
                    new_objects[i].features.caption_emb,
                    current_tracks[j].features.caption_emb,
                )
                centroid_scores[i][j] = np.linalg.norm(
                    new_objects[i].features.centroid
                    - current_tracks[j].features.centroid
                )

        score_matrix = np.mean(
            np.stack(
                [
                    np.where(visual_scores > 0.8, 1, 0),
                    np.where(caption_scores > 0.8, 1, 0),
                    np.where(centroid_scores < 0.4, 1, 0),
                ]
            ),
            0,
        )

        # Compute the matches
        matched = np.max(score_matrix, 1) > 0.5
        matched_nodes = np.argmax(score_matrix, 1)

        return matched, matched_nodes

    def find_neighbours(self, current_tracks, distance_threshold=1):
        track_ids = list(current_tracks.keys())
        track_values = list(current_tracks.values())
        n_tracks = len(track_ids)
        neighbours = np.zeros((n_tracks, n_tracks))
        for i in range(n_tracks):
            for j in range(i + 1, n_tracks):
                box_i = track_values[i].features.bbox_3d
                box_j = track_values[j].features.bbox_3d

                # iou = pointcloud.compute_3d_iou(box_i, box_j)
                distance = pointcloud.compute_3d_bbox_distance(box_i, box_j)

                neighbours[i][j] = neighbours[j][i] = int(distance < distance_threshold)
        return neighbours

    def infer_hierarchy_relationships(
        self,
        current_tracks,
        full_pcd,
        neighbour_thresh=1,
        downsample_voxel_size=0.02,
        in_threshold=0.8,
        on_threshold=0.8,
    ):
        track_ids = list(current_tracks.keys())
        track_values = list(current_tracks.values())
        n_tracks = len(track_ids)

        neighbour_matrix = self.find_neighbours(current_tracks, neighbour_thresh)
        in_matrix = np.zeros((n_tracks, n_tracks))
        on_matrix = np.zeros((n_tracks, n_tracks))

        # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
        track_pcds = [
            np.asarray(trk.compute_local_pcd(full_pcd).points, dtype=np.float32)
            for trk in track_values
        ]
        indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in track_pcds]
        for index, arr in zip(indices, track_pcds):
            index.add(arr)

        track_top_pcds = [
            pointcloud.get_top_points(
                np.asarray(trk.compute_local_pcd(full_pcd).points)
            )
            for trk in track_values
        ]
        track_bottom_pcds = [
            pointcloud.get_bottom_points(
                np.asarray(trk.compute_local_pcd(full_pcd).points)
            )
            for trk in track_values
        ]
        top_indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in track_top_pcds]
        for index, arr in zip(top_indices, track_top_pcds):
            index.add(arr)

        for i in range(n_tracks):
            for j in range(n_tracks):
                if not neighbour_matrix[i][j]:
                    continue

                # Probability track_i is in track_j
                D, I = indices[j].search(track_pcds[i], 1)
                overlap = (
                    D < downsample_voxel_size**2
                ).sum()  # D is the squared distance
                in_matrix[i, j] = overlap / len(track_pcds[i])

                # Probability track_i is on track_j
                D, I = top_indices[j].search(track_bottom_pcds[i], 1)
                overlap = (
                    D < downsample_voxel_size**2
                ).sum()  # D is the squared distance
                on_matrix[i, j] = overlap / len(track_bottom_pcds[i])

        in_matrix = np.where(in_matrix > in_threshold, in_matrix, 0)
        on_matrix = np.where(on_matrix > on_threshold, on_matrix, 0)
        hierarchy_matrix = np.logical_or(on_matrix, in_matrix)
        return neighbour_matrix, in_matrix, on_matrix, hierarchy_matrix

    # def infer_hierarchy_relationships(self, current_tracks, full_pcd):
    #     ontop_matrix = self.compute_ontop_matrix(current_tracks, full_pcd)
    #     containedin_matrix = self.compute_containedin_matrix(current_tracks, full_pcd)

    #     ontop_matrix = np.where(ontop_matrix > self.threshold_support, 1, 0)
    #     containedin_matrix = np.where(containedin_matrix > self.threshold_containment, 1, 0)

    #     parent_matrix = np.logical_or(ontop_matrix, containedin_matrix)

    #     parent_prob = np.max(parent_matrix, 1)
    #     parent_indxs = np.argmax(parent_matrix, 1)
    #     parent_ids = [current_tracks[idx].id for idx in parent_indxs]

    #     track_to_parent = {}
    #     i = 0
    #     for id in current_tracks.keys():
    #         if parent_prob[i] < 0.5:
    #             track_to_parent[id] = None
    #         else:
    #             track_to_parent[id] = parent_ids[i]
    #         i += 1
    #     return track_to_parent

    def compute_containedin_matrix(self, tracks, full_pcd):
        downsample_voxel_size = self.downsample_voxel_size
        track_values = list(tracks.values())

        # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
        track_pcds = [
            np.asarray(trk.compute_local_pcd(full_pcd).points, dtype=np.float32)
            for trk in track_values
        ]
        indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in track_pcds]

        # Add the points from the numpy arrays to the corresponding FAISS indices
        for index, arr in zip(indices, track_pcds):
            index.add(arr)

        overlap_matrix = np.zeros((len(track_values), len(track_values)))
        # Compute the pairwise overlaps
        for i in range(len(track_values)):
            for j in range(len(track_values)):
                if i == j:
                    continue
                box_i = track_values[i].features.bbox_3d
                box_j = track_values[j].features.bbox_3d

                # Skip if the boxes do not overlap at all (saves computation)
                iou = pointcloud.compute_3d_iou(box_i, box_j)
                if iou == 0:
                    continue

                # # Use range_search to find points within the threshold
                # _, I = indices[j].range_search(point_arrays[i], threshold ** 2)
                D, I = indices[j].search(track_pcds[i], 1)

                # # If any points are found within the threshold, increase overlap count
                # overlap += sum([len(i) for i in I])

                overlap = (
                    D < downsample_voxel_size**2
                ).sum()  # D is the squared distance

                # Calculate the ratio of points within the threshold
                overlap_matrix[i, j] = overlap / len(track_pcds[i])

        return overlap_matrix

    def compute_ontop_matrix(self, tracks, full_pcd):
        downsample_voxel_size = self.downsample_voxel_size
        track_values = list(tracks.values())

        # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
        track_top_pcds = [
            pointcloud.get_top_points(
                np.asarray(trk.compute_local_pcd(full_pcd).points)
            )
            for trk in track_values
        ]
        track_bottom_pcds = [
            pointcloud.get_bottom_points(
                np.asarray(trk.compute_local_pcd(full_pcd).points)
            )
            for trk in track_values
        ]

        indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in track_top_pcds]

        # Add the points from the numpy arrays to the corresponding FAISS indices
        for index, arr in zip(indices, track_top_pcds):
            index.add(arr)

        overlap_matrix = np.zeros((len(track_bottom_pcds), len(track_top_pcds)))
        # Compute the pairwise overlaps
        for i in range(len(track_values)):
            for j in range(len(track_values)):
                if i == j:
                    continue
                box_i = track_values[i].features.bbox_3d
                box_j = track_values[j].features.bbox_3d

                # Skip if the boxes do not overlap at all (saves computation)
                iou = pointcloud.compute_3d_iou(box_i, box_j)
                if iou == 0:
                    continue

                # # Use range_search to find points within the threshold
                # _, I = indices[j].range_search(point_arrays[i], threshold ** 2)
                D, I = indices[j].search(track_bottom_pcds[i], 1)

                # # If any points are found within the threshold, increase overlap count
                # overlap += sum([len(i) for i in I])

                overlap = (
                    D < downsample_voxel_size**2
                ).sum()  # D is the squared distance

                # Calculate the ratio of points within the threshold
                overlap_matrix[i, j] = overlap / len(track_bottom_pcds[i])

        return overlap_matrix


if __name__ == "__main__":
    from perception import Perceptor
    from scenegraph import SceneGraph

    scene_graph = SceneGraph()
    scene_graph.load_pcd(folder="/pub3/qasim/hm3d/data/ham-sg/000-hm3d-BFRyYbPCCPE")

    perceptor = Perceptor()
    similarity = FeatureComputer()

    img1, objects1 = load_objects(
        "/pub3/qasim/hm3d/data/ham-sg/000-hm3d-BFRyYbPCCPE/detections", 0
    )
    # _, objects1_clip = similarity.compute_clip_features(
    #     perceptor.img, image_crops=objects1['crops']
    #     )
    # objects1_clip = similarity.compute_clip_features_avg(
    #     img1, masks=objects1.get_field("mask")
    # )
    similarity.compute_features(img1, scene_graph.geometry_map, objects1)
    img2, objects2 = load_objects(
        "/pub3/qasim/hm3d/data/ham-sg/000-hm3d-BFRyYbPCCPE/detections", 1
    )
    # _, objects2_clip = similarity.compute_clip_features(
    #     perceptor.img, image_crops=objects2['crops']
    #     )
    # objects2_clip = similarity.compute_clip_features_avg(
    #     img2, masks=objects2.get_field("mask")
    # )
    similarity.compute_features(img2, scene_graph.geometry_map, objects2)

    print("Clip Similarities:")
    for i in range(len(objects1)):
        for j in range(len(objects2)):
            print(
                objects1[i].label,
                "-",
                objects2[j].label,
                np.dot(
                    objects1[i].features.visual_emb, objects2[j].features.visual_emb
                ),
                np.dot(
                    objects1[i].features.caption_emb, objects2[j].features.caption_emb
                ),
                np.linalg.norm(
                    objects1[i].features.centroid - objects2[j].features.centroid
                ),
            )
