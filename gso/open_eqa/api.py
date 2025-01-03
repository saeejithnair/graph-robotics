import argparse
import base64
import copy
import io
import json
import os
import time
import traceback
from abc import ABC
from pathlib import Path
from typing import Optional

import cv2
import google.auth
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from vertexai.generative_models import GenerativeModel, Part

import read_graphs
import scene_graph.utils as utils
from scene_graph.detection import Detection, DetectionList, Edge, load_objects
from scene_graph.features import FeatureComputer
from scene_graph.perception import Perceptor
from scene_graph.pointcloud import create_depth_cloud
from scene_graph.relationship_scorer import RelationshipScorer
from scene_graph.semantic_tree import SemanticTree
from scene_graph.track import associate_dets_to_tracks, get_trackid_by_name


class API(ABC):
    def __init__(self, prompt_reasoning_loop, prompt_reasoning_final, device="cuda"):
        self.device = device
        self.feature_computer = FeatureComputer(device)
        self.feature_computer.init()
        self.relationship_scorer = RelationshipScorer(downsample_voxel_size=0.02)
        self.prompt_reasoning_loop = prompt_reasoning_loop
        self.prompt_reasoning_final = prompt_reasoning_final

    def call(
        self,
        request_json,
        dataset,
        result_path,
        semantic_tree: SemanticTree,
        obj_pcd_max_points,
        downsample_voxel_size=0.02,
    ):
        raise NotImplementedError()


class API_TextualQA(API):
    def __init__(self, prompt_reasoning_loop, prompt_reasoning_final, device="cuda"):
        super().__init__(prompt_reasoning_loop, prompt_reasoning_final, device)
        self.perceptor = PerceptorAPI(
            json_detection_key="Detections",
            json_other_keys=["Finding"],
            prompt_file="open_eqa/prompts/gso/refine_textual_response.txt",
            device=self.device,
        )
        self.perceptor.init()

    def call(
        self,
        request_json,
        dataset,
        result_path,
        semantic_tree: SemanticTree,
        obj_pcd_max_points,
    ):
        keyframe_id = int(request_json["frame_id"])
        assert keyframe_id < len(dataset)
        query = request_json["query"]

        text_prompt = self.perceptor.text_prompt_template
        text_prompt = text_prompt.format(request=query)
        color_tensor, depth_tensor, intrinsics, *_ = dataset[keyframe_id]
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        depth_cloud = create_depth_cloud(depth_array, dataset.get_cam_K())
        unt_pose = dataset.poses[keyframe_id]
        trans_pose = unt_pose.cpu().numpy()
        color_np = color_tensor.cpu().numpy()  # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8)  # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        detections, response = self.perceptor.perceive(color_np, text_prompt)
        detections.extract_pcd(
            depth_cloud,
            color_np,
            semantic_tree.geometry_map,
            trans_pose,
            obj_pcd_max_points,
        )
        self.feature_computer.compute_features(
            image_rgb, semantic_tree.geometry_map, detections
        )

        det_matched, matched_tracks = features.associate_dets_to_tracks(
            detections, semantic_tree.get_tracks()
        )

        semantic_tree.integrate_detections(
            detections, det_matched, matched_tracks, keyframe_id
        )

        navigation_log_refinement = dict(Request=query)
        navigation_log_refinement.update(response)
        semantic_tree.integrate_refinement_log(
            query, navigation_log_refinement, keyframe_id
        )

        # perceptor.save_results(
        #     llm_response, detections, perception_result_dir, frame_idx
        # )
        # detections.save(perception_result_dir / str(frame_idx), perceptor.img)

        api_log = dict()
        api_log["call"] = query
        api_log["keyframe_path"] = dataset.color_paths[keyframe_id]
        api_log["response"] = response

        return semantic_tree, response, api_log


class API_GraphAPI(API):
    def __init__(self, prompt_reasoning_loop, prompt_reasoning_final, device="cuda"):
        super().__init__(prompt_reasoning_loop, prompt_reasoning_final, device)
        self.find_objects_perceptor = PerceptorAPI(
            json_detection_key="Detections",
            json_other_keys=["your notes"],
            prompt_file="open_eqa/prompts/gso/refine_find_objects_v2.txt",
            device=self.device,
            with_edges=True,
        )
        self.analyze_objects_perceptor = PerceptorAPI(
            json_detection_key=None,
            json_other_keys=["name", "your notes"],
            prompt_file="open_eqa/prompts/gso/refine_analyze_objects.txt",
            device=self.device,
        )
        self.find_objects_perceptor.init()
        self.analyze_objects_perceptor.init()
        self.relationship_scorer = RelationshipScorer(downsample_voxel_size=0.02)

    def call(
        self,
        request,
        dataset,
        result_path,
        semantic_tree: SemanticTree,
        obj_pcd_max_points,
        downsample_voxel_size=0.02,
    ):
        keyframe_id = int(request["frame_id"])
        assert keyframe_id < len(dataset)
        query = request["query"]

        color_tensor, depth_tensor, intrinsics, *_ = dataset[keyframe_id]
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        depth_cloud = create_depth_cloud(depth_array, dataset.get_cam_K())
        unt_pose = dataset.poses[keyframe_id]
        trans_pose = unt_pose.cpu().numpy()
        color_np = color_tensor.cpu().numpy()  # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8)  # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        _, prev_detections = load_objects(result_path, keyframe_id)
        prev_detections_json = extract_detections_prompts(
            prev_detections, semantic_tree.tracks
        )

        if request["type"] == "add_nodes_to_graph":
            text_prompt = self.find_objects_perceptor.text_prompt_template
            text_prompt = text_prompt.format(
                query=query, prev_detections=prev_detections_json
            )
            detections, response = self.find_objects_perceptor.perceive(
                color_np, text_prompt
            )
            detections.extract_pcd(
                depth_cloud,
                color_np,
                semantic_tree.geometry_map,
                downsample_voxel_size=downsample_voxel_size,
                dbscan_remove_noise=True,
                dbscan_eps=0.1,
                dbscan_min_points=10,
                trans_pose=trans_pose,
                obj_pcd_max_points=obj_pcd_max_points,
            )
            self.feature_computer.compute_features(
                image_rgb, semantic_tree.geometry_map, detections
            )
            det_matched, matched_tracks = associate_dets_to_tracks(
                detections,
                semantic_tree.get_tracks(),
                semantic_tree.geometry_map,
                downsample_voxel_size=downsample_voxel_size,
            )
            semantic_tree.integrate_detections(
                detections, det_matched, matched_tracks, keyframe_id
            )

            for i in range(len(response)):
                id = get_trackid_by_name(
                    semantic_tree.tracks, detections[i].matched_track_name
                )
                if id is None:
                    continue
                semantic_tree.tracks[id].notes = response[i]["your notes"]

            hierarchy_matrix, hierarchy_type_matrix = (
                self.relationship_scorer.infer_hierarchy_vlm(semantic_tree.tracks)
            )
            semantic_tree.compute_node_levels(hierarchy_matrix, hierarchy_type_matrix)

            for det in prev_detections:
                if det.label in detections.get_field("label"):
                    continue
                detections.add_object(det)
            self.find_objects_perceptor.save_results(
                prev_detections, result_path / "detections-refine", keyframe_id
            )

        elif request["type"] == "inspect_nodes_in_graph":
            text_prompt = self.analyze_objects_perceptor.text_prompt_template
            text_prompt = text_prompt.format(
                query=query, detections=prev_detections_json
            )
            detections, response = self.analyze_objects_perceptor.perceive(
                color_np, text_prompt
            )
            for i in range(len(response)):
                id = get_trackid_by_name(semantic_tree.tracks, response[i]["name"])
                if id is None:
                    continue
                semantic_tree.tracks[id].notes = response[i]["your notes"]
        else:
            raise Exception("Unknown request type: " + request["type"])

        navigation_log_refinement = dict(request=request)
        navigation_log_refinement["response"] = response
        semantic_tree.integrate_refinement_log(
            query, navigation_log_refinement, keyframe_id
        )

        # perceptor.save_results(
        #     llm_response, detections, perception_result_dir, frame_idx
        # )
        # detections.save(perception_result_dir / str(frame_idx), perceptor.img)

        api_log = dict()
        api_log["call_type"] = request["type"]
        api_log["query"] = query
        api_log["keyframe_path"] = dataset.color_paths[keyframe_id]
        api_log["response"] = response

        return semantic_tree, response, api_log


class PerceptorAPI(Perceptor):
    def perceive(self, img, text_prompt):
        self.img = np.uint8(img)

        detections, llm_response = self.ask_question_gemini(
            self.img,
            text_prompt,
            self.img.shape,
        )

        if not self.json_detection_key:
            detection_list = DetectionList()
            return detection_list, llm_response

        response = []

        self.num_detections = len(detections)

        self.mask_predictor.set_image(np.uint8(img))
        detection_list = DetectionList()
        for i in range(self.num_detections):
            box = detections[i]["bbox"]
            sam_masks, iou_preds, _ = self.mask_predictor.predict(
                box=np.array(box), multimask_output=False
            )
            crop = utils.get_crop(self.img, box)
            if crop.shape[0] <= 1 or crop.shape[1] <= 1:
                continue

            edges = []
            if self.with_edges:
                for rel in detections[i]["relationships"]:
                    edges.append(
                        Edge(
                            rel["relationship_type"].lower(),
                            detections[i]["label"].lower(),
                            rel["related_object_label"].lower(),
                            (
                                rel["related_object_spatial_caption"]
                                if "related_object_spatial_caption" in rel
                                else None
                            ),
                        )
                    )
            detection_list.add_object(
                Detection(
                    mask=sam_masks[0],
                    crop=crop,
                    label=detections[i]["label"].lower(),
                    visual_caption=detections[i]["visual caption"],
                    spatial_caption=detections[i]["spatial caption"],
                    bbox=box,
                    confidence=detections[i]["confidence"],
                    notes=detections[i]["your notes"],
                    edges=edges,
                )
            )
            response.append(llm_response[self.json_detection_key][i])

        detection_names = detection_list.get_field("label")
        detection_list.filter_edges(self.edge_types, detection_names)

        return detection_list, response

    def verify_json_response(self, response):
        # This code is very hacked. This is because of find_objects prompt returns a list, which I process into a JSON where 'keys' =  detections.
        assert "```json" in response
        response = response.replace("```json", "")
        response = response.replace("```", "")
        response = response.strip()
        response = json.loads(response)
        for obj in response:
            if self.json_detection_key:
                assert "label" in obj
                assert "visual caption" in obj
                assert "spatial caption" in obj
                assert "bbox" in obj
                assert "confidence" in obj
                assert "your notes" in obj
                assert len(obj["bbox"]) == 4
                assert not ("/" in obj["label"])
                if self.with_edges:
                    assert "relationships" in obj
                    for rel in obj["relationships"]:
                        assert "related_object_label" in rel
                        assert "relationship_type" in rel
            else:
                for key in self.json_other_keys:
                    assert key in obj
        if self.json_detection_key:
            return {self.json_detection_key: response}
        else:
            return response


def extract_scene_prompts(
    semantic_tree: SemanticTree,
    dataset,
    edge_types=["inside", "on", "part of", "attached to"],
    prompt_img_interleaved=False,
    prompt_img_seperate=False,
    prompt_video=True,
    video_uri=None,
    fps=5,
):
    navigation_log = []
    images_prompt = []

    for i in range(len(semantic_tree.navigation_log)):
        log = copy.deepcopy(semantic_tree.navigation_log[i])

        if prompt_video:
            log["Timestamp"] = get_frame_timestamp(log["Frame Index"], fps)

        if log["Generic Mapping"] is not None:
            log["Generic Mapping"]["Visible Scene Graph Nodes"] = log[
                "Generic Mapping"
            ]["Detections"]
            del log["Generic Mapping"]["Detections"]
        navigation_log.append(log)

    if prompt_img_interleaved or prompt_img_seperate:
        for frame_id in semantic_tree.visual_memory:
            with Image.open(dataset.color_paths[frame_id]) as image:
                resized_image = image.resize((512, 512))
                buffered = io.BytesIO()
                resized_image.save(buffered, format="JPEG")
                image_data = buffered.getvalue()
                base64_encoded_data = base64.b64encode(image_data).decode("utf-8")
                images_prompt.append(base64_encoded_data)
            if prompt_img_interleaved:
                log["Image"] = base64_encoded_data
            else:
                images_prompt.append(base64_encoded_data)

    if prompt_video:
        if video_uri is None:
            video_uri = create_gemini_video_prompt(
                "video", dataset.color_paths, fps=fps
            )
        images_prompt.append(video_uri)

    graph = []
    not_in_graph = set(semantic_tree.track_ids)
    in_graph = set()
    level = 0
    while len(not_in_graph) > 0:
        for id in not_in_graph:
            if semantic_tree.tracks[id].level == level:
                node = {
                    "name": semantic_tree.tracks[id].name,
                    "visual caption": semantic_tree.tracks[id].captions[0],
                    "your notes": semantic_tree.tracks[id].notes,
                    "times scene": semantic_tree.tracks[id].times_scene,
                    "hierarchy level": semantic_tree.tracks[id].level,
                    "centroid": semantic_tree.tracks[id].features.centroid.tolist(),
                }
                for type in edge_types:
                    node["Items " + type] = []
                children_id = semantic_tree.get_children_ids(
                    id, semantic_tree.hierarchy_matrix
                )
                for child_id in children_id:
                    if semantic_tree.tracks[child_id].level >= level:
                        continue
                    for i in range(len(graph)):
                        if graph[i]["name"] == semantic_tree.tracks[child_id].name:
                            node[
                                "Items "
                                + semantic_tree.hierarchy_type_matrix[id][child_id]
                            ].append(graph.pop(i))
                            break
                graph.append(node)
                in_graph.add(id)
        not_in_graph = not_in_graph - in_graph
        level += 1
    return graph, navigation_log, images_prompt, semantic_tree.visual_memory


def extract_detections_prompts(detections: DetectionList, tracks):
    prev_detections_json = []
    for obj in detections:
        id = get_trackid_by_name(tracks, obj.matched_track_name)
        prev_detections_json.append(
            {
                "track name": obj.matched_track_name,
                "visual caption": obj.visual_caption,
                "spatial caption": obj.spatial_caption,
                "bbox": obj.bbox,
                "edges": [str(edge) for edge in obj.edges],
                "your notes": tracks[id].notes,
            }
        )
    return prev_detections_json


def get_frame_timestamp(frame_number, fps):
    """Calculates the timestamp for a given frame number in (minutes, seconds) format.

    Args:
      frame_number: The frame number (starting from 0).
      fps: Frames per second of the video.

    Returns:
      The timestamp in the format MM:SS.
    """

    seconds = frame_number / fps
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes):02d}:{float(seconds):02.0f}"


def create_gemini_video_prompt(name, image_paths, fps=5):
    """Converts a list of image paths into a video and uploads it to Gemini.

    Args:
      image_paths: A list of paths to image files.
      fps: Frames per second for the output video.

    Returns:
      The URI of the uploaded video file.
    """
    path = name + ".avi"

    # Load the first image to get dimensions
    first_image = cv2.imread(image_paths[0])
    height, width, _ = first_image.shape

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    # Write images to the video
    for image_path in image_paths:
        image = cv2.imread(image_path)
        video_writer.write(image)

    video_writer.release()

    # Upload the video to Gemini
    print("Uploading video...")
    video_file = genai.upload_file(
        path=path,
    )
    print(f"Completed upload: {video_file.uri}")

    while video_file.state.name == "PROCESSING":
        print(".", end="")
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)
    return video_file.uri
