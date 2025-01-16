import argparse
import base64
import copy
import io
import json
import os
import random
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
from scene_graph.perception import PerceptorWithTextPrompt
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
        detections_folder,
        semantic_tree: SemanticTree,
        obj_pcd_max_points,
        downsample_voxel_size=0.02,
    ):
        raise NotImplementedError()


class API_TextualQA(API):
    def __init__(self, prompt_reasoning_loop, prompt_reasoning_final, device="cuda"):
        super().__init__(prompt_reasoning_loop, prompt_reasoning_final, device)
        self.perceptor = PerceptorWithTextPrompt(
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
            detections, det_matched, matched_tracks, keyframe_id, consolidate=False
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
        self.find_objects_perceptor = PerceptorWithTextPrompt(
            json_detection_key="Detections",
            json_other_keys=["your notes"],
            prompt_file="open_eqa/prompts/gso/refine_find_objects_v2.txt",
            device=self.device,
            with_edges=False,
        )
        self.analyze_frame_perceptor = PerceptorWithTextPrompt(
            json_detection_key="Detections",
            json_other_keys=["your notes"],
            prompt_file="open_eqa/prompts/gso/analyze_frame.txt",
            device=self.device,
            with_edges=False,
        )
        self.analyze_objects_perceptor = PerceptorWithTextPrompt(
            json_detection_key=None,
            json_other_keys=["name", "your notes"],
            prompt_file="open_eqa/prompts/gso/refine_analyze_objects.txt",
            device=self.device,
        )
        self.find_objects_perceptor.init()
        self.analyze_objects_perceptor.init()
        self.analyze_frame_perceptor.init()
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
        clarify_class_valid = True
        if request["type"] == "analyze_objects_in_frame" and "nodes" in request:
            for name in request["nodes"].split(","):
                name = name.strip()
                if (
                    not name
                    in semantic_tree.navigation_log[keyframe_id]["Generic Mapping"][
                        "Detections"
                    ]
                ):
                    clarify_class_valid = False
                    break
        if not clarify_class_valid and request["type"] == "analyze_objects_in_frame":
            request["type"] = "find_objects_in_frame"

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

        _, prev_detections = load_objects(
            result_path, keyframe_id, temp_refine=semantic_tree.temp_refine_dir
        )
        prev_detections_json = extract_detections_prompts(
            prev_detections, semantic_tree.tracks
        )

        if (
            request["type"] == "analyze_frame"
            or request["type"] == "find_objects_in_frame"
        ):
            if request["type"] == "find_objects_in_frame":
                text_prompt = self.find_objects_perceptor.text_prompt_template
                text_prompt = text_prompt.format(
                    query=query, prev_detections=prev_detections_json
                )
                detections, response = self.find_objects_perceptor.perceive(
                    color_np, text_prompt
                )
            else:
                text_prompt = self.analyze_frame_perceptor.text_prompt_template
                text_prompt = text_prompt.format(query=query)
                detections, response = self.analyze_frame_perceptor.perceive(
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
            detections, edges = semantic_tree.integrate_detections(
                detections,
                det_matched,
                matched_tracks,
                keyframe_id,
                img=color_np,
                consolidate=False,
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
            if request["type"] == "find_objects_in_frame":
                self.find_objects_perceptor.save_results(
                    detections, result_path / "temp-refine", keyframe_id
                )
            else:
                self.analyze_frame_perceptor.save_results(
                    detections, result_path / "temp-refine", keyframe_id
                )
        elif request["type"] == "analyze_objects_in_frame":
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

            if not keyframe_id in semantic_tree.visual_memory:
                remove_id = random.choice(semantic_tree.visual_memory)
                semantic_tree.visual_memory.remove(remove_id)
                semantic_tree.visual_memory.append(keyframe_id)
                print("Swapped ", remove_id, "for", keyframe_id)
                semantic_tree.visual_memory.sort()
        elif request["type"] == "swap_image":
            assert request["removed_frame_id"] in semantic_tree.visual_memory
            assert request["insert_frame_id"] < len(dataset)
            semantic_tree.visual_memory.remove(int(request["removed_frame_id"]))
            semantic_tree.visual_memory.append(int(request["insert_frame_id"]))
        else:
            raise Exception("Unknown request type: " + request["type"])

        navigation_log_refinement = dict(request=request)
        navigation_log_refinement["response"] = response
        semantic_tree.integrate_refinement_log(
            query, navigation_log_refinement, keyframe_id, detections=detections
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


def extract_scene_prompts(
    semantic_tree: SemanticTree,
    dataset,
    edge_types=[
        "enclosed within",
        "resting on top of",
        "directly connected to",
        "subpart of",
    ],
    prompt_img_interleaved=False,
    prompt_img_seperate=False,
    prompt_video=False,
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

            # del log["Generic Mapping"]["Estimated Current Location"]

            # Replace the Estimated Current Location with the Present Room
            # log["Generic Mapping"]["Estimated Current Location"] = semantic_tree.rooms[
            #     log["Generic Mapping"]["Present Room"]
            # ].get_semantic_name()
            # del log["Generic Mapping"]["Present Room"]

            semantic_tree

        navigation_log.append(log)

    if prompt_img_interleaved or prompt_img_seperate:
        for frame_id in semantic_tree.visual_memory:
            with Image.open(dataset.color_paths[frame_id]) as image:
                resized_image = image.resize((1000, 1000))
                buffered = io.BytesIO()
                resized_image.save(buffered, format="JPEG")
                image_data = buffered.getvalue()
                base64_encoded_data = base64.b64encode(image_data).decode("utf-8")
                # images_prompt.append(base64_encoded_data)
            if prompt_img_interleaved:
                log["Image"] = base64_encoded_data
            else:
                # images_prompt.append(base64_encoded_data)
                images_prompt.append(dataset.color_paths[frame_id])

    if prompt_video:
        if video_uri is None:
            video_uri = create_gemini_video_prompt(
                "video", dataset.color_paths, fps=fps
            )
        images_prompt.append(video_uri)

    graph = []

    # Option 1: Nested JSON
    # not_in_graph = set(semantic_tree.track_ids)
    # in_graph = set()
    # level = 0
    # while len(not_in_graph) > 0:
    #     for id in not_in_graph:
    #         if semantic_tree.tracks[id].level == level:
    #             node = {
    #                 "name": semantic_tree.tracks[id].name,
    #                 "visual caption": semantic_tree.tracks[id].captions[0],
    #                 "your notes": semantic_tree.tracks[id].notes,
    #                 "times scene": semantic_tree.tracks[id].times_scene,
    #                 "hierarchy level": semantic_tree.tracks[id].level,
    #                 "centroid": semantic_tree.tracks[id].features.centroid.tolist(),
    #             }
    #             for type in edge_types:
    #                 node["Items " + type] = []
    #             children_id = semantic_tree.get_children_ids(
    #                 id, semantic_tree.hierarchy_matrix
    #             )
    #             for child_id in children_id:
    #                 if semantic_tree.tracks[child_id].level >= level:
    #                     continue
    #                 for i in range(len(graph)):
    #                     if graph[i]["name"] == semantic_tree.tracks[child_id].name:
    #                         node[
    #                             "Items "
    #                             + semantic_tree.hierarchy_type_matrix[id][child_id]
    #                         ].append(graph.pop(i))
    #                         break
    #             graph.append(node)
    #             in_graph.add(id)
    #     not_in_graph = not_in_graph - in_graph
    #     level += 1

    # Option 2: Flat Edges
    for track in semantic_tree.tracks.values():
        edges = [e.json() for e in track.edges]
        _ = [e.pop("frame_id") for e in edges]
        _ = [e.pop("subject") for e in edges]
        graph.append(
            {
                "name": track.name,
                "visual caption": track.captions[0],
                "your notes": track.notes,
                "times scene": track.times_scene,
                "hierarchy level": track.level,
                "centroid": track.features.centroid.tolist(),
                "relationships": [e.json() for e in track.edges],
            }
        )

    # # Option 3: Edges at in seperate key
    # graph = {"Nodes": []}
    # edges = []
    # for track in semantic_tree.tracks.values():
    #     graph["Nodes"].append(
    #         {
    #             "name": track.name,
    #             "visual caption": track.captions[0],
    #             "your notes": track.notes,
    #             "times scene": track.times_scene,
    #             "hierarchy level": track.level,
    #             "centroid": track.features.centroid.tolist(),
    #         }
    #     )
    #     new_edges = [e.json() for e in track.edges]
    #     _ = [e.pop("frame_id") for e in new_edges]
    #     _ = [e.pop("subject") for e in new_edges]
    #     edges += new_edges
    # graph["Relationships"] = edges

    # Option 4: Hierarchy with room nodes
    # edges = []
    # for room in semantic_tree.rooms:
    #     graph[room.get_semantic_name()] = {"Objects": []}
    # for track in semantic_tree.tracks.values():
    #     room_name = semantic_tree.rooms[track.room_id].get_semantic_name()
    #     graph[room_name]["Objects"].append(
    #         {
    #             "name": track.name,
    #             "visual caption": track.captions[0],
    #             "your notes": track.notes,
    #             "times scene": track.times_scene,
    #             "hierarchy level": track.level,
    #             "centroid": track.features.centroid.tolist(),
    #             "edges": [e.json() for e in track.edges],
    #         }
    #     )

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
