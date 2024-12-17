import argparse
import copy
import json
import os
import time
import traceback
from abc import ABC
from pathlib import Path
from typing import Optional

import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import vertexai
from PIL import Image
from vertexai.generative_models import GenerativeModel, Part

import read_graphs
import scene_graph.utils as utils
from scene_graph.object import Object, ObjectList, load_objects
from scene_graph.perception import Perceptor
from scene_graph.pointcloud import create_depth_cloud
from scene_graph.relationship_scorer import FeatureComputer, RelationshipScorer
from scene_graph.semantic_tree import SemanticTree
from scene_graph.track import get_trackid_by_name


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
        perception_result_path,
        semantic_tree: SemanticTree,
        obj_pcd_max_points,
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
        perception_result_path,
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
        detections.extract_local_pcd(
            depth_cloud,
            color_np,
            semantic_tree.geometry_map,
            trans_pose,
            obj_pcd_max_points,
        )
        self.feature_computer.compute_features(
            image_rgb, semantic_tree.geometry_map, detections
        )

        det_matched, matched_tracks = self.relationship_scorer.associate_dets_to_tracks(
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
            json_other_keys=[],
            prompt_file="open_eqa/prompts/gso/refine_find_objects.txt",
            device=self.device,
        )
        self.analyze_objects_perceptor = PerceptorAPI(
            json_detection_key=None,
            json_other_keys=["name", "your notes"],
            prompt_file="open_eqa/prompts/gso/refine_analyze_objects.txt",
            device=self.device,
        )
        self.find_objects_perceptor.init()
        self.analyze_objects_perceptor.init()

    def call(
        self,
        request,
        dataset,
        perception_result_path,
        semantic_tree: SemanticTree,
        obj_pcd_max_points,
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

        _, prev_detections = load_objects(perception_result_path, keyframe_id)
        prev_detections = extract_detections_prompts(
            prev_detections, semantic_tree.tracks
        )

        if request["type"] == "find_objects":
            text_prompt = self.find_objects_perceptor.text_prompt_template
            text_prompt = text_prompt.format(
                query=query, prev_detections=prev_detections
            )
            detections, response = self.find_objects_perceptor.perceive(
                color_np, text_prompt
            )
            detections.extract_local_pcd(
                depth_cloud,
                color_np,
                semantic_tree.geometry_map,
                trans_pose,
                obj_pcd_max_points,
            )
            self.feature_computer.compute_features(
                image_rgb, semantic_tree.geometry_map, detections
            )
            det_matched, matched_tracks = (
                self.relationship_scorer.associate_dets_to_tracks(
                    detections, semantic_tree.get_tracks()
                )
            )
            semantic_tree.integrate_detections(
                detections, det_matched, matched_tracks, keyframe_id
            )
            # Bug
            # raise Exception(
            #     "This doesnt work because of you didn;t load the objects in perception "
            # )
            # detections.save(perception_result_path / str(keyframe_id), color_np)

        elif request["type"] == "analyse_objects":
            text_prompt = self.analyze_objects_perceptor.text_prompt_template
            text_prompt = text_prompt.format(query=query, detections=prev_detections)
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

        navigation_log_refinement = dict(Request=request)
        navigation_log_refinement.update(response)
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

        object_detections, llm_response = self.ask_question_gemini(
            self.img,
            text_prompt,
            self.img.shape,
        )
        response = llm_response

        self.num_detections = len(object_detections)

        self.mask_predictor.set_image(np.uint8(img))
        object_list = ObjectList()
        for i in range(self.num_detections):
            box = object_detections[i]["bbox"]
            sam_masks, iou_preds, _ = self.mask_predictor.predict(
                box=np.array(box), multimask_output=False
            )
            if iou_preds < 0.7:
                continue
            object_list.add_object(
                Object(
                    mask=sam_masks[0],
                    crop=utils.get_crop(self.img, box),
                    label=object_detections[i]["label"],
                    visual_caption=object_detections[i]["visual caption"],
                    bbox=box,
                    confidence=object_detections[i]["confidence"],
                )
            )
        return object_list, response

    def verify_and_format_response(self, response):
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
                assert len(obj["bbox"]) == 4
                assert not ("/" in obj["label"])
            else:
                for key in self.json_other_keys:
                    assert key in obj
        if self.json_detection_key:
            return {self.json_detection_key: response}
        else:
            return response


def extract_scene_prompts(semantic_tree: SemanticTree):
    graph = []
    for i in semantic_tree.track_ids:
        graph.append(
            {
                "name": semantic_tree.tracks[i].name,
                "visual caption": semantic_tree.tracks[i].captions[0],
                "your notes": semantic_tree.tracks[i].notes,
                "times scene": semantic_tree.tracks[i].times_scene,
                "centroid": semantic_tree.tracks[i].features.centroid.tolist(),
            }
        )
    navigation_log = []
    for i in range(len(semantic_tree.navigation_log)):
        log = copy.deepcopy(semantic_tree.navigation_log[i])
        if log["Generic Mapping"] is not None:
            log["Generic Mapping"]["Visible Scene Graph Nodes"] = log[
                "Generic Mapping"
            ]["Detections"]
            del log["Generic Mapping"]["Detections"]
        navigation_log.append(log)
    return graph, navigation_log


def extract_detections_prompts(detections: ObjectList, tracks):
    prev_detections_json = []
    for obj in detections:
        id = get_trackid_by_name(tracks, obj.matched_track_name)
        prev_detections_json.append(
            {
                "track name": obj.matched_track_name,
                "visual caption": obj.visual_caption,
                "spatial caption": obj.spatial_caption,
                "bbox": obj.bbox,
                "your notes": tracks[id].notes,
            }
        )
    return prev_detections_json
