import random
import time
from abc import ABC

import cv2
import google.generativeai as genai
import numpy as np
from PIL import Image

from embodied_memory.detection import DetectionList, load_objects
from embodied_memory.detection_feature_extractor import DetectionFeatureExtractor
from embodied_memory.embodied_memory import EmbodiedMemory
from embodied_memory.pointcloud import create_depth_cloud
from embodied_memory.relationship_scorer import HierarchyExtractor
from embodied_memory.rooms import assign_nodes_to_rooms
from embodied_memory.scene_graph import associate_dets_to_nodes, get_nodeid_by_name
from embodied_memory.vlm_detectors import VLMPrompterAPI

from .prompt_formatting import extract_detections_prompts


class API(ABC):
    """
    Base class for APIs
    """

    def __init__(self, prompt_reasoning_loop, prompt_reasoning_final, device="cuda"):
        self.device = device
        self.feature_computer = DetectionFeatureExtractor(device)
        self.feature_computer.init()
        self.relationship_scorer = HierarchyExtractor(downsample_voxel_size=0.02)
        self.prompt_reasoning_loop = prompt_reasoning_loop
        self.prompt_reasoning_final = prompt_reasoning_final

    def get_declaration(self):
        # returns the function declarations available to the reasoning agent
        raise NotImplementedError()

    def get_system_prompt(self):
        # returns the system prompt for the reasoning agent associated with this reasoning agent
        raise NotImplementedError()

    def call(
        self,
        request_json,
        dataset,
        result_dir_detections,
        temp_workspace_dir,
        detections_folder,
        embodied_memory: EmbodiedMemory,
        obj_pcd_max_points,
        downsample_voxel_size=0.02,
    ):
        raise NotImplementedError()


class NodeLevelAPI(API):
    def __init__(
        self,
        gemini_model,
        prompt_reasoning_loop=None,
        prompt_reasoning_final=None,
        device="cuda",
    ):
        super().__init__(prompt_reasoning_loop, prompt_reasoning_final, device)
        self.find_objects_perceptor = VLMPrompterAPI(
            response_detection_key="Detections",
            response_other_keys=["your notes"],
            prompt_file="eqa/prompts/graph-pad/api_find_objects.txt",
            device=self.device,
            gemini_model=gemini_model,
            with_edges=False,
        )
        self.analyze_objects_perceptor = VLMPrompterAPI(
            response_detection_key=None,
            response_other_keys=["name", "your notes"],
            prompt_file="eqa/prompts/graph-pad/api_analyze_objects.txt",
            gemini_model=gemini_model,
            device=self.device,
        )
        self.find_objects_perceptor.init()
        self.analyze_objects_perceptor.init()
        self.hierarchy_extractor = HierarchyExtractor(downsample_voxel_size=0.02)

    def get_declaration(self):
        return [
            {
                "name": "find_objects_in_frame",
                "description": "Adds new objects to the Scene Graph. Searches a scene for items present in the scene but not yet represented in the Scene Graph.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frame_id": {
                            "type": "integer",
                            "description": "Frame index of image to search. Can refer to any frame in the Observation Log.",
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query (e.g., object characteristics, function, location)",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Explanation of why this is the best Tool to use. Also explain why you choose the particular frame to search instead of others.",
                        },
                    },
                    "required": ["frame_id", "query"],
                },
            },
            {
                "name": "analyze_objects_in_frame",
                "description": "Analyzes existing detections in a specific frame based on a query. Adds resulting insights to the 'query-relevant notes' of the visible detections in the requested frame to the Scratch Pad. Returns an image frame and updated Scratch Pad. Only examines existing detections as described in the Scene Graph, does not search or find new objects.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frame_id": {
                            "type": "integer",
                            "description": "Frame index of image to examine. Can refer to any frame in the Observation Log. ",
                        },
                        "query": {
                            "type": "string",
                            "description": "Analysis query (e.g., object state, features, comparisons, spatial relationships)",
                        },
                        "nodes": {
                            "type": "string",
                            "description": "Labels of objects your query is relevent to. Should correspond to objects in the Scene Graph. use the Observation Log to ensure that the requested nodes are indeed visible in the requested frame.",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Explanation of why this is the best Tool to use. Also explain why you choose the particular frame to search instead of others.",
                        },
                    },
                    "required": ["frame_id", "query", "nodes"],
                },
            },
        ]

    def get_system_prompt(self):
        with open(
            "eqa/prompts/graph-pad/system_prompt_nodelevel.txt",
            "r",
        ) as f:
            system_prompt = f.read().strip()
        return system_prompt

    def call(
        self,
        request,
        dataset,
        result_dir_detections,
        temp_workspace_dir,
        embodied_memory: EmbodiedMemory,
        obj_pcd_max_points,
        downsample_voxel_size=0.02,
    ):
        keyframe_id = int(request["frame_id"])
        assert keyframe_id < len(dataset)

        # check if the analyze_objects_in_frame is valid, if not then set call find_objects_in_frame instead
        valid_analyze_call = True
        if request["type"] == "analyze_objects_in_frame" and "nodes" in request:
            try:
                request["nodes"] = request["nodes"][0].split(",")
                for name in request["nodes"]:
                    name = name.strip()
                    if (
                        not name
                        in embodied_memory.navigation_log[keyframe_id][
                            "General Frame Info"
                        ]["Detections"]
                    ):
                        valid_analyze_call = False
                        break
            except:
                pass
        if not valid_analyze_call and request["type"] == "analyze_objects_in_frame":
            request["type"] = "find_objects_in_frame"

        # extract data about the keyframe
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
        new_nodes = []

        # extract the previous detections for the keyframe. these will be update.
        _, prev_detections = load_objects(
            result_dir_detections, keyframe_id, temp_dir=temp_workspace_dir
        )
        prev_detections_json = extract_detections_prompts(
            prev_detections, embodied_memory.scene_graph
        )

        if request["type"] == "find_objects_in_frame":
            # Prompt the VLM to detect new objects
            text_prompt = self.find_objects_perceptor.text_prompt_template
            text_prompt = text_prompt.format(
                query=query, prev_detections=prev_detections_json
            )
            detections, response = self.find_objects_perceptor.perceive(
                color_np, text_prompt
            )

            # process the detections, similar to how they were processed during embodied memory creation
            detections.extract_pcd(
                depth_cloud,
                color_np,
                downsample_voxel_size=downsample_voxel_size,
                dbscan_remove_noise=True,
                dbscan_eps=0.1,
                dbscan_min_points=10,
                trans_pose=trans_pose,
                obj_pcd_max_points=obj_pcd_max_points,
            )
            self.feature_computer.compute_features(image_rgb, detections)
            det_matched, matched_nodes = associate_dets_to_nodes(
                detections,
                embodied_memory.scene_graph.get_nodes(),
                downsample_voxel_size=downsample_voxel_size,
            )
            detections, edges = embodied_memory.update_scene_graph(
                detections,
                det_matched,
                matched_nodes,
                keyframe_id,
                img=color_np,
                consolidate=False,
            )

            for i in range(len(response)):
                id = get_nodeid_by_name(
                    embodied_memory.scene_graph, detections[i].matched_node_name
                )
                if id is None:
                    continue
                embodied_memory.scene_graph[id].notes = response[i]["your notes"]

            hierarchy_matrix, hierarchy_type_matrix = (
                self.hierarchy_extractor.infer_hierarchy(embodied_memory.scene_graph)
            )
            embodied_memory.compute_node_levels(hierarchy_matrix, hierarchy_type_matrix)

            embodied_memory.scene_graph = assign_nodes_to_rooms(
                embodied_memory.scene_graph,
                embodied_memory.full_scene_pcd.pcd,
                embodied_memory.floors,
            )
            for det in prev_detections:
                if det.label in detections.get_field("label"):
                    continue
                detections.add_object(det)

            # Save the new detections into the temp workspace
            self.find_objects_perceptor.save_detection_results(
                detections, temp_workspace_dir, keyframe_id
            )

            # Update new_nodes
            for i in range(len(response)):
                if det_matched[i]:
                    continue
                id = get_nodeid_by_name(
                    embodied_memory.scene_graph, detections[i].matched_node_name
                )
                new_nodes.append(
                    {
                        "name": detections[i].matched_node_name,
                        "visual caption": detections[i].visual_caption[0],
                        "times scene": embodied_memory.scene_graph[id].times_scene,
                        "room": embodied_memory.scene_graph[id].room_id,
                        "centroid": embodied_memory.scene_graph[
                            id
                        ].features.centroid.tolist(),
                    }
                )

        elif request["type"] == "analyze_objects_in_frame":
            # Ask VLM to caption each of the detected objects with information related to a query
            text_prompt = self.analyze_objects_perceptor.text_prompt_template
            text_prompt = text_prompt.format(
                query=query, detections=prev_detections_json
            )
            detections, response = self.analyze_objects_perceptor.perceive(
                color_np, text_prompt
            )

            # update the scene graph with the new node detections
            for i in range(len(response)):
                id = get_nodeid_by_name(
                    embodied_memory.scene_graph, response[i]["name"]
                )
                if id is None:
                    continue
                embodied_memory.scene_graph[id].notes = response[i]["your notes"]
        else:
            raise Exception("Unknown request type: " + request["type"])

        navigation_log_refinement = dict(request=request, response=response)
        embodied_memory.navigation_log.add_api_log(
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

        return embodied_memory, response, api_log, new_nodes


class FrameLevelAPI(API):
    def __init__(
        self,
        gemini_model,
        prompt_reasoning_loop=None,
        prompt_reasoning_final=None,
        device="cuda",
    ):
        super().__init__(prompt_reasoning_loop, prompt_reasoning_final, device)
        self.analyze_frame_perceptor = VLMPrompterAPI(
            response_detection_key="Detections",
            response_other_keys=["your notes"],
            prompt_file="eqa/prompts/graph-pad/api_analyze_frame.txt",
            device=self.device,
            gemini_model=gemini_model,
            with_edges=False,
        )
        self.analyze_frame_perceptor.init()
        self.hierarchy_extractor = HierarchyExtractor(downsample_voxel_size=0.02)

    def get_declaration(self):
        return [
            {
                "name": "analyze_frame",
                "description": "Analyzes an image based on a given query. Returns an image of the requested frame, an updated Scene Graph, updated Scratch Pad, and updated Observation Log.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frame_id": {
                            "type": "integer",
                            "description": "Frame index of the image to analyze. Can refer to any frame in the Observation Log.",
                        },
                        "query": {
                            "type": "string",
                            "description": "The analysis query (e.g., 'What is the relationship between the table and the chair?', 'Is the door open?', 'What is the material of the countertop?')",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Explanation why you choose to search the requested frame and why you choose the requested query.",
                        },
                    },
                    "required": ["frame_id", "query"],
                },
            }
        ]

    def get_system_prompt(self):
        with open(
            "eqa/prompts/graph-pad/system_prompt_framelevel.txt",
            "r",
        ) as f:
            system_prompt = f.read().strip()
        return system_prompt

    def call(
        self,
        request,
        dataset,
        result_dir_detections,
        temp_workspace_dir,
        embodied_memory: EmbodiedMemory,
        obj_pcd_max_points,
        downsample_voxel_size=0.02,
    ):
        keyframe_id = int(request["frame_id"])
        assert keyframe_id < len(dataset)
        clarify_class_valid = True
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
        new_nodes = []

        # Load the previous detections
        _, prev_detections = load_objects(
            result_dir_detections, keyframe_id, temp_dir=temp_workspace_dir
        )
        prev_detections_json = extract_detections_prompts(
            prev_detections, embodied_memory.scene_graph
        )

        # Given the query, detect new objects and caption them
        text_prompt = self.analyze_frame_perceptor.text_prompt_template
        text_prompt = text_prompt.format(query=query)
        detections, response = self.analyze_frame_perceptor.perceive(
            color_np, text_prompt
        )

        # Process the detections, similar to how they were processed during embodied memory creation
        detections.extract_pcd(
            depth_cloud,
            color_np,
            downsample_voxel_size=downsample_voxel_size,
            dbscan_remove_noise=True,
            dbscan_eps=0.1,
            dbscan_min_points=10,
            trans_pose=trans_pose,
            obj_pcd_max_points=obj_pcd_max_points,
        )
        self.feature_computer.compute_features(image_rgb, detections)
        det_matched, matched_nodes = associate_dets_to_nodes(
            detections,
            embodied_memory.scene_graph.get_nodes(),
            downsample_voxel_size=downsample_voxel_size,
        )
        detections, edges = embodied_memory.update_scene_graph(
            detections,
            det_matched,
            matched_nodes,
            keyframe_id,
            img=color_np,
            consolidate=False,
        )

        for i in range(len(response)):
            id = get_nodeid_by_name(
                embodied_memory.scene_graph, detections[i].matched_node_name
            )
            if id is None:
                continue
            embodied_memory.scene_graph[id].notes = response[i]["your notes"]

        hierarchy_matrix, hierarchy_type_matrix = (
            self.hierarchy_extractor.infer_hierarchy(embodied_memory.scene_graph)
        )
        embodied_memory.compute_node_levels(hierarchy_matrix, hierarchy_type_matrix)

        embodied_memory.scene_graph = assign_nodes_to_rooms(
            embodied_memory.scene_graph,
            embodied_memory.full_scene_pcd.pcd,
            embodied_memory.floors,
        )
        for det in prev_detections:
            if det.label in detections.get_field("label"):
                continue
            detections.add_object(det)

        # Save the resulting detections in the temporary workspace
        self.analyze_frame_perceptor.save_detection_results(
            detections, temp_workspace_dir, keyframe_id
        )

        # Update new_nodes
        for i in range(len(response)):
            if det_matched[i]:
                continue
            id = get_nodeid_by_name(
                embodied_memory.scene_graph, detections[i].matched_node_name
            )
            new_nodes.append(
                {
                    "name": detections[i].matched_node_name,
                    "visual caption": detections[i].visual_caption[0],
                    "times scene": embodied_memory.scene_graph[id].times_scene,
                    "room": embodied_memory.scene_graph[id].room_id,
                    "centroid": embodied_memory.scene_graph[
                        id
                    ].features.centroid.tolist(),
                }
            )

        navigation_log_refinement = dict(request=request, response=response)
        embodied_memory.navigation_log.add_api_log(
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

        return embodied_memory, response, api_log, new_nodes
