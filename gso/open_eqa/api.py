import base64
import copy
import io
import random
import time
from abc import ABC

import cv2
import google.generativeai as genai
import numpy as np
from PIL import Image

from scene_graph.detection import DetectionList, load_objects
from scene_graph.detection_feature_extractor import DetectionFeatureExtractor
from scene_graph.edges import EDGE_TYPES
from scene_graph.embodied_memory import EmbodiedMemory
from scene_graph.perception import VLMPrompterAPI
from scene_graph.pointcloud import create_depth_cloud
from scene_graph.relationship_scorer import HierarchyExtractor
from scene_graph.rooms import assign_nodes_to_rooms
from scene_graph.scene_graph import associate_dets_to_nodes, get_nodeid_by_name


class API(ABC):
    def __init__(self, prompt_reasoning_loop, prompt_reasoning_final, device="cuda"):
        self.device = device
        self.feature_computer = DetectionFeatureExtractor(device)
        self.feature_computer.init()
        self.relationship_scorer = HierarchyExtractor(downsample_voxel_size=0.02)
        self.prompt_reasoning_loop = prompt_reasoning_loop
        self.prompt_reasoning_final = prompt_reasoning_final

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


def get_api_declaration(api_type):
    if api_type == "frame_level":
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
    else:
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


class API_GraphAPI(API):
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
            prompt_file="open_eqa/prompts/gso/refine_find_objects_v2.txt",
            device=self.device,
            gemini_model=gemini_model,
            with_edges=False,
        )
        self.analyze_frame_perceptor = VLMPrompterAPI(
            response_detection_key="Detections",
            response_other_keys=["your notes"],
            prompt_file="open_eqa/prompts/gso/analyze_frame.txt",
            device=self.device,
            gemini_model=gemini_model,
            with_edges=False,
        )
        self.analyze_objects_perceptor = VLMPrompterAPI(
            response_detection_key=None,
            response_other_keys=["name", "your notes"],
            prompt_file="open_eqa/prompts/gso/refine_analyze_objects.txt",
            gemini_model=gemini_model,
            device=self.device,
        )
        self.find_objects_perceptor.init()
        self.analyze_objects_perceptor.init()
        self.analyze_frame_perceptor.init()
        self.hierarchy_extractor = HierarchyExtractor(downsample_voxel_size=0.02)

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
        if request["type"] == "analyze_objects_in_frame" and "nodes" in request:
            try:
                request["nodes"] = request["nodes"][0].split(",")
                for name in request["nodes"]:
                    name = name.strip()
                    if (
                        not name
                        in embodied_memory.navigation_log[keyframe_id][
                            "Generic Mapping"
                        ]["Detections"]
                    ):
                        clarify_class_valid = False
                        break
            except:
                pass
        if not clarify_class_valid and request["type"] == "analyze_objects_in_frame":
            request["type"] = "analyze_frame"

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

        _, prev_detections = load_objects(
            result_dir_detections, keyframe_id, temp_dir=temp_workspace_dir
        )
        prev_detections_json = extract_detections_prompts(
            prev_detections, embodied_memory.scene_graph
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
            if request["type"] == "find_objects_in_frame":
                self.find_objects_perceptor.save_detection_results(
                    detections, temp_workspace_dir, keyframe_id
                )
            else:
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

        elif request["type"] == "analyze_objects_in_frame":
            text_prompt = self.analyze_objects_perceptor.text_prompt_template
            text_prompt = text_prompt.format(
                query=query, detections=prev_detections_json
            )
            detections, response = self.analyze_objects_perceptor.perceive(
                color_np, text_prompt
            )
            for i in range(len(response)):
                id = get_nodeid_by_name(
                    embodied_memory.scene_graph, response[i]["name"]
                )
                if id is None:
                    continue
                embodied_memory.scene_graph[id].notes = response[i]["your notes"]

            if not keyframe_id in embodied_memory.visual_memory:
                remove_id = random.choice(embodied_memory.visual_memory)
                embodied_memory.visual_memory.remove(remove_id)
                embodied_memory.visual_memory.append(keyframe_id)
                print("Swapped ", remove_id, "for", keyframe_id)
                embodied_memory.visual_memory.sort()
        else:
            raise Exception("Unknown request type: " + request["type"])

        navigation_log_refinement = dict(request=request)
        navigation_log_refinement["response"] = response
        embodied_memory.integrate_refinement_log(
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


def search_name_recursive(graph, search_name):
    for i in range(len(graph)):
        if graph[i]["name"] == search_name:
            return graph.pop(i)
        for type in EDGE_TYPES:
            rel_name = "Items " + type
            node = search_name_recursive(graph[i][rel_name], search_name, EDGE_TYPES)
            if not (node is None):
                return node
    return None


def extract_field_recursive(graph, field):
    ret = []
    for i in range(len(graph)):
        if graph[i] == None:
            graph.pop(i)
            continue
        node = {
            "name": graph[i]["name"],
            "query-relevant notes": graph[i]["query-relevant notes"],
        }
        del graph[i]["query-relevant notes"]
        for type in EDGE_TYPES:
            rel_name = "Items " + type
            node[rel_name] = extract_field_recursive(graph[i][rel_name], field)
        ret.append(node)
    return ret


def get_room_name(room_name):
    if room_name == "unknown" or not " " in room_name or not "_" in room_name:
        return room_name
    floor_name = room_name.split(" ")[-1][0]
    id = room_name.split(" ")[-1][-1]
    room_string = " ".join(room_name.split(" ")[:-1]) + f" {id} on floor {floor_name}"
    return room_string


def extract_scene_prompts(
    embodied_memory: EmbodiedMemory,
    dataset,
    prompt_img_interleaved=False,
    prompt_img_seperate=False,
    prompt_video=False,
    video_uri=None,
    fps=5,
):
    navigation_log = []
    images_prompt = []

    for i in range(len(embodied_memory.navigation_log)):
        log = copy.deepcopy(embodied_memory.navigation_log[i])

        if prompt_video:
            log["Timestamp"] = get_frame_timestamp(log["Frame Index"], fps)

        if log["Generic Mapping"] is not None:
            log["Generic Mapping"]["Visible Scene Graph Nodes"] = log[
                "Generic Mapping"
            ]["Detections"]
            del log["Generic Mapping"]["Detections"]

            # del log["Generic Mapping"]["Estimated Current Location"]

            # Replace the Estimated Current Location with the Present Room
            log["Generic Mapping"]["Current Room"] = "unknown"
            if log["Generic Mapping"]["Present Room"] != "unknown":
                log["Generic Mapping"]["Current Room"] = get_room_name(
                    log["Generic Mapping"]["Present Room"]
                )
            del log["Generic Mapping"]["Present Room"]

            for k in log["Generic Mapping"]:
                log[k] = log["Generic Mapping"][k]
            del log["Generic Mapping"]
            del log["Focused Analyses and Search"]

            navigation_log.append(log)
        # navigation_log.append(log)

    if prompt_img_interleaved or prompt_img_seperate:
        for frame_id in embodied_memory.visual_memory:
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
    scratchpad = []

    # Option 1: Nested JSON
    # not_in_graph = set(embodied_memory.scene_graph.get_node_ids())
    # in_graph = set()
    # level = 0
    # while len(not_in_graph) > 0:
    #     for id in not_in_graph:
    #         if embodied_memory.scene_graph[id].level == level:
    #             node = {
    #                 "name": embodied_memory.scene_graph[id].name,
    #                 "visual caption": embodied_memory.scene_graph[id].captions[0],
    #                 "room": get_room_name(embodied_memory.scene_graph[id].room_id),
    #                 # "hierarchy level": embodied_memory.scene_graph[id].level,
    #                 "centroid": ", ".join(
    #                     f"{x:.4f}"
    #                     for x in embodied_memory.scene_graph[id].features.centroid.tolist()
    #                 ),
    #                 "query-relevant notes": embodied_memory.scene_graph[id].notes,
    #             }
    #             for type in EDGE_TYPES:
    #                 node["Items " + type] = []

    #             children_id = embodied_memory.get_children_ids(
    #                 id, embodied_memory.hierarchy_matrix
    #             )
    #             for child_id in children_id:
    #                 if embodied_memory.scene_graph[child_id].level >= level:
    #                     continue
    #                 child_node = search_name_recursive(
    #                     graph, embodied_memory.scene_graph[child_id].name
    #                 )
    #                 node[
    #                     "Items " + embodied_memory.hierarchy_type_matrix[id][child_id]
    #                 ].append(child_node)
    #             graph.append(node)
    #             in_graph.add(id)
    #     not_in_graph = not_in_graph - in_graph
    #     level += 1
    # scratchpad = extract_field_recursive(graph, "query-relevant notes")

    # Option 2: Flat Edges
    # for node in embodied_memory.scene_graph.nodes():
    #     edges = [e.json() for e in node.edges]
    #     _ = [e.pop("frame_id") for e in edges]
    #     _ = [e.pop("subject") for e in edges]
    #     graph.append(
    #         {
    #             "name": node.name,
    #             "visual caption": node.captions[0],
    #             "your notes": node.notes,
    #             "times scene": node.times_scene,
    #             "room": get_room_name(node.room_id),
    #             "hierarchy level": node.level,
    #             "centroid": node.features.centroid.tolist(),
    #             "relationships": edges,
    #         }
    #     )

    # # Option 3: Edges at in seperate key
    # graph = {"Nodes": []}
    # edges = []
    # for node in embodied_memory.scene_graph.get_nodes():
    #     graph["Nodes"].append(
    #         {
    #             "name": node.name,
    #             "visual caption": node.captions[0],
    #             "your notes": node.notes,
    #             "times scene": node.times_scene,
    #             "room": get_room_name(node.room_id),
    #             "hierarchy level": node.level,
    #             "centroid": node.features.centroid.tolist(),
    #         }
    #     )
    #     new_edges = [e.json() for e in node.edges]
    #     _ = [e.pop("frame_id") for e in new_edges]
    #     _ = [e.pop("subject") for e in new_edges]
    #     edges += new_edges
    # graph["Relationships"] = edges

    # Option 4: Hierarchy with room nodes
    # graph = {"unknown": {"Objects": []}}
    # for id in embodied_memory.rooms:
    #     graph[get_room_name(embodied_memory.rooms[id].name)] = {"Objects": []}
    # scratchpad = copy.deepcopy(graph)
    # for node in embodied_memory.scene_graph.get_nodes():
    #     room_name = get_room_name(node.room_id)
    #     if not room_name in graph:
    #         room_name = "unknown"
    #     edges = [e.json() for e in node.edges]
    #     _ = [e.pop("frame_id") for e in edges]
    #     _ = [e.pop("subject") for e in edges]
    #     graph[room_name]["Objects"].append(
    #         {
    #             "name": node.name,
    #             "visual caption": node.captions[0],
    #             # "times scene": node.times_scene,
    #             # "hierarchy level": node.level,
    #             "centroid": ", ".join(
    #                 f"{x:.4f}" for x in node.features.centroid.tolist()
    #             ),
    #             "edges": edges,
    #         }
    #     )
    #     scratchpad[room_name]["Objects"].append(
    #         {
    #             "name": node.name,
    #             "query-relevant notes": node.notes,
    #         }
    #     )

    # Option 5: Flat Edges
    for node in embodied_memory.scene_graph.get_nodes():
        edges = [e.json() for e in node.edges]
        _ = [e.pop("frame_id") for e in edges]
        _ = [e.pop("subject") for e in edges]
        graph.append(
            {
                "name": node.name,
                "visual caption": node.captions[0],
                "times scene": node.times_scene,
                "room": get_room_name(node.room_id),
                # "hierarchy level": node.level,
                "centroid": node.features.centroid.tolist(),
                "relationships": edges,
            }
        )
        scratchpad.append(
            {
                "name": node.name,
                "query-relevant notes": node.notes,
            }
        )
    return (
        graph,
        scratchpad,
        navigation_log,
        images_prompt,
        embodied_memory.visual_memory,
    )


def extract_detections_prompts(detections: DetectionList, scene_graph):
    prev_detections_json = []
    for obj in detections:
        id = get_nodeid_by_name(scene_graph, obj.matched_node_name)
        prev_detections_json.append(
            {
                "track name": obj.matched_node_name,
                "visual caption": obj.visual_caption,
                "spatial caption": obj.spatial_caption,
                "bbox": obj.bbox,
                "edges": [str(edge) for edge in obj.edges],
                "your notes": scene_graph[id].notes,
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
