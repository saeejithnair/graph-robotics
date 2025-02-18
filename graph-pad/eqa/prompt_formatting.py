import base64
import copy
import io

import cv2
import google.generativeai as genai
from PIL import Image

from embodied_memory.detection import DetectionList
from embodied_memory.edges import EDGE_TYPES
from embodied_memory.embodied_memory import EmbodiedMemory
from embodied_memory.scene_graph import get_nodeid_by_name


def extract_qa_prompts(
    embodied_memory: EmbodiedMemory,
    dataset,
    prompt_img_interleaved=False,
    prompt_img_seperate=True,
):
    navigation_log = []
    images_prompt = []

    for i in range(len(embodied_memory.navigation_log)):
        log = copy.deepcopy(embodied_memory.navigation_log[i])

        if log["General Frame Info"] is not None:
            log["General Frame Info"]["Visible Scene Graph Nodes"] = log[
                "General Frame Info"
            ]["Detections"]
            del log["General Frame Info"]["Detections"]

            # del log["General Frame Info"]["Estimated Current Location"]

            # Replace the Estimated Current Location with the Present Room
            log["General Frame Info"]["Current Room"] = "unknown"
            if log["General Frame Info"]["Present Room"] != "unknown":
                log["General Frame Info"]["Current Room"] = get_room_name(
                    log["General Frame Info"]["Present Room"]
                )
            del log["General Frame Info"]["Present Room"]

            for k in log["General Frame Info"]:
                log[k] = log["General Frame Info"][k]
            del log["General Frame Info"]
            del log["Focused Analyses and Search"]

            navigation_log.append(log)
        # navigation_log.append(log)

    visual_memory = embodied_memory.navigation_log.get_evenly_spaced_idxs(
        embodied_memory.visual_memory_size
    )
    if prompt_img_interleaved or prompt_img_seperate:
        for frame_id in visual_memory:
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
        visual_memory,
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


def get_room_name(room_name):
    if room_name == "unknown" or not " " in room_name or not "_" in room_name:
        return room_name
    floor_name = room_name.split(" ")[-1][0]
    id = room_name.split(" ")[-1][-1]
    room_string = " ".join(room_name.split(" ")[:-1]) + f" {id} on floor {floor_name}"
    return room_string


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


def format_qa_prompt(question, graph, navigation_log, scratchpad):
    return f"Your question is: {question} \nYour current Scene Graph: {graph} \nYour current Observation Log: {navigation_log} \nYour current ScratchPad: {scratchpad}"
