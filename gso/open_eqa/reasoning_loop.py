import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Optional

import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import vertexai
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerativeModel,
    Image,
    Part,
    Tool,
)

import read_graphs
from scene_graph.perception import call_gemini

from .api import API, create_gemini_video_prompt, extract_scene_prompts


def parse_output(output: str) -> str:
    start_idx = output.find("A:")
    if start_idx == -1:
        raise ValueError("Invalid output string: {}".format(output))
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return output[start_idx:].replace("A:", "").strip()
    return output[start_idx:end_idx].replace("A:", "").strip()


api_declarations = [
    {
        "name": "analyze_frame",
        "description": "Analyzes an image frame based on a query and updates the Scene Graph with the findings.",
        "parameters": {
            "type": "object",
            "properties": {
                "frame_id": {
                    "type": "integer",
                    "description": "Frame index of the image to analyze",
                },
                "query": {
                    "type": "string",
                    "description": "The analysis query (e.g., 'What is the relationship between the table and the chair?', 'Is the door open?', 'What is the material of the countertop?')",
                },
                "justification": {
                    "type": "string",
                    "description": "Explanation why you choose to search the requested frame and why you choose the requested query.",
                },
            },
            "required": ["frame_id", "query", "justification"],
        },
    },
    # {
    #     "name": "find_objects_in_frame",
    #     "description": "Adds new objects to the Scene Graph by discovering items present in the scene but not yet represented.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "frame_id": {
    #                 "type": "integer",
    #                 "description": "Frame index of image to search",
    #             },
    #             "query": {
    #                 "type": "string",
    #                 "description": "Search query (e.g., object characteristics, function, location)",
    #             },
    #             "justification": {
    #                 "type": "string",
    #                 "description": "Explanation of why this is the best Tool to use. Also explain why you choose the particular frame to search instead of others.",
    #             },
    #         },
    #         "required": ["frame_id", "query", "justification"],
    #     },
    # },
    # {
    #     "name": "analyze_objects_in_frame",
    #     "description": "Analyzes an image to clarify the attributes of existing Scene Graph nodes. Does not add new nodes to the Scene Graph.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "frame_id": {
    #                 "type": "integer",
    #                 "description": "Frame index of image to search",
    #             },
    #             "query": {
    #                 "type": "string",
    #                 "description": "Analysis query (e.g., object state, features, comparisons, spatial relationships)",
    #             },
    #             "nodes": {
    #                 "type": "string",
    #                 "description": "Labels of objects your query is relevent to. Should correspond to objects in the Scene Graph.",
    #             },
    #             "justification": {
    #                 "type": "string",
    #                 "description": "Explanation of why this is the best Tool to use. Also explain why you choose the particular frame to search instead of others.",
    #             },
    #         },
    #         "required": ["frame_id", "query", "justification", "nodes"],
    #     },
    # },
]
system_prompt = """
You are an AI agent specializing in scene understanding and reasoning. Your task is to answer questions about the scene, but only after ensuring the Scene Graph is sufficiently detailed and accurate. You must rely on your Scene Exploration Tool to enhance the Scene Graph before providing answers. Avoid answering questions directly unless the Scene Graph is explicitly complete and precise.

You will be provided with:
1. Scene Graph: Your internal structured representation of the scene, including objects, attributes, and spatial relationships. This graph is often incomplete, imprecise, or incorrect and must be expanded or refined before answering questions.
2. Frame Memory: Raw images of the scene. Use these images as context to understand the Scene Graph. Do not directly base answers directly on the images. 
3. Observation Log:  A summary of prior observations and analyses. Use this log to plan your exploration and tool usage but do not base answers directly on it.

When answering a question, assume that the initial Scene Graph may lack the necessary information. Use the Scene Exploration Tool called analyze_frame to improve it:
   - analyze_frame: Analyzes an image frame based on a query and updates the Scene Graph with the findings. This can add new attributes to existing nodes or establish relationships between existing nodes.
   
Keep using your Scene Graph Exploration Tool unless you are certain that the Scene Graph already contains complete and accurate information to answer the question. Once the Scene Graph provides sufficient detail to answer the question conclusively, provide your answer and your justification.
"""


def reasoning_loop_only_graph(
    question: str,
    api: API,
    dataset,
    graph_result_path: str,
    result_path: str,
    obj_pcd_max_points=5000,
    downsample_voxel_size=0.02,
    gemini_model: str = "gemini-1.5-flash-latest",
    gemini_seed: int = 1234,
    gemini_max_tokens: int = 128,
    gemini_temperature: float = 0.2,
    force: bool = False,
    load_floors=False,
    load_rooms=False,
) -> Optional[str]:

    semantic_tree = read_graphs.read_hamsg_flatgraph(
        graph_result_path, load_floors=load_floors, load_rooms=load_rooms
    )
    semantic_tree.compute_node_levels()
    with open(api.prompt_reasoning_loop, "r") as f:
        text_prompt_template = f.read().strip()
    with open(api.prompt_reasoning_final, "r") as f:
        text_prompt_final_template = f.read().strip()

    # vertexai.init(project='302560724657', location="northamerica-northeast2")
    # model = GenerativeModel(model_name=gemini_model)
    # text_prompt = Part.from_text(text_prompt)
    # image_prompt = [Part.from_image(vertexai.generative_models.Image.load_from_file(i)) for i in init_images]
    # prompt = [text_prompt] + image_prompt

    prompt_img_seperate = True
    prompt_video, video_fps = False, 1
    if prompt_video:
        video_uri = create_gemini_video_prompt(
            "video", dataset.color_paths, fps=video_fps
        )
    else:
        video_uri = None

    vertexai.init(project="total-byte-432318-q3")
    model = GenerativeModel(
        model_name="gemini-1.5-flash-002",  # "gemini-1.5-flash-002" "gemini-2.0-flash-exp"
        tools=[Tool([FunctionDeclaration(**a) for a in api_declarations])],
        system_instruction=system_prompt,
    )
    chat = model.start_chat()

    answer = None
    api_logs = []

    graph_prompt, navigation_log_prompt, images_prompt, frame_ids = (
        extract_scene_prompts(
            semantic_tree,
            dataset,
            video_uri=video_uri,
            fps=video_fps,
            prompt_video=prompt_video,
            prompt_img_seperate=prompt_img_seperate,
        )
    )
    prompt = [
        Part.from_text(
            text_prompt_template.format(
                question=question,
                graph=json.dumps(graph_prompt, indent=2),
                navigation_log=json.dumps(navigation_log_prompt, indent=2),
            )
        )
    ]
    for i, id in enumerate(frame_ids):
        prompt.append(Part.from_text(f"Frame Index {id}:"))
        prompt.append(Part.from_image(Image.load_from_file(images_prompt[i])))
    prompt = Content(role="user", parts=prompt)
    response = chat.send_message(prompt)
    for i in range(20):
        parts = response.candidates[0].content.parts
        if len(parts) == 1 and not "function_call" in parts[0].to_dict():
            response = chat.send_message(
                f"In a few words, summarize your answer to the question '{question}'? Do not include any explanation or justification for your answer."
            )
            answer = response.candidates[0].content.parts[0].text.strip()
            break
        call_response = []
        for part in parts:
            if not "function_call" in part.to_dict():
                continue
            tool = part.function_call
            keyframe = int(tool.args["frame_id"])
            if not keyframe in semantic_tree.visual_memory:
                call_response.append(Part.from_text(f"Frame Index {keyframe}:"))
                call_response.append(
                    Part.from_image(Image.load_from_file(dataset.color_paths[keyframe]))
                )
            semantic_tree, api_response, api_log = api.call(
                {"type": tool.name, **tool.args},
                dataset,
                result_path,
                semantic_tree,
                obj_pcd_max_points,
                downsample_voxel_size=downsample_voxel_size,
            )
            api_logs.append(api_log)
            graph_prompt, navigation_log_prompt, images_prompt, frame_ids = (
                extract_scene_prompts(semantic_tree, dataset, prompt_video=prompt_video)
            )
            text_response = {
                "Updated SceneGraph": json.dumps(graph_prompt, indent=2),
                "Updated Observation Log": json.dumps(navigation_log_prompt, indent=2),
            }
            call_response.append(
                Part.from_function_response(name=tool.name, response=text_response)
            )
        response = chat.send_message(
            Content(
                role="function_response",
                parts=call_response,
            )
        )
    if answer == None:
        graph_prompt, navigation_log_prompt, images_prompt, frame_ids = (
            extract_scene_prompts(semantic_tree, dataset, prompt_video=prompt_video)
        )
        # Bug, this prompt should return a JSON
        text_prompt_final = text_prompt_final_template.format(
            question=question,
            graph=json.dumps(graph_prompt, indent=2),
            navigation_log=json.dumps(navigation_log_prompt, indent=2),
            frame_ids=frame_ids,
        )
        answer = call_gemini(model, [text_prompt_final] + images_prompt)
        answer = json.loads(answer.replace("```json", "").replace("```", ""))
    return answer, api_logs


def images_and_graph(
    question: str,
    image_paths,
    graph_path: str,
    gemini_model: str = "gemini-1.5-flash-latest",
    gemini_seed: int = 1234,
    gemini_max_tokens: int = 128,
    gemini_temperature: float = 0.2,
    force: bool = False,
) -> Optional[str]:
    import utils

    graph, keyframes = read_graphs.read_conceptgraph(graph_path, image_paths)
    # graph, navigation_log, valid_keyframes = read_graphs.read_hamsg_flatgraph(
    #     graph_path, image_paths
    # )

    with open("open_eqa/sg-reasoner/prompts/images_with_graph.txt", "r") as f:
        text_prompt_template = f.read().strip()
    with open("sg-reasoner/prompts/images_with_graph_final.txt", "r") as f:
        text_prompt_final = f.read().strip()

    # init_images = utils.sample_images(keyframes, n=5)
    # image_prompt = [utils.open_image(i) for i in init_images]

    # vertexai.init(project='302560724657', location="northamerica-northeast2")
    # model = GenerativeModel(model_name=gemini_model)
    # text_prompt = Part.from_text(text_prompt)
    # image_prompt = [Part.from_image(vertexai.generative_models.Image.load_from_file(i)) for i in init_images]
    # prompt = [text_prompt] + image_prompt

    model = genai.GenerativeModel(model_name="gemini-1.5-flash-002")
    text_prompt = text_prompt_template.format(
        question=question, graph=json.dumps(graph)
    )

    answer = None
    for i in range(3):
        response = call_gemini(model, [text_prompt] + image_prompt)
        if response.startswith("Get Image:"):
            objects = response.removeprefix("Get Image: ").strip().split(", ")
            new_keyframes = utils.sample_images(
                utils.subset_dict(keyframes, objects), n=4
            )
            image_prompt = image_prompt + [utils.open_image(i) for i in new_keyframes]
        else:
            answer = response
            break
    if answer == None:
        answer = call_gemini(model, [text_prompt_final] + image_prompt)
    return answer.removeprefix("Answer: ").strip()

    try:
        raise Exception("Pause")
        prompt = load_prompt("blind-llm")
        set_openai_key(key=openai_key)
        messages = prepare_openai_messages(prompt.format(question=question))
        output = call_openai_api(
            messages=messages,
            model=gemini_model,
            seed=gemini_seed,
            max_tokens=gemini_max_tokens,
            temperature=gemini_temperature,
        )
        return parse_output(output)
    except Exception as e:
        if not force:
            traceback.print_exc()
            raise e
