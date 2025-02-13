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
from scene_graph.relationship_scorer import RelationshipScorer
from scene_graph.semantic_tree import SemanticTree

from .api import API, create_gemini_video_prompt, extract_scene_prompts


def parse_output(output: str) -> str:
    start_idx = output.find("A:")
    if start_idx == -1:
        raise ValueError("Invalid output string: {}".format(output))
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return output[start_idx:].replace("A:", "").strip()
    return output[start_idx:end_idx].replace("A:", "").strip()


# with open("open_eqa/prompts/gso/system_prompt_single-api.txt", "r") as f:
#     system_prompt = f.read().strip()
# with open("open_eqa/prompts/gso/system_prompt_multi-api.txt", "r") as f:
#     system_prompt = f.read().strip()


def format_scene_jsons(question, graph, navigation_log, scratchpad):
    return f"Your question is: {question} \nYour current Scene Graph: {graph} \nYour current Observation Log: {navigation_log} \nYour current ScratchPad: {scratchpad}"


def inference_time_search(
    question: str,
    api: API,
    dataset,
    gemini_model,
    graph_result_path: str,
    result_path: str,
    device,
    api_declarations,
    visual_memory_size,
    max_search_depth,
    system_prompt,
    skip_frame,
    obj_pcd_max_points=5000,
    downsample_voxel_size=0.02,
    gemini_seed: int = 1234,
    gemini_max_tokens: int = 128,
    gemini_temperature: float = 0.2,
    force: bool = False,
    load_floors=False,
    load_rooms=False,
) -> Optional[str]:

    semantic_tree = SemanticTree(visual_memory_size, device=device)
    semantic_tree.load(
        graph_result_path, load_floors=load_floors, load_rooms=load_rooms
    )

    relationship_scorer = RelationshipScorer(downsample_voxel_size=0.02)
    hierarchy_matrix, hierarchy_type_matrix = relationship_scorer.infer_hierarchy_vlm(
        semantic_tree.tracks
    )
    semantic_tree.compute_node_levels(hierarchy_matrix, hierarchy_type_matrix)

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
    # vertexai.init(project="robotics-447921")

    model = GenerativeModel(
        model_name=gemini_model,
        tools=[Tool([FunctionDeclaration(**a) for a in api_declarations])],
        system_instruction=system_prompt,
    )
    chat = model.start_chat()

    answer = None
    api_logs = []
    token_counts = []

    # graph_prompt, navigation_log_prompt, images_prompt, frame_ids = (
    #     extract_scene_prompts(
    #         semantic_tree,
    #         dataset,
    #         video_uri=video_uri,
    #         fps=video_fps,
    #         prompt_video=prompt_video,
    #         prompt_img_seperate=prompt_img_seperate,
    #     )
    # )
    graph_prompt, scratchpad_prompt, navigation_log_prompt, images_prompt, frame_ids = (
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
            format_scene_jsons(
                question=question,
                graph=json.dumps(graph_prompt),
                navigation_log=json.dumps(navigation_log_prompt),
                scratchpad=scratchpad_prompt,
            )
        )
    ]

    for i, id in enumerate(frame_ids):
        prompt.append(Part.from_text(f"Frame Index {id}:"))
        prompt.append(Part.from_image(Image.load_from_file(images_prompt[i])))
    # for i in range(0, len(dataset), 5):
    #     prompt.append(Part.from_text(f"Frame Index {i}:"))
    #     prompt.append(Part.from_image(Image.load_from_file(dataset.color_paths[i])))

    prompt = Content(role="user", parts=prompt)
    response = chat.send_message(prompt)
    token_counts.append(
        {
            "prompt_token_count": response.usage_metadata.prompt_token_count,
            "candidates_token_count": response.usage_metadata.candidates_token_count,
        }
    )
    for i in range(max_search_depth):
        parts = response.candidates[0].content.parts
        if len(parts) == 1 and not "function_call" in parts[0].to_dict():
            response = chat.send_message(
                f"In a few words, summarize your answer to the question '{question}'? Do not include any explanation or justification for your answer. If you are uncertain in your answer, then state your most likely answer."
            )
            answer = response.candidates[0].content.parts[0].text.strip()
            break
        call_response = []
        for part in parts:
            if not "function_call" in part.to_dict():
                continue
            tool = part.function_call
            keyframe = int(tool.args["frame_id"])
            # if not keyframe in semantic_tree.visual_memory:
            call_response.append(Part.from_text(f"Frame Index {keyframe}:"))
            call_response.append(
                Part.from_image(Image.load_from_file(dataset.color_paths[keyframe]))
            )
            semantic_tree, api_response, api_log, new_nodes = api.call(
                {"type": tool.name, **tool.args},
                dataset,
                result_path,
                semantic_tree,
                obj_pcd_max_points,
                downsample_voxel_size=downsample_voxel_size,
            )
            api_logs.append(api_log)
            (
                graph_prompt,
                scratchpad_prompt,
                navigation_log_prompt,
                images_prompt,
                frame_ids,
            ) = extract_scene_prompts(semantic_tree, dataset, prompt_video=prompt_video)
            text_response = {
                "New SceneGraph": json.dumps(graph_prompt),
                # f"New Observation Log for frame {keyframe}": json.dumps(navigation_log_prompt[keyframe % skip_frame]),
                f"New Observation Log": json.dumps(navigation_log_prompt),
                "New Scratchpad": json.dumps(scratchpad_prompt),
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
        token_counts.append(
            {
                "prompt_token_count": response.usage_metadata.prompt_token_count,
                "candidates_token_count": response.usage_metadata.candidates_token_count,
            }
        )
    if answer == None:
        response = chat.send_message(
            f"In a few words, summarize your answer to the question '{question}'? Do not include any explanation or justification for your answer. If you are uncertain in your answer, then state your most likely answer."
        )
        answer = response.candidates[0].content.parts[0].text.strip()
        token_counts.append(
            {
                "prompt_token_count": response.usage_metadata.prompt_token_count,
                "candidates_token_count": response.usage_metadata.candidates_token_count,
            }
        )
    return answer, api_logs, token_counts
