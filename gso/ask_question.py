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
from vertexai.generative_models import GenerativeModel, Part

import read_graphs
from scene_graph.perception import Perceptor
from scene_graph.pointcloud import create_depth_cloud
from scene_graph.relationship_scorer import FeatureComputer, RelationshipScorer
from scene_graph.semantic_tree import SemanticTree


def parse_output(output: str) -> str:
    start_idx = output.find("A:")
    if start_idx == -1:
        raise ValueError("Invalid output string: {}".format(output))
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return output[start_idx:].replace("A:", "").strip()
    return output[start_idx:end_idx].replace("A:", "").strip()


def call_gemini(model, prompt):
    response = None
    try:
        response = model.generate_content(prompt)
    except:
        time.sleep(30)
        response = model.generate_content(prompt)
    try:
        response = response.text
    except:
        response = "No response"
    return response


def flat_graph(
    question: str,
    api,
    dataset,
    graph_path: str,
    obj_pcd_max_points=5000,
    gemini_model: str = "gemini-1.5-flash-latest",
    gemini_seed: int = 1234,
    gemini_max_tokens: int = 128,
    gemini_temperature: float = 0.2,
    force: bool = False,
) -> Optional[str]:

    semantic_tree = read_graphs.read_hamsg_flatgraph(graph_path)
    graph, navigation_log, valid_keyframes = (
        semantic_tree.get_summary_for_questionanswer()
    )
    with open("open_eqa/prompts/structure/flat_graph_v3.txt", "r") as f:
        text_prompt_template = f.read().strip()
    with open("open_eqa/prompts/structure/flat_graph_final.txt", "r") as f:
        text_prompt_final = f.read().strip()

    # vertexai.init(project='302560724657', location="northamerica-northeast2")
    # model = GenerativeModel(model_name=gemini_model)
    # text_prompt = Part.from_text(text_prompt)
    # image_prompt = [Part.from_image(vertexai.generative_models.Image.load_from_file(i)) for i in init_images]
    # prompt = [text_prompt] + image_prompt

    model = genai.GenerativeModel(model_name=gemini_model)

    answer = None
    debuglog = [dict(question=question)]
    for i in range(6):
        text_prompt = text_prompt_template.format(
            question=question,
            graph=json.dumps(graph, indent=4),
            navigation_log=json.dumps(navigation_log, indent=4),
        )
        response = call_gemini(model, [text_prompt]).strip()
        if response.startswith("Answer:"):
            answer = response.removeprefix("Answer: ")
            break
        else:
            try:
                api_log = dict()
                api_log["call"] = response
                response = response.split(": ")[1].strip()
                response = response.split(",")
                keyframe = int(response[0].strip())
                assert keyframe in valid_keyframes
                question = (
                    ",".join(response[1:]).replace("'", "").replace('"', "").strip()
                )
                semantic_tree, response = api.call(
                    keyframe, question, dataset, semantic_tree, obj_pcd_max_points
                )
                api_log["keyframe_path"] = dataset.color_paths[keyframe]
                api_log["response"] = response
                debuglog.append(api_log)
            except Exception as e:
                print("Refinement:", e)
                traceback.print_exc()
                continue

    if answer == None:
        answer = call_gemini(model, [text_prompt_final])
    return answer.strip(), debuglog


class API_NoStructure:
    def __init__(self, device="cuda:2"):
        self.perceptor = Perceptor(
            prompt_file="open_eqa/prompts/api/api_no_structure.txt", device=device
        )
        self.feature_computer = FeatureComputer(device)
        self.perceptor.init()
        self.feature_computer.init()
        self.relationship_scorer = RelationshipScorer(downsample_voxel_size=0.02)

    def call(
        self,
        keyframe_id,
        request,
        dataset,
        semantic_tree: SemanticTree,
        obj_pcd_max_points,
    ):
        text_prompt = self.perceptor.text_prompt_template
        text_prompt = text_prompt.format(request=request)
        color_tensor, depth_tensor, intrinsics, *_ = dataset[keyframe_id]
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        depth_cloud = create_depth_cloud(depth_array, dataset.get_cam_K())
        unt_pose = dataset.poses[keyframe_id]
        trans_pose = unt_pose.cpu().numpy()
        color_np = color_tensor.cpu().numpy()  # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8)  # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        detections, response = self.perceptor.refine(color_np, text_prompt)
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

        navigation_log_refinement = dict(Request=request)
        navigation_log_refinement.update(response)
        semantic_tree.integrate_refinement_log(
            request, navigation_log_refinement, keyframe_id
        )

        # perceptor.save_results(
        #     llm_response, detections, perception_result_dir, frame_idx
        # )
        # detections.save(perception_result_dir / str(frame_idx), perceptor.img)

        return semantic_tree, response


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
        text_prompt_final = (
            f.read().strip().format(question=question, graph=json.dumps(graph))
        )

    # init_images = utils.sample_images(keyframes, n=5)
    # image_prompt = [utils.open_image(i) for i in init_images]

    # vertexai.init(project='302560724657', location="northamerica-northeast2")
    # model = GenerativeModel(model_name=gemini_model)
    # text_prompt = Part.from_text(text_prompt)
    # image_prompt = [Part.from_image(vertexai.generative_models.Image.load_from_file(i)) for i in init_images]
    # prompt = [text_prompt] + image_prompt

    model = genai.GenerativeModel(model_name=gemini_model)
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
