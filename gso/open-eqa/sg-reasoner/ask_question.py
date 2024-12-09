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
import read_graphs
import tqdm
import utils
import vertexai
from vertexai.generative_models import GenerativeModel, Part


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
        time.sleep(60)
        response = model.generate_content(prompt)
    try:
        response = response.text
    except:
        response = "No response"
    return response


def flat_structure(
    question: str,
    image_paths,
    graph_path: str,
    gemini_model: str = "gemini-1.5-flash-latest",
    gemini_seed: int = 1234,
    gemini_max_tokens: int = 128,
    gemini_temperature: float = 0.2,
    force: bool = False,
) -> Optional[str]:

    graph, navigation_log, valid_keyframes = read_graphs.read_hamsg_flatgraph(
        graph_path
    )

    with open("sg-reasoner/prompts/structure_flat.txt", "r") as f:
        text_prompt_template = f.read().strip()
    with open("sg-reasoner/prompts/structure_flat_final.txt", "r") as f:
        text_prompt_final = f.read().strip()

    # vertexai.init(project='302560724657', location="northamerica-northeast2")
    # model = GenerativeModel(model_name=gemini_model)
    # text_prompt = Part.from_text(text_prompt)
    # image_prompt = [Part.from_image(vertexai.generative_models.Image.load_from_file(i)) for i in init_images]
    # prompt = [text_prompt] + image_prompt

    model = genai.GenerativeModel(model_name=gemini_model)
    text_prompt = text_prompt_template.format(
        question=question,
        graph=json.dumps(graph, indent=4),
        navigation_log=json.dumps(navigation_log, indent=4),
    )

    answer = None
    for i in range(3):
        response = call_gemini(model, [text_prompt]).strip()
        if response.startswith("Ask:"):
            response = response.removeprefix("Ask: ").strip().split(",")
            keyframe = int(response[0].strip())
            assert keyframe in valid_keyframes
            question = response[1].strip().replace("'", "").replace('"', "")
            print(response, keyframe, question)
        else:
            answer = response.removeprefix("Answer: ")
            break
    if answer == None:
        answer = call_gemini(model, [text_prompt_final])
    return answer.strip()


def images_with_structure(
    question: str,
    image_paths,
    graph_path: str,
    gemini_model: str = "gemini-1.5-flash-latest",
    gemini_seed: int = 1234,
    gemini_max_tokens: int = 128,
    gemini_temperature: float = 0.2,
    force: bool = False,
) -> Optional[str]:

    graph, keyframes = read_graphs.read_conceptgraph(graph_path, image_paths)
    # graph, navigation_log, valid_keyframes = read_graphs.read_hamsg_flatgraph(
    #     graph_path, image_paths
    # )

    with open("sg-reasoner/prompts/structure_flat.txt", "r") as f:
        text_prompt_template = f.read().strip()
    with open("sg-reasoner/prompts/structure_flat_final.txt", "r") as f:
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
