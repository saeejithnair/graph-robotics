import argparse
import json
import os
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import google.generativeai as genai
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import vertexai
from omegaconf import DictConfig
from vertexai.generative_models import GenerativeModel, Part

import embodied_memory.utils as utils
import eqa.reasoning_loop as reasoning_loop
from embodied_memory.datasets import get_dataset
from eqa.api import FrameLevelAPI, NodeLevelAPI


# A logger for this file
@hydra.main(
    version_base=None,
    config_path="embodied_memory/configs/mapping",
    config_name="hm3d_mapping",
)
def main(cfg: DictConfig):
    # check for GOOGLE_API_KEY api key
    with open("../api_keys/gemini_key.txt") as f:
        GOOGLE_API_KEY = f.read().strip()
        # genai.set_api_key(GOOGLE_API_KEY)

    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        "/home/qasim/Projects/graph-robotics/api_keys/total-byte-432318-q3-78e6d4aa6497.json"
    )

    cfg = utils.process_cfg(cfg)

    # load questions
    with open(cfg.questions) as f:
        questions = json.load(f)

    # read the openeqa184 question set
    openeqa_184 = []
    with open("eqa/data/openeqa_184.txt", "r") as f:
        for line in f.readlines():
            if "#" in line or len(line.strip()) == 0:
                continue
            openeqa_184.append(line.strip())

    # only run particular questions
    subset_questions = []
    if cfg.use_184_subset:
        subset_questions = openeqa_184
    subset_questions = [
        # "f2e82760-5c3c-41b1-88b6-85921b9e7b32",
        # "7447d782-d1a7-4c87-86dc-b5eafc5a0f76",
    ]
    print("running {:,} questions".format(len(questions)))
    if len(subset_questions) > 0:
        new_questions = []
        for q in subset_questions:
            for i in range(len(questions)):
                if questions[i]["question_id"] == q:
                    new_questions.append(questions[i])
                    break
        questions = new_questions

    # initialize output file names
    output_path = Path(cfg.questions_output_dir) / (
        Path(cfg.questions).stem
        + "-{}-{}-depth{}-vsize{}.json".format(
            cfg.questions_model,
            cfg.api_type,
            cfg.max_search_depth,
            cfg.visual_memory_size,
        )
    )
    log_path = Path(cfg.questions_output_dir) / (
        "Log-{}-{}-depth{}-vsize{}.json".format(
            cfg.questions_model,
            cfg.api_type,
            cfg.max_search_depth,
            cfg.visual_memory_size,
        )
    )
    print("Saving results to ", output_path)

    # load past results
    results, debuglogs = [], []
    if output_path.exists() and not cfg.questions_ignore_old_results:
        results = json.load(output_path.open())
        print("found {:,} existing results".format(len(results)))
        if log_path.exists() and not cfg.questions_ignore_old_results:
            debuglogs = json.load(log_path.open())
            print("found {:,} existing results".format(len(results)))
    completed = [item["question_id"] for item in results]

    # create api object
    assert cfg.api_type in ["frame_level", "node_level"]
    if cfg.api_type == "node_level":
        api = NodeLevelAPI(
            device=cfg.device,
            gemini_model=cfg.questions_model,
        )
    else:
        api = FrameLevelAPI(
            device=cfg.device,
            gemini_model=cfg.questions_model,  # "gemini-1.   5-flash-002" "gemini-2.0-flash-exp"
        )
    system_prompt = api.get_system_prompt()

    for idx, item in enumerate(tqdm.tqdm(questions)):
        if cfg.questions_dry_run and idx >= 5:
            break

        # skip completed questions
        question_id = item["question_id"]
        if question_id in completed:
            continue  # skip existing
        scene_id = os.path.basename(item["episode_history"])

        dataset = get_dataset(
            dataconfig=cfg.dataset_config,
            start=cfg.start,
            end=cfg.end,
            stride=cfg.stride,
            basedir=cfg.dataset_root,
            sequence=scene_id,
            desired_height=cfg.image_height,
            desired_width=cfg.image_width,
            device=cfg.device,
            dtype=torch.float,
            # relative_pose=True
        )

        # generate answer, try atleast a couple times before failing
        question = item["question"]
        answer = None
        for i in range(3):
            try:
                answer, api_logs, token_counts = reasoning_loop.answer_question(
                    question=question,
                    api=api,
                    dataset=dataset,
                    system_prompt=system_prompt,
                    cfg=cfg,
                    result_dir_embodied_memory=Path(cfg.result_root)
                    / scene_id
                    / "embodied_memory",
                    result_dir_detections=Path(cfg.result_root)
                    / scene_id
                    / "detections",
                    temp_workspace_dir=Path(cfg.result_root)
                    / scene_id
                    / "qa-temp-workspace",
                )
                break
            except:
                time.sleep(30)
                traceback.print_exc()
                pass
        if answer == None:
            raise Exception("Failed to generate answer")

        # Write the results to disk
        results.append({"question_id": question_id, "answer": answer})
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        # Write the debug logs to disk
        debuglog = OrderedDict(
            {
                "question": question,
                "gt_answer": item["answer"],
                "pred_answer": answer,
                # "pred_answer": answer["answer"],
                # "pred_answer_evidence": answer["evidence"],
                # "pred_answer_justification": answer["justification"],
            }
        )
        if "explain_info_sufficiency" in answer:
            debuglog["explain_info_sufficiency"] = answer["explain_info_sufficiency"]
        debuglog["question_id"] = question_id
        debuglog["category"] = item["category"]
        debuglog["episode_history"] = item["episode_history"]
        debuglog["api_logs"] = api_logs
        debuglog["token_count"] = token_counts
        debuglogs.append(debuglog)
        with open(log_path, "w") as f:
            json.dump(debuglogs, f, indent=2)

    # save at end for redundancy
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print("saving {:,} answers".format(len(results)))


if __name__ == "__main__":
    main()
