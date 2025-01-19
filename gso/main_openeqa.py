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

import open_eqa.reasoning_loop as reasoning_loop
import scene_graph.utils as utils
from open_eqa.api import API_GraphAPI
from open_eqa.reasoning_loop import reasoning_loop_only_graph
from scene_graph.datasets import get_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=Path,
        default="data/open-eqa-v0.json",
        help="path to EQ3A questions (default: data/open-eqa-v0.json)",
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default="/pub3/qasim/hm3d/data/ham-sg/",
        help="path to precreated scene graphs",
    )
    parser.add_argument(
        "--frames-directory",
        type=Path,
        default="/pub3/qasim/hm3d/data/frames",
        help="path image frames (default: data/frames/)",
    )
    parser.add_argument(
        "--use-annotated-frames",
        action="store_false",
        default=False,
        help="use vlm annotated frames or not",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1000,
        help="image size (default: 512)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="gpt seed (default: 1234)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="gpt temperature (default: 0.2)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default="results",
        help="output directory (default: results)",
    )
    parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="continue running on API errors (default: false)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=20,
        help="continue running on API errors (default: false)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only process the first 5 questions",
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (
        args.questions.stem + "-{}-{}.json".format(args.model, args.seed)
    )
    return args


# A logger for this file
@hydra.main(
    version_base=None,
    config_path="scene_graph/configs/mapping",
    config_name="hm3d_mapping",
)
def main(cfg: DictConfig):
    # check for GOOGLE_API_KEY api key
    with open("/home/qasim/Projects/graph-robotics/api_keys/gemini_key.txt") as f:
        GOOGLE_API_KEY = f.read().strip()
        # genai.set_api_key(GOOGLE_API_KEY)

    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        "/home/qasim/Projects/graph-robotics/api_keys/total-byte-432318-q3-78e6d4aa6497.json"
        # "/home/qasim/Projects/graph-robotics/api_keys/robotics-447921.json"
    )

    cfg = utils.process_cfg(cfg)

    # load questions
    with open(cfg.questions) as f:
        questions = json.load(f)
    print("found {:,} questions".format(len(questions)))
    # only run particular questions

    subset_questions = [
        # "a86ef102-5500-4fbf-8fae-cdbeb20a3b7b",
        # "35bfcf60-4227-4bbb-9213-e1bf643b2325",
        # "6fddec60-f221-4944-88c3-a132dfbfd1ed",
        # "025257b6-8b7e-4f6f-aacc-1788069cbfad",
        # "915cb310-31be-4114-846c-242fc59b581d",
        # "ecad68d2-a16f-4a3e-b8a1-a70ec1c5cf00",
        # "0c6b5dce-8baf-4d95-b7ca-26fe915c4bd7",
        # "21f8ed6a-4ca4-465f-986d-c4390984d8a9",
        # "5460114d-e885-4eae-8bdc-a273deb3df0a",
        # "77c6644e-6018-4ef3-a683-276d3d2af67f",
        # "21f8ed6a-4ca4-465f-986d-c4390984d8a9",
        # "500e6924-0ea3-45c8-89ce-db3d37e142bf",
        # "ecad68d2-a16f-4a3e-b8a1-a70ec1c5cf00",
        # "5c1d30b9-5827-49fc-b74f-5ee9909a75b8",
    ]
    # subset_questions = [
    #     "d36c5ac4-65b9-4979-881c-56c7d0870a50",
    #     "e177ec9a-7855-471d-9f81-3b42d5a91911",
    #     "37f74dbf-05d9-44c0-a5d0-6d996d16ccc5",
    #     "b4de266c-5361-46b7-a098-167d6ee4d5c1",
    #     "225a293a-2a24-43da-a783-7012d968d731",
    #     "08e8e5fd-31a3-466b-afd6-fa171f1d9de4",
    # ]
    if subset_questions and len(subset_questions) > 0:
        new_questions = []
        for q in subset_questions:
            for i in range(len(questions)):
                if questions[i]["question_id"] == q:
                    new_questions.append(questions[i])
                    break

        questions = new_questions
    # questions[0], questions[1] = questions[1], questions[0]

    output_path = Path(cfg.questions_output_dir) / (
        Path(cfg.questions).stem
        + "-{}-{}.json".format(cfg.questions_model, cfg.questions_exp_name)
    )
    debuglog_path = Path(cfg.questions_output_dir) / (
        "APIDebugLog-{}-{}.json".format(cfg.questions_model, cfg.questions_exp_name)
    )
    print("Saving results to ", output_path)

    # load results
    results, debuglogs = [], []
    if output_path.exists() and not cfg.questions_ignore_old_results:
        results = json.load(output_path.open())
        print("found {:,} existing results".format(len(results)))
        if debuglog_path.exists() and not cfg.questions_ignore_old_results:
            debuglogs = json.load(debuglog_path.open())
            print("found {:,} existing results".format(len(results)))
    completed = [item["question_id"] for item in results]

    scenes_available = os.listdir(cfg.questions_graph_dir)

    device = "cpu"
    # api = API_TextualQA(
    #     "open_eqa/prompts/gso/flat_graph_v3.txt",
    #     "open_eqa/prompts/gso/flat_graph_final.txt",
    #     device,
    # )
    api = API_GraphAPI(
        prompt_reasoning_loop="open_eqa/prompts/gso/flat_graph_v13_withimages.txt",
        prompt_reasoning_final="open_eqa/prompts/gso/flat_graph_final.txt",
        device=device,
        gemini_model=cfg.questions_model,  # "gemini-1.   5-flash-002" "gemini-2.0-flash-exp"
    )

    # process data
    for idx, item in enumerate(tqdm.tqdm(questions)):
        if cfg.questions_dry_run and idx >= 5:
            break
        if idx >= 54:
            break

        # skip completed questions
        question_id = item["question_id"]
        if question_id in completed:
            continue  # skip existing
        scene_id = os.path.basename(item["episode_history"])
        if not scene_id in scenes_available:
            continue

        dataset = get_dataset(
            dataconfig=cfg.dataset_config,
            start=cfg.start,
            end=cfg.end,
            stride=cfg.stride,
            basedir=cfg.dataset_root,
            sequence=scene_id,
            desired_height=cfg.image_height,
            desired_width=cfg.image_width,
            device=device,
            dtype=torch.float,
            # relative_pose=True
        )

        # generate answer
        question = item["question"]
        answer, api_logs = reasoning_loop.reasoning_loop_only_graph(
            question=question,
            api=api,
            dataset=dataset,
            graph_result_path=Path(cfg.questions_graph_dir) / scene_id,
            result_path=Path(cfg.result_root) / scene_id,
            obj_pcd_max_points=cfg.obj_pcd_max_points,
            downsample_voxel_size=cfg.downsample_voxel_size,
            gemini_model=cfg.questions_model,
            visual_memory_size=cfg.visual_memory_size,
            max_search_depth=cfg.max_search_depth,
            load_floors=True,
            load_rooms=True,
        )

        # store results
        debuglog = OrderedDict(
            {
                "question": question,
                "answer": item["answer"],
                "pred_answer": answer,
                # "pred_answer": answer["answer"],
                # "pred_answer_evidence": answer["evidence"],
                # "pred_answer_justification": answer["justification"],
            }
        )
        if "explain_info_sufficiency" in answer:
            debuglog["explain_info_sufficiency"] = answer["explain_info_sufficiency"]
        debuglog["question_id"] = (question_id,)
        debuglog["category"] = item["category"]
        debuglog["episode_history"] = item["episode_history"]
        debuglog["api_logs"] = api_logs
        debuglogs.append(debuglog)
        with open(debuglog_path, "w") as f:
            json.dump(debuglogs, f, indent=2)
        results.append({"question_id": question_id, "answer": answer})
        # results.append({"question_id": question_id, "answer": answer["answer"]})
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    # save at end (redundant)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print("saving {:,} answers".format(len(results)))


if __name__ == "__main__":
    main()
