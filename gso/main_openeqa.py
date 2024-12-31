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
from open_eqa.api import API_GraphAPI, API_TextualQA
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
        default=512,
        help="image size (default: 512)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-flash-latest",
        help="Gemini model (default: gpt-4-0613)",
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
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        "/home/qasim/Projects/graph-robotics/api_keys/total-byte-432318-q3-78e6d4aa6497.json"
    )

    cfg = utils.process_cfg(cfg)

    # load questions
    with open(cfg.questions) as f:
        questions = json.load(f)
    print("found {:,} questions".format(len(questions)))
    # only run particular questions
    subset_questions = [
        # "41ad08e1-cb46-4599-8beb-def3b0f91e34",
        # "35bfcf60-4227-4bbb-9213-e1bf643b2325",
        # "9104773d-a2f0-4be3-a370-f7bf6e242ff7",
        # "6fddec60-f221-4944-88c3-a132dfbfd1ed",
        # "915cb310-31be-4114-846c-242fc59b581d",
        # "bbb83d84-289c-4f1d-a470-772e5c823a90",
        # "6be2fe87-f20c-48a2-a8fb-161362d86e2a",
        # "96d7f5ef-14b2-432a-8b85-21a0621e41a4",
        # "915cb310-31be-4114-846c-242fc59b581d",
        # "41ad08e1-cb46-4599-8beb-def3b0f91e34",
        # "06745779-f1b8-458b-8daa-ccc18d8958c8",
        # "35bfcf60-4227-4bbb-9213-e1bf643b2325",
        # "87019892-7a7f-4ce0-b1e6-4c0d6e87b90a",
        # "025257b6-8b7e-4f6f-aacc-1788069cbfad",
        # "28ea4932-55bf-44b3-9a48-73fde896b8ce",
        # "77c6644e-6018-4ef3-a683-276d3d2af67f",
        # "564d876c-44f1-4915-8d4b-ab5a7728494f",
        # "500e6924-0ea3-45c8-89ce-db3d37e142bf",
        # "5c1d30b9-5827-49fc-b74f-5ee9909a75b8",
        # "8d7f1dd7-9764-4603-918b-58eefcb0b10e",
        # "f2e82760-5c3c-41b1-88b6-85921b9e7b32",
    ]
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
    # scenes_available = [
    #     "000-hm3d-BFRyYbPCCPE",
    #     "002-hm3d-wcojb4TFT35",
    #     "003-hm3d-c5eTyR3Rxyh",
    # ]

    device = "cpu"
    # api = API_TextualQA(
    #     "open_eqa/prompts/gso/flat_graph_v3.txt",
    #     "open_eqa/prompts/gso/flat_graph_final.txt",
    #     device,
    # )
    api = API_GraphAPI(
        "open_eqa/prompts/gso/flat_graph_v8_rename.txt",
        "open_eqa/prompts/gso/flat_graph_final.txt",
        device,
    )

    # process data
    for idx, item in enumerate(tqdm.tqdm(questions)):
        if cfg.questions_dry_run and idx >= 5:
            break
        if idx >= 240:
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
        )

        # store results
        debuglog = OrderedDict(
            {
                "question": question,
                "answer": item["answer"],
                "pred_answer": answer["answer"],
                "pred_answer_evidence": answer["evidence"],
                "pred_answer_justification": answer["justification"],
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
        results.append({"question_id": question_id, "answer": answer["answer"]})
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    # save at end (redundant)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print("saving {:,} answers".format(len(results)))


if __name__ == "__main__":
    main()
