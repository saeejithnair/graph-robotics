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
        default=1000,
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
    print("found {:,} questions".format(len(questions)))
    # only run particular questions

    subset_questions = [
        "f2e82760-5c3c-41b1-88b6-85921b9e7b32",
        "79344680-6b45-4531-8789-ad0f5ef85b3b",
        "35bfcf60-4227-4bbb-9213-e1bf643b2325",
        "87019892-7a7f-4ce0-b1e6-4c0d6e87b90a",
        "28ea4932-55bf-44b3-9a48-73fde896b8ce",
        "915cb310-31be-4114-846c-242fc59b581d",
        "d7019200-5300-459e-a2c1-b54d5ec0a80b",
        "9fc2d987-fc9c-4771-a0d4-07d24e2b7910",
        "06745779-f1b8-458b-8daa-ccc18d8958c8",
        "884e2405-85f6-446f-a0ab-a03e537e5184",
        "cec83622-f735-435a-b054-1e201a48652c",
        "9104773d-a2f0-4be3-a370-f7bf6e242ff7",
        "81739eb9-ef32-42e3-b1cf-fa8e826813ef",
        "d9737f88-11c2-454d-aa40-a2e1f3111505",
        "cbb81160-b858-49d6-a234-a94edb4d0b1d",
        "143ade55-0216-455d-8e2f-6aa249596986",
        "f144699a-0af1-4830-9b28-a2b82777a92e",
        "10d6d14b-ef30-42b6-89d7-b79eb4ce9b5d",
        "fa4fdac0-a749-4744-a739-3eff882a2862",
        "17d99f84-d536-4623-b428-d298f8669f09",
        "8548aacb-669f-4341-a21e-0426e5dc3b42",
        "6e4d210c-e7b0-4e71-96e9-d4f58f17b3ea",
        "0748efce-1cd1-4b0f-89c3-1248e49bc8fc",
        "26bd014e-529f-4deb-bcfd-261f35ac7ff2",
        "e2a55cb0-a883-4dd6-9b2f-239d92ebd8bc",
        "fd4dad2f-19fc-4a3f-87ac-d210657d1add",
        "566776fa-aab8-4084-b40b-57f69167c97c",
        "83356063-ea59-4894-a077-5c2a6fb1042e",
        "4dbd213e-56cd-481a-8ff5-ed9a8d636dbc",
        "ae19adeb-498a-4814-b955-e0af05623f9b",
        "7ebac357-a338-4ce0-975a-62141e90a3c3",
        "3825d20a-c58a-458a-971c-f1714dfccd9d",
        "07c4017d-db5a-447a-8086-17d9472e7100",
        "e45cb43e-25a2-46cd-818a-5267984ceafa",
        "d36c5ac4-65b9-4979-881c-56c7d0870a50",
        "15ef0e88-83c5-41dd-9a1f-cf9feb3dafbb",
        "225a293a-2a24-43da-a783-7012d968d731",
        "6b8f1b52-25fa-47bc-a3a6-a2a43e834605",
        "cd5a15ff-d222-4745-8238-eee9afb27b79",
        "41693f7e-4192-495e-9b4e-b238432c6424",
        "31561566-4e70-40c7-9c5b-583ddc2d39cf",
        "4decde3d-5ab8-43db-893c-c3f3f80bcc76",
        "36ad6cce-7cd1-429e-b75a-581dc6849603",
        "2449be8f-1320-4061-beb0-2797f5766c73",
        "8471794d-32cd-4989-8cec-91118eb43b67",
        "d5b18be3-2d0c-4653-9706-7c33159de7a9",
        "0f56a54b-eac3-4b98-b118-ddcbac7735e7",
        "bfe3dd93-acd5-42ab-a492-f5914c1a3b26",
        "987aac58-d2a4-417e-84fc-4b1ede9934a9",
        "447e4e2d-7010-4672-b8e0-eb5246430499",
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
        "open_eqa/prompts/gso/flat_graph_v13_withimages.txt",
        "open_eqa/prompts/gso/flat_graph_final.txt",
        device,
    )

    # process data
    for idx, item in enumerate(tqdm.tqdm(questions)):
        if cfg.questions_dry_run and idx >= 5:
            break
        if idx >= 250:
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
