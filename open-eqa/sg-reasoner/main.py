import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Optional

import ask_question
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import read_graphs
import tqdm
import utils
import vertexai
from vertexai.generative_models import GenerativeModel, Part


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=Path,
        default="data/open-eqa-v0.json",
        help="path to EQA questions (default: data/open-eqa-v0.json)",
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


def main(args: argparse.Namespace):
    # check for GOOGLE_API_KEY api key
    with open("/home/qasim/Projects/open-eqa/sg-reasoner/keys/gemini_key.txt") as f:
        GOOGLE_API_KEY = f.read().strip()
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        "/home/qasim/Projects/open-eqa/sg-reasoner/keys/total-byte-432318-q3-78e6d4aa6497.json"
    )

    # load questions
    questions = json.load(args.questions.open("r"))
    print("found {:,} questions".format(len(questions)))

    # only run particular questions
    subset_questions = [
        # "73bcd8d2-cd39-46e1-a63a-54acfae8d060",
        # "f2e82760-5c3c-41b1-88b6-85921b9e7b32",
        # "500e6924-0ea3-45c8-89ce-db3d37e142bf",
        # "c8808989-9b62-4764-814e-520fd40c22cd",
        # "0c6b5dce-8baf-4d95-b7ca-26fe915c4bd7",
    ]
    if subset_questions and len(subset_questions) > 0:
        questions = [q for q in questions if q["question_id"] in subset_questions]

    # load results
    results = []
    if False:  # args.output_path.exists():
        results = json.load(args.output_path.open())
        print("found {:,} existing results".format(len(results)))
    completed = [item["question_id"] for item in results]

    scenes_available = os.listdir(args.graph_dir)

    print("Saving results to ", args.output_path)

    # process data
    for idx, item in enumerate(tqdm.tqdm(questions)):
        if args.dry_run and idx >= 5:
            break

        # skip completed questions
        question_id = item["question_id"]
        if question_id in completed:
            continue  # skip existing
        scene_id = os.path.basename(item["episode_history"])
        if not scene_id in scenes_available:
            continue

        # extract scene paths
        if args.use_annotated_frames:
            folder = args.graph_dir / scene_id / "exps/s_detections/vis"
            frames = sorted(folder.glob("*-rgbannotated_for_vlm.jpg"))
        else:
            folder = args.frames_directory / scene_id
            frames = sorted(folder.glob("*-rgb.png"))
        indices = [int(s.stem[:5]) for s in frames]
        paths = {int(indices[i]): str(frames[i]) for i in range(len(frames))}

        # generate answer
        question = item["question"]
        answer = ask_question.flat_structure(
            question=question,
            image_paths=paths,
            graph_path=args.graph_dir / scene_id,
            gemini_model=args.model,
            gemini_seed=args.seed,
            gemini_temperature=args.temperature,
            force=args.force,
        )

        # store results
        results.append({"question_id": question_id, "answer": answer})
        json.dump(results, args.output_path.open("w"), indent=2)

    # save at end (redundant)
    json.dump(results, args.output_path.open("w"), indent=2)
    print("saving {:,} answers".format(len(results)))


if __name__ == "__main__":
    main(parse_args())
