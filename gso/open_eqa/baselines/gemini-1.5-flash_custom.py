# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import tqdm
from openeqa.utils.google_utils import call_google_api, set_google_key
from openeqa.utils.prompt_utils import load_prompt
from PIL import Image, PngImagePlugin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="open_eqa/data/open-eqa-v0.json",
        help="path to EQA dataset (default: data/open-eqa-v0.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-flash-002",
        help="Google model (default: gemini-1.5-flash-latest)",
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
        "--step-size",
        type=int,
        default=7,
        help="number of frames (default: 15)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1000,
        help="image size (default: 512)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default="open_eqa/results",
        help="output directory (default: data/results)",
    )
    parser.add_argument(
        "--force",
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
        "baseline-" + args.dataset.stem + "-{}.json".format(args.model)
    )
    return args


def parse_gemini_output(input: str, output: str) -> str:
    start_idx = output.find("A:")
    if start_idx == -1:
        return output.replace("A:", "").strip()
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return output[start_idx:].replace("A:", "").strip()
    return output[start_idx:end_idx].replace("A:", "").strip()


def ask_question(
    frame_paths: List,
    question: str,
    image_size: int,
    google_model: str,
    google_key: Optional[str] = None,
    force: bool = False,
) -> Optional[str]:
    try:
        set_google_key(key=google_key)

        frames = [cv2.imread(p) for p in frame_paths]
        size = max(frames[0].shape)
        frames = [
            cv2.resize(img, dsize=None, fx=image_size / size, fy=image_size / size)
            for img in frames
        ]
        frames = [Image.fromarray(img) for img in frames]

        prompt = load_prompt("gemini-pro-vision")
        prefix, suffix = prompt.split("User Query:")
        suffix = "User Query:" + suffix.format(question=question)

        messages = []
        messages += [prefix]
        messages += frames
        messages += [suffix]

        output = call_google_api(
            message=messages,
            model=google_model,
        )
        return parse_gemini_output(input, output)
    except Exception as e:
        time.sleep(60)
        return ask_question(
            frame_paths,
            question,
            image_size,
            google_model,
            google_key,
            force,
        )
        if not force:
            traceback.print_exc()
            raise e


def main(args: argparse.Namespace):
    # check for google api key
    # check for GOOGLE_API_KEY api key
    with open("/home/qasim/Projects/graph-robotics/api_keys/gemini_key.txt") as f:
        GOOGLE_API_KEY = f.read().strip()
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        "/home/qasim/Projects/graph-robotics/api_keys/total-byte-432318-q3-78e6d4aa6497.json"
    )

    # load dataset
    dataset = json.load(args.dataset.open("r"))
    print("found {:,} questions".format(len(dataset)))

    # load results
    results = []
    if False:  # args.output_path.exists():
        results = json.load(args.output_path.open())
        print("found {:,} existing results".format(len(results)))
    completed = [item["question_id"] for item in results]

    scenes_available = os.listdir(args.graph_dir)

    subset_questions = [
        # "41ad08e1-cb46-4599-8beb-def3b0f91e34",
        # "35bfcf60-4227-4bbb-9213-e1bf643b2325",
        # "9104773d-a2f0-4be3-a370-f7bf6e242ff7",
        # "6fddec60-f221-4944-88c3-a132dfbfd1ed",
        # "915cb310-31be-4114-846c-242fc59b581d",
        # "bbb83d84-289c-4f1d-a470-772e5c823a90",
        # "6be2fe87-f20c-48a2-a8fb-161362d86e2a",
        # "96d7f5ef-14b2-432a-8b85-21a0621e41a4",
        # "06745779-f1b8-458b-8daa-ccc18d8958c8",
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

    # process data
    for idx, item in enumerate(tqdm.tqdm(dataset)):
        if args.dry_run and idx >= 5:
            break
        if idx >= 200:
            break
        # skip completed questions
        question_id = item["question_id"]
        if question_id in completed:
            continue  # skip existing
        if len(subset_questions) > 0 and not (question_id in subset_questions):
            print("skipping")
            continue
        scene_id = os.path.basename(item["episode_history"])
        if not scene_id in scenes_available:
            continue

        # extract scene paths
        folder = args.frames_directory / scene_id
        frames = sorted(folder.glob("*-rgb.png"))
        indices = [
            i for i in range(0, len(frames), args.step_size)
        ]  # np.round(np.linspace(0, len(frames) - 1, args.num_frames)).astype(int)
        paths = [str(frames[i]) for i in indices]

        # generate answer
        question = item["question"]
        answer = ask_question(
            question=question,
            frame_paths=paths,
            image_size=args.image_size,
            google_model=args.model,
            google_key=GOOGLE_API_KEY,
            force=args.force,
        )

        # store results
        results.append(
            {"question_id": question_id, "gt_answer": item["answer"], "answer": answer}
        )
        json.dump(results, args.output_path.open("w"), indent=2)

    # save at end (redundant)
    json.dump(results, args.output_path.open("w"), indent=2)
    print("saving {:,} answers".format(len(results)))


if __name__ == "__main__":
    main(parse_args())
