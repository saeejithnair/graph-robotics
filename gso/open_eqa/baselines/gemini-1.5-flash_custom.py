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

openeqa_184 = [
    "f2e82760-5c3c-41b1-88b6-85921b9e7b32",
    "79344680-6b45-4531-8789-ad0f5ef85b3b",
    "ecad68d2-a16f-4a3e-b8a1-a70ec1c5cf00",
    "e816a1f9-af6c-4901-8d54-4ddaa2a60dc3",
    "5460114d-e885-4eae-8bdc-a273deb3df0a",
    "a86ef102-5500-4fbf-8fae-cdbeb20a3b7b",
    "438e8c6f-f27d-4d3d-b13d-6f41c2981c2a",
    "9b4a7fbb-680d-4e39-8d60-7b1e521f3108",
    "7f15d867-8ba6-47fa-9bca-9d9ae64046b7",
    "025257b6-8b7e-4f6f-aacc-1788069cbfad",
    "23fb241e-989a-4299-a3fb-8d41f7156397",
    "915cb310-31be-4114-846c-242fc59b581d",
    "d7019200-5300-459e-a2c1-b54d5ec0a80b",
    "6be2fe87-f20c-48a2-a8fb-161362d86e2a",
    "f776a834-1e21-4442-8834-18b6f9d6cfad",
    "9a0fe947-4c0b-47b8-a1dc-414f2d555c67",
    "38ce32f5-3c19-46c3-94e6-79efa00a6fbe",
    "0e49111c-608d-4d02-aacb-3705bdd0ca5e",
    "41f53d99-4277-4dec-892e-8e52a2cc7402",
    "10d6d14b-ef30-42b6-89d7-b79eb4ce9b5d",
    "59128ef6-1338-49a8-ab06-191971bb1815",
    "d6142b7c-27e3-4aef-bca7-2cfddca328f4",
    "24228768-d745-4796-990f-2b5d8aeb4827",
    "22c31dab-ea65-4752-b541-edcdb3c67108",
    "8548aacb-669f-4341-a21e-0426e5dc3b42",
    "98a31a80-3f7b-416a-ba1a-fc1858523860",
    "6e4d210c-e7b0-4e71-96e9-d4f58f17b3ea",
    "8c26c6d7-4d26-4069-9829-53f01c6d0cae",
    "352d1df4-83c8-430c-8d6e-f8b477d7e1c1",
    "28694964-f409-42ee-b3a1-22b17c7f3408",
    "b9fa3fcf-34f1-4eb5-a6d1-3fb4465ade39",
    "853d340b-c69d-4371-894c-5e1151844b14",
    "f26d0764-cfaa-4d85-8adf-8be0a3c1864d",
    "5cc891f2-c7fd-478a-bbc0-03a4b7c66472",
    "c5a16c11-e855-4abe-bfe5-33df48982386",
    "69376f0e-ffd7-4d04-aad3-6089bacfc1d3",
    "4a0b1006-0209-4e6a-a0fa-dab6835b6605",
    "8985bd89-1b04-4328-869e-75c416eab90b",
    "0b48b97e-4a15-4181-bff3-8852f09f2f3e",
    "07c4017d-db5a-447a-8086-17d9472e7100",
    "d36c5ac4-65b9-4979-881c-56c7d0870a50",
    "16345ba0-9217-4f07-a79d-bbb965bc69a4",
    "15ef0e88-83c5-41dd-9a1f-cf9feb3dafbb",
    "b4de266c-5361-46b7-a098-167d6ee4d5c1",
    "08e8e5fd-31a3-466b-afd6-fa171f1d9de4",
    "6b8f1b52-25fa-47bc-a3a6-a2a43e834605",
    "9acfbdd3-bc51-4010-ae1e-a28a949731d5",
    "bf4960d4-469f-49bc-8594-9b994049fa77",
    "6a13d4a2-4866-40e7-8f10-d1ec12573dc2",
    "41693f7e-4192-495e-9b4e-b238432c6424",
    "4decde3d-5ab8-43db-893c-c3f3f80bcc76",
    "6bff2ba3-5b68-4d77-a302-1640cc06dd15",
    "36ad6cce-7cd1-429e-b75a-581dc6849603",
    "2449be8f-1320-4061-beb0-2797f5766c73",
    "8471794d-32cd-4989-8cec-91118eb43b67",
    "d5b18be3-2d0c-4653-9706-7c33159de7a9",
    "447e4e2d-7010-4672-b8e0-eb5246430499",
    "872e8692-1e2a-4f7e-8ceb-7a85378be97d",
    "4d127d5e-1a90-468c-93a0-0473c2d1623f",
    "8345f4b2-0850-495b-a957-16cb9cd66f4e",
    "26bd014e-529f-4deb-bcfd-261f35ac7ff2",
    "e2a55cb0-a883-4dd6-9b2f-239d92ebd8bc",
    "8b7d2afd-2a77-4f2b-afe1-b1751d890db4",
    "2f73aba9-c2c7-4f97-b3e4-2435960763b9",
    "a605c40f-96e7-4bec-a1cb-6d48e88e39cd",
    "3a5be057-47d2-4f78-98a9-729ef19b3d8b",
    "4dbd213e-56cd-481a-8ff5-ed9a8d636dbc",
    "ae19adeb-498a-4814-b955-e0af05623f9b",
    "4cc4212e-0db2-421f-8bb5-93817e58f9b4",
    "633ca326-2027-4316-8c20-ef4debde39d2",
    "7131770c-d338-4dfa-b778-0dd5a00a4ef1",
    "e22162c7-9c4d-46f0-8dd5-560a9c4f0dad",
    "e87b90d9-77d5-4f99-b44a-ad1d11480334",
    "13d097e7-12c7-48e0-92c4-9667fc7f9c60",
    "b38045c3-cf3d-43e3-8fee-a085b89a4d3a",
    "5a8b3936-43e0-4474-ac15-efaf488265a1",
    "cb38e809-d5b7-471d-b412-5bc13bd20413",
    "50d2cdeb-86e2-46d7-9c32-ef91e66176f0",
    "0c81b6f8-2d46-4e07-a9d3-a019729d5570",
    "bc1af4f3-5241-4606-8315-ca46d88d7d84",
    "06c9b25c-b117-4b8f-a052-6fd71b2bb043",
    "7ad70423-200c-42a8-8e6a-c471f171782e",
    "5fc88f40-890b-4a30-8b97-d404e8f5c330",
    "65ec009e-d173-4e49-9168-f48fd20308f1",
    "46a04f3a-56a5-4547-9cd9-c683919c0eb1",
    "0df60236-15ad-4166-a31a-a98d14214fdb",
    "d0165de2-29aa-44d1-8689-ff13cd573c79",
    "ba5f1c9b-9a41-4a84-829b-f9b8ccd19b69",
    "f17869a2-2a4d-4ce4-b262-cb69618e3394",
    "e0d20472-8fa6-4e8d-880d-22d4eed3fbb8",
    "fc9d2a18-6197-4c8b-abd8-be0c493e5450",
    "c1b2ccf5-b56d-4ced-9cec-eaf62fedc675",
    "a5c5bb29-700a-4ef5-b17d-aaa47bb0ef3f",
    "b05e7b30-6a4d-4381-9d05-a42ed0c90e30",
    "1dcdd225-eba2-4ba1-97b6-c4cdc7ca4e9b",
    "1eb05aa7-89a0-4e9f-a06d-e05a2e4e8e63",
    "11574d0e-54bb-4900-b230-0f76f1f43266",
    "df5a6203-24a0-40d7-b178-31fb02db71ef",
    "182db45a-eeda-4ccd-841b-20ce864f5c1e",
    "4446bd7d-25fa-4996-9b23-9337b8512f07",
    "11da38f3-c32f-4443-bd33-6a2c1ec22a64",
    "cdb2760c-33d0-4e19-8ddc-494f9874dfb3",
    "98f5190a-b4b0-4bcb-83d0-43dfc39dba85",
    "860923a7-097b-4df5-8a62-59975c3d2a83",
    "7473836e-84ba-4d9c-a86e-4da760d670f4",
    "226ab7fe-8b53-4842-b313-1e7644771cb2",
    "c7c8c496-b3d2-4370-b321-d4274ffda701",
    "44a23f96-b643-4e1b-94ad-48687d0f38b7",
    "04c770eb-c1a7-44c4-b91f-8aa24e2dbed9",
    "0ef0ebd1-db05-4f87-adc7-d01a640c1eed",
    "de97a986-30c3-4e0b-92dc-77ba1900cf8d",
    "acebe630-9d99-4897-bd0f-028038e5baaa",
    "8a914303-067c-44ba-b8a3-2fd72d3f4396",
    "babe466a-4ee8-4afd-a8f9-964793a5d425",
    "41db3bb6-0bb5-4fcb-95b1-f19a32be4184",
    "b70465b8-53a3-436f-b12f-2d8cdf8f1856",
    "206ca121-9185-484d-ab22-acfb082b1359",
    "c64f520c-6450-413c-99da-979be386ff86",
    "33639e66-332d-4824-82ef-e1bf13e94ccb",
    "56c62311-8d4b-470d-a716-49fef718fcff",
    "b0740f05-dbf1-4835-b16e-62d01d371a78",
    "872e9d7f-752d-47bb-aa7d-50a23be3ea69",
    "d6d33031-738b-462d-ac53-2c2df150083e",
    "197342c3-490c-4d6c-9fc9-e4003bc61c17",
    "6b7a2be4-beaa-4023-b031-81f8ecb5c94f",
    "4f65eebc-f602-44ae-8c37-e903f5d940c4",
    "2e1f37e6-0259-4cdb-817e-ba1d015458f6",
    "59df90ad-e54a-48a3-8ac6-7c00e48f0b3d",
    "225c132a-1ec6-47b3-8f5e-887b91168b93",
    "ce8acdaa-800e-4c8a-a3a2-42297c2b9526",
    "62627a1e-e41b-480d-9608-48a154b260bc",
    "62ee487f-ee36-4126-b427-41d7447da702",
    "991967d0-f7ba-4b8e-af60-16c7b9ca00a5",
    "95762878-541e-4f5c-b071-abe79a0393f3",
    "af709fd8-dca2-4697-9548-07aa9d157d8e",
    "297ec2f6-52fe-4dd2-a325-587510d53de7",
    "a4d9802f-5339-4b61-8c9a-42256441b86d",
    "d843b020-4415-4efd-95e7-903f96d4eb26",
    "122417bb-6bcd-4d2f-87dc-96be6ba6c262",
    "5cccc0a8-288e-460a-ad2c-d36fcbaee644",
    "da85d7b4-f3d3-44a4-ac2d-de022e39ff45",
    "bc0caf6a-7684-4730-bc58-3717c1e57b38",
    "911693d9-2d28-4ff2-83a9-c67b83753831",
    "9b2d06e5-ca78-4519-a9ca-75c06209b770",
    "af4b62be-5f12-4180-8a3a-665152a7dfd9",
    "8dded29b-3c01-43bf-846b-b09b9b4ea439",
    "8c57fa88-0550-4808-b081-095c709d68a8",
    "6f9d6ab6-d566-46d8-bd98-ad1c6460c2a8",
    "eb6335ed-c49e-408e-abcd-cce9636ec2b8",
    "d5f844fc-81cd-465e-aa90-e8ff8658c861",
    "3412275e-e797-462e-820d-030317d9e323",
    "d9be5488-237b-41e0-bfac-3ba299d64203",
    "cf7a6ff1-4a97-4b6c-b78e-a70f40cdd80f",
    "2f6546fe-af9d-4986-bbf6-3c189353126a",
    "a8cc7ee8-36ea-4726-bb0e-2642ffc2c2d0",
    "e36087a0-f638-4769-8055-dc357e706c71",
    "e3f6ebae-2b21-4356-856b-52a54fc45b60",
    "6faa9052-c5ae-44b9-a024-ab14474d0c29",
    "de9bd341-0754-4c6c-9558-c973832c3942",
    "ba31f08f-0721-4773-b3da-fdeef9dad06f",
    "2b7089df-2398-43e7-9262-1c2a8069c524",
    "49723897-3ce8-4944-80ac-35f430386b4f",
    "e6f70056-2a9c-429c-9570-f136d2eb4120",
    "da17ae0d-58f0-4099-8bd6-4537e67d93f9",
    "6e2f5803-5dca-4853-85d5-468e8f27ce89",
    "6767409d-f832-4f59-87d1-2dfc3c66d343",
    "66648ca6-3619-4e93-98bb-f4606a842144",
    "961fa9de-6a12-49e7-8e69-2590b96242af",
    "e25996f5-9a95-4b55-a357-a71d65acede3",
    "d16c927b-8d75-4743-8018-97320c76b351",
    "a8c02803-de6a-4dd8-97b1-98301dbda075",
    "7d868374-5434-40f3-a95d-66548d092d6d",
    "15d330b7-11bd-4b29-8263-5235cab34c21",
    "3321cf87-c5fe-46cc-90c2-33d114503de6",
    "1fcfa31c-43d6-4c9a-acb6-21f019956e1c",
    "fa7906e8-12fb-4511-9b6d-9b514a3e63f9",
    "d5c19ea7-5931-4501-a3cf-bed0eb161a9f",
    "0bc41aa3-c14f-4117-92ff-868fda0e5e4b",
    "27fd907f-7c89-4e0a-9c6c-73ba570b0df6",
    "cd8dd632-4431-44b9-9cbb-6eec2317344c",
    "2d2cc029-bad4-4dd3-9dc6-aeceb0207e2a",
    "8de58b75-8369-4185-b39f-82838fc29d87",
    "b41c3183-c6cb-4bc6-a554-13e27532b2ad",
    "f739f880-79fc-4066-9ca1-b04943433974",
]


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
        default="gemini-2.0-flash-exp",
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
        "--num-frames",
        type=int,
        default=50,
        help="number of frames (default: 15)",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=3,
        help="number of frames (default: 15)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=920,
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
        GOOGLE_API_KEY = "AIzaSyCJtymXxwho9G9womXQD12HDZJNBbZjVqU"
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    #     "/home/qasim/Projects/graph-robotics/api_keys/total-byte-432318-q3-78e6d4aa6497.json"
    # )

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
    subset_questions = openeqa_184
    # process data
    for idx, item in enumerate(tqdm.tqdm(dataset)):
        if args.dry_run and idx >= 5:
            break
        # if idx >= 54:
        #     break
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
        indices = np.round(np.linspace(0, len(frames) - 1, args.num_frames)).astype(
            int
        )  # [ i for i in range(0, len(frames), args.step_size) ]
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
