import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Optional
import numpy as np
import tqdm
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import PIL
import google.generativeai as genai
import time
import matplotlib.pyplot as plt

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
        default="/pub3/qasim/hm3d/data/concept-graphs",
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
        help="use vlm annotated frames or not"
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

def parse_output(output: str) -> str:
    start_idx = output.find("A:")
    if start_idx == -1:
        raise ValueError("Invalid output string: {}".format(output))
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return output[start_idx:].replace("A:", "").strip()
    return output[start_idx:end_idx].replace("A:", "").strip()

def process_scenegraph(graph_path, image_paths):
    f = open(graph_path / 'exps/r_mapping/obj_json_r_mapping.json')
    graph = json.load(f)
    f.close()
    graph_keyframes = {}
    for obj in graph:
        keyframes = graph[obj]['image_idxs']
        keyframes = list(set(keyframes))
        keyframes = [image_paths[i] for i in keyframes]
        graph_keyframes[obj] = keyframes
        graph[obj] = {
            'id' : obj, # graph[obj]['id'],
            'caption' : graph[obj]['object_tag'],
            'bbox_center' : graph[obj]['bbox_center'],
            'bbox_volume' : graph[obj]['bbox_volume']
        }
    return graph, graph_keyframes

def sample_images(image_paths, n=3):
    flattened_paths = [x for xs in image_paths.values() for x in xs]
    freqs = {i:flattened_paths.count(i) for i in set(flattened_paths)}
    n = min(n, len(freqs.keys()))
    most_freq_image = max(freqs, key=freqs.get)
    freqs.pop(most_freq_image)
    
    probs = np.array(list(freqs.values())) / np.sum(list(freqs.values()))
    images = np.random.choice(list(freqs.keys()), size=n-1, replace=False, p = probs)
    return [most_freq_image] + list(images)

def subset_dict(full_dict, search_keys):
    subset_dict = {}
    for key in search_keys:
        if not key in full_dict.keys():
            print('Key', key, 'not found in dict')
            continue
        subset_dict[key] = full_dict[key]
    return subset_dict
def open_image(path, new_size=(800,480)):
    return PIL.Image.open(path).resize(new_size)
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
        response = 'No response' 
    return response

def ask_question(
    question: str,
    image_paths,
    graph_path: str,
    gemini_model: str = "gemini-1.5-flash-latest",
    gemini_seed: int = 1234,
    gemini_max_tokens: int = 128,
    gemini_temperature: float = 0.2,
    force: bool = False 
) -> Optional[str]:
    
    graph, keyframes = process_scenegraph(graph_path, image_paths)
    
    with open('sg-reasoner/prompts/question_prompt2.txt', 'r') as f:
        text_prompt_template = f.read().strip()
    with open('sg-reasoner/prompts/question_prompt2_final.txt', 'r') as f:
        text_prompt_final = f.read().strip().format(
            question=question, 
            graph=json.dumps(graph))
    
    
    init_images = sample_images(keyframes, n=5)
    
    # vertexai.init(project='302560724657', location="northamerica-northeast2")
    # model = GenerativeModel(model_name=gemini_model)
    model = genai.GenerativeModel(model_name=gemini_model)
    
    # text_prompt = Part.from_text(text_prompt)
    # image_prompt = [Part.from_image(vertexai.generative_models.Image.load_from_file(i)) for i in init_images]
    # prompt = [text_prompt] + image_prompt
    
    image_prompt = [open_image(i) for i in init_images]
    text_prompt = text_prompt_template.format(
            question=question, 
            graph=json.dumps(graph))
    
    answer = None
    for i in range(3):
        response = call_gemini(model, [text_prompt] + image_prompt) 
        if response.startswith('Get Image:'):
            objects = response.removeprefix('Get Image: ').strip().split(', ')
            new_keyframes = sample_images(subset_dict(keyframes, objects), n=4)
            image_prompt = image_prompt + [open_image(i) for i in new_keyframes]
        else:
            answer = response
            break
    if answer == None:
        answer = call_gemini(model, [text_prompt_final] + image_prompt)
    return answer.removeprefix('Answer: ').strip()

    try:
        raise Exception('Pause')
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
        

def main(args: argparse.Namespace):
    # check for GOOGLE_API_KEY api key
    with open('/home/qasim/Projects/open-eqa/sg-reasoner/keys/gemini_key.txt') as f:
        GOOGLE_API_KEY = f.read().strip()
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/qasim/Projects/open-eqa/sg-reasoner/keys/total-byte-432318-q3-78e6d4aa6497.json'
    
    # load questions
    questions = json.load(args.questions.open("r"))
    print("found {:,} questions".format(len(questions)))
    
    # only run particular questions
    subset_questions = [
        '73bcd8d2-cd39-46e1-a63a-54acfae8d060',
        'f2e82760-5c3c-41b1-88b6-85921b9e7b32', '500e6924-0ea3-45c8-89ce-db3d37e142bf',
        'c8808989-9b62-4764-814e-520fd40c22cd',
        '0c6b5dce-8baf-4d95-b7ca-26fe915c4bd7',
    ] 
    if subset_questions and len(subset_questions) > 0:
        questions =  [q for q in questions if q['question_id'] in subset_questions]

    # load results
    results = []
    if False: # args.output_path.exists():
        results = json.load(args.output_path.open())
        print("found {:,} existing results".format(len(results)))
    completed = [item["question_id"] for item in results]

    scenes_available = os.listdir(args.graph_dir)

    print('Saving results to ', args.output_path)
    
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
            folder = args.graph_dir / scene_id / 'exps/s_detections/vis'
            frames = sorted(folder.glob("*-rgbannotated_for_vlm.jpg"))
        else:
            folder = args.frames_directory / scene_id
            frames = sorted(folder.glob("*-rgb.png"))
        indices = np.round(np.linspace(0, len(frames) - 1, num=len(frames))).astype(int)
        paths = {i:str(frames[i]) for i in indices} 

        # generate answer
        question = item["question"]
        answer = ask_question(
            question=question,
            image_paths=paths,
            graph_path = args.graph_dir / scene_id,
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