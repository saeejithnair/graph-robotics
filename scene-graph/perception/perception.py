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
import PIL
import PIL.Image
import supervision as sv
import tqdm
import vertexai
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from vertexai.generative_models import GenerativeModel, Part

import utils
from object import Object, ObjectList

class Perceptor:
    def __init__(
        self,
        device="cuda:0",
        sam_model_type="vit_h",
        sam_checkpoint_path="checkpoints/sam_vit_h_4b8939.pth",
    ):
        self.device = device
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
        self.sam.to(device=device)
        self.mask_predictor = SamPredictor(self.sam)
        # self.prompt_file = 'perception/prompts/generic_spatial_mapping_claude.txt'
        self.prompt_file = (
            "perception/prompts/generic_spatial_mapping_encouragement.txt"
        )

    def init(self):
        utils.init_gemini()

    def process_response(self, response, img_size, img=None):
        try:
            assert "```json" in response
            response = response.replace("```json", "")
            response = response.replace("```", "")
            response = response.strip()
            response = json.loads(response)
            raw_response = response
            bboxes = [d["bbox"] for d in response]
            width, height = img_size[1], img_size[0]
            for i, box in enumerate(bboxes):
                assert len(box) == 4
                # convert to xyxy format of supervision
                ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
                x1 = int(xmin / 1000 * width)
                y1 = int(ymin / 1000 * height)
                x2 = int(xmax / 1000 * width)
                y2 = int(ymax / 1000 * height)
                bbox = [x1, y1, x2, y2]
                # box = [ y1, x1, y2, x2]
                # box = np.array(box)

                # resize bbox to match image size
                # normalize_size = np.concatenate([img_size, img_size])
                # resized_bbox = ((box/1000)*normalize_size).astype(np.int64)

                response[i]["bbox"] = bbox
            return (response, raw_response)
        except:
            return False

    def ask_question_gemini(
        self,
        img,
        img_size,
        gemini_model: str = "gemini-1.5-flash-latest",
    ) -> Optional[str]:

        with open(self.prompt_file, "r") as f:
            text_prompt = f.read().strip()

        # vertexai.init(project='302560724657', location="northamerica-northeast2")
        # model = GenerativeModel(model_name=gemini_model)
        model = genai.GenerativeModel(model_name=gemini_model)

        # text_prompt = Part.from_text(text_prompt)
        # image_prompt = [Part.from_image(vertexai.generative_models.Image.load_from_file(i)) for i in init_images]
        # prompt = [text_prompt] + image_prompt

        if isinstance(img, (np.ndarray, np.generic)):
            img = Image.fromarray(np.uint8(img))
        image_prompt = [img]
        answer = None
        for _ in range(6):
            response = utils.call_gemini(model, [text_prompt] + image_prompt)
            result = self.process_response(response, img_size, img=img)
            if result:
                response, llm_response = result
                if response:
                    answer = response
                    break
        if not answer:
            raise Exception("Model could not generate bounding boxes")
        return answer, llm_response  # .removeprefix('Answer: ').strip()

    def process(self, img):
        self.img = np.uint8(img)
        object_detections, self.llm_response = self.ask_question_gemini(self.img, self.img.shape)
        self.num_detections = len(object_detections)

        self.mask_predictor.set_image(np.uint8(img))
        object_list = ObjectList()
        for i in range(len(object_detections)):
            box = object_detections[i]["bbox"]
            sam_masks, _, _ = self.mask_predictor.predict(
                box=np.array(box), multimask_output=False
            )
            object_list.add_object(Object(
                mask=sam_masks[0],
                crop=utils.get_crop(self.img, box),
                label=object_detections[i]["label"],
                caption=object_detections[i]["caption"],
                bbox=box,
                confidence=object_detections[i]["confidence"]
            ))

        self.object_list = object_list
        return self.object_list, self.llm_response

    def get_object(self, i: int):
        return dict(
            mask=self.objects["masks"][i],
            crop=self.objects["crops"][i],
            label=self.objects["labels"][i],
            caption=self.objects["captions"][i],
            bbox=self.objects["bboxes"][i],
            confidence=self.objects["confidences"][i],
        )

    def save_results(self, result_dir, frame):
        frame_dir = Path(result_dir) / str(frame)
        os.makedirs(frame_dir, exist_ok=True)
        json_path = frame_dir / "llm_detections.txt"
        annotated_img_path = frame_dir / "annotated_llm_detections.png"
        img_path = frame_dir / "input_image.png"
        with open(json_path, "w") as f:
            json.dump(self.llm_response, f, indent=4)
        
        annotated_img = utils.annotate_img_boxes(
            self.img, 
            self.object_list.get_field('bbox'),
            self.object_list.get_field('label')
            )
        plt.imsave(annotated_img_path, np.uint8(annotated_img))
        plt.imsave(img_path, np.uint8(self.img))

    # def save_results(self, result_dir, frame):

    #     frame_dir = Path(result_dir) / str(frame)
    #     os.makedirs(frame_dir, exist_ok=True)
    #     json_path = frame_dir / "llm_detections.txt"
    #     annotated_img_path = frame_dir / "annotated_llm_detections.png"
    #     img_path = frame_dir / "input_img.png"
    #     mask_path = frame_dir / "masks.npz"

    #     json_data = []

    #     for i in range(self.num_detections):
    #         obj = self.get_object(i)

    #         mask = np.array(obj["mask"])
    #         crop = self.annotate_img(
    #             np.copy(obj["crop"]), self.get_crop(mask, obj["bbox"])
    #         )
    #         plt.imsave(
    #             frame_dir
    #             / ("annotated_bbox_crop-" + str(i) + "-" + obj["label"] + ".png"),
    #             np.uint8(crop),
    #         )

    #         data = {
    #             "label": obj["label"],
    #             "caption": obj["caption"],
    #             "bbox": obj["bbox"],
    #             "confidence": obj["confidence"],
    #         }
    #         json_data.append(data)

    #     np.savez(mask_path, masks=self.objects["masks"])
    #     with open(json_path, "w") as f:
    #         json.dump(json_data, f, indent=4)
    #     annotated_image = self.annotate_img(self.img, self.objects["masks"])
    #     plt.imsave(annotated_img_path, np.uint8(annotated_image))
    #     plt.imsave(img_path, np.uint8(self.img))

    # def load_results(self, result_dir, frame):
    #     frame_dir = Path(result_dir) / str(frame)
    #     json_path = frame_dir / "llm_detections.txt"
    #     masks_path = frame_dir / "masks.npz"
    #     img_path = frame_dir / "input_img.png"

    #     if not frame_dir.exists():
    #         raise FileNotFoundError(f"Frame directory {frame_dir} does not exist.")
    #     if not json_path.exists() or not masks_path.exists():
    #         raise FileNotFoundError(f"Detection files not found in {frame_dir}.")

    #     self.img = PIL.Image.open(img_path).convert("RGB")
    #     with open(json_path, "r") as f:
    #         json_data = json.load(f)
    #     self.num_detections = len(json_data)

    #     crops = []
    #     img_np = np.asarray(self.img)
    #     for i in range(self.num_detections):
    #         box = json_data[i]["bbox"]
    #         crops.append(self.get_crop(img_np, box))

    #     self.objects = dict(
    #         masks=np.load(masks_path)["masks"],
    #         crops=crops,
    #         labels=[o["label"] for o in json_data],
    #         captions=[o["caption"] for o in json_data],
    #         bboxes=[o["bbox"] for o in json_data],
    #         confidences=[o["confidence"] for o in json_data],
    #     )


if __name__ == "__main__":
    # Old Unit test
    # import sys
    # sys.path.append('/home/qasim/Projects/graph-robotics/scene-graph')
    # from images import open_image
    # from utils import init_gemini, utils.call_gemini
    # init_gemini()
    # sample_folder = '/home/qasim/Projects/graph-robotics/scene-graph/perception/sample_images'
    # result_folder = '/home/qasim/Projects/graph-robotics/scene-graph/perception/sample_results'
    # img_resize = (768,432)
    # device='cuda:0'
    # sam_model_type = "vit_h"
    # sam_checkpoint_path = "checkpoints/sam_vit_h_4b8939.pth"
    # perceptor = Perceptor()
    # for file in os.listdir(sample_folder):
    #     img = open_image(os.path.join(sample_folder, file))
    #     perceptor.process(img)
    #     perceptor.save_results(
    #         os.path.join(result_folder, file)[:-4] + '.txt',
    #         os.path.join(result_folder, file))

    # Testing loading function

    perceptor = Perceptor()
    perceptor.load_results(
        "/pub3/qasim/hm3d/data/ham-sg/000-hm3d-BFRyYbPCCPE/detections", 1
    )
    print(perceptor)
