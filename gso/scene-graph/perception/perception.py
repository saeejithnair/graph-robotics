import argparse
import json
import math
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
from scipy.spatial.transform import Rotation
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
        prompt_file="generic_spatial_mapping_localize.txt",
    ):
        self.device = device
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
        self.sam.to(device=device)
        self.mask_predictor = SamPredictor(self.sam)
        # self.prompt_file = 'perception/prompts/generic_spatial_mapping_claude.txt'
        self.prompt_file = os.path.join("perception/prompts", prompt_file)

    def init(self):
        utils.init_gemini()
        self.img = None
        self.past_pose = None
        self.past_response = None

    def reset(self):
        self.img = None
        self.past_pose = None
        self.past_response = None

    def format_objects_response(self, response, img_size, img=None):
        bboxes = [d["bbox"] for d in response]
        width, height = img_size[1], img_size[0]
        for i, box in enumerate(bboxes):
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
        return response

    def verify_response(self, response):
        assert "```json" in response
        response = response.replace("```json", "")
        response = response.replace("```", "")
        response = response.strip()
        response = json.loads(response)
        assert "Current Location" in response
        assert "View" in response
        assert "Novelty" in response
        assert "DetectionList" in response
        for obj in response["DetectionList"]:
            assert "label" in obj
            assert "caption" in obj
            assert "spatial caption" in obj
            assert "bbox" in obj
            assert "confidence" in obj
            assert len(obj["bbox"]) == 4
            assert not ("/" in obj["label"])
        return response

    def ask_question_gemini(
        self,
        img,
        img_size,
        navigation_log=[],
        past_response=None,
        gemini_model: str = "gemini-1.5-flash-latest",
    ) -> Optional[str]:
        with open(self.prompt_file, "r") as f:
            text_prompt = f.read().strip()

        if not (past_response is None):
            text_prompt += (
                "Here was your response on the previous frame:\n"
                + json.dumps(past_response)
            )

        # vertexai.init(project='302560724657', location="northamerica-northeast2")
        # model = GenerativeModel(model_name=gemini_model)
        model = genai.GenerativeModel(model_name=gemini_model)

        # text_prompt = Part.from_text(text_prompt)
        # image_prompt = [Part.from_image(vertexai.generative_models.Image.load_from_file(i)) for i in init_images]
        # prompt = [text_prompt] + image_prompt
        # navigation_prompt = json.dumps(navigation_log)
        # navigation_prompt = [
        #     "Here is a navigation log you created, containing all the information from all the previous frames you have seen"
        #     + navigation_prompt
        # ]
        if isinstance(img, (np.ndarray, np.generic)):
            img = Image.fromarray(np.uint8(img))
        image_prompt = [img.resize((1000, 1000))]
        prompt = [text_prompt] + image_prompt

        llm_response, object_detections = None, None
        for _ in range(6):
            try:
                response = utils.call_gemini(model, prompt)
                llm_response = self.verify_response(response)
                object_detections = self.format_objects_response(
                    llm_response["DetectionList"], img_size, img=img
                )
                break
            except AssertionError as e:
                print("VLM Error:", e)
                traceback.print_exc()
                continue
        if llm_response is None:
            raise Exception("Model could not generate bounding boxes")
        return (
            object_detections,
            llm_response,
        )  # .removeprefix('Answer: ').strip()

    def perceive(self, img, pose, navigation_log=None):
        self.img = np.uint8(img)
        if self.past_pose is None:
            relative_pose = np.zeros(pose.shape)
        else:
            relative_pose = pose - self.past_pose

        response = {"Relative Motion": self.categorize_pose(relative_pose)}
        object_detections, llm_response = self.ask_question_gemini(
            self.img, self.img.shape, navigation_log, self.past_response
        )
        response.update(llm_response)

        self.past_response = llm_response
        self.past_pose = pose

        self.num_detections = len(object_detections)

        self.mask_predictor.set_image(np.uint8(img))
        object_list = ObjectList()
        for i in range(self.num_detections):
            box = object_detections[i]["bbox"]
            sam_masks, iou_preds, _ = self.mask_predictor.predict(
                box=np.array(box), multimask_output=False
            )
            if iou_preds < 0.7:
                continue
            object_list.add_object(
                Object(
                    mask=sam_masks[0],
                    crop=utils.get_crop(self.img, box),
                    label=object_detections[i]["label"],
                    caption=object_detections[i]["caption"],
                    bbox=box,
                    confidence=object_detections[i]["confidence"],
                )
            )

        return object_list, response

    def get_object(self, i: int):
        return dict(
            mask=self.objects["masks"][i],
            crop=self.objects["crops"][i],
            label=self.objects["labels"][i],
            caption=self.objects["captions"][i],
            bbox=self.objects["bboxes"][i],
            confidence=self.objects["confidences"][i],
        )

    def save_results(self, llm_response, object_list, result_dir, frame):
        frame_dir = Path(result_dir) / str(frame)
        os.makedirs(frame_dir, exist_ok=True)
        json_path = frame_dir / "llm_response.txt"
        annotated_img_path = frame_dir / "annotated_llm_detections.png"
        img_path = frame_dir / "input_image.png"
        with open(json_path, "w") as f:
            json.dump(llm_response, f, indent=4)

        img = np.copy(self.img)
        plt.imsave(img_path, np.uint8(img))

        if len(object_list) > 0:
            annotated_img = utils.annotate_img_boxes(
                img,
                object_list.get_field("bbox"),
                object_list.get_field("label"),
            )
            plt.imsave(annotated_img_path, np.uint8(annotated_img))

    def categorize_pose(self, pose_matrix):
        """
        Categorize a 4x4 homogeneous transformation matrix into a descriptive action string.

        Args:
        pose_matrix (numpy.ndarray): 4x4 homogeneous transformation matrix.

        Returns:
        str: Categorized pose description.
        """
        # Validate input
        if not isinstance(pose_matrix, np.ndarray) or pose_matrix.shape != (4, 4):
            raise ValueError("Input must be a 4x4 numpy array")

        # Extract translation components
        translation = pose_matrix[:3, 3]

        # Extract rotation matrix (top-left 3x3 submatrix)
        rotation = pose_matrix[:3, :3]

        # Function to compute Euler angles from a rotation matrix (ZYX convention)
        def rotation_to_euler(R):
            sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            singular = sy < 1e-6

            if not singular:
                x = np.arctan2(R[2, 1], R[2, 2])
                y = np.arctan2(-R[2, 0], sy)
                z = np.arctan2(R[1, 0], R[0, 0])
            else:
                x = np.arctan2(-R[1, 2], R[1, 1])
                y = np.arctan2(-R[2, 0], sy)
                z = 0

            return x, y, z

        # Compute rotation angles
        roll, yaw, pitch = rotation_to_euler(rotation)

        # Categorize rotation
        rotation_actions = []
        if abs(roll) > 0.2 and abs(roll - round(roll / math.pi) * math.pi) > 0.1:
            rotation_actions.append("roll left" if roll > 0 else "roll right")
        if abs(pitch) > 0.2 and abs(pitch - round(pitch / math.pi) * math.pi) > 0.1:
            rotation_actions.append("pitch up" if pitch > 0 else "pitch down")
        if abs(yaw) > 0.2 and abs(yaw - round(yaw / math.pi) * math.pi) > 0.1:
            rotation_actions.append("left turn" if yaw > 0 else "right turn")

        # Compute translation magnitude
        trans_mag = np.linalg.norm(translation)

        # Categorize translation
        translation_actions = []
        if trans_mag > 1e-4:
            # Normalize translation vector
            trans_norm = translation / trans_mag
            directions = {
                (1, 0, 0): "move forward",
                (-1, 0, 0): "move backward",
                (0, 1, 0): "move left",
                (0, -1, 0): "move right",
                (0, 0, 1): "move up",
                (0, 0, -1): "move down",
            }

            # Find closest axis direction
            best_match = max(directions.keys(), key=lambda ax: np.dot(ax, trans_norm))
            translation_actions.append(directions[best_match])

        # Combine rotation and translation actions
        actions = rotation_actions + translation_actions

        # Join actions descriptively
        return ", ".join(actions) if actions else "stationary"

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
