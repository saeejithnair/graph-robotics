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
import google.generativeai as genai
import time
import matplotlib.pyplot as plt
import PIL 
import supervision as sv 
from segment_anything import sam_model_registry, SamPredictor
from utils import init_gemini, call_gemini
from PIL import Image

class Perceptor:
    def __init__(self,
                img_resize = (768,432),
                device='cuda:0',
                sam_model_type = "vit_h",
                sam_checkpoint_path = "checkpoints/sam_vit_h_4b8939.pth"):
        self.img_resize = img_resize
        self.device = device
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
        self.sam.to(device=device)
        self.mask_predictor = SamPredictor(self.sam)
        
    def init(self):
        init_gemini()
        
    def process_response(self, 
                         response,
                         img_size):
        try:
            assert '```json' in response
            response = response.replace('```json', '')
            response = response.replace('```', '')
            response = response.strip()
            response = json.loads(response)
            bboxes = [d['bbox'] for d in response]
            for i,box in enumerate(bboxes):
                assert len(box) == 4
                # convert to xyxy format of supervision
                box = [box[1], box[0], box[3], box[2]]
                box = np.array(box)
                
                # resize bbox to match image size
                normalize_size = np.concatenate([img_size, img_size])
                resized_bbox = ((box/1000)*normalize_size).astype(np.int64)
                
                response[i]['bbox'] = resized_bbox.tolist()
            return response
        except:
            return False

    def ask_question_gemini(
        self,
        img,
        img_size,
        gemini_model: str = "gemini-1.5-flash-latest",
        ) -> Optional[str]:

        with open('perception/prompts/generic_spatial_mapping_claude.txt', 'r') as f:
            text_prompt = f.read().strip()
        
        # vertexai.init(project='302560724657', location="northamerica-northeast2")
        # model = GenerativeModel(model_name=gemini_model)
        model = genai.GenerativeModel(model_name=gemini_model)
        
        # text_prompt = Part.from_text(text_prompt)
        # image_prompt = [Part.from_image(vertexai.generative_models.Image.load_from_file(i)) for i in init_images]
        # prompt = [text_prompt] + image_prompt
        
        img = Image.fromarray(np.uint8(img))
        image_prompt = [img]
        answer = None
        for _ in range(6):
            response = call_gemini(model, [text_prompt] + image_prompt) 
            response = self.process_response(response, img_size)
            if response : 
                answer = response       
                break
        if not answer:
            raise Exception('Model could not generate bounding boxes') 
        return answer # .removeprefix('Answer: ').strip()

    def process(self, img):
        self.img = img
        self.object_detections = self.ask_question_gemini(img, img.shape[:2])
        self.num_detections = len(self.object_detections)

        self.mask_predictor.set_image(np.uint8(img))
        masks = []
        crops = []
        labels = []
        captions = []
        bboxes = []
        confidences = []
        for i in range(len(self.object_detections)):
            box = self.object_detections[i]['bbox']
            sam_masks, _, _ = self.mask_predictor.predict(
                box=np.array(box),
                multimask_output=False
            )
            masks.append(sam_masks[0])
            crops.append(img[box[0]:box[2], box[1]:box[3]])
            labels.append(self.object_detections[i]['label'])
            captions.append(self.object_detections[i]['caption'])
            bboxes.append(box)
            confidences.append(self.object_detections[i]['confidence'])
        
        self.objects = dict(
            masks= masks,
            crops= crops,
            labels= labels,
            captions= captions,
            bboxes = bboxes,
            confidences = confidences
        )
        
    def get_object(self, i: int):
        return dict(
            mask= self.objects['masks'][i],
            crop= self.objects['crops'][i],
            label= self.objects['labels'][i],
            caption= self.objects['captions'][i],
            bbox = self.objects['bboxes'][i],
            confidence = self.objects['confidences'][i]
        )
    
    def annotate_img(self):
        labels = self.objects['labels']
        detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=np.array(self.objects['masks'])),
                mask=np.array(self.objects['masks']),
            )
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        # label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
        annotated_image = self.img.copy()
        annotated_image = box_annotator.annotate(annotated_image, 
                                            detections=detections,
                                            labels=labels
                                            )

        annotated_image = mask_annotator.annotate(annotated_image, 
                                            detections=detections,
                                            )
        return annotated_image
    
    def save_results(self, result_dir):
        
        json_path = os.path.join(result_dir, "llm_detections.txt")
        annotated_img_path = os.path.join(result_dir, "llm_detections_annotated.png")
        
        json_data = []
        for i in range(self.num_detections):
            obj = self.get_object(i)
            
            frame_dir = os.path.join(result_dir, str(i))
            
            data = {
                'label': obj['label'],
                'caption': obj['caption'],
                'bbox': obj['bbox'],
                'confidence': obj['confidence'],
            }
            json_data.append(data)
            
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        annotated_image = self.annotate_img()
        plt.imsave(annotated_img_path, np.uint8(annotated_image))
        

if __name__ == "__main__":
    import sys
    sys.path.append('/home/qasim/Projects/graph-robotics/scene-graph')
    from images import open_image
    from utils import init_gemini, call_gemini
    init_gemini()
    sample_folder = '/home/qasim/Projects/graph-robotics/scene-graph/perception/sample_images'
    result_folder = '/home/qasim/Projects/graph-robotics/scene-graph/perception/sample_results'
    img_resize = (768,432) 
    device='cuda:0'
    sam_model_type = "vit_h"
    sam_checkpoint_path = "checkpoints/sam_vit_h_4b8939.pth"
    perceptor = Perceptor()
    for file in os.listdir(sample_folder):
        img = open_image(os.path.join(sample_folder, file))
        perceptor.process(img)
        perceptor.save_results(
            os.path.join(result_folder, file)[:-4] + '.txt', 
            os.path.join(result_folder, file))
    