import json
import os
import pickle
from pathlib import Path

import supervision as sv
import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import numpy as np

from . import utils
from .pointcloud import dynamic_downsample, find_nearest_points, denoise_pcd
from .utils import get_crop

class Features:
    def __init__(
        self, visual_emb=None, caption_emb=None, centroid=None, bbox_3d=None
    ) -> None:
        self.visual_emb = np.array(visual_emb)
        self.caption_emb = np.array(caption_emb)
        self.centroid = np.array(centroid)
        self.bbox_3d = bbox_3d


class FeatureComputer:
    def __init__(self, device="cuda") -> None:
        os.environ["HF_HOME"] = os.path.join(os.getcwd(), "checkpoints")
        self.device = device

    def init(self):
        self.clip_processor = AutoProcessor.from_pretrained(
            "google/siglip-base-patch16-224"
        )
        self.clip_model = AutoModel.from_pretrained(
            "google/siglip-base-patch16-224"
        ).to(self.device)
        self.sentence_tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/bge-small-en-v1.5"
        )
        self.sentence_model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")

    def compute_features(self, img, objects):
        if len(objects) == 0:
            return

        visual_features = self.compute_clip_features_avg(img, objects.get_field("mask"))
        caption_features = self.compute_sentence_features(
            objects.get_field("visual_caption")
        )
        centroids = []
        bboxes_3d = []
        for i in range(len(objects)):
            local_pcd = objects[i].compute_local_pcd()
            centroids.append(np.mean(local_pcd.points, 0))
            bboxes_3d.append(self.get_bounding_box(local_pcd))

        for i in range(len(objects)):
            objects[i].features = Features(
                visual_features[i], caption_features[i], centroids[i], bboxes_3d[i]
            )

    def get_bounding_box(self, pcd):
        try:
            return pcd.get_oriented_bounding_box(robust=True)
        except RuntimeError as e:
            print(f"Met {e}, use axis aligned bounding box instead")
            return pcd.get_axis_aligned_bounding_box()

    def compute_sentence_features(self, captions):
        encoded_input = self.sentence_tokenizer(
            captions, padding=True, truncation=True, return_tensors="pt"
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.sentence_model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1
        )
        return sentence_embeddings

    def compute_clip_features(self, image, image_crops=None, bboxes=None, padding=5):
        if image_crops == None:
            image_crops = []
            for idx in range(len(bboxes)):
                x_min, y_min, x_max, y_max = bboxes[idx]
                image_width, image_height = image.size
                left_padding = min(padding, x_min)
                top_padding = min(padding, y_min)
                right_padding = min(padding, image_width - x_max)
                bottom_padding = min(padding, image_height - y_max)

                x_min -= left_padding
                y_min -= top_padding
                x_max += right_padding
                y_max += bottom_padding

                cropped_image = get_crop(image, (x_min, y_min, x_max, y_max))

                image_crops.append(cropped_image)

        # Convert lists to batches
        preprocessed_images_batch = self.clip_processor(
            images=image_crops, return_tensors="pt"
        ).to(self.device)

        # Batch inference
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(
                **preprocessed_images_batch
            )
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # text_features = clip_model.encode_text(text_tokens_batch)
            # text_features /= text_features.norm(dim=-1, keepdim=True)

        # Convert to numpy
        image_feats = image_features.detach().cpu().numpy()
        # text_feats = text_features.cpu().numpy()
        # image_feats = []

        return image_crops, image_feats

    def compute_clip_features_avg(self, image, masks, bbox_crops=None):
        bbox_masks = sv.mask_to_xyxy(np.array(masks))
        masks_crops = []
        masks_crops_nocontext = []
        for i in range(len(masks)):
            mask_box = bbox_masks[i]
            mask_crop = get_crop(image, mask_box)
            mask = get_crop(masks[i], mask_box)
            # mask = sv.crop_image(masks[i], mask_box)

            mask_crop_nocontext = np.where(mask[:, :, np.newaxis], mask_crop, 0)

            masks_crops.append(mask_crop)
            masks_crops_nocontext.append(mask_crop_nocontext)

        feats = [
            self.compute_clip_features(image, image_crops=masks_crops)[1],
            self.compute_clip_features(image, image_crops=masks_crops_nocontext)[1],
        ]

        if bbox_crops:
            feats.append(self.compute_clip_features(image, image_crops=bbox_crops))

        feat_vector = np.mean(np.stack(feats, 0), 0)
        return feat_vector
    