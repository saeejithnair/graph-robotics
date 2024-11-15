import numpy as np
import torch
from PIL import Image


class Similarity:
    
    def __init__(self) -> None:
        pass

    def compute_clip_features_batched(
        image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device
    ):
        image = Image.fromarray(image)
        padding = 20  # Adjust the padding amount as needed

        image_crops = []
        preprocessed_images = []
        text_tokens = []

        # Prepare data for batch processing
        for idx in range(len(detections.xyxy)):
            x_min, y_min, x_max, y_max = detections.xyxy[idx]
            image_width, image_height = image.size
            left_padding = min(padding, x_min)
            top_padding = min(padding, y_min)
            right_padding = min(padding, image_width - x_max)
            bottom_padding = min(padding, image_height - y_max)

            x_min -= left_padding
            y_min -= top_padding
            x_max += right_padding
            y_max += bottom_padding

            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0)
            preprocessed_images.append(preprocessed_image)

            class_id = detections.class_id[idx]
            text_tokens.append(classes[class_id])
            image_crops.append(cropped_image)

        # Convert lists to batches
        preprocessed_images_batch = torch.cat(preprocessed_images, dim=0).to(device)
        text_tokens_batch = clip_tokenizer(text_tokens).to(device)

        # Batch inference
        with torch.no_grad():
            image_features = clip_model.encode_image(preprocessed_images_batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # text_features = clip_model.encode_text(text_tokens_batch)
            # text_features /= text_features.norm(dim=-1, keepdim=True)

        # Convert to numpy
        image_feats = image_features.cpu().numpy()
        # text_feats = text_features.cpu().numpy()
        # image_feats = []
        text_feats = []

        return image_crops, image_feats, text_feats
