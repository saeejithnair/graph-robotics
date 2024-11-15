import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import io
import base64
from typing import List, Dict
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize YOLO model (YOLOv8)
yolo_model = YOLO('yolov8n.pt')  # Ensure the 'yolov8n.pt' model is downloaded

# Initialize CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize BLIP model for captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate caption using BLIP
def generate_caption(image: Image.Image) -> str:
    inputs = blip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=50)  # Adjust max_new_tokens as needed
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

# Function to process the image and create scene graph
def process_image(image: Image.Image) -> Dict:
    # Run YOLO detection
    results = yolo_model(image)
    detections = results[0]

    scene_graph = []
    draw = ImageDraw.Draw(image)

    # Load a font for better text rendering
    try:
        font = ImageFont.truetype("arial.ttf", size=15)
    except IOError:
        font = ImageFont.load_default()

    for idx, det in enumerate(detections.boxes):
        class_id = int(det.cls[0])
        class_label = yolo_model.names[class_id]
        bbox = det.xyxy[0].tolist()  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, bbox)

        # Crop the object from the image
        crop = image.crop((x1, y1, x2, y2))

        # Generate CLIP embedding
        clip_inputs = clip_processor(images=crop, return_tensors="pt")
        with torch.no_grad():
            clip_outputs = clip_model.get_image_features(**clip_inputs)
        clip_embedding = clip_outputs[0].cpu().numpy()

        # Generate caption
        caption = generate_caption(crop)

        # Create a scene graph node as a dictionary
        node = {
            "id": idx,
            "class_label": class_label,
            "bbox": bbox,
            "crop_image": crop,  # PIL Image
            "clip_embedding": clip_embedding.tolist(),
            "caption": caption
        }
        scene_graph.append(node)

        # Draw bounding box and label on the image
        draw.rectangle(bbox, outline="red", width=2)
        draw.text((x1, max(y1 - 15, 0)), class_label, fill="red", font=font)

    # Prepare detected objects list
    detected_objects = [{
        "id": node["id"],
        "class_label": node["class_label"],
        "bbox": node["bbox"]
    } for node in scene_graph]

    # Store scene graph in metadata
    metadata = {
        "scene_graph": scene_graph,
        "detected_objects": detected_objects
    }

    return image, detected_objects, metadata

# Function to handle image upload and processing
def upload_image(image: Image.Image):
    if image is None:
        return None, None, None
    image_with_boxes, detected_objects, metadata = process_image(image)
    return image_with_boxes, detected_objects, metadata

# Function to handle click on the image
def on_click(select_data, metadata):
    if metadata is None or 'scene_graph' not in metadata:
        return {"Info": "No metadata available."}

    if select_data is None:
        return {"Info": "No selection data available."}

    scene_graph: List[Dict] = metadata["scene_graph"]

    x, y = select_data.get('x'), select_data.get('y')

    if x is None or y is None:
        return {"Info": "Invalid selection data."}

    # Iterate through the scene graph to find if the click is within any bounding box
    for node in scene_graph:
        x1, y1, x2, y2 = node['bbox']
        if x1 <= x <= x2 and y1 <= y <= y2:
            # Found the clicked object
            # Convert crop image to base64
            buffered = io.BytesIO()
            node['crop_image'].save(buffered, format="PNG")
            crop_bytes = buffered.getvalue()
            crop_base64 = base64.b64encode(crop_bytes).decode()

            # Prepare the information dictionary
            info = {
                "Class Label": node['class_label'],
                "Bounding Box": node['bbox'],
                "CLIP Embedding": node['clip_embedding'],
                "Caption": node['caption'],
                "Cropped Image": f"data:image/png;base64,{crop_base64}"
            }
            return info
    return {"Info": "No object detected at this location."}

# Function to display detailed information with image
def display_info(info):
    if info is None:
        return gr.Markdown("**No information available.**"), gr.Markdown(""), gr.Markdown(""), gr.Image()

    if "Info" in info:
        return gr.Markdown(f"**{info['Info']}**"), gr.Markdown(""), gr.Markdown(""), gr.Image()

    class_label = info.get("Class Label", "N/A")
    bbox = info.get("Bounding Box", "N/A")
    caption = info.get("Caption", "N/A")
    clip_embedding = info.get("CLIP Embedding", "N/A")
    cropped_image = info.get("Cropped Image", None)

    return (
        gr.Markdown(f"**Class Label:** {class_label}"),
        gr.Markdown(f"**Bounding Box:** {bbox}"),
        gr.Markdown(f"**Caption:** {caption}"),
        gr.Image(cropped_image)
    )

# Gradio Blocks Interface
with gr.Blocks() as demo:
    gr.Markdown("## Hierarchical Scene Graph Generator")
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image")
            upload_button = gr.Button("Process Image")
            detected_objects = gr.JSON(label="Detected Objects")
        with gr.Column(scale=2):
            # The Image component to display detected objects with bounding boxes
            image_output = gr.Image(label="Detected Objects Image", interactive=True)
            # Panels to display information
            with gr.Accordion("Object Information", open=False):
                class_label_md = gr.Markdown()
                bbox_md = gr.Markdown()
                caption_md = gr.Markdown()
                cropped_image = gr.Image()

    # Hidden state to store metadata
    metadata_state = gr.State()

    # Define the upload button click event
    upload_button.click(
        fn=upload_image,
        inputs=image_input,
        outputs=[image_output, detected_objects, metadata_state]
    )

    # Define the image select event without using lambda
    image_output.select(
        fn=on_click,
        inputs=[gr.JSON(), metadata_state],
        outputs=gr.JSON()
    ).then(
        fn=display_info,
        inputs=gr.JSON(),
        outputs=[class_label_md, bbox_md, caption_md, cropped_image]
    )

demo.launch()
