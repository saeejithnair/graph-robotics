import json
import warnings
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import PIL
import supervision as sv
import torch

# Add at the top of the file
_label_to_class_id = {}


def _update_label_mapping(labels):
    """Updates global label mapping with new labels"""
    global _label_to_class_id
    for label in labels:
        if label not in _label_to_class_id:
            _label_to_class_id[label] = len(_label_to_class_id)
    return _label_to_class_id


def open_image(path, new_size=None):
    if new_size:
        return PIL.Image.open(path).resize(new_size)
    else:
        return PIL.Image.open(path)


def normalize_image(rgb: Union[torch.Tensor, np.ndarray]):
    r"""Normalizes RGB image values from :math:`[0, 255]` range to :math:`[0, 1]` range.

    Args:
        rgb (torch.Tensor or numpy.ndarray): RGB image in range :math:`[0, 255]`

    Returns:
        torch.Tensor or numpy.ndarray: Normalized RGB image in range :math:`[0, 1]`

    Shape:
        - rgb: :math:`(*)` (any shape)
        - Output: Same shape as input :math:`(*)`
    """
    if torch.is_tensor(rgb):
        return rgb.float() / 255
    elif isinstance(rgb, np.ndarray):
        return rgb.astype(float) / 255
    else:
        raise TypeError("Unsupported input rgb type: %r" % type(rgb))


def channels_first(rgb: Union[torch.Tensor, np.ndarray]):
    r"""Converts from channels last representation :math:`(*, H, W, C)` to channels first representation
    :math:`(*, C, H, W)`

    Args:
        rgb (torch.Tensor or numpy.ndarray): :math:`(*, H, W, C)` ordering `(*, height, width, channels)`

    Returns:
        torch.Tensor or numpy.ndarray: :math:`(*, C, H, W)` ordering

    Shape:
        - rgb: :math:`(*, H, W, C)`
        - Output: :math:`(*, C, H, W)`
    """
    if not (isinstance(rgb, np.ndarray) or torch.is_tensor(rgb)):
        raise TypeError("Unsupported input rgb type {}".format(type(rgb)))

    if rgb.ndim < 3:
        raise ValueError(
            "Input rgb must contain atleast 3 dims, but had {} dims.".format(rgb.ndim)
        )
    if rgb.shape[-3] < rgb.shape[-1]:
        msg = "Are you sure that the input is correct? Number of channels exceeds height of image: %r > %r"
        warnings.warn(msg % (rgb.shape[-1], rgb.shape[-3]))
    ordering = list(range(rgb.ndim))
    ordering[-2], ordering[-1], ordering[-3] = ordering[-3], ordering[-2], ordering[-1]

    if isinstance(rgb, np.ndarray):
        return np.ascontiguousarray(rgb.transpose(*ordering))
    elif torch.is_tensor(rgb):
        return rgb.permute(*ordering).contiguous()


def annotate_img_masks(img, masks, labels):
    if not type(masks) is list:
        masks = [masks]
    if isinstance(labels, str):
        labels = [labels] * len(masks)
    if len(masks) == 0:
        return img.copy()

    # Use global mapping
    class_ids = np.array([_label_to_class_id[label] for label in labels])

    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=np.array(masks)),
        mask=np.array(masks),
        class_id=class_ids,
    )
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    # label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
    annotated_image = img.copy()
    annotated_image = box_annotator.annotate(annotated_image, detections=detections)
    annotated_image = mask_annotator.annotate(
        annotated_image,
        detections=detections,
    )
    return annotated_image


def annotate_img_boxes(img, box, labels):
    # Update global mapping
    _update_label_mapping(labels)
    class_ids = np.array([_label_to_class_id[label] for label in labels])
    # create detections with bounding boxes and class ids
    detections = sv.Detections(xyxy=np.array(box), class_id=class_ids)
    box_annotator = sv.BoxAnnotator()
    annotated_image = img.copy()
    annotated_image = box_annotator.annotate(
        annotated_image,
        detections=detections,
    )
    return annotated_image


def get_crop(img, xyxy):
    if type(img) is np.ndarray:
        return sv.crop_image(
            image=img,
            xyxy=xyxy,
        )
    else:
        return img.crop(xyxy)


class ObjectClasses:
    """
    Manages object classes and their associated colors, allowing for exclusion of background classes.

    This class facilitates the creation or loading of a color map from a specified file containing
    class names. It also manages background classes based on configuration, allowing for their
    inclusion or exclusion. Background classes are ["wall", "floor", "ceiling"] by default.

    Attributes:
        classes_file_path (str): Path to the file containing class names, one per line.

    Usage:
        obj_classes = ObjectClasses(classes_file_path, skip_bg=True)
        model.set_classes(obj_classes.get_classes_arr())
        some_class_color = obj_classes.get_class_color(index or class_name)
    """

    def __init__(self, classes_file_path, bg_classes, skip_bg):
        self.classes_file_path = Path(classes_file_path)
        self.bg_classes = bg_classes
        self.skip_bg = skip_bg
        self.classes, self.class_to_color = self._load_or_create_colors()

    def _load_or_create_colors(self):
        with open(self.classes_file_path, "r") as f:
            all_classes = [cls.strip() for cls in f.readlines()]

        # Filter classes based on the skip_bg parameter
        if self.skip_bg:
            classes = [cls for cls in all_classes if cls not in self.bg_classes]
        else:
            classes = all_classes

        colors_file_path = (
            self.classes_file_path.parent / f"{self.classes_file_path.stem}_colors.json"
        )
        if colors_file_path.exists():
            with open(colors_file_path, "r") as f:
                class_to_color = json.load(f)
            # Ensure color map only includes relevant classes
            class_to_color = {
                cls: class_to_color[cls] for cls in classes if cls in class_to_color
            }
        else:
            class_to_color = {
                class_name: list(np.random.rand(3).tolist()) for class_name in classes
            }
            with open(colors_file_path, "w") as f:
                json.dump(class_to_color, f)

        return classes, class_to_color

    def get_classes_arr(self):
        """
        Returns the list of class names, excluding background classes if configured to do so.
        """
        return self.classes

    def get_bg_classes_arr(self):
        """
        Returns the list of background class names, if configured to do so.
        """
        return self.bg_classes

    def get_class_color(self, key):
        """
        Retrieves the color associated with a given class name or index.

        Args:
            key (int or str): The index or name of the class.

        Returns:
            list: The color (RGB values) associated with the class.
        """
        if isinstance(key, int):
            if key < 0 or key >= len(self.classes):
                raise IndexError("Class index out of range.")
            class_name = self.classes[key]
        elif isinstance(key, str):
            class_name = key
            if class_name not in self.classes:
                raise ValueError(f"{class_name} is not a valid class name.")
        else:
            raise ValueError("Key must be an integer index or a string class name.")
        return self.class_to_color.get(
            class_name, [0, 0, 0]
        )  # Default color for undefined classes

    def get_class_color_dict_by_index(self):
        """
        Returns a dictionary of class colors, just like self.class_to_color, but indexed by class index.
        """
        return {str(i): self.get_class_color(i) for i in range(len(self.classes))}
