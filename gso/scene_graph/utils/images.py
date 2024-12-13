import warnings
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
