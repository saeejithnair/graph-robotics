import json

import numpy as np
import open3d as o3d


def save_oriented_bounding_boxes(bbox_list, file_path):
    """
    Save a list of OrientedBoundingBox objects to a JSON file.
    """
    data = []
    for bbox in bbox_list:
        bbox_data = {
            "center": bbox.center.tolist(),
            "extent": bbox.extent.tolist(),
            "R": bbox.R.tolist(),  # Rotation matrix
        }
        data.append(bbox_data)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def load_oriented_bounding_boxes(file_path):
    """
    Load a list of OrientedBoundingBox objects from a JSON file.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    bbox_list = []
    for bbox_data in data:
        bbox = o3d.geometry.OrientedBoundingBox(
            center=bbox_data["center"],
            extent=bbox_data["extent"],
            R=bbox_data["R"],
        )
        bbox_list.append(bbox)

    return bbox_list


# Example usage
if __name__ == "__main__":
    bbox1 = o3d.geometry.OrientedBoundingBox(
        center=[0, 0, 0], extent=[1, 1, 1], R=np.eye(3)
    )
    bbox2 = o3d.geometry.OrientedBoundingBox(
        center=[1, 1, 1], extent=[2, 2, 2], R=np.eye(3)
    )
    bbox_list = [bbox1, bbox2]

    # Save the list
    save_oriented_bounding_boxes(bbox_list, "bounding_boxes.json")

    # Load the list
    loaded_bboxes = load_oriented_bounding_boxes("bounding_boxes.json")
    print(loaded_bboxes)
