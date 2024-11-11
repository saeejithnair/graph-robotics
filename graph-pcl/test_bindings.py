# test_bindings.py

import sys
import os
import scene_graph
import numpy as np
import cv2

def main():
    # Create a SceneGraphManager instance
    manager = scene_graph.SceneGraphManager()

    # Add nodes
    print("Adding nodes...")
    manager.add_node("Parent")
    manager.add_node("Child1")
    manager.add_node("Child2")

    # Get parent and child nodes
    parent = manager.get_node("Parent")
    child1 = manager.get_node("Child1")
    child2 = manager.get_node("Child2")

    # Establish hierarchy
    print("Establishing hierarchy...")
    parent.add_child(child1)
    parent.add_child(child2)

    # Verify hierarchy
    print(f"Parent's children: {[child.label for child in parent.children]}")
    print(f"Child1's parent: {child1.parent.label if child1.parent else 'None'}")
    print(f"Child2's parent: {child2.parent.label if child2.parent else 'None'}")

    # Add points to Child1 using PointXYZ
    print("Adding points to Child1 using PointXYZ...")
    point1 = scene_graph.PointXYZ(1.0, 2.0, 3.0)
    point2 = scene_graph.PointXYZ(4.0, 5.0, 6.0)
    child1.add_point_reference(point1)
    child1.add_point_reference(point2)
    child1.calculate_bounding_box()

    # Print PointXYZ objects to verify __repr__
    print(f"Point1: {point1}")
    print(f"Point2: {point2}")

    # Get bounding box
    bbox_min = child1.bounding_box_min
    bbox_max = child1.bounding_box_max
    print(f"Child1 Bounding Box Min: {bbox_min}")
    print(f"Child1 Bounding Box Max: {bbox_max}")

    # Add image crops to Child1 (dummy images)
    print("Adding image crops to Child1...")
    dummy_image1 = np.zeros((100, 100, 3), dtype=np.uint8)  # RGB image
    dummy_image2 = np.ones((50, 50, 1), dtype=np.uint8) * 255  # Grayscale image
    child1.add_image_crop(dummy_image1)
    child1.add_image_crop(dummy_image2)

    # Retrieve image crops
    image_crops = child1.image_crops
    print(f"Child1 has {len(image_crops)} image crops.")

    # Optionally, display one image crop using OpenCV
    # Uncomment the following lines if you want to visualize the image crops
    # for i, crop in enumerate(image_crops):
    #     cv2.imshow(f'Image Crop {i+1}', crop)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Remove a child
    print("Removing Child1 from Parent...")
    parent.remove_child(child1)

    # Verify removal
    print(f"Parent's children after removal: {[child.label for child in parent.children]}")
    print(f"Child1's parent after removal: {child1.parent.label if child1.parent else 'None'}")

    # List all nodes
    all_nodes = manager.get_all_nodes()
    print("All nodes in the scene graph:")
    for node in all_nodes:
        parent_label = node.parent.label if node.parent else "None"
        print(f"- {node.label}, Parent: {parent_label}")

if __name__ == "__main__":
    # Ensure the Python interpreter can find the scene_graph module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(current_dir, "build")
    sys.path.insert(0, build_dir)
    main()
