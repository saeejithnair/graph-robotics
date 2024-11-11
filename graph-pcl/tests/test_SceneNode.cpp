// tests/test_SceneNode.cpp

#include "SceneNode.h"
#include <cassert>
#include <iostream>

int main() {
    // Create a SceneNode using std::make_shared
    auto node = std::make_shared<SceneNode>("TestNode");
    assert(node->getLabel() == "TestNode");

    // Set and get label
    node->setLabel("UpdatedNode");
    assert(node->getLabel() == "UpdatedNode");

    // Add points using std::shared_ptr
    auto p1 = std::make_shared<pcl::PointXYZ>(1.0, 2.0, 3.0);
    auto p2 = std::make_shared<pcl::PointXYZ>(4.0, 5.0, 6.0);
    node->addPointReference(p1);
    node->addPointReference(p2);

    auto points = node->getPointReferences();
    assert(points.size() == 2);
    assert(points[0]->x == 1.0f);
    assert(points[1]->y == 5.0f);

    // Calculate bounding box
    node->calculateBoundingBox();
    Eigen::Vector3f bboxMin = node->getBoundingBoxMin();
    Eigen::Vector3f bboxMax = node->getBoundingBoxMax();
    assert(bboxMin[0] == 1.0f && bboxMin[1] == 2.0f && bboxMin[2] == 3.0f);
    assert(bboxMax[0] == 4.0f && bboxMax[1] == 5.0f && bboxMax[2] == 6.0f);

    // Test hierarchy
    auto child = std::make_shared<SceneNode>("ChildNode");
    node->addChild(child);
    assert(child->getParent() == node);
    assert(node->getChildren().size() == 1);
    assert(node->getChildren()[0] == child);

    node->removeChild(child);
    assert(child->getParent() == nullptr);
    assert(node->getChildren().empty());

    std::cout << "All tests passed successfully!" << std::endl;
    return 0;
}
