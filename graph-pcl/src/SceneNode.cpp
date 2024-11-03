// src/SceneNode.cpp

#include "SceneNode.h"
#include <algorithm>

// Constructor
SceneNode::SceneNode(const std::string& label) : label_(label), boundingBoxMin_(Eigen::Vector3f::Zero()), boundingBoxMax_(Eigen::Vector3f::Zero()) {}

// Getters and Setters
std::string SceneNode::getLabel() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return label_;
}

void SceneNode::setLabel(const std::string& label) {
    std::lock_guard<std::mutex> lock(mutex_);
    label_ = label;
}

Eigen::Vector3f SceneNode::getBoundingBoxMin() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return boundingBoxMin_;
}

Eigen::Vector3f SceneNode::getBoundingBoxMax() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return boundingBoxMax_;
}

void SceneNode::setBoundingBoxMin(const Eigen::Vector3f& min) {
    std::lock_guard<std::mutex> lock(mutex_);
    boundingBoxMin_ = min;
}

void SceneNode::setBoundingBoxMax(const Eigen::Vector3f& max) {
    std::lock_guard<std::mutex> lock(mutex_);
    boundingBoxMax_ = max;
}

std::vector<float> SceneNode::getClipEmbeddings() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return clipEmbeddings_;
}

void SceneNode::setClipEmbeddings(const std::vector<float>& embeddings) {
    std::lock_guard<std::mutex> lock(mutex_);
    clipEmbeddings_ = embeddings;
}

std::vector<cv::Mat> SceneNode::getImageCrops() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return imageCrops_;
}

void SceneNode::addImageCrop(const cv::Mat& crop) {
    std::lock_guard<std::mutex> lock(mutex_);
    imageCrops_.push_back(crop.clone());
}

// Point References
void SceneNode::addPointReference(PointXYZPtr point) {
    std::lock_guard<std::mutex> lock(mutex_);
    pointReferences_.push_back(point);
}

void SceneNode::removePointReference(PointXYZPtr point) {
    std::lock_guard<std::mutex> lock(mutex_);
    pointReferences_.erase(std::remove(pointReferences_.begin(), pointReferences_.end(), point), pointReferences_.end());
}

std::vector<SceneNode::PointXYZPtr> SceneNode::getPointReferences() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pointReferences_;
}

// Hierarchy Management
void SceneNode::addChild(std::shared_ptr<SceneNode> child) {
    std::lock_guard<std::mutex> lock(mutex_);
    children_.push_back(child);
    child->setParent(shared_from_this()); // Now works because of inheritance
}

void SceneNode::removeChild(std::shared_ptr<SceneNode> child) {
    std::lock_guard<std::mutex> lock(mutex_);
    children_.erase(std::remove(children_.begin(), children_.end(), child), children_.end());
    child->setParent(nullptr);
}

std::shared_ptr<SceneNode> SceneNode::getParent() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return parent_.lock();
}

std::vector<std::shared_ptr<SceneNode>> SceneNode::getChildren() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return children_;
}

void SceneNode::setParent(std::shared_ptr<SceneNode> parent) {
    std::lock_guard<std::mutex> lock(mutex_);
    parent_ = parent;
}

// Bounding Box Calculation
void SceneNode::calculateBoundingBox() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pointReferences_.empty()) {
        boundingBoxMin_ = Eigen::Vector3f::Zero();
        boundingBoxMax_ = Eigen::Vector3f::Zero();
        return;
    }

    float minX = pointReferences_[0]->x;
    float minY = pointReferences_[0]->y;
    float minZ = pointReferences_[0]->z;
    float maxX = pointReferences_[0]->x;
    float maxY = pointReferences_[0]->y;
    float maxZ = pointReferences_[0]->z;

    for (const auto& point : pointReferences_) {
        if (point->x < minX) minX = point->x;
        if (point->y < minY) minY = point->y;
        if (point->z < minZ) minZ = point->z;
        if (point->x > maxX) maxX = point->x;
        if (point->y > maxY) maxY = point->y;
        if (point->z > maxZ) maxZ = point->z;
    }

    boundingBoxMin_ << minX, minY, minZ;
    boundingBoxMax_ << maxX, maxY, maxZ;
}
