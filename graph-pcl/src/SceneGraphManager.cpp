// src/SceneGraphManager.cpp

#include "SceneGraphManager.h"
#include <iostream>

// Constructor
SceneGraphManager::SceneGraphManager() {}

// Destructor
SceneGraphManager::~SceneGraphManager() {}

// Add a node to the scene graph
bool SceneGraphManager::addNode(const std::string& label) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (nodes_.find(label) != nodes_.end()) {
        std::cerr << "Node with label '" << label << "' already exists." << std::endl;
        return false;
    }
    auto node = std::make_shared<SceneNode>(label);
    nodes_[label] = node;
    return true;
}

// Remove a node from the scene graph
bool SceneGraphManager::removeNode(const std::string& label) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = nodes_.find(label);
    if (it == nodes_.end()) {
        std::cerr << "Node with label '" << label << "' does not exist." << std::endl;
        return false;
    }
    // Remove the node from its parent's children if it has a parent
    auto parent = it->second->getParent();
    if (parent) {
        parent->removeChild(it->second);
    }
    // Remove the node from all children
    auto children = it->second->getChildren();
    for (auto& child : children) {
        child->setParent(nullptr);
    }
    nodes_.erase(it);
    return true;
}

// Get a node by label
std::shared_ptr<SceneNode> SceneGraphManager::getNode(const std::string& label) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = nodes_.find(label);
    if (it != nodes_.end()) {
        return it->second;
    }
    return nullptr;
}

// Query all nodes
std::vector<std::shared_ptr<SceneNode>> SceneGraphManager::getAllNodes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<SceneNode>> allNodes;
    for (const auto& pair : nodes_) {
        allNodes.push_back(pair.second);
    }
    return allNodes;
}
