// include/SceneGraphManager.h

#ifndef SCENEGRAPHMANAGER_H
#define SCENEGRAPHMANAGER_H

#include "SceneNode.h"
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>

class SceneGraphManager {
public:
    // Constructor
    SceneGraphManager();

    // Destructor
    ~SceneGraphManager();

    // Add a node to the scene graph
    bool addNode(const std::string& label);

    // Remove a node from the scene graph
    bool removeNode(const std::string& label);

    // Get a node by label
    std::shared_ptr<SceneNode> getNode(const std::string& label) const;

    // Query all nodes
    std::vector<std::shared_ptr<SceneNode>> getAllNodes() const;

private:
    // Map to store nodes by their label
    std::unordered_map<std::string, std::shared_ptr<SceneNode>> nodes_;

    // Mutex for thread safety
    mutable std::mutex mutex_;
};

#endif // SCENEGRAPHMANAGER_H
