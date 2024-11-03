// include/SceneNode.h

#ifndef SCENENODE_H
#define SCENENODE_H

#include <string>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <mutex>

// Inherit from std::enable_shared_from_this
class SceneNode : public std::enable_shared_from_this<SceneNode> {
public:
    using PointXYZPtr = pcl::PointXYZ*;
    using PointCloudPtr = std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>;

    // Constructor
    SceneNode(const std::string& label);

    // Getters and Setters
    std::string getLabel() const;
    void setLabel(const std::string& label);

    // Bounding Box now uses two Vector3f for min and max
    Eigen::Vector3f getBoundingBoxMin() const;
    Eigen::Vector3f getBoundingBoxMax() const;
    void setBoundingBoxMin(const Eigen::Vector3f& min);
    void setBoundingBoxMax(const Eigen::Vector3f& max);

    std::vector<float> getClipEmbeddings() const;
    void setClipEmbeddings(const std::vector<float>& embeddings);

    std::vector<cv::Mat> getImageCrops() const;
    void addImageCrop(const cv::Mat& crop);

    // Point References
    void addPointReference(PointXYZPtr point);
    void removePointReference(PointXYZPtr point);
    std::vector<PointXYZPtr> getPointReferences() const;

    // Hierarchy Management
    void addChild(std::shared_ptr<SceneNode> child);
    void removeChild(std::shared_ptr<SceneNode> child);
    std::shared_ptr<SceneNode> getParent() const;
    std::vector<std::shared_ptr<SceneNode>> getChildren() const;
    void setParent(std::shared_ptr<SceneNode> parent);

    // Bounding Box Calculation
    void calculateBoundingBox();

private:
    // Attributes
    std::string label_;
    Eigen::Vector3f boundingBoxMin_;
    Eigen::Vector3f boundingBoxMax_;
    std::vector<float> clipEmbeddings_;
    std::vector<cv::Mat> imageCrops_;

    // Point References
    std::vector<PointXYZPtr> pointReferences_;

    // Hierarchy
    std::weak_ptr<SceneNode> parent_;
    std::vector<std::shared_ptr<SceneNode>> children_;

    // Mutex for thread safety
    mutable std::mutex mutex_;
};

#endif // SCENENODE_H
