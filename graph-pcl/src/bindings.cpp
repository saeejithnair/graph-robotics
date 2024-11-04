// src/bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include "SceneNode.h"
#include "SceneGraphManager.h"

#include <pcl/point_types.h> // Ensure this is included

namespace py = pybind11;

// Helper function to convert cv::Mat to NumPy array
py::array_t<unsigned char> mat_to_numpy(const cv::Mat& mat) {
    if (mat.empty()) {
        return py::array();
    }
    // Ensure the image is in a format compatible with NumPy
    cv::Mat mat_converted;
    if (mat.channels() == 1) {
        mat_converted = mat.clone();
    } else if (mat.channels() == 3) {
        cv::cvtColor(mat, mat_converted, cv::COLOR_BGR2RGB);
    } else {
        throw std::runtime_error("Unsupported number of channels in cv::Mat");
    }

    // Create a capsule to manage the lifetime of mat_converted
    py::capsule mat_holder(new cv::Mat(mat_converted), [](void *m) { delete static_cast<cv::Mat*>(m); });

    return py::array_t<unsigned char>(
        { static_cast<size_t>(mat_converted.rows), static_cast<size_t>(mat_converted.cols), static_cast<size_t>(mat_converted.channels()) },
        { static_cast<size_t>(mat_converted.step[0]),
          static_cast<size_t>(mat_converted.step[1]),
          static_cast<size_t>(mat_converted.elemSize1()) },
        mat_converted.data,
        mat_holder // Use the capsule instead of py::cast
    );
}

PYBIND11_MODULE(scene_graph, m) {
    m.doc() = "Hierarchical Scene Graph for Robotics";

    // Bind pcl::PointXYZ with __repr__
    py::class_<pcl::PointXYZ, std::shared_ptr<pcl::PointXYZ>>(m, "PointXYZ")
        .def(py::init<>())
        .def(py::init<float, float, float>(), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("x", &pcl::PointXYZ::x)
        .def_readwrite("y", &pcl::PointXYZ::y)
        .def_readwrite("z", &pcl::PointXYZ::z)
        .def("__repr__", [](const pcl::PointXYZ &p) {
            return "<PointXYZ x=" + std::to_string(p.x) + 
                   " y=" + std::to_string(p.y) + 
                   " z=" + std::to_string(p.z) + ">";
        });

    // Bind SceneNode with updated add_image_crop
    py::class_<SceneNode, std::shared_ptr<SceneNode>>(m, "SceneNode")
        .def(py::init<const std::string&>(), py::arg("label"))
        .def_property("label", &SceneNode::getLabel, &SceneNode::setLabel)
        .def_property("bounding_box_min", &SceneNode::getBoundingBoxMin, &SceneNode::setBoundingBoxMin)
        .def_property("bounding_box_max", &SceneNode::getBoundingBoxMax, &SceneNode::setBoundingBoxMax)
        .def_property("clip_embeddings", &SceneNode::getClipEmbeddings, &SceneNode::setClipEmbeddings)
        .def_property_readonly("image_crops", [](const SceneNode& self) {
            std::vector<cv::Mat> crops = self.getImageCrops();
            std::vector<py::array> numpy_crops;
            for (const auto& crop : crops) {
                numpy_crops.push_back(mat_to_numpy(crop));
            }
            return numpy_crops;
        })
        // Modify add_image_crop to accept NumPy array and convert to cv::Mat
        .def("add_image_crop", [](SceneNode &self, py::array_t<unsigned char> array) {
            // Convert NumPy array to cv::Mat
            py::buffer_info buf = array.request();
            if (buf.ndim != 3) {
                throw std::runtime_error("NumPy array must have 3 dimensions (height, width, channels)");
            }
            int height = buf.shape[0];
            int width = buf.shape[1];
            int channels = buf.shape[2];
            
            // Validate data type
            if (buf.format != py::format_descriptor<unsigned char>::format()) {
                throw std::runtime_error("NumPy array must be of type unsigned char (uint8)");
            }

            // Validate channels (assuming 1 or 3)
            if (channels != 1 && channels != 3) {
                throw std::runtime_error("NumPy array must have 1 or 3 channels");
            }

            // Create cv::Mat without copying data
            cv::Mat mat(height, width, channels == 1 ? CV_8UC1 : CV_8UC3, (unsigned char*)buf.ptr);

            // Clone the data to ensure it persists beyond the scope
            cv::Mat mat_copy = mat.clone();
            self.addImageCrop(mat_copy);
        }, py::arg("crop"))
        .def("add_child", &SceneNode::addChild, py::arg("child"))
        .def("remove_child", &SceneNode::removeChild, py::arg("child"))
        .def_property_readonly("children", &SceneNode::getChildren)
        .def_property_readonly("parent", &SceneNode::getParent)
        .def("add_point_reference", &SceneNode::addPointReference, py::arg("point"))
        .def("remove_point_reference", &SceneNode::removePointReference, py::arg("point"))
        .def("add_point", &SceneNode::addPoint, py::arg("x"), py::arg("y"), py::arg("z"))
        .def_property_readonly("point_references", &SceneNode::getPointReferences)
        .def("calculate_bounding_box", &SceneNode::calculateBoundingBox)
        .def("__repr__", [](const SceneNode &node) {
            return "<SceneNode label='" + node.getLabel() + "'>";
        });

    // Bind SceneGraphManager
    py::class_<SceneGraphManager, std::shared_ptr<SceneGraphManager>>(m, "SceneGraphManager")
        .def(py::init<>())
        .def("add_node", &SceneGraphManager::addNode, py::arg("label"))
        .def("remove_node", &SceneGraphManager::removeNode, py::arg("label"))
        .def("get_node", &SceneGraphManager::getNode, py::arg("label"))
        .def("get_all_nodes", &SceneGraphManager::getAllNodes)
        .def("__repr__", [](const SceneGraphManager &manager) {
            return "<SceneGraphManager>";
        });
}
