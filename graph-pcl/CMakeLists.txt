# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(HierarchicalSceneGraph)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PCL
find_package(PCL 1.10 REQUIRED)
if(NOT PCL_FOUND)
    message(FATAL_ERROR "PCL not found")
endif()

# Find Eigen
find_package(Eigen3 REQUIRED)
if(NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 not found")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found")
endif()

# Include directories
include_directories(
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    include
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Source files
file(GLOB SOURCES "src/*.cpp")

# Create library
add_library(SceneGraphLib ${SOURCES})

# Link libraries
target_link_libraries(SceneGraphLib
    ${PCL_LIBRARIES}
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

# Add executable for testing
add_executable(test_SceneNode tests/test_SceneNode.cpp)
target_link_libraries(test_SceneNode SceneGraphLib)

# Enable testing
enable_testing()
add_test(NAME SceneNodeTest COMMAND test_SceneNode)
