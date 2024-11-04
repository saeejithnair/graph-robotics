# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(HierarchicalSceneGraph)

# Set C++ standard
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

# Find Python3 with Development components
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
if(NOT Python3_FOUND)
    message(FATAL_ERROR "Python3 not found")
endif()

# Display Python include directories for debugging
message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")

# Add Pybind11 from submodule
add_subdirectory(extern/pybind11)

# Include directories
include_directories(
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    include
)

# Link directories
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Source files for SceneGraphLib (exclude bindings.cpp)
file(GLOB LIB_SOURCES "src/*.cpp")
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp")

# Create library
add_library(SceneGraphLib ${LIB_SOURCES})

# Link libraries
target_link_libraries(SceneGraphLib
    PUBLIC ${PCL_LIBRARIES}
    PUBLIC Eigen3::Eigen
    PUBLIC ${OpenCV_LIBS}
)

# Create Python module using Pybind11 (bindings.cpp exclusively)
pybind11_add_module(scene_graph src/bindings.cpp)

# Link the SceneGraphLib and Python libraries to the Python module
target_link_libraries(scene_graph PRIVATE SceneGraphLib Python3::Python)

# Add executable for testing
add_executable(test_SceneNode tests/test_SceneNode.cpp)
target_link_libraries(test_SceneNode SceneGraphLib)

# Enable testing
enable_testing()
add_test(NAME SceneNodeTest COMMAND test_SceneNode)
