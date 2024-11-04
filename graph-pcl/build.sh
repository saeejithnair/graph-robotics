#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Initialize Conda in the script
source /home/smnair/miniconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate graph-pcl

# Navigate to the project directory
cd "$(dirname "$0")"

# Remove and recreate the build directory
rm -rf build
mkdir build
cd build

# Configure CMake with explicit Python paths
cmake .. \
  -DPython3_EXECUTABLE=$(which python) \
  -DPython3_INCLUDE_DIR=$(python -c "from sysconfig import get_paths as gp; print(gp()['include'])") \
  -DPython3_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.11.so \
  -DCMAKE_PREFIX_PATH=$(conda info --base)/envs/graph-pcl

# # Build the project
make
