# Setup

```bash
sudo apt-get update
sudo apt-get install build-essential cmake git
sudo apt-get install libpcl-dev libeigen3-dev libopencv-dev

conda create -n graph-pcl python=3.11 -y
conda activate graph-pcl
pip install -e .
```

# Build
```bash
./build.sh
```
