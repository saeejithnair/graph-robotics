## Installation
Create graphpad environemnts
```
conda create -n graphpad python=3.9
conda activate graphpad
pip install -r requirements.txt
```
Next install the HOV-SG repo, don't forget to run
```
pip install -e .
```
Download the SAM checkpoint sam_vit_h_4b8939.pth and put it in the folder "checkpoints".
Download the [OpenEQA](https://github.com/facebookresearch/open-eqa/tree/main) dataset open-eqa-v0.json and put it in the folder eqa/data/open-eqa-v0.json. 
Download the HM3D dataset and ScanNet dataset, following the instructions on the OpenEQA repo. 
Make a account on Google Cloud. Put your gemini key in graph-robotics/api_keys/gemini_key.txt. Put your service file in graph-robotics/api_keys/total-byte-432318-q3-78e6d4aa6497.json. 
# Running
Edit the paths in config files `base_paths.yaml`, `hm3d_mapping.yaml`,  `scannet_mapping.yaml`. 

- `main_embodied_memory.py`: Creates the embodied memory (scene graph and navigation log) of the scenes listed in graph-robotics/graphpad/subset_scene_ids.txt. It is run offline, before EQA.
- `main_openeqa.py`: Runs the EQA loop on the OpenEQA benchmark.
- `open_eqa/evaluate-predictions.py`: Evaluates a set of OpenEQA predictions, and outputs a metrics file in eqa

See the launch.json for how to run each of these files
## Code overview
The eqa folder contains the code needed for EQA tasks, and embodied_memory folder contains code for creating the embodied memory (scene graph and navigation log).