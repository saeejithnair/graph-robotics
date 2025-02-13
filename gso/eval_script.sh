#!/bin/bash
python open_eqa/evaluate-predictions.py --force --results=open_eqa/results/baseline-open-eqa-v0-gemini-2.0-flash-exp.json
# python open_eqa/evaluate-predictions.py --force --results=open_eqa/results/open-eqa-v0-gemini-2.0-flash-exp-frame_level-depth20-vsize3.json

# python open_eqa/evaluate-predictions.py --force --results=open_eqa/results/open-eqa-v0-gemini-2.0-flash-exp-frame_level-depth20-vsize6.json # 45.5
# python open_eqa/evaluate-predictions.py --force --results=open_eqa/results/open-eqa-v0-gemini-2.0-flash-exp-frame_level-depth20-vsize4.json # 48.2
# python open_eqa/evaluate-predictions.py --force --results=open_eqa/results/open-eqa-v0-gemini-2.0-flash-exp-node_level-depth20-vsize4.json # 46.1
