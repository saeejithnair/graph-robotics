#!/bin/bash

# python main_openeqa.py visual_memory_size=4 max_search_depth=20 questions_exp_name=only_graph
python main_openeqa1.py visual_memory_size=5 max_search_depth=20 questions_exp_name=scannet_room_fix

# vsize experiment
# python main_openeqa1.py visual_memory_size=2
# python main_openeqa1.py visual_memory_size=3
# python main_openeqa1.py visual_memory_size=4
# python main_openeqa1.py visual_memory_size=5
# python main_openeqa1.py visual_memory_size=6
# search depth experiment
# python main_openeqa.py visual_memory_size=4 max_search_depth=1
# python main_openeqa.py visual_memory_size=4 max_search_depth=2
# python main_openeqa.py visual_memory_size=4 max_search_depth=3
# python main_openeqa.py visual_memory_size=4 max_search_depth=4
# python main_openeqa.py visual_memory_size=4 max_search_depth=5
# python main_openeqa.py visual_memory_size=4 max_search_depth=6
# python main_openeqa.py visual_memory_size=4 max_search_depth=20

# python main_openeqa.py visual_memory_size=4 questions_exp_name=room-hierarchy
# python main_openeqa.py api_type=frame_level
# python main_openeqa.py api_type=node_level
# python main_openeqa.py visual_memory_size=6
# python main_openeqa.py visual_memory_size=3
# python main_openeqa.py visual_memory_size=5
