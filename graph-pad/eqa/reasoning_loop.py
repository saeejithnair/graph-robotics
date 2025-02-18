import json
import os
import shutil
from typing import Optional

import vertexai
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerativeModel,
    Image,
    Part,
    Tool,
)

from embodied_memory.detection import DetectionList
from embodied_memory.edges import EDGE_TYPES
from embodied_memory.embodied_memory import EmbodiedMemory
from embodied_memory.relationship_scorer import HierarchyExtractor

from .api import API
from .prompt_formatting import extract_qa_prompts, format_qa_prompt


def answer_question(
    question: str,
    api: API,
    dataset,
    system_prompt,
    cfg,
    result_dir_embodied_memory: str,
    result_dir_detections: str,
    temp_workspace_dir: str,
) -> Optional[str]:
    # Extract the relevant parameters from the cfg
    gemini_model = cfg.questions_model
    obj_pcd_max_points = cfg.obj_pcd_max_points
    downsample_voxel_size = cfg.downsample_voxel_size
    device = cfg.device
    load_floors = True
    load_rooms = True
    visual_memory_size = cfg.visual_memory_size
    max_search_depth = cfg.max_search_depth
    api_declarations = api.get_declaration()

    # Create the embodied memory
    embodied_memory = EmbodiedMemory(visual_memory_size, cfg.room_types, device=device)
    embodied_memory.load(
        result_dir_embodied_memory, load_floors=load_floors, load_rooms=load_rooms
    )

    # re-compute node levels because of a prior bug. This is not needed with the new scripts, and can be removed.
    hierarchy_extractor = HierarchyExtractor(downsample_voxel_size=0.02)
    hierarchy_matrix, hierarchy_type_matrix = hierarchy_extractor.infer_hierarchy(
        embodied_memory.scene_graph
    )
    embodied_memory.compute_node_levels(hierarchy_matrix, hierarchy_type_matrix)

    # start the vertex ai chat
    vertexai.init(project="total-byte-432318-q3")
    model = GenerativeModel(
        model_name=gemini_model,
        tools=[Tool([FunctionDeclaration(**a) for a in api_declarations])],
        system_instruction=system_prompt,
    )
    chat = model.start_chat()

    # create shared variables
    answer = None
    api_logs = []
    token_counts = []
    shutil.rmtree(temp_workspace_dir, ignore_errors=True)
    os.makedirs(temp_workspace_dir)

    # extract the structured memory components from the emboded memory
    graph_prompt, scratchpad_prompt, navigation_log_prompt, images_prompt, frame_ids = (
        extract_qa_prompts(
            embodied_memory,
            dataset,
            prompt_img_seperate=True,
        )
    )

    # format the initial prompt
    prompt = [
        Part.from_text(
            format_qa_prompt(
                question=question,
                graph=json.dumps(graph_prompt),
                navigation_log=json.dumps(navigation_log_prompt),
                scratchpad=scratchpad_prompt,
            )
        )
    ]
    for i, id in enumerate(frame_ids):
        prompt.append(Part.from_text(f"Frame Index {id}:"))
        prompt.append(Part.from_image(Image.load_from_file(images_prompt[i])))
    prompt = Content(role="user", parts=prompt)

    # send the message to the chat
    response = chat.send_message(prompt)
    token_counts.append(
        {
            "prompt_token_count": response.usage_metadata.prompt_token_count,
            "candidates_token_count": response.usage_metadata.candidates_token_count,
        }
    )

    # execute the reasoning loop for max_search_depth iterations
    for i in range(max_search_depth):
        parts = response.candidates[0].content.parts

        # Check if the vlm chooses not to use any search tools, and is thus ready to answer the question.
        if len(parts) == 1 and not "function_call" in parts[0].to_dict():
            # Ask it to summarize its answer in a few words.
            response = chat.send_message(
                f"In a few words, summarize your answer to the question '{question}'? Do not include any explanation or justification for your answer. If you are uncertain in your answer, then state your most likely answer."
            )
            answer = response.candidates[0].content.parts[0].text.strip()
            break

        # Process all the VLMs function calls
        call_response = []
        for part in parts:
            if not "function_call" in part.to_dict():
                continue
            tool = part.function_call
            keyframe = int(tool.args["frame_id"])

            # append the requested image to the function response
            call_response.append(Part.from_text(f"Frame Index {keyframe}:"))
            call_response.append(
                Part.from_image(Image.load_from_file(dataset.color_paths[keyframe]))
            )

            # call the function
            embodied_memory, api_response, api_log, new_nodes = api.call(
                {"type": tool.name, **tool.args},
                dataset,
                result_dir_detections,
                temp_workspace_dir,
                embodied_memory,
                obj_pcd_max_points=obj_pcd_max_points,
                downsample_voxel_size=downsample_voxel_size,
            )

            # log details to for the debug log
            api_logs.append(api_log)

            # extract the updated structured memory for prompting
            (
                graph_prompt,
                scratchpad_prompt,
                navigation_log_prompt,
                images_prompt,
                frame_ids,
            ) = extract_qa_prompts(embodied_memory, dataset)

            # append the new structured memory to the call_response
            text_response = {
                "New SceneGraph": json.dumps(graph_prompt),
                # f"New Observation Log for frame {keyframe}": json.dumps(navigation_log_prompt[keyframe % skip_frame]),
                f"New Observation Log": json.dumps(navigation_log_prompt),
                "New Scratchpad": json.dumps(scratchpad_prompt),
            }
            call_response.append(
                Part.from_function_response(name=tool.name, response=text_response)
            )

        # send call_response to the reasoning agent
        response = chat.send_message(
            Content(
                role="function_response",
                parts=call_response,
            )
        )

        # update the token counts
        token_counts.append(
            {
                "prompt_token_count": response.usage_metadata.prompt_token_count,
                "candidates_token_count": response.usage_metadata.candidates_token_count,
            }
        )

    # If there still isn't an answer after max_search_depth, then ask the reasoning agent to take a guess summarizing its best idea
    if answer == None:
        response = chat.send_message(
            f"In a few words, summarize your answer to the question '{question}'? Do not include any explanation or justification for your answer. If you are uncertain in your answer, then state your most likely answer."
        )
        answer = response.candidates[0].content.parts[0].text.strip()
        token_counts.append(
            {
                "prompt_token_count": response.usage_metadata.prompt_token_count,
                "candidates_token_count": response.usage_metadata.candidates_token_count,
            }
        )

    # Remove the temporary workspace
    shutil.rmtree(temp_workspace_dir, ignore_errors=True)

    return answer, api_logs, token_counts
