import yaml
import os
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Load task definitions
def load_tasks(filename: str) -> Dict:
    with open(filename, "r") as file:
        return yaml.safe_load(file)


atomic_tasks = load_tasks("atomic_tasks.yaml")
composite_tasks = load_tasks("composite_tasks.yaml")


# Define the structure for our task decomposition
class AtomicTask(BaseModel):
    name: str
    description: str


class TaskDecomposition(BaseModel):
    task_sequence: List[AtomicTask] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


def create_gpt4_prompt(composite_task: Dict) -> str:
    prompt = f"""You are an AI assistant tasked with decomposing complex kitchen tasks into a sequence of atomic actions. 
    You have access to the following atomic tasks:

    {yaml.dump(atomic_tasks, default_flow_style=False)}

    Your goal is to break down the following composite task into a sequence of these atomic tasks:

    Task: {composite_task['name']}
    Description: {composite_task['description']}

    Please provide the decomposition as a sequence of atomic tasks, represented by their names and descriptions.
    If a step cannot be represented by the available atomic tasks, use the most relevant atomic task and add a note explaining the limitation.

    Respond with a JSON object with the following structure:
    {{
        "task_sequence": [
            {{"name": "AtomicTaskName", "description": "Description of how this atomic task is used"}},
            ...
        ],
        "notes": [
            "Any additional notes or explanations",
            ...
        ]
    }}
    """
    print(f"Prompt for '{composite_task['name']}':\n{prompt}")
    return prompt


def parse_response(response_content: str) -> TaskDecomposition:
    try:
        data = json.loads(response_content)
        if "decomposition" in data:
            # Handle the case where the model wrapped the task sequence in a "decomposition" key
            data["task_sequence"] = data.pop("decomposition")
        return TaskDecomposition(**data)
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract structured data from the text
        tasks = []
        notes = []
        lines = response_content.split("\n")
        for line in lines:
            if line.startswith("- ") or line.strip().isdigit():
                # This looks like a task
                parts = line.split(":", 1)
                if len(parts) == 2:
                    name = parts[0].strip("- ").strip()
                    description = parts[1].strip()
                    tasks.append(AtomicTask(name=name, description=description))
            elif line.lower().startswith("note:") or line.lower().startswith(
                "additional note:"
            ):
                notes.append(line.split(":", 1)[1].strip())
        return TaskDecomposition(task_sequence=tasks, notes=notes)


def decompose_task(composite_task: Dict) -> TaskDecomposition:
    prompt = create_gpt4_prompt(composite_task)

    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",  # Use the latest available model
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant that decomposes complex tasks into simpler ones. Always respond with a JSON object following the specified schema.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=500,
    )

    # Parse the response into our TaskDecomposition structure
    return parse_response(response.choices[0].message.content)


def process_composite_task(task_name: str) -> TaskDecomposition:
    composite_task = next(
        (
            task
            for task in composite_tasks["composite_tasks"]
            if task["name"] == task_name
        ),
        None,
    )
    if not composite_task:
        raise ValueError(f"Composite task '{task_name}' not found")

    return decompose_task(composite_task)


# Example usage
if __name__ == "__main__":
    task_name = "PrepareCoffee"
    result = process_composite_task(task_name)
    print(f"Task Decomposition for '{task_name}':")
    for i, task in enumerate(result.task_sequence, 1):
        print(f"{i}. {task.name}: {task.description}")
    if result.notes:
        print("\nNotes:")
        for note in result.notes:
            print(f"- {note}")
