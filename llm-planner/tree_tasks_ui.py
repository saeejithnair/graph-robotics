import streamlit as st
import yaml
import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import networkx as nx
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, ValidationError

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Pydantic models for structured output
class Task(BaseModel):
    name: str
    is_atomic: bool
    subtasks: List["Task"] = Field(default_factory=list)


class TaskDecomposition(BaseModel):
    root: Task


@st.cache_data
def load_tasks(filename: str) -> Dict:
    with open(filename, "r") as file:
        return yaml.safe_load(file)


atomic_tasks = load_tasks("atomic_tasks.yaml")
composite_tasks = load_tasks("composite_tasks.yaml")


def gpt4_call(prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def gpt4_structured_call(prompt: str, model: Any) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ],
        functions=[
            {
                "name": "decompose_scenario",
                "description": "Decompose a scenario into a tree of tasks",
                "parameters": TaskDecomposition.model_json_schema(),
            }
        ],
        function_call={"name": "decompose_scenario"},
        temperature=0.7,
        max_tokens=1000,
    )

    if response.choices[0].message.function_call:
        return json.loads(response.choices[0].message.function_call.arguments)
    else:
        st.error("No function call found in the API response.")
        return {}


def create_scenario(selected_tasks: List[str]) -> str:
    task_descriptions = [
        task["description"]
        for task in composite_tasks["composite_tasks"]
        if task["name"] in selected_tasks
    ]
    prompt = f"""Create a realistic scenario that incorporates the following tasks:
    {yaml.dump(task_descriptions)}
    
    Provide a concise but coherent narrative that naturally includes all these tasks."""

    return gpt4_call(prompt)


def decompose_scenario(scenario: str) -> Optional[TaskDecomposition]:
    prompt = f"""Decompose the following scenario into a detailed tree of tasks:
    {scenario}
    
    Available atomic tasks:
    {yaml.dump(atomic_tasks['atomic_tasks'])}
    
    Guidelines:
    1. Create a detailed, multi-level decomposition. Go at least 3-4 levels deep where appropriate.
    2. Use as many relevant atomic tasks as possible. Don't skip small steps.
    3. Avoid repeating tasks like NavigateKitchen unless absolutely necessary.
    4. Figure out the main tasks mentioned in the scenario and focus on them.
    5. Break down each main task into smaller subtasks, and then into atomic tasks.
    
    Provide the decomposition as a tree structure where each node is a task (either composite or atomic) and can have subtasks."""

    decomposition_dict = gpt4_structured_call(prompt, TaskDecomposition)

    if decomposition_dict:
        try:
            return TaskDecomposition.parse_obj(decomposition_dict)
        except ValidationError as e:
            st.error(f"Error parsing the decomposition: {e}")
            return None
    else:
        st.error("Failed to generate a valid decomposition.")
        return None


def create_graph_from_tree(tree: TaskDecomposition) -> nx.DiGraph:
    G = nx.DiGraph()

    def add_nodes(node: Task, parent: str = None):
        G.add_node(node.name, is_atomic=node.is_atomic)
        if parent:
            G.add_edge(parent, node.name)
        for subtask in node.subtasks:
            add_nodes(subtask, node.name)

    add_nodes(tree.root)
    return G


def visualize_tree(graph: nx.DiGraph):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph)
    node_colors = [
        "lightblue" if graph.nodes[node]["is_atomic"] else "lightgreen"
        for node in graph.nodes()
    ]
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=3000,
        font_size=8,
        font_weight="bold",
    )

    plt.title("Task Decomposition Tree")
    plt.axis("off")
    plt.tight_layout()
    return plt


def plan_execution(tree: TaskDecomposition, urgency_prompt: str) -> List[str]:
    tree_json = tree.json()
    prompt = f"""Given the following task decomposition tree:
    {tree_json}
    
    And considering this urgency context:
    {urgency_prompt}
    
    Provide a sequential plan of atomic tasks in the order they should be executed.
    Return the plan as a numbered list of atomic task names."""

    plan = gpt4_call(prompt)

    # Split the plan into lines and clean them
    lines = [line.strip() for line in plan.split("\n") if line.strip()]

    # Process each line to extract the task name
    tasks = []
    for line in lines:
        # Remove any leading numbers or symbols
        task = line.lstrip("0123456789.- ")
        # Remove any trailing punctuation
        task = task.rstrip(".:,;")
        tasks.append(task)

    return tasks


st.title("Advanced Task Planning App")

# Task selection
task_names = [task["name"] for task in composite_tasks["composite_tasks"]]
selected_tasks = st.multiselect(
    "Select composite tasks to include in the scenario", task_names
)

if st.button("Generate Scenario and Decompose"):
    if not selected_tasks:
        st.warning("Please select at least one task.")
    else:
        st.header("Generated Scenario")
        scenario = create_scenario(selected_tasks)
        st.write(scenario)

        st.header("Task Decomposition")
        with st.spinner("Decomposing scenario into tasks..."):
            decomposition = decompose_scenario(scenario)

        if decomposition:
            st.subheader("Decomposition Tree")
            st.json(decomposition.dict())

            st.subheader("Tree Visualization")
            graph = create_graph_from_tree(decomposition)
            fig = visualize_tree(graph)
            st.pyplot(fig)

            st.session_state.decomposition = decomposition
        else:
            st.error("Failed to generate a valid decomposition. Please try again.")

if "decomposition" in st.session_state:
    st.header("Execution Planning")
    urgency_prompt = st.text_area("Enter urgency context for task execution:")

    if st.button("Generate Execution Plan"):
        with st.spinner("Generating execution plan..."):
            plan = plan_execution(st.session_state.decomposition, urgency_prompt)

        st.subheader("Final Execution Plan")
        for i, task in enumerate(plan, 1):
            st.write(f"{i}. {task}")

st.sidebar.title("About")
st.sidebar.info(
    "This app demonstrates advanced task planning using GPT-4. "
    "Select tasks to generate a scenario, decompose it into atomic tasks, "
    "and plan the execution based on urgency."
)
