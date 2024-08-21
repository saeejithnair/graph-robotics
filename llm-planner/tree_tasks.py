import yaml
import os
from typing import List, Dict, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import networkx as nx
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_tasks(filename: str) -> Dict:
    with open(filename, "r") as file:
        return yaml.safe_load(file)


atomic_tasks = load_tasks("atomic_tasks.yaml")
composite_tasks = load_tasks("composite_tasks.yaml")


def generate_decomposition_thoughts(
    task: Dict, num_thoughts: int = 3
) -> List[List[str]]:
    prompt = f"""Decompose the following task into a sequence of atomic steps:
    Task: {task['name']}
    Description: {task['description']}
    
    Available atomic tasks:
    {yaml.dump(atomic_tasks['atomic_tasks'])}
    
    Provide {num_thoughts} different decompositions, each as a list of atomic task names."""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant that decomposes complex tasks into simpler ones.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=500,
    )

    # Parse the response to extract the decompositions
    decompositions = []
    current_decomposition = []
    for line in response.choices[0].message.content.split("\n"):
        if line.startswith(("1.", "2.", "3.")):
            if current_decomposition:
                decompositions.append(current_decomposition)
                current_decomposition = []
        elif line.strip():
            current_decomposition.append(line.strip())
    if current_decomposition:
        decompositions.append(current_decomposition)

    return decompositions


def evaluate_decomposition(task: Dict, decomposition: List[str]) -> float:
    prompt = f"""Evaluate the following decomposition for the task:
    Task: {task['name']}
    Description: {task['description']}
    
    Decomposition:
    {yaml.dump(decomposition)}
    
    Rate the decomposition on a scale from 0 to 1, where 1 is perfect and 0 is completely wrong.
    Provide your rating as a single number between 0 and 1, followed by a brief explanation."""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant that evaluates task decompositions.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=100,
    )

    # Extract the rating from the response
    content = response.choices[0].message.content.strip()
    try:
        # Try to extract the first number from the response
        rating = float(
            next(word for word in content.split() if word.replace(".", "").isdigit())
        )
        return rating
    except StopIteration:
        print(
            f"Warning: Could not find a numeric rating in the response. Using default rating of 0.5."
        )
        print(f"Full response: {content}")
        return 0.5  # Default rating if we can't parse a number


def decompose_task(
    task: Dict, max_depth: int = 3, num_thoughts: int = 3
) -> Tuple[List[str], nx.DiGraph]:
    def search(current_decomposition: List[str], depth: int) -> Tuple[List[str], float]:
        if depth >= max_depth:
            return current_decomposition, evaluate_decomposition(
                task, current_decomposition
            )

        thoughts = generate_decomposition_thoughts(task, num_thoughts)
        best_decomposition = None
        best_score = -1

        for thought in thoughts:
            new_decomposition = current_decomposition + thought
            score = evaluate_decomposition(task, new_decomposition)
            if score > best_score:
                best_decomposition = new_decomposition
                best_score = score

        return search(best_decomposition, depth + 1)

    final_decomposition, final_score = search([], 0)

    # Create a graph of the decomposition
    G = nx.DiGraph()
    G.add_node(task["name"], desc=task["description"])
    for i, step in enumerate(final_decomposition):
        G.add_node(f"Step {i+1}: {step}")
        if i == 0:
            G.add_edge(task["name"], f"Step {i+1}: {step}")
        else:
            G.add_edge(f"Step {i}: {final_decomposition[i-1]}", f"Step {i+1}: {step}")

    return final_decomposition, G


def process_all_tasks() -> Dict[str, Tuple[List[str], nx.DiGraph]]:
    results = {}
    for task in composite_tasks["composite_tasks"]:
        print(f"Processing task: {task['name']}")
        decomposition, graph = decompose_task(task)
        results[task["name"]] = (decomposition, graph)
    return results


def visualize_decomposition(task_name: str, graph: nx.DiGraph):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=3000,
        font_size=8,
        font_weight="bold",
    )

    labels = nx.get_node_attributes(graph, "desc")
    nx.draw_networkx_labels(graph, pos, labels, font_size=6)

    plt.title(f"Task Decomposition: {task_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        f"{task_name.replace(' ', '_')}_decomposition.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


if __name__ == "__main__":
    results = process_all_tasks()

    for task_name, (decomposition, graph) in results.items():
        print(f"\nTask: {task_name}")
        print("Decomposition:")
        for i, step in enumerate(decomposition, 1):
            print(f"{i}. {step}")

        visualize_decomposition(task_name, graph)
        print(f"Visualization saved as {task_name.replace(' ', '_')}_decomposition.png")
