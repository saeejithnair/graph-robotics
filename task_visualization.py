# %%
import networkx as nx
import matplotlib.pyplot as plt
from task_decomposition import *


def visualize_task_graph(task_decomposition: TaskDecomposition, task_name: str):
    G = nx.DiGraph()

    # Add nodes
    G.add_node("Start", pos=(0, 0))
    G.add_node("End", pos=(len(task_decomposition.task_sequence) + 1, 0))

    for i, task in enumerate(task_decomposition.task_sequence, 1):
        G.add_node(task.name, pos=(i, 0))
        if i == 1:
            G.add_edge("Start", task.name)
        else:
            G.add_edge(task_decomposition.task_sequence[i - 2].name, task.name)
        if i == len(task_decomposition.task_sequence):
            G.add_edge(task.name, "End")

    # Set up the plot
    plt.figure(figsize=(12, 6))
    pos = nx.get_node_attributes(G, "pos")

    # Draw the graph
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=3000,
        font_size=8,
        font_weight="bold",
        arrows=True,
        edge_color="gray",
    )

    # Add task descriptions as labels
    labels = {
        task.name: f"{task.name}\n{task.description}"
        for task in task_decomposition.task_sequence
    }
    labels["Start"] = "Start"
    labels["End"] = "End"
    nx.draw_networkx_labels(G, pos, labels, font_size=6)

    # Set title
    plt.title(f"Task Graph: {task_name}")

    # Show the plot
    plt.axis("off")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{task_name}_task_graph.png", dpi=300, bbox_inches="tight")


# %%
# task_name = "PrepareCoffee"
# result = process_composite_task(task_name)
# print(f"Task Decomposition for '{task_name}':")
# for i, task in enumerate(result.task_sequence, 1):
#     print(f"{i}. {task.name}: {task.description}")
# if result.notes:
#     print("\nNotes:")
#     for note in result.notes:
#         print(f"- {note}")

# # Visualize the task graph
# visualize_task_graph(result, task_name)

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

    # Visualize the task graph
    visualize_task_graph(result, task_name)
