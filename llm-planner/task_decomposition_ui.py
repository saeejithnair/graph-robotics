import streamlit as st
import graphviz
from task_decomposition import process_composite_task, load_tasks

# Load task definitions
composite_tasks = load_tasks("composite_tasks.yaml")


def create_task_graph(task_sequence):
    dot = graphviz.Digraph()
    dot.attr(rankdir="TB", size="8,8")

    for i, task in enumerate(task_sequence):
        dot.node(
            f"task_{i}",
            f"{task.name}\n{task.description}",
            shape="box",
            style="filled",
            fillcolor="lightblue",
        )
        if i > 0:
            dot.edge(f"task_{i-1}", f"task_{i}")

    return dot


st.title("Kitchen Task Decomposition")

# Task selection
task_names = [task["name"] for task in composite_tasks["composite_tasks"]]
selected_task = st.selectbox("Select a task", task_names)

if st.button("Decompose Task"):
    with st.spinner("Decomposing task..."):
        result = process_composite_task(selected_task)

    st.subheader("Task Decomposition:")
    for i, task in enumerate(result.task_sequence, 1):
        st.write(f"{i}. **{task.name}**: {task.description}")

    if result.notes:
        st.subheader("Notes:")
        for note in result.notes:
            st.write(f"- {note}")

    st.subheader("Task Graph:")
    task_graph = create_task_graph(result.task_sequence)
    st.graphviz_chart(task_graph)

st.sidebar.title("About")
st.sidebar.info(
    "This app demonstrates task decomposition for kitchen tasks using AI. "
    "Select a task from the dropdown and click 'Decompose Task' to see the breakdown."
)
