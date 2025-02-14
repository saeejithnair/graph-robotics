import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import open3d as o3d
import rerun as rr
from networkx.drawing.nx_agraph import graphviz_layout
from PIL.ImageColor import getrgb
from pyvis.network import Network


class Visualizer3D:
    def __init__(self, level2color):
        self.level2color = {i: getrgb(level2color[i]) for i in level2color}

    def init(self):
        rr.init("visualize-ham-sg", spawn=True)
        # rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Set an up-axis
        rr.log(
            "world/xyz",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )

    def add_full_scene_pcd(self, points, colors=None):
        if not colors:
            colors = [getrgb("white") for _ in range(len(points))]
        rr.log(
            "full_scene_pcd",
            rr.Points3D(np.asarray(points), colors=np.asarray(colors), radii=0.01),
        )

    def add_semanticnode(
        self,
        id,
        label,
        caption,
        location,
        children_locs,
        children_labels,
        points_xyz,
        points_colors,
        level,
        geometry_color=getrgb("gray"),
    ):
        entity_path = "embodied_memory"
        semantic_color = self.level2color[level % len(self.level2color)]
        name = str(id) + "_" + label.replace(" ", "_")
        rr.log(
            entity_path + "/node_centroids/" + name,
            rr.Points3D(
                np.asarray([location]),
                colors=np.asarray(semantic_color),
                radii=0.1,
                labels=[name],
            ),
            rr.AnyValues(
                uuid=str(id),
                label=label,
                caption=caption,
            ),
        )
        rr.log(
            entity_path + "/node_point_cloud_lines/" + name,
            rr.LineStrips3D(
                np.array([(location, i) for i in points_xyz]),
                colors=np.asarray(geometry_color),
            ),
        )
        rr.log(
            entity_path + "/node_point_cloud/" + name,
            rr.Points3D(np.asarray(points_xyz), colors=np.asarray(points_colors), radii=0.01),
        )
        rr.log(
            entity_path + "/node_children/" + name,
            rr.LineStrips3D(
                np.array([(location, i) for i in children_locs]),
                colors=np.asarray(semantic_color),
            ),
            rr.AnyValues(children_labels=" ,".join(children_labels)),
        )


class Visualizer2D:
    def __init__(self, level2color) -> None:
        self.level2color = level2color
        self.in_graph = pyvisGraph(self.level2color, "In")
        self.on_graph = pyvisGraph(self.level2color, "On")
        self.hierarchy_graph = pyvisGraph(self.level2color, "Hierarchy Graph")

    def init(self):
        self.in_graph.init()
        self.on_graph.init()
        self.hierarchy_graph.init()

    def visualize(
        self,
        folder,
        ids,
        labels,
        levels,
        hierarchy_matrix,
        hierarchy_type_matrix=None,
        in_matrix=None,
        on_matrix=None,
    ):
        for i in range(len(labels)):
            labels[i] = str(ids[i]) + "-" + labels[i].replace(" ", "_")
        for i in range(len(labels)):
            self.in_graph.add_node(labels[i], levels[i])
            self.on_graph.add_node(labels[i], levels[i])
            self.hierarchy_graph.add_node(labels[i], levels[i])
        for i in range(len(labels)):
            for j in range(len(labels)):
                if (not in_matrix is None) and in_matrix[i][j] > 0:
                    self.in_graph.add_edge(labels[i], labels[j], weight=in_matrix[i][j])
                if (not on_matrix is None) and on_matrix[i][j] > 0:
                    self.on_graph.add_edge(labels[i], labels[j], weight=on_matrix[i][j])
                if hierarchy_matrix[i][j] > 0:
                    self.hierarchy_graph.add_edge(labels[i], labels[j], title=','.join(hierarchy_type_matrix[i][j]))
        if not in_matrix is None:
            self.in_graph.save_graph(folder)
        if not on_matrix is None:
            self.on_graph.save_graph(folder)
        self.hierarchy_graph.save_graph(folder)


class nxGraph:
    def __init__(self, level2color, name="Graph") -> None:
        self.level2color = level2color
        self.name = name

    def init(self):
        self.G = nx.DiGraph()
        self.node_colors = []

    def add_node(self, label, level):
        self.G.add_node(label)
        self.node_colors.append(self.level2color[level])

    def add_edge(self, parent, child, weight=None):
        if weight:
            self.G.add_edge(parent, child, weight=weight)
        else:
            self.G.add_edge(parent, child)

    def save_graph(self, path):
        # Use a layout suited for trees
        pos = nx.spring_layout(self.G)  # Use "dot" for hierarchical layout
        # Draw the graph with edge weights
        nx.draw(
            self.G,
            pos,
            with_labels=True,
            node_color=self.node_colors,
            font_size=10,
            arrows=True,
            # node_size=3000,
            # font_weight='bold',
            # arrowsize=20,
        )
        # Draw edge labels to show weights
        edge_labels = nx.get_edge_attributes(self.G, "weight")
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)
        plt.title(self.name)
        plt.savefig(os.path.join(path, self.name + ".png"))


class pyvisGraph:
    def __init__(self, level2color, name="Graph") -> None:
        self.level2color = level2color
        self.name = name

    def init(self):
        self.G = Network(height="750px", width="100%", directed=True)  # Directed graph
        self.G.set_options(
            """
            {
            "nodes": {
                "shape": "dot",
                "size": 20,
                "font": {
                "size": 14,
                "bold": true
                }
            },
            "edges": {
                "arrows": {
                "to": {
                    "enabled": true,
                    "type": "arrow"
                }
                },
                "size": 10,
                "smooth": false
            },
            "layout": {
                "hierarchical": {
                "enabled": true,
                "levelSeparation": 150,
                "nodeSpacing": 150,
                "treeSpacing": 200,
                "direction": "UD",
                "sortMethod": "directed"
                }
            },
            "physics": {
                "hierarchicalRepulsion": {
                "centralGravity": 0.0,
                "springLength": 100,
                "springConstant": 0.01,
                "nodeDistance": 120,
                "damping": 0.09
                }
            }
            }
            """
        )

    def add_node(self, label, level):
        # Add a node with a specific color based on the level
        color = self.level2color[level]
        self.G.add_node(label, label=label, color=color, level=level)

    def add_edge(self, parent, child, title=None, weight=None):
        if not (weight is None) and not (title is None):
            self.G.add_edge(parent, child, title=title, value=weight)
        elif weight:
            self.G.add_edge(parent, child, title=f"Weight: {weight}", value=weight)
        elif title:
            self.G.add_edge(parent, child, title=title)
        else:
            self.G.add_edge(parent, child)

    def save_graph(self, path):
        # Save the graph as an interactive HTML file
        output_file = os.path.join(path, f"{self.name}.html")
        self.G.write_html(output_file, notebook=False)
        print(f"Graph saved as {output_file}")


class _UnitTestNode:
    def __init__(
        self,
        id,
        level,
        location=None,
        label=None,
        caption=None,
        local_pcd_idxs=None,
        pcd=None,
        height_by_level=False,
    ) -> None:
        self.id = id
        self.level = level
        self.local_pcd_idxs = local_pcd_idxs
        self.label = f"node" if not label else label
        self.caption = "None" if not caption else caption
        if not location:
            mean_pcd = np.mean(np.asarray(pcd.points)[local_pcd_idxs], 0)
            height = None
            if height_by_level:
                height = self.level * 0.5 + 4  # heursitic
            else:
                max_pcd_height = np.max(-np.asarray(pcd.points)[local_pcd_idxs], 0)[1]
                height = max_pcd_height + self.level * 0.3 + 0.5
            location = [mean_pcd[0], -1 * height, mean_pcd[2]]
        self.location = location

    def add_local_pcd_idxs(self, local_pcd_idxs):
        self.local_pcd_idxs = local_pcd_idxs


if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud(
        "/home/qasim/Projects/graph-robotics/gso/scene_graph/outputs/pcd/000-hm3d-BFRyYbPCCPE.pcd"
    )
    visualizer = Visualizer3D()
    nodes = [
        _UnitTestNode(
            0,
            0,
            local_pcd_idxs=[i for i in range(150)],
            caption="test caption",
            pcd=pcd,
        ),
        _UnitTestNode(1, 1, local_pcd_idxs=[i for i in range(500, 1000)], pcd=pcd),
        _UnitTestNode(2, 0, local_pcd_idxs=[i for i in range(1500, 2000)], pcd=pcd),
        _UnitTestNode(3, 2, local_pcd_idxs=[i for i in range(3000, 3500)], pcd=pcd),
    ]
    children = {0: [], 1: [0], 2: [], 3: [1]}

    def visualize_node(visualizer, pcd, nodes, idx):
        visualizer.add_semanticnode(
            id=nodes[idx].id,
            label=nodes[idx].label,
            caption=nodes[idx].caption,
            location=nodes[idx].location,
            level=nodes[idx].level,
            children_locs=[nodes[c].location for c in children[idx]],
            children_labels=[nodes[c].label for c in children[idx]],
            points= get_points_at_idxs(pcd, nodes[idx].local_pcd_idxs),
        )

    def get_points_at_idxs(pcd, idxs):
        return np.asarray(pcd.points)[idxs]

    visualizer.init()
    visualizer.add_full_scene_pcd(pcd.points, pcd.colors)
    visualize_node(visualizer, pcd, nodes, 0)
    visualize_node(visualizer, pcd, nodes, 1)
    visualize_node(visualizer, pcd, nodes, 2)
    visualize_node(visualizer, pcd, nodes, 3)
