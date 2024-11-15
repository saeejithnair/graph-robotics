import numpy as np
import open3d as o3d
import rerun as rr
from PIL.ImageColor import getrgb

class Visualizer3D:
    def __init__(self):
        level2color = {
            0: 'yellow',
            1: 'green',
            2: 'red',
            3: 'purple',
            4: 'orange',
            5: 'brown',
            6: 'turquoise',
            7: 'blue'
        }
        self.level2color = {i:getrgb(level2color[i]) for i in level2color}
    
    def init(self):
        rr.init("visualize-ham-sg", spawn=True)
        # rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Set an up-axis
        rr.log(
            "world/xyz",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            )
        )
        
    def add_geometry_map(self, points, colors=None):
        if not colors:
            colors = [getrgb('white') for _ in range(len(points))]
        rr.log(
            "geometry_map",
            rr.Points3D(np.asarray(points), 
                        colors=np.asarray(colors), 
                        radii=0.01)
        )
    
    def add_semanticnode(self, id, label, caption, location, children_locs, children_labels,
                points, level, geometry_color=getrgb('gray')):
        entity_path = "semantic_tree"
        semantic_color=self.level2color[level % len(self.level2color)]
        name = str(id) + '_' + label
        rr.log(
            entity_path + '/node_centroids/'+name,
            rr.Points3D(
                np.asarray([location]), 
                colors=np.asarray(semantic_color), 
                radii=0.1,
                labels=[name]),
            rr.AnyValues(
                uuid = str(id),
                label= label,
                caption= caption,
            ))
        rr.log(
            entity_path + '/node_point_cloud/'+name,
            rr.LineStrips3D(
                np.array([(location, i) for i in points]),
                colors=np.asarray(geometry_color))
            )
        rr.log(
            entity_path + '/node_children/'+name,
            rr.LineStrips3D(
                np.array([(location, i) for i in children_locs]),
                colors=np.asarray(semantic_color),
            ),
            rr.AnyValues(
                children_labels = ' ,'.join(children_labels)
            )
            )
     

class _UnitTestNode:
    def __init__(self, id, level, location=None, label=None, 
                 caption=None, point_idxs=None, pcd=None, height_by_level=False) -> None:
        self.id = id
        self.level = level
        self.point_idxs = point_idxs
        self.label = f'node' if not label else label
        self.caption = 'None' if not caption else caption
        if not location:
            mean_pcd = np.mean(np.asarray(pcd.points)[point_idxs], 0)
            height = None
            if height_by_level:
                height = self.level*0.5 + 4 # heursitic
            else:
                max_pcd_height = np.max(- np.asarray(pcd.points)[point_idxs], 0)[1]
                height = max_pcd_height + self.level*0.3 + 0.5
            location = [mean_pcd[0], -1 * height, mean_pcd[2]]
        self.location = location
    def add_point_idxs(self, point_idxs):
        self.point_idxs = point_idxs

if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud("/home/qasim/Projects/graph-robotics/scene-graph/outputs/pcd/000-hm3d-BFRyYbPCCPE.pcd")
    visualizer = Visualizer3D()
    nodes = [
        _UnitTestNode(0, 0, point_idxs=[i for i in range(150)], caption='test caption', pcd=pcd),
        _UnitTestNode(1, 1, point_idxs=[i for i in range(500, 1000)], pcd=pcd),
        _UnitTestNode(2, 0, point_idxs=[i for i in range(1500, 2000)], pcd=pcd),
        _UnitTestNode(3, 2, point_idxs=[i for i in range(3000, 3500)], pcd=pcd),
    ]
    children = {
        0: [],
        1: [0],
        2: [],
        3: [1]
    }

    def visualize_node(visualizer, pcd, nodes, idx):
        visualizer.add_semanticnode(
            id=nodes[idx].id,
            label=nodes[idx].label,
            caption=nodes[idx].caption,
            location=nodes[idx].location,
            level = nodes[idx].level,
            children_locs=[nodes[c].location for c in children[idx]],
            children_labels=[nodes[c].label for c in children[idx]],
            points=get_points_at_idxs(pcd, nodes[idx].point_idxs),
        )
    def get_points_at_idxs(pcd, idxs):
        return np.asarray(pcd.points)[idxs]

    visualizer.init()
    visualizer.add_geometry_map(pcd.points, pcd.colors)
    visualize_node( visualizer, pcd, nodes, 0)
    visualize_node( visualizer, pcd, nodes, 1)
    visualize_node( visualizer, pcd, nodes, 2)
    visualize_node( visualizer, pcd, nodes, 3)