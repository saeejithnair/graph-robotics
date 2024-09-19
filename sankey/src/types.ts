export interface Node {
  id: string;
  label: string;
  children: Node[];
}

export interface SankeyNode extends d3.SankeyNode<Node, any> {
  x0?: number;
  x1?: number;
  y0?: number;
  y1?: number;
}

export interface SceneGraphUpdate {
  nodeId: string;
  newParentId: string;
}