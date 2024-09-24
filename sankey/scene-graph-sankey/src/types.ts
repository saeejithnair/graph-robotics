import { SankeyNode as D3SankeyNode, SankeyLink as D3SankeyLink } from 'd3-sankey';

export interface Node {
  id: string;
  label: string;
  children: Node[];
}

export interface SankeyNodeExtra {
  id: string;
  name: string;
  depth: number;
  value: number;
}

export type SankeyNode = D3SankeyNode<SankeyNodeExtra, SankeyLink>;

export interface SankeyLink extends D3SankeyLink<SankeyNode, SankeyNode> {
  source: SankeyNode;
  target: SankeyNode;
  value: number;
}

export interface SceneGraphUpdate {
  nodeId: string;
  newParentId: string;
}