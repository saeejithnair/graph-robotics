interface RawNode {
  id: string;
  label: string;
  children?: RawNode[];
}

interface ProcessedNode {
  id: string;
  name: string;
}

interface ProcessedLink {
  source: string;
  target: string;
  value: number;
}

interface ProcessedData {
  nodes: ProcessedNode[];
  links: ProcessedLink[];
}

function flattenNodes(node: RawNode, nodes: ProcessedNode[] = [], links: ProcessedLink[] = []): void {
  nodes.push({ id: node.id, name: node.label });

  if (node.children) {
    for (const child of node.children) {
      flattenNodes(child, nodes, links);
      links.push({ 
        source: node.id, 
        target: child.id, 
        value: 1 
      });
    }
  }
}

export function processData(rawData: RawNode): ProcessedData {
  const nodes: ProcessedNode[] = [];
  const links: ProcessedLink[] = [];

  flattenNodes(rawData, nodes, links);

  return { nodes, links };
}