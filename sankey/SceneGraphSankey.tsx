import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { useSelector } from 'react-redux';

interface Node {
  id: string;
  label: string;
  children: Node[];
}

interface SankeyNode extends d3.SankeyNode<Node, any> {
  x0?: number;
  x1?: number;
  y0?: number;
  y1?: number;
}

interface Props {
  data: Node;
  width: number;
  height: number;
}

const SceneGraphSankey: React.FC<Props> = ({ data, width, height }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [nodes, setNodes] = useState<SankeyNode[]>([]);
  const [links, setLinks] = useState<d3.SankeyLink<Node, SankeyNode>[]>([]);

  const sceneGraphUpdates = useSelector((state: RootState) => state.sceneGraph.updates);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const sankey = d3.sankey<Node, d3.SankeyLink<Node, SankeyNode>>()
      .nodeWidth(15)
      .nodePadding(10)
      .extent([[1, 1], [width - 1, height - 6]]);

    const { nodes: updatedNodes, links: updatedLinks } = sankey({
      nodes: flattenHierarchy(data),
      links: getLinks(data),
    });

    setNodes(updatedNodes);
    setLinks(updatedLinks);

    const link = svg.append('g')
      .attr('fill', 'none')
      .attr('stroke', '#000')
      .attr('stroke-opacity', 0.2)
      .selectAll('path')
      .data(updatedLinks)
      .join('path')
      .attr('d', d3.sankeyLinkHorizontal())
      .attr('stroke-width', d => Math.max(1, d.width));

    const node = svg.append('g')
      .selectAll('rect')
      .data(updatedNodes)
      .join('rect')
      .attr('x', d => d.x0!)
      .attr('y', d => d.y0!)
      .attr('height', d => d.y1! - d.y0!)
      .attr('width', d => d.x1! - d.x0!)
      .attr('fill', '#69b3a2');

    node.append('title')
      .text(d => `${d.name}\n${d.value}`);

    svg.append('g')
      .attr('font-family', 'sans-serif')
      .attr('font-size', 10)
      .selectAll('text')
      .data(updatedNodes)
      .join('text')
      .attr('x', d => d.x0! < width / 2 ? d.x1! + 6 : d.x0! - 6)
      .attr('y', d => (d.y1! + d.y0!) / 2)
      .attr('dy', '0.35em')
      .attr('text-anchor', d => d.x0! < width / 2 ? 'start' : 'end')
      .text(d => d.name);

    // Animate transitions
    svg.selectAll('rect')
      .data(updatedNodes)
      .transition()
      .duration(500)
      .attr('x', d => d.x0!)
      .attr('y', d => d.y0!)
      .attr('height', d => d.y1! - d.y0!)
      .attr('width', d => d.x1! - d.x0!);

    svg.selectAll('path')
      .data(updatedLinks)
      .transition()
      .duration(500)
      .attr('d', d3.sankeyLinkHorizontal());

    // Update text positions
    svg.selectAll('text')
      .data(updatedNodes)
      .transition()
      .duration(500)
      .attr('x', d => d.x0! < width / 2 ? d.x1! + 6 : d.x0! - 6)
      .attr('y', d => (d.y1! + d.y0!) / 2);
  }, [sceneGraphUpdates]);

  const flattenHierarchy = (node: Node, depth = 0): Node[] => {
    const result: Node[] = [{ ...node, depth }];
    if (node.children) {
      node.children.forEach(child => {
        result.push(...flattenHierarchy(child, depth + 1));
      });
    }
    return result;
  };

  const getLinks = (node: Node): { source: string; target: string; value: number }[] => {
    const links: { source: string; target: string; value: number }[] = [];
    if (node.children) {
      node.children.forEach(child => {
        links.push({ source: node.id, target: child.id, value: 1 });
        links.push(...getLinks(child));
      });
    }
    return links;
  };

  return <svg ref={svgRef} width={width} height={height} />;
};

export default SceneGraphSankey;