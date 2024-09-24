import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import * as d3Sankey from 'd3-sankey';
import { useSelector } from 'react-redux';
import { RootState } from '../store/store';
import { Node, SankeyNode, SankeyLink } from '../types';
import { Paper, Typography, Box } from '@mui/material';
import { useTheme } from '@mui/material/styles';

interface Props {
  width: number;
  height: number;
}

const SceneGraphSankey: React.FC<Props> = ({ width, height }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const data = useSelector((state: RootState) => state.sceneGraph.data);
  const theme = useTheme();

  useEffect(() => {
    if (!svgRef.current || !data.id) {
      console.log("No data available or SVG ref is null");
      return;
    }

    console.log("Data being passed to prepareDataForSankey:", data); // Log the data

    const { nodes: newNodes, links: newLinks } = prepareDataForSankey(data);
    console.log("New Nodes:", newNodes); // Log the new nodes
    console.log("New Links:", newLinks); // Log the new links

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const chart = SankeyChart({ nodes: newNodes, links: newLinks }, {
      nodeGroup: (d: SankeyNode) => d.id.split(/\W/)[0],
      nodeAlign: d3Sankey.sankeyJustify,
      linkColor: "source-target",
      width,
      height,
      nodeLabel: (d: SankeyNode) => d.name,
      colors: d3.schemeCategory10
    });

    svg.node()?.appendChild(chart);

  }, [data, width, height, theme]);

  const prepareDataForSankey = (node: Node): { nodes: SankeyNode[], links: SankeyLink[] } => {
    if (!node || !node.id) {
      console.warn("Invalid node data:", node);
      return { nodes: [], links: [] }; // Return empty arrays if data is invalid
    }

    const nodes: SankeyNode[] = [];
    const links: SankeyLink[] = [];
    const nodeMap: { [key: string]: SankeyNode } = {};
    
    const addNode = (n: Node, depth: number): SankeyNode => {
      const sankeyNode: SankeyNode = { 
        id: n.id,
        name: n.label,
        depth,
        value: 1
      } as SankeyNode;
      nodes.push(sankeyNode);
      nodeMap[n.id] = sankeyNode;
      
      n.children.forEach((child) => {
        const childNode = addNode(child, depth + 1);
        links.push({
          source: sankeyNode,
          target: childNode,
          value: 1
        } as SankeyLink);
      });
      
      return sankeyNode;
    };
    
    addNode(node, 0);
    
    return { nodes, links };
  };

  function SankeyChart(
    { nodes, links }: { nodes: SankeyNode[], links: SankeyLink[] },
    {
      format = ",",
      align = "justify",
      nodeId = (d: SankeyNode) => d.id,
      nodeGroup,
      nodeGroups,
      nodeLabel,
      nodeTitle = (d: SankeyNode) => `${d.name}\n${format(d.value)}`,
      nodeAlign = align,
      nodeWidth = 15,
      nodePadding = 10,
      nodeLabelPadding = 6,
      nodeStroke = "currentColor",
      nodeStrokeWidth,
      nodeStrokeOpacity,
      nodeStrokeLinejoin,
      linkSource = ({source}: SankeyLink) => source,
      linkTarget = ({target}: SankeyLink) => target,
      linkValue = ({value}: SankeyLink) => value,
      linkPath = d3Sankey.sankeyLinkHorizontal(),
      linkTitle = (d: SankeyLink) => `${d.source.name} → ${d.target.name}\n${format(d.value)}`,
      linkColor = "source-target",
      linkStrokeOpacity = 0.5,
      linkMixBlendMode = "multiply",
      colors = d3.schemeTableau10,
      width = 640,
      height = 400,
      marginTop = 5,
      marginRight = 1,
      marginBottom = 5,
      marginLeft = 1,
    }: any = {}
  ) {
    if (typeof nodeAlign !== "function") nodeAlign = {
      left: d3Sankey.sankeyLeft,
      right: d3Sankey.sankeyRight,
      center: d3Sankey.sankeyCenter
    }[nodeAlign as string] ?? d3Sankey.sankeyJustify;

    const LS = d3.map(links, linkSource);
    const LT = d3.map(links, linkTarget);
    const LV = d3.map(links, linkValue);
    if (nodes === undefined) nodes = Array.from(d3.union(LS, LT), id => ({id, name: id, depth: 0, value: 0} as SankeyNode));
    const N = d3.map(nodes, nodeId);
    const G = nodeGroup == null ? null : d3.map(nodes, nodeGroup) as string[];

    nodes = d3.map(nodes, (d, i) => ({...d, id: N[i]} as SankeyNode));
    links = d3.map(links, (d, i) => ({...d, source: LS[i], target: LT[i], value: LV[i]} as SankeyLink));

    if (!G && ["source", "target", "source-target"].includes(linkColor)) linkColor = "currentColor";

    if (G && nodeGroups === undefined) nodeGroups = G;

    const color = nodeGroup == null ? null : d3.scaleOrdinal<string, string>(nodeGroups as Iterable<string>, colors as ReadonlyArray<string>);

    d3Sankey.sankey()
        .nodeId((d: any) => d.id)
        .nodeAlign(nodeAlign)
        .nodeWidth(nodeWidth)
        .nodePadding(nodePadding)
        .extent([[marginLeft, marginTop], [width - marginRight, height - marginBottom]])
      ({nodes, links});

    const Tl = nodeLabel === undefined ? N : nodeLabel == null ? null : d3.map(nodes, nodeLabel);
    const Tt = nodeTitle == null ? null : d3.map(nodes, nodeTitle);
    const Lt = linkTitle == null ? null : d3.map(links, linkTitle);

    const uid = `O-${Math.random().toString(16).slice(2)}`;

    const svg = d3.create("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height])
        .attr("style", "max-width: 100%; height: auto; height: intrinsic;");

    const node = svg.append("g")
        .attr("stroke", nodeStroke)
        .attr("stroke-width", nodeStrokeWidth)
        .attr("stroke-opacity", nodeStrokeOpacity)
        .attr("stroke-linejoin", nodeStrokeLinejoin)
      .selectAll("rect")
      .data(nodes)
      .join("rect")
        .attr("x", (d: any) => d.x0)
        .attr("y", (d: any) => d.y0)
        .attr("height", (d: any) => d.y1 - d.y0)
        .attr("width", (d: any) => d.x1 - d.x0);

    if (G) node.attr("fill", (d: SankeyNode, i: number) => 
      color && G[i] ? color(G[i]) : "gray" // Default color if undefined
    );

    if (Tt) node.append("title").text(({index: i}: any) => Tt[i] as string);

    const link = svg.append("g")
        .attr("fill", "none")
        .attr("stroke-opacity", linkStrokeOpacity)
      .selectAll("g")
      .data(links)
      .join("g")
        .style("mix-blend-mode", linkMixBlendMode);

    if (linkColor === "source-target") link.append("linearGradient")
        .attr("id", (d: any) => `${uid}-link-${d.index}`)
        .attr("gradientUnits", "userSpaceOnUse")
        .attr("x1", (d: any) => d.source.x1)
        .attr("x2", (d: any) => d.target.x0)
        .call((gradient: any) => gradient.append("stop")
            .attr("offset", "0%")
            .attr("stop-color", ({source: {index: i}}: any) => color && G && G[i] ? color(G[i]) : "gray")) // Default color if undefined
        .call((gradient: any) => gradient.append("stop")
            .attr("offset", "100%")
            .attr("stop-color", ({target: {index: i}}: any) => color && G && G[i] ? color(G[i]) : "gray")); // Default color if undefined

    link.append("path")
        .attr("d", linkPath)
        .attr("stroke", (d: any) => linkColor === "source-target" ? `url(#${uid}-link-${d.index})`
            : linkColor === "source" ? (color && G && G[d.source.index] ? color(G[d.source.index]) : "gray") // Default color if undefined
            : linkColor === "target" ? (color && G && G[d.target.index] ? color(G[d.target.index]) : "gray") // Default color if undefined
            : linkColor)
        .attr("stroke-width", ({width}: any) => Math.max(1, width))
        .call(Lt ? (path: any) => path.append("title").text(({index: i}: any) => Lt[i] as string) : () => {});

    if (Tl) svg.append("g")
        .attr("font-family", "sans-serif")
        .attr("font-size", 10)
      .selectAll("text")
      .data(nodes)
      .join("text")
        .attr("x", (d: any) => d.x0 < width / 2 ? d.x1 + nodeLabelPadding : d.x0 - nodeLabelPadding)
        .attr("y", (d: any) => (d.y1 + d.y0) / 2)
        .attr("dy", "0.35em")
        .attr("text-anchor", (d: any) => d.x0 < width / 2 ? "start" : "end")
        .text(({index: i}: any) => Tl[i] as string);

    return Object.assign(svg.node() as SVGSVGElement, {scales: {color}});
  }

  if (!data.id) {
    return <div>No data available</div>;
  }

  return (
    <Paper elevation={3} sx={{ p: 2, bgcolor: theme.palette.background.paper }}>
      <Typography variant="h6" gutterBottom>Scene Graph Visualization</Typography>
      <Box sx={{ width, height, overflow: 'hidden' }}>
        <svg ref={svgRef} width={width} height={height} />
      </Box>
    </Paper>
  );
};

export default SceneGraphSankey;