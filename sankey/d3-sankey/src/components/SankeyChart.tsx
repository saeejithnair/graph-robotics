import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { sankey, sankeyLinkHorizontal, SankeyNode, SankeyLink } from 'd3-sankey';

interface NodeDatum {
  name: string;
}

interface LinkDatum {
  source: number;
  target: number;
  value: number;
}

type Node = SankeyNode<NodeDatum, LinkDatum>;
type Link = SankeyLink<NodeDatum, LinkDatum>;

interface SankeyChartProps {
  data: { nodes: NodeDatum[]; links: LinkDatum[] };
}

const SankeyChart: React.FC<SankeyChartProps> = ({ data }) => {
  const chartRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!chartRef.current) return;

    const margin = { top: 1, right: 1, bottom: 6, left: 1 };
    const width = 960 - margin.left - margin.right;
    const height = 500 - margin.top - margin.bottom;

    const formatNumber = d3.format(",.0f");
    const format = (d: number) => `${formatNumber(d)} TWh`;
    const color = d3.scaleOrdinal(d3.schemeCategory10);

    const svg = d3.select(chartRef.current)
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const sankeyGenerator = sankey<NodeDatum, LinkDatum>()
      .nodeWidth(15)
      .nodePadding(10)
      .extent([[0, 0], [width, height]]);

    const { nodes, links } = sankeyGenerator(data);

    // Gradient definitions
    const defs = svg.append("defs");

    links.forEach((link, i) => {
      const gradient = defs.append("linearGradient")
        .attr("id", `gradient-${i}`)
        .attr("gradientUnits", "userSpaceOnUse")
        .attr("x1", (link.source as Node).x1 ?? 0)
        .attr("x2", (link.target as Node).x0 ?? 0);

      gradient.append("stop")
        .attr("offset", "0%")
        .attr("stop-color", color((link.source as Node).name.replace(/ .*/, "")));

      gradient.append("stop")
        .attr("offset", "100%")
        .attr("stop-color", color((link.target as Node).name.replace(/ .*/, "")));
    });

    // Links
    const link = svg.append("g")
      .selectAll(".link")
      .data(links)
      .enter().append("path")
      .attr("class", "link")
      .attr("d", sankeyLinkHorizontal())
      .attr("stroke-width", d => Math.max(1, d.width ?? 0))
      .attr("stroke", (d, i) => `url(#gradient-${i})`)
      .style("opacity", 0.5)
      .sort((a, b) => (b.width ?? 0) - (a.width ?? 0));

    link.append("title")
      .text(d => `${(d.source as Node).name} → ${(d.target as Node).name}\n${format(d.value)}`);

    // Nodes
    const node = svg.append("g")
      .selectAll(".node")
      .data(nodes)
      .enter().append("g")
      .attr("class", "node")
      .attr("transform", d => `translate(${d.x0},${d.y0})`);

    node.append("rect")
      .attr("height", d => (d.y1 ?? 0) - (d.y0 ?? 0))
      .attr("width", sankeyGenerator.nodeWidth())
      .style("fill", d => color(d.name.replace(/ .*/, "")) as string)
      .style("stroke", d => d3.rgb(color(d.name.replace(/ .*/, "")) as string).darker(2).toString())
      .append("title")
      .text(d => `${d.name}\n${format(d.value!)}`);

    node.append("text")
      .attr("x", -6)
      .attr("y", d => ((d.y1 ?? 0) - (d.y0 ?? 0)) / 2)
      .attr("dy", ".35em")
      .attr("text-anchor", "end")
      .attr("transform", null)
      .text(d => d.name)
      .filter(d => (d.x0 ?? 0) < width / 2)
      .attr("x", 6 + sankeyGenerator.nodeWidth())
      .attr("text-anchor", "start");

    // Animation
    function branchAnimate(nodeData: Node) {
      const links = svg.selectAll<SVGPathElement, Link>(".link")
        .filter(d => d.source === nodeData);

      links
        .style("opacity", 1)
        .transition()
        .duration(400)
        .ease(d3.easeLinear)
        .style("stroke-dashoffset", 0)
        .on("end", function(d) {
          branchAnimate(d.target as Node);
        });
    }

    svg.selectAll<SVGPathElement, Link>(".link")
      .style("stroke-dasharray", function() {
        return this.getTotalLength() + " " + this.getTotalLength();
      })
      .style("stroke-dashoffset", function() {
        return this.getTotalLength();
      });

    node.on("mouseover", (event, d) => {
      svg.selectAll(".link").style("opacity", 0.05);
      branchAnimate(d);
    }).on("mouseout", () => {
      svg.selectAll<SVGPathElement, Link>(".link")
        .interrupt()
        .style("opacity", 0.5)
        .style("stroke-dashoffset", function() {
          return this.getTotalLength();
        });
    });

  }, [data]);

  return <svg ref={chartRef} style={{ height: '500px', width: '960px', margin: 'auto' }} />;
};

export default SankeyChart;