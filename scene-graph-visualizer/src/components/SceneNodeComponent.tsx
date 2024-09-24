import React, { useState } from 'react';
import { SceneNode } from '../types';

interface SceneNodeComponentProps {
  node: SceneNode;
  depth: number;
}

const SceneNodeComponent: React.FC<SceneNodeComponentProps> = ({ node, depth }) => {
  const [isExpanded, setIsExpanded] = useState(depth < 2);

  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div style={{ marginLeft: `${depth * 20}px` }}>
      <div onClick={toggleExpand} style={{ cursor: 'pointer' }}>
        {node.children.length > 0 && (isExpanded ? '▼' : '►')} {node.label}
      </div>
      {isExpanded && node.children.map(child => (
        <SceneNodeComponent key={child.id} node={child} depth={depth + 1} />
      ))}
    </div>
  );
};

export default SceneNodeComponent;
