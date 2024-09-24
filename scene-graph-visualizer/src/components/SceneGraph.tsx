import React, { useState, useRef, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import * as THREE from 'three';
import { SceneNode } from '../types';

interface NodeProps {
  node: SceneNode;
  position: [number, number, number];
  depth: number;
  onClick: (node: SceneNode) => void;
  isFocused: boolean;
  color: string;
}

const Node: React.FC<NodeProps> = ({ node, position, depth, onClick, isFocused, color }) => {
  const ref = useRef<THREE.Mesh>(null);
  const [hovered, setHover] = useState(false);

  const scale = Math.max(0.2, 1 - depth * 0.2);

  return (
    <group position={position}>
      {/* Invisible hitbox for better clickability */}
      <mesh
        scale={1.5}
        onClick={(e) => {
          e.stopPropagation();
          onClick(node);
        }}
        onPointerOver={() => setHover(true)}
        onPointerOut={() => setHover(false)}
      >
        <sphereGeometry args={[0.5, 32, 32]} />
        <meshBasicMaterial visible={false} />
      </mesh>
      {/* Visible node */}
      <mesh
        ref={ref}
        scale={hovered || isFocused ? scale * 1.2 : scale}
      >
        <cylinderGeometry args={[0.5, 0.5, 0.2, 32]} />
        <meshStandardMaterial color={color} />
      </mesh>
      <Html
        center
        style={{
          backgroundColor: 'rgba(0,0,0,0.5)',
          padding: '2px 5px',
          borderRadius: '3px',
          transform: 'translateY(-20px)'
        }}
      >
        <div style={{ color: 'white', fontSize: '12px' }}>{node.label}</div>
      </Html>
    </group>
  );
};

interface SceneContentProps {
  root: SceneNode;
}

const SceneContent: React.FC<SceneContentProps> = ({ root }) => {
  const [focusedNode, setFocusedNode] = useState<SceneNode>(root);
  const [nodeStack, setNodeStack] = useState<SceneNode[]>([root]);
  const { camera } = useThree();
  const controlsRef = useRef<any>();

  const handleNodeClick = (node: SceneNode) => {
    setFocusedNode(node);
    setNodeStack(prevStack => [...prevStack, node]);
  };

  const handleBackClick = () => {
    if (nodeStack.length > 1) {
      const newStack = nodeStack.slice(0, -1);
      setNodeStack(newStack);
      setFocusedNode(newStack[newStack.length - 1]);
    }
  };

  useEffect(() => {
    if (controlsRef.current) {
      const targetPosition = new THREE.Vector3(...focusedNode.position);
      targetPosition.y += 5;
      controlsRef.current.target.copy(new THREE.Vector3(...focusedNode.position));
      camera.position.copy(targetPosition);
      controlsRef.current.update();
    }
  }, [focusedNode, camera]);

  const getNodeColor = (depth: number, index: number, parentHue?: number): string => {
    if (depth === 0) {
      return `hsl(0, 100%, 50%)`; // Root node is red
    } else if (depth === 1) {
      const hue = (index * 360 / focusedNode.children.length) % 360;
      return `hsl(${hue}, 100%, 50%)`;
    } else {
      return `hsl(${parentHue}, 100%, ${70 + depth * 10}%)`;
    }
  };

  const renderNodes = (node: SceneNode, basePosition: [number, number, number], spread: number, depth: number, index: number = 0, parentHue?: number) => {
    if (depth > 2) return null; // Only render up to 3 levels deep

    const color = getNodeColor(depth, index, parentHue);
    const hue = depth === 1 ? parseInt(color.match(/hsl\((\d+),/)?.[1] || '0') : parentHue;

    return (
      <group key={node.id}>
        <Node 
          node={node} 
          position={basePosition} 
          depth={depth} 
          onClick={handleNodeClick}
          isFocused={node.id === focusedNode.id}
          color={color}
        />
        {node.children.map((child, childIndex) => {
          const angle = (childIndex / node.children.length) * Math.PI * 2;
          const radius = spread;
          const childPosition: [number, number, number] = [
            basePosition[0] + Math.cos(angle) * radius,
            basePosition[1],
            basePosition[2] + Math.sin(angle) * radius
          ];
          return renderNodes(child, childPosition, spread * 0.6, depth + 1, childIndex, hue);
        })}
      </group>
    );
  };

  return (
    <>
      <ambientLight intensity={0.5} />
      <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} />
      <pointLight position={[-10, -10, -10]} />
      {renderNodes(focusedNode, [0, 0, 0], 5, 0)}
      <OrbitControls 
        ref={controlsRef}
        maxPolarAngle={Math.PI / 2}
        minDistance={2}
        maxDistance={20}
      />
      <Html fullscreen>
        <div style={{ position: 'absolute', top: '10px', left: '10px' }}>
          <button onClick={handleBackClick} disabled={nodeStack.length <= 1}>
            Back
          </button>
        </div>
      </Html>
    </>
  );
};

interface SceneGraphProps {
  root: SceneNode;
}

const SceneGraph: React.FC<SceneGraphProps> = ({ root }) => {
  return (
    <div style={{ width: '100%', height: '80vh' }}>
      <Canvas camera={{ position: [0, 15, 0], fov: 75 }}>
        <SceneContent root={root} />
      </Canvas>
    </div>
  );
};

export default SceneGraph;
