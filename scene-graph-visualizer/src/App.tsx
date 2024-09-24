// src/App.tsx
import React from 'react';
import SceneGraph from './components/SceneGraph';
import { generateMockSceneGraph } from './utils/mockSceneGraph';

const App: React.FC = () => {
  const mockSceneGraph = generateMockSceneGraph();

  return (
    <div className="App">
      <h1>3D Scene Graph Visualizer</h1>
      <SceneGraph root={mockSceneGraph} />
    </div>
  );
};

export default App;
