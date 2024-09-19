import React, { useEffect } from 'react';
import { useDispatch } from 'react-redux';
import SceneGraphSankey from './components/SceneGraphSankey';
import { setSceneGraphData, updateNodeParent } from './store/sceneGraphSlice';
import mockData from './mock_graph.json';

const App: React.FC = () => {
  const dispatch = useDispatch();

  useEffect(() => {
    dispatch(setSceneGraphData(mockData));
  }, [dispatch]);

  const handleMoveNode = () => {
    // Example: Move a node to a new parent
    dispatch(updateNodeParent({ nodeId: 'sofa-cushion-1', newParentId: 'ground-floor-kitchen' }));
  };

  return (
    <div>
      <h1>Scene Graph Visualization</h1>
      <button onClick={handleMoveNode}>Move Node</button>
      <SceneGraphSankey width={800} height={600} />
    </div>
  );
};

export default App;