import React, { useEffect } from 'react';
import SceneGraphSankey from './components/SceneGraphSankey';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from './store/store';
import { setSceneGraphData } from './store/sceneGraphSlice'; // Update this line
import mockData from './mock_graph.json'; // Ensure this is the correct path

const App: React.FC = () => {
  const dispatch = useDispatch();
  const data = useSelector((state: RootState) => state.sceneGraph.data);

  useEffect(() => {
    dispatch(setSceneGraphData(mockData)); // Ensure this is being called
  }, [dispatch]);

  return (
    <div>
      <SceneGraphSankey width={800} height={600} />
    </div>
  );
};

export default App;