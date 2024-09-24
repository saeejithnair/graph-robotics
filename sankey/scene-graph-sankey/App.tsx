import React from 'react';
import SceneGraphSankey from './SceneGraphSankey';

const App: React.FC = () => {
  const data = {
    // Your scene graph data here
  };

  return (
    <div>
      <SceneGraphSankey data={data} width={800} height={600} />
    </div>
  );
};

export default App;