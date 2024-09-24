import React, { useEffect, useState } from 'react';
import SankeyChart from './components/SankeyChart';
import './App.css';

interface NodeDatum {
  name: string;
}

interface LinkDatum {
  source: number;
  target: number;
  value: number;
}

interface SankeyData {
  nodes: NodeDatum[];
  links: LinkDatum[];
}

const App: React.FC = () => {
  const [data, setData] = useState<SankeyData | null>(null);
  const [animate, setAnimate] = useState(true); // Add state for animation toggle

  useEffect(() => {
    fetch('/energy.json')
      .then(response => response.json())
      .then((rawData: SankeyData) => {
        setData(rawData);
      });
  }, []);

  return (
    <div className="App">
      <h1>Scene Graph Sankey Diagram</h1>
      <button onClick={() => setAnimate(!animate)}>
        {animate ? 'Disable Animations' : 'Enable Animations'}
      </button>
      {data && <SankeyChart data={data} animate={animate} />} {/* Pass animate prop */}
    </div>
  );
};

export default App;
