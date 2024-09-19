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

  useEffect(() => {
    fetch('/energy.json')
      .then(response => response.json())
      .then((rawData: SankeyData) => {
        setData(rawData);
      });
  }, []);

  return (
    <div className="App">
      <h1>Energy Flow Sankey Diagram</h1>
      {data && <SankeyChart data={data} />}
    </div>
  );
};

export default App;
