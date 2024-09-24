// src/components/Controls.tsx
import { OrbitControls } from '@react-three/drei';
import React from 'react';

const Controls: React.FC = () => {
  return (
    <>
      <OrbitControls enablePan enableZoom enableRotate />
    </>
  );
};

export default Controls;
