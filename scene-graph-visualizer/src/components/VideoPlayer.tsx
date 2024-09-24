// src/components/VideoPlayer.tsx
import React, { useRef } from 'react';
import { useSceneGraphStore, SceneGraphState } from '../store/useSceneGraphStore';
import { mapSpatialDataToNodes } from '../utils/mapSpatialData';

const VideoPlayer: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const setNodes = useSceneGraphStore((state) => state.setNodes);
  const setSceneGraph = useSceneGraphStore((state: SceneGraphState) => state.setSceneGraph);

  const handleFrameChange = () => {
    const currentFrameData = {}; // Get spatial data for the current frame
    const nodes = mapSpatialDataToNodes(currentFrameData);
    setNodes(nodes);
  };

  return (
    <video
      ref={videoRef}
      src="path_to_video.mp4"
      onTimeUpdate={handleFrameChange}
      controls
    />
  );
};

export default VideoPlayer;
