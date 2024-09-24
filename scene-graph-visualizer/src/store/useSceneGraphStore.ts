// src/store/useSceneGraphStore.ts
import { create } from 'zustand';
import { SceneNode } from '../types';
import { generateMockSceneGraph } from '../utils/mockSceneGraph';

export interface SceneGraphState {
  sceneGraph: SceneNode | null;
  setSceneGraph: (sceneGraph: SceneNode) => void;
  setNodes: (nodes: SceneNode[]) => void;
}

export const useSceneGraphStore = create<SceneGraphState>((set) => ({
  sceneGraph: generateMockSceneGraph(),
  setSceneGraph: (sceneGraph: SceneNode) => set({ sceneGraph }),
  setNodes: (nodes: SceneNode[]) => set((state) => ({
    sceneGraph: state.sceneGraph ? { ...state.sceneGraph, children: nodes } : null
  })),
}));
