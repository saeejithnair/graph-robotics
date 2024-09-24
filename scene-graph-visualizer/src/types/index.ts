// src/types/index.ts
export interface SceneNode {
    id: string;
    label: string;
    state?: string;
    position: [number, number, number];
    children?: SceneNode[];
  }
  