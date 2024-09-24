// src/types.ts
export interface SceneNode {
    id: string;
    label: string;
    description: string;
    position: [number, number, number];
    children: SceneNode[];
}
