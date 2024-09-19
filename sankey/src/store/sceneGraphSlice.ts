import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { Node, SceneGraphUpdate } from '../types';

interface SceneGraphState {
  data: Node;
  updates: SceneGraphUpdate[];
}

const initialState: SceneGraphState = {
  data: {} as Node,
  updates: [],
};

const sceneGraphSlice = createSlice({
  name: 'sceneGraph',
  initialState,
  reducers: {
    setSceneGraphData: (state, action: PayloadAction<Node>) => {
      state.data = action.payload;
    },
    updateNodeParent: (state, action: PayloadAction<SceneGraphUpdate>) => {
      state.updates.push(action.payload);
      // Update the actual data structure
      updateNodeParentInTree(state.data, action.payload);
    },
  },
});

function updateNodeParentInTree(node: Node, update: SceneGraphUpdate) {
  if (node.id === update.newParentId) {
    const childNode = findAndRemoveNode(node, update.nodeId);
    if (childNode) {
      node.children.push(childNode);
    }
    return true;
  }
  for (const child of node.children) {
    if (updateNodeParentInTree(child, update)) {
      return true;
    }
  }
  return false;
}

function findAndRemoveNode(node: Node, nodeId: string): Node | null {
  for (let i = 0; i < node.children.length; i++) {
    if (node.children[i].id === nodeId) {
      return node.children.splice(i, 1)[0];
    }
    const foundNode = findAndRemoveNode(node.children[i], nodeId);
    if (foundNode) {
      return foundNode;
    }
  }
  return null;
}

export const { setSceneGraphData, updateNodeParent } = sceneGraphSlice.actions;
export default sceneGraphSlice.reducer;