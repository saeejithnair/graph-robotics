import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { Node } from '../types';

interface SceneGraphState {
  data: Node;
}

const initialState: SceneGraphState = {
  data: {} as Node,
};

const sceneGraphSlice = createSlice({
  name: 'sceneGraph',
  initialState,
  reducers: {
    setSceneGraphData: (state, action: PayloadAction<Node>) => {
      state.data = action.payload;
    },
  },
});

export const { setSceneGraphData } = sceneGraphSlice.actions;

export default sceneGraphSlice.reducer;