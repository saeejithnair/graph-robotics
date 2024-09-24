import { configureStore } from '@reduxjs/toolkit';
import sceneGraphReducer from './sceneGraphSlice';

export const store = configureStore({
  reducer: {
    sceneGraph: sceneGraphReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;