/**
 * Redux Store Configuration for Bloomberg Terminal
 */

import { configureStore } from '@reduxjs/toolkit';
import marketDataReducer from './slices/marketDataSlice';
import portfolioReducer from './slices/portfolioSlice';
import ordersReducer from './slices/ordersSlice';
import riskReducer from './slices/riskSlice';
import agentsReducer from './slices/agentsSlice';
import systemReducer from './slices/systemSlice';
import uiReducer from './slices/uiSlice';

export const store = configureStore({
  reducer: {
    marketData: marketDataReducer,
    portfolio: portfolioReducer,
    orders: ordersReducer,
    risk: riskReducer,
    agents: agentsReducer,
    system: systemReducer,
    ui: uiReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST'],
        ignoredPaths: ['register'],
      },
    }),
  devTools: process.env.NODE_ENV !== 'production',
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;