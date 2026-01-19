import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { Position, PortfolioMetrics } from '../../types';

interface PortfolioState {
  positions: Record<string, Position>;
  metrics: PortfolioMetrics | null;
  loading: boolean;
  lastUpdated: number;
}

const initialState: PortfolioState = {
  positions: {},
  metrics: null,
  loading: false,
  lastUpdated: 0
};

const portfolioSlice = createSlice({
  name: 'portfolio',
  initialState,
  reducers: {
    updatePositions: (state, action: PayloadAction<Position[]>) => {
      state.positions = {};
      action.payload.forEach(position => {
        state.positions[position.symbol] = position;
      });
      state.lastUpdated = Date.now();
    },
    updateMetrics: (state, action: PayloadAction<PortfolioMetrics>) => {
      state.metrics = action.payload;
      state.lastUpdated = Date.now();
    }
  }
});

export const { updatePositions, updateMetrics } = portfolioSlice.actions;
export default portfolioSlice.reducer;