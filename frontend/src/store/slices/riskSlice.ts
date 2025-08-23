import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RiskMetrics, RiskEvent } from '../../types';

interface RiskState {
  metrics: RiskMetrics | null;
  events: RiskEvent[];
  loading: boolean;
  lastUpdated: number;
}

const initialState: RiskState = {
  metrics: null,
  events: [],
  loading: false,
  lastUpdated: 0
};

const riskSlice = createSlice({
  name: 'risk',
  initialState,
  reducers: {
    updateRiskMetrics: (state, action: PayloadAction<RiskMetrics>) => {
      state.metrics = action.payload;
      state.lastUpdated = Date.now();
    },
    addRiskEvent: (state, action: PayloadAction<RiskEvent>) => {
      state.events.unshift(action.payload);
      // Keep only last 100 events
      if (state.events.length > 100) {
        state.events = state.events.slice(0, 100);
      }
      state.lastUpdated = Date.now();
    }
  }
});

export const { updateRiskMetrics, addRiskEvent } = riskSlice.actions;
export default riskSlice.reducer;