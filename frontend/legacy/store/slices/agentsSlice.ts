import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { Agent, AgentSignal } from '../../types';

interface AgentsState {
  agents: Record<string, Agent>;
  signals: AgentSignal[];
  loading: boolean;
  lastUpdated: number;
}

const initialState: AgentsState = {
  agents: {},
  signals: [],
  loading: false,
  lastUpdated: 0
};

const agentsSlice = createSlice({
  name: 'agents',
  initialState,
  reducers: {
    updateAgent: (state, action: PayloadAction<Agent>) => {
      state.agents[action.payload.name] = action.payload;
      state.lastUpdated = Date.now();
    },
    addSignal: (state, action: PayloadAction<AgentSignal>) => {
      state.signals.unshift(action.payload);
      // Keep only last 500 signals
      if (state.signals.length > 500) {
        state.signals = state.signals.slice(0, 500);
      }
      state.lastUpdated = Date.now();
    }
  }
});

export const { updateAgent, addSignal } = agentsSlice.actions;
export default agentsSlice.reducer;