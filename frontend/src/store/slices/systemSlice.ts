import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { SystemStatus } from '../../types';

interface SystemState {
  status: SystemStatus | null;
  loading: boolean;
  lastUpdated: number;
}

const initialState: SystemState = {
  status: null,
  loading: false,
  lastUpdated: 0
};

const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    updateSystemStatus: (state, action: PayloadAction<SystemStatus>) => {
      state.status = action.payload;
      state.lastUpdated = Date.now();
    }
  }
});

export const { updateSystemStatus } = systemSlice.actions;
export default systemSlice.reducer;