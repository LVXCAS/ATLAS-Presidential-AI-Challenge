import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { Order, Trade } from '../../types';

interface OrdersState {
  orders: Record<string, Order>;
  trades: Trade[];
  loading: boolean;
  lastUpdated: number;
}

const initialState: OrdersState = {
  orders: {},
  trades: [],
  loading: false,
  lastUpdated: 0
};

const ordersSlice = createSlice({
  name: 'orders',
  initialState,
  reducers: {
    updateOrder: (state, action: PayloadAction<Order>) => {
      state.orders[action.payload.id] = action.payload;
      state.lastUpdated = Date.now();
    },
    addTrade: (state, action: PayloadAction<Trade>) => {
      state.trades.unshift(action.payload);
      // Keep only last 1000 trades
      if (state.trades.length > 1000) {
        state.trades = state.trades.slice(0, 1000);
      }
      state.lastUpdated = Date.now();
    }
  }
});

export const { updateOrder, addTrade } = ordersSlice.actions;
export default ordersSlice.reducer;