import { create } from 'zustand';
import type { MarketData, Position, Order, AgentSignal, RiskMetrics, NewsItem, SystemStatus, TrainingStatus, AgentPerformance } from '../types';

interface TradingState {
  // Market Data
  marketData: { [symbol: string]: MarketData };
  positions: Position[];
  orders: Order[];
  
  // AI/ML Data
  agentSignals: AgentSignal[];
  agentPerformance: AgentPerformance[];
  trainingStatus: TrainingStatus;
  
  // Risk & Analytics
  riskMetrics: RiskMetrics | null;
  portfolioPnL: number;
  
  // News & Sentiment
  newsItems: NewsItem[];
  
  // System Status
  systemStatus: SystemStatus | null;
  connected: boolean;
  
  // WebSocket
  ws: WebSocket | null;
  
  // Actions
  setMarketData: (data: { [symbol: string]: MarketData }) => void;
  setPositions: (positions: Position[]) => void;
  setOrders: (orders: Order[]) => void;
  addAgentSignal: (signal: AgentSignal) => void;
  setRiskMetrics: (metrics: RiskMetrics) => void;
  setSystemStatus: (status: SystemStatus) => void;
  setConnected: (connected: boolean) => void;
  initWebSocket: () => void;
  placeOrder: (symbol: string, side: 'BUY' | 'SELL', quantity: number) => Promise<void>;
  toggleAgent: (agentName: string) => Promise<void>;
}

export const useTradingStore = create<TradingState>((set, get) => ({
  // Initial state
  marketData: {},
  positions: [],
  orders: [],
  agentSignals: [],
  agentPerformance: [],
  trainingStatus: {
    is_training: false,
    current_epoch: 0,
    total_epochs: 0,
    training_loss: 0,
    validation_loss: 0,
    model_accuracy: 0,
    eta_completion: '',
    best_model_version: ''
  },
  riskMetrics: null,
  portfolioPnL: 0,
  newsItems: [],
  systemStatus: null,
  connected: false,
  ws: null,

  // Actions
  setMarketData: (data) => set({ marketData: data }),
  setPositions: (positions) => set({ positions }),
  setOrders: (orders) => set({ orders }),
  addAgentSignal: (signal) => set((state) => ({ 
    agentSignals: [...state.agentSignals.slice(-99), signal] 
  })),
  setRiskMetrics: (metrics) => set({ riskMetrics: metrics }),
  setSystemStatus: (status) => set({ systemStatus: status }),
  setConnected: (connected) => set({ connected }),

  initWebSocket: () => {
    const ws = new WebSocket('ws://localhost:8000/ws');
    
    ws.onopen = () => {
      set({ connected: true });
      console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.market_data) {
          get().setMarketData(data.market_data);
        }
        
        if (data.positions) {
          get().setPositions(data.positions);
        }
        
        if (data.agent_signals) {
          data.agent_signals.forEach((signal: AgentSignal) => {
            get().addAgentSignal(signal);
          });
        }
        
        if (data.risk_metrics) {
          get().setRiskMetrics(data.risk_metrics);
        }
        
        if (data.system_status) {
          get().setSystemStatus(data.system_status);
        }
      } catch (error) {
        console.error('WebSocket message error:', error);
      }
    };
    
    ws.onclose = () => {
      set({ connected: false });
      // Reconnect after 5 seconds
      setTimeout(() => {
        get().initWebSocket();
      }, 5000);
    };
    
    set({ ws });
  },

  placeOrder: async (symbol, side, quantity) => {
    try {
      const response = await fetch('http://localhost:8000/api/orders', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, side, quantity })
      });
      
      if (!response.ok) {
        throw new Error('Order failed');
      }
    } catch (error) {
      console.error('Order placement error:', error);
    }
  },

  toggleAgent: async (agentName) => {
    try {
      await fetch(`http://localhost:8000/api/agents/${agentName}/toggle`, {
        method: 'POST'
      });
    } catch (error) {
      console.error('Agent toggle error:', error);
    }
  }
}));