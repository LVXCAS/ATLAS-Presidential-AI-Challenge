// @ts-nocheck
/**
 * WebSocket Service for Real-time Trading Terminal
 * Connects to the backend WebSocket for live market data, agent signals, and system updates
 */

import { v4 as uuidv4 } from 'uuid';

export interface MarketDataMessage {
  type: 'market_data';
  symbol: string;
  data: {
    price: number;
    change: number;
    changePercent: number;
    volume: number;
    high: number;
    low: number;
    open: number;
    vwap?: number;
    iv?: number;
    timestamp: string;
  };
  timestamp: number;
  initial?: boolean;
}

export interface AgentSignalMessage {
  type: 'agent_signal';
  agent: string;
  symbol: string;
  data: {
    signal: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    price_target?: number;
    stop_loss?: number;
    position_size?: number;
  };
  timestamp: number;
}

export interface PortfolioUpdateMessage {
  type: 'portfolio_update';
  data: {
    total_value: number;
    pnl: number;
    exposure: number;
    positions: any[];
  };
  timestamp: number;
}

export interface RiskAlertMessage {
  type: 'risk_alert';
  data: {
    alert_type: string;
    severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
    message: string;
    value?: number;
    threshold?: number;
  };
  severity: string;
  timestamp: number;
}

export interface SystemStatusMessage {
  type: 'system_status';
  data: {
    status: 'HEALTHY' | 'DEGRADED' | 'ERROR';
    services: Record<string, boolean>;
    latency: number;
    memory_usage: number;
    cpu_usage: number;
  };
  timestamp: number;
}

export interface ConnectionMessage {
  type: 'connection';
  status: 'connected' | 'disconnected';
  client_id: string;
  server_time: number;
  message: string;
}

export interface PingMessage {
  type: 'ping';
  timestamp: number;
}

export interface NewsUpdateMessage {
  type: 'news_update';
  data: {
    title: string;
    summary: string;
    sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
    impact: 'HIGH' | 'MEDIUM' | 'LOW';
    timestamp: string;
    source: string;
    symbols: string[];
    tags: string[];
    sentiment_score: number;
    confidence: number;
    reach: number;
    virality: number;
  };
  timestamp: number;
}

export interface SentimentUpdateMessage {
  type: 'sentiment_update';
  data: {
    overall?: number;
    sectors?: Record<string, number>;
    fear_greed?: number;
  };
  timestamp: number;
}

export interface SocialBuzzMessage {
  type: 'social_buzz';
  data: Array<{
    symbol: string;
    platform: string;
    mentions: number;
    sentiment: number;
  }>;
  timestamp: number;
}

export interface PerformanceUpdateMessage {
  type: 'performance_update';
  data: {
    metrics?: Array<{
      label: string;
      value: string;
      change: string;
      positive: boolean | null;
    }>;
    equity_curve?: number[];
    strategy_performance?: Array<{
      name: string;
      active: boolean;
      pnl: number;
      winRate: number;
      sharpe: number;
      trades: number;
      maxDrawdown: number;
      allocation: number;
    }>;
  };
  timestamp: number;
}

export interface TradeExecutionMessage {
  type: 'trade_execution';
  data: {
    symbol: string;
    side: 'BUY' | 'SELL';
    quantity: number;
    price: number;
    time: string;
    strategy: string;
    pnl: number;
  };
  timestamp: number;
}

export interface OrderUpdateMessage {
  type: 'order_update';
  data: {
    id: string;
    symbol: string;
    side: 'BUY' | 'SELL';
    quantity: number;
    price: string | number;
    filled: number;
    remaining: number;
    type: string;
    time: string;
    status: 'NEW' | 'PARTIALLY_FILLED' | 'FILLED' | 'CANCELLED';
  };
  timestamp: number;
}

export type WebSocketMessage = 
  | MarketDataMessage 
  | AgentSignalMessage 
  | PortfolioUpdateMessage 
  | RiskAlertMessage 
  | SystemStatusMessage 
  | ConnectionMessage 
  | PingMessage
  | NewsUpdateMessage
  | SentimentUpdateMessage
  | SocialBuzzMessage
  | PerformanceUpdateMessage
  | TradeExecutionMessage
  | OrderUpdateMessage;

export interface WebSocketServiceConfig {
  url: string;
  autoReconnect: boolean;
  reconnectInterval: number;
  maxReconnectAttempts: number;
  pingInterval: number;
}

export class WebSocketService {
  private ws: WebSocket | null = null;
  private clientId: string;
  private config: WebSocketServiceConfig;
  private reconnectAttempts: number = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private pingTimer: NodeJS.Timeout | null = null;
  private isConnected: boolean = false;
  private subscribedSymbols: Set<string> = new Set();
  
  // Event handlers
  private messageHandlers: Map<string, ((message: any) => void)[]> = new Map();
  private connectionHandlers: ((connected: boolean) => void)[] = [];
  private errorHandlers: ((error: Error) => void)[] = [];

  constructor(config: Partial<WebSocketServiceConfig> = {}) {
    this.clientId = uuidv4();
    this.config = {
      url: 'ws://localhost:8000',
      autoReconnect: true,
      reconnectInterval: 5000,
      maxReconnectAttempts: 10,
      pingInterval: 30000,
      ...config
    };
  }

  /**
   * Connect to the WebSocket server
   */
  public connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const wsUrl = `${this.config.url}/ws/${this.clientId}`;
        console.log(`[WebSocket] Connecting to ${wsUrl}`);
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
          console.log(`[WebSocket] Connected as client ${this.clientId}`);
          this.isConnected = true;
          this.reconnectAttempts = 0;
          
          // Start ping timer
          this.startPingTimer();
          
          // Notify connection handlers
          this.connectionHandlers.forEach(handler => handler(true));
          
          // Resubscribe to symbols if reconnecting
          if (this.subscribedSymbols.size > 0) {
            this.subscribeToSymbols(Array.from(this.subscribedSymbols));
          }
          
          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event.data);
        };

        this.ws.onclose = (event) => {
          console.log(`[WebSocket] Connection closed: ${event.code} - ${event.reason}`);
          this.isConnected = false;
          this.cleanup();
          
          // Notify connection handlers
          this.connectionHandlers.forEach(handler => handler(false));
          
          // Auto-reconnect if enabled
          if (this.config.autoReconnect && this.reconnectAttempts < this.config.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          console.error(`[WebSocket] Error:`, error);
          const errorObj = new Error(`WebSocket connection failed`);
          this.errorHandlers.forEach(handler => handler(errorObj));
          reject(errorObj);
        };

      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Disconnect from the WebSocket server
   */
  public disconnect(): void {
    console.log('[WebSocket] Disconnecting...');
    this.config.autoReconnect = false;
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    
    this.cleanup();
    
    if (this.ws) {
      this.ws.close(1000, 'Manual disconnect');
      this.ws = null;
    }
  }

  /**
   * Subscribe to market data for specific symbols
   */
  public subscribeToSymbols(symbols: string[]): void {
    if (!this.isConnected || !this.ws) {
      console.warn('[WebSocket] Cannot subscribe: not connected');
      return;
    }

    const symbolsToSubscribe = symbols.filter(symbol => !this.subscribedSymbols.has(symbol));
    
    if (symbolsToSubscribe.length > 0) {
      const message = `SUBSCRIBE:${symbolsToSubscribe.join(',')}`;
      this.ws.send(message);
      
      symbolsToSubscribe.forEach(symbol => this.subscribedSymbols.add(symbol));
      console.log(`[WebSocket] Subscribed to symbols: ${symbolsToSubscribe.join(', ')}`);
    }
  }

  /**
   * Unsubscribe from market data for specific symbols
   */
  public unsubscribeFromSymbols(symbols: string[]): void {
    if (!this.isConnected || !this.ws) {
      console.warn('[WebSocket] Cannot unsubscribe: not connected');
      return;
    }

    const symbolsToUnsubscribe = symbols.filter(symbol => this.subscribedSymbols.has(symbol));
    
    if (symbolsToUnsubscribe.length > 0) {
      const message = `UNSUBSCRIBE:${symbolsToUnsubscribe.join(',')}`;
      this.ws.send(message);
      
      symbolsToUnsubscribe.forEach(symbol => this.subscribedSymbols.delete(symbol));
      console.log(`[WebSocket] Unsubscribed from symbols: ${symbolsToUnsubscribe.join(', ')}`);
    }
  }

  /**
   * Add message handler for specific message types
   */
  public onMessage<T extends WebSocketMessage>(
    messageType: T['type'], 
    handler: (message: T) => void
  ): void {
    if (!this.messageHandlers.has(messageType)) {
      this.messageHandlers.set(messageType, []);
    }
    this.messageHandlers.get(messageType)!.push(handler);
  }

  /**
   * Add connection status handler
   */
  public onConnection(handler: (connected: boolean) => void): void {
    this.connectionHandlers.push(handler);
  }

  /**
   * Add error handler
   */
  public onError(handler: (error: Error) => void): void {
    this.errorHandlers.push(handler);
  }

  /**
   * Remove all handlers
   */
  public removeAllHandlers(): void {
    this.messageHandlers.clear();
    this.connectionHandlers = [];
    this.errorHandlers = [];
  }

  /**
   * Get connection status
   */
  public getConnectionStatus(): { connected: boolean; clientId: string; subscribedSymbols: string[] } {
    return {
      connected: this.isConnected,
      clientId: this.clientId,
      subscribedSymbols: Array.from(this.subscribedSymbols)
    };
  }

  // Private methods

  private handleMessage(rawData: string): void {
    try {
      const message: WebSocketMessage = JSON.parse(rawData);
      
      // Handle ping messages
      if (message.type === 'ping') {
        this.ws?.send('PONG');
        return;
      }
      
      // Route message to appropriate handlers
      const handlers = this.messageHandlers.get(message.type);
      if (handlers) {
        handlers.forEach(handler => {
          try {
            handler(message);
          } catch (error) {
            console.error(`[WebSocket] Error in message handler for ${message.type}:`, error);
          }
        });
      } else {
        console.log(`[WebSocket] Received unhandled message type: ${message.type}`);
      }
      
    } catch (error) {
      console.error('[WebSocket] Error parsing message:', error, rawData);
    }
  }

  private startPingTimer(): void {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
    }
    
    this.pingTimer = setInterval(() => {
      if (this.isConnected && this.ws) {
        this.ws.send('PING');
      }
    }, this.config.pingInterval);
  }

  private cleanup(): void {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }

  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    console.log(`[WebSocket] Scheduling reconnect attempt ${this.reconnectAttempts}/${this.config.maxReconnectAttempts} in ${this.config.reconnectInterval}ms`);
    
    this.reconnectTimer = setTimeout(() => {
      console.log(`[WebSocket] Attempting to reconnect (${this.reconnectAttempts}/${this.config.maxReconnectAttempts})`);
      this.connect().catch(error => {
        console.error(`[WebSocket] Reconnect attempt failed:`, error);
      });
    }, this.config.reconnectInterval);
  }
}

// Create singleton instance
export const webSocketService = new WebSocketService({
  url: import.meta.env.VITE_WEBSOCKET_URL || 'ws://localhost:8000'
});

export default webSocketService;