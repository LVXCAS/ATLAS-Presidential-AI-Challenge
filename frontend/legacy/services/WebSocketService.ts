/**
 * WebSocket Service for Bloomberg Terminal
 * High-performance real-time data connection management
 */

import { io, Socket } from 'socket.io-client';

export type MessageHandler = (data: any) => void;
export type ConnectionHandler = (connected: boolean) => void;

export interface WebSocketConfig {
  url: string;
  autoReconnect: boolean;
  maxReconnectAttempts: number;
  reconnectInterval: number;
  pingInterval: number;
}

export interface SubscriptionManager {
  subscriptions: Set<string>;
  messageHandlers: Map<string, Set<MessageHandler>>;
  connectionHandlers: Set<ConnectionHandler>;
}

export class WebSocketService {
  private socket: Socket | null = null;
  private config: WebSocketConfig;
  private subscriptionManager: SubscriptionManager;
  private reconnectCount = 0;
  private isConnecting = false;
  private connectionState: 'connected' | 'disconnected' | 'connecting' = 'disconnected';
  private clientId: string;
  private lastMessageTime = 0;
  private pingInterval?: NodeJS.Timeout;
  private reconnectTimeout?: NodeJS.Timeout;

  constructor(config: Partial<WebSocketConfig> = {}) {
    this.config = {
      url: config.url || 'ws://localhost:8001',
      autoReconnect: config.autoReconnect ?? true,
      maxReconnectAttempts: config.maxReconnectAttempts || 10,
      reconnectInterval: config.reconnectInterval || 5000,
      pingInterval: config.pingInterval || 30000
    };

    this.subscriptionManager = {
      subscriptions: new Set(),
      messageHandlers: new Map(),
      connectionHandlers: new Set()
    };

    this.clientId = this.generateClientId();
    
    // Auto-connect on instantiation
    this.connect();
  }

  private generateClientId(): string {
    return `bloomberg_terminal_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  public async connect(): Promise<void> {
    if (this.isConnecting || this.connectionState === 'connected') {
      return;
    }

    this.isConnecting = true;
    this.connectionState = 'connecting';
    this.notifyConnectionHandlers(false);

    try {
      console.log(`[WebSocket] Connecting to ${this.config.url}/ws/${this.clientId}...`);

      this.socket = io(this.config.url, {
        transports: ['websocket'],
        autoConnect: true,
        reconnection: false, // We handle reconnection ourselves
        timeout: 5000,
        query: {
          clientId: this.clientId
        }
      });

      this.setupEventListeners();
      
    } catch (error) {
      console.error('[WebSocket] Connection failed:', error);
      this.handleConnectionError(error);
    } finally {
      this.isConnecting = false;
    }
  }

  private setupEventListeners(): void {
    if (!this.socket) return;

    // Connection events
    this.socket.on('connect', () => {
      console.log(`[WebSocket] Connected with ID: ${this.clientId}`);
      this.connectionState = 'connected';
      this.reconnectCount = 0;
      this.notifyConnectionHandlers(true);
      this.startPinging();
      this.resubscribeAll();
    });

    this.socket.on('disconnect', (reason) => {
      console.log(`[WebSocket] Disconnected: ${reason}`);
      this.connectionState = 'disconnected';
      this.notifyConnectionHandlers(false);
      this.stopPinging();
      
      if (this.config.autoReconnect && reason !== 'io client disconnect') {
        this.scheduleReconnect();
      }
    });

    this.socket.on('connect_error', (error) => {
      console.error('[WebSocket] Connection error:', error);
      this.handleConnectionError(error);
    });

    // Message handlers
    this.socket.on('message', (message) => {
      this.handleMessage(message);
    });

    this.socket.on('market_data', (data) => {
      this.handleMessage({ type: 'market_data', data });
    });

    this.socket.on('quote_data', (data) => {
      this.handleMessage({ type: 'quote_data', data });
    });

    this.socket.on('indicators', (data) => {
      this.handleMessage({ type: 'indicators', data });
    });

    this.socket.on('portfolio_update', (data) => {
      this.handleMessage({ type: 'portfolio_update', data });
    });

    this.socket.on('risk_alert', (data) => {
      this.handleMessage({ type: 'risk_alert', data });
    });

    this.socket.on('agent_signal', (data) => {
      this.handleMessage({ type: 'agent_signal', data });
    });

    this.socket.on('system_status', (data) => {
      this.handleMessage({ type: 'system_status', data });
    });

    this.socket.on('pong', () => {
      // Handle pong response
      this.lastMessageTime = Date.now();
    });

    this.socket.on('error', (error) => {
      console.error('[WebSocket] Socket error:', error);
    });
  }

  private handleMessage(message: any): void {
    try {
      this.lastMessageTime = Date.now();
      
      // Add timestamp if not present
      if (!message.timestamp) {
        message.timestamp = this.lastMessageTime;
      }

      // Route message to appropriate handlers
      const messageType = message.type || 'unknown';
      const handlers = this.subscriptionManager.messageHandlers.get(messageType);

      if (handlers && handlers.size > 0) {
        handlers.forEach(handler => {
          try {
            handler(message);
          } catch (error) {
            console.error(`[WebSocket] Error in message handler for ${messageType}:`, error);
          }
        });
      }

      // Also route to generic message handlers
      const genericHandlers = this.subscriptionManager.messageHandlers.get('*');
      if (genericHandlers && genericHandlers.size > 0) {
        genericHandlers.forEach(handler => {
          try {
            handler(message);
          } catch (error) {
            console.error('[WebSocket] Error in generic message handler:', error);
          }
        });
      }

    } catch (error) {
      console.error('[WebSocket] Error handling message:', error);
    }
  }

  private handleConnectionError(error: any): void {
    this.connectionState = 'disconnected';
    this.isConnecting = false;
    this.notifyConnectionHandlers(false);

    if (this.config.autoReconnect && this.reconnectCount < this.config.maxReconnectAttempts) {
      this.scheduleReconnect();
    } else {
      console.error('[WebSocket] Max reconnect attempts reached or auto-reconnect disabled');
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }

    const delay = this.config.reconnectInterval * Math.pow(2, Math.min(this.reconnectCount, 5)); // Exponential backoff
    this.reconnectCount++;

    console.log(`[WebSocket] Scheduling reconnect attempt ${this.reconnectCount} in ${delay}ms`);

    this.reconnectTimeout = setTimeout(() => {
      if (this.connectionState !== 'connected') {
        this.connect();
      }
    }, delay);
  }

  private startPinging(): void {
    this.stopPinging(); // Clear existing interval
    
    this.pingInterval = setInterval(() => {
      if (this.socket && this.connectionState === 'connected') {
        this.socket.emit('ping', { timestamp: Date.now() });
        
        // Check if we haven't received any messages recently
        if (Date.now() - this.lastMessageTime > this.config.pingInterval * 2) {
          console.warn('[WebSocket] No recent messages, connection might be stale');
        }
      }
    }, this.config.pingInterval);
  }

  private stopPinging(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = undefined;
    }
  }

  private notifyConnectionHandlers(connected: boolean): void {
    this.subscriptionManager.connectionHandlers.forEach(handler => {
      try {
        handler(connected);
      } catch (error) {
        console.error('[WebSocket] Error in connection handler:', error);
      }
    });
  }

  private resubscribeAll(): void {
    if (this.subscriptionManager.subscriptions.size === 0) return;

    const symbols = Array.from(this.subscriptionManager.subscriptions);
    console.log(`[WebSocket] Resubscribing to ${symbols.length} symbols`);
    this.subscribeToSymbols(symbols);
  }

  // Public API methods

  public subscribeToSymbols(symbols: string[]): void {
    if (!this.socket || this.connectionState !== 'connected') {
      console.warn('[WebSocket] Cannot subscribe - not connected');
      symbols.forEach(symbol => this.subscriptionManager.subscriptions.add(symbol));
      return;
    }

    const newSymbols = symbols.filter(symbol => !this.subscriptionManager.subscriptions.has(symbol));
    
    if (newSymbols.length === 0) return;

    newSymbols.forEach(symbol => this.subscriptionManager.subscriptions.add(symbol));
    
    this.socket.emit('message', `SUBSCRIBE:${newSymbols.join(',')}`);
    console.log(`[WebSocket] Subscribed to symbols: ${newSymbols.join(', ')}`);
  }

  public unsubscribeFromSymbols(symbols: string[]): void {
    if (!this.socket || this.connectionState !== 'connected') {
      symbols.forEach(symbol => this.subscriptionManager.subscriptions.delete(symbol));
      return;
    }

    const existingSymbols = symbols.filter(symbol => this.subscriptionManager.subscriptions.has(symbol));
    
    if (existingSymbols.length === 0) return;

    existingSymbols.forEach(symbol => this.subscriptionManager.subscriptions.delete(symbol));
    
    this.socket.emit('message', `UNSUBSCRIBE:${existingSymbols.join(',')}`);
    console.log(`[WebSocket] Unsubscribed from symbols: ${existingSymbols.join(', ')}`);
  }

  public addMessageHandler(messageType: string, handler: MessageHandler): void {
    if (!this.subscriptionManager.messageHandlers.has(messageType)) {
      this.subscriptionManager.messageHandlers.set(messageType, new Set());
    }
    this.subscriptionManager.messageHandlers.get(messageType)!.add(handler);
  }

  public removeMessageHandler(messageType: string, handler: MessageHandler): void {
    const handlers = this.subscriptionManager.messageHandlers.get(messageType);
    if (handlers) {
      handlers.delete(handler);
      if (handlers.size === 0) {
        this.subscriptionManager.messageHandlers.delete(messageType);
      }
    }
  }

  public addConnectionHandler(handler: ConnectionHandler): void {
    this.subscriptionManager.connectionHandlers.add(handler);
  }

  public removeConnectionHandler(handler: ConnectionHandler): void {
    this.subscriptionManager.connectionHandlers.delete(handler);
  }

  public sendMessage(message: any): void {
    if (!this.socket || this.connectionState !== 'connected') {
      console.warn('[WebSocket] Cannot send message - not connected');
      return;
    }

    this.socket.emit('message', message);
  }

  public disconnect(): void {
    console.log('[WebSocket] Manually disconnecting...');
    
    this.config.autoReconnect = false;
    this.stopPinging();
    
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }

    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }

    this.connectionState = 'disconnected';
    this.notifyConnectionHandlers(false);
  }

  // Getters
  public get connected(): boolean {
    return this.connectionState === 'connected';
  }

  public get connecting(): boolean {
    return this.connectionState === 'connecting';
  }

  public get subscribedSymbols(): string[] {
    return Array.from(this.subscriptionManager.subscriptions);
  }

  public getConnectionInfo() {
    return {
      clientId: this.clientId,
      connected: this.connected,
      reconnectCount: this.reconnectCount,
      lastMessage: this.lastMessageTime,
      subscriptions: this.subscribedSymbols.length
    };
  }
}

// Singleton instance
let webSocketService: WebSocketService | null = null;

export const getWebSocketService = (config?: Partial<WebSocketConfig>): WebSocketService => {
  if (!webSocketService) {
    webSocketService = new WebSocketService(config);
  }
  return webSocketService;
};

export default WebSocketService;