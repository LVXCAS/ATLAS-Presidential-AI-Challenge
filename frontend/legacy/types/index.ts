/**
 * Type definitions for Bloomberg Terminal Application
 */

// Market Data Types
export interface MarketData {
  symbol: string;
  price: number;
  timestamp: number;
  volume: number;
  change?: number;
  changePercent?: number;
  open?: number;
  high?: number;
  low?: number;
  close?: number;
  vwap?: number;
}

export interface Quote {
  symbol: string;
  bid: number;
  ask: number;
  bidSize: number;
  askSize: number;
  spread: number;
  timestamp: number;
}

export interface TickData {
  symbol: string;
  price: number;
  size: number;
  timestamp: number;
  side?: 'BUY' | 'SELL';
  exchange?: string;
}

export interface OHLCV {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// Technical Indicators
export interface TechnicalIndicators {
  symbol: string;
  timestamp: number;
  sma20?: number;
  sma50?: number;
  sma200?: number;
  ema12?: number;
  ema26?: number;
  rsi?: number;
  macd?: number;
  macdSignal?: number;
  macdHistogram?: number;
  bollinger?: {
    upper: number;
    middle: number;
    lower: number;
  };
  stochastic?: {
    k: number;
    d: number;
  };
  atr?: number;
  vwap?: number;
}

// Order Management Types
export enum OrderSide {
  BUY = 'BUY',
  SELL = 'SELL'
}

export enum OrderType {
  MARKET = 'MARKET',
  LIMIT = 'LIMIT',
  STOP = 'STOP',
  STOP_LIMIT = 'STOP_LIMIT'
}

export enum OrderStatus {
  PENDING_NEW = 'PENDING_NEW',
  NEW = 'NEW',
  PARTIALLY_FILLED = 'PARTIALLY_FILLED',
  FILLED = 'FILLED',
  CANCELLED = 'CANCELLED',
  REJECTED = 'REJECTED',
  EXPIRED = 'EXPIRED'
}

export interface Order {
  id: string;
  clientOrderId: string;
  brokerOrderId?: string;
  symbol: string;
  side: OrderSide;
  quantity: number;
  orderType: OrderType;
  price?: number;
  stopPrice?: number;
  timeInForce: string;
  status: OrderStatus;
  filledQuantity: number;
  averageFillPrice?: number;
  commission: number;
  createdAt: string;
  updatedAt: string;
  agentName?: string;
  strategyName?: string;
  rejectionReason?: string;
}

export interface Trade {
  id: string;
  orderId: string;
  symbol: string;
  side: OrderSide;
  quantity: number;
  price: number;
  commission: number;
  timestamp: string;
  agentName?: string;
  strategyName?: string;
}

// Position Management
export interface Position {
  symbol: string;
  quantity: number;
  averagePrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  realizedPnL: number;
  totalPnL: number;
  positionValue: number;
  costBasis: number;
  percentOfPortfolio: number;
  dayChange: number;
  dayChangePercent: number;
}

// Portfolio Types
export interface PortfolioMetrics {
  totalValue: number;
  cashBalance: number;
  equityValue: number;
  buyingPower: number;
  dayPnL: number;
  totalPnL: number;
  positionsCount: number;
  longExposure: number;
  shortExposure: number;
  netExposure: number;
  grossExposure: number;
  beta?: number;
  sharpeRatio?: number;
  maxDrawdown?: number;
  var95?: number;
  expectedShortfall?: number;
  timestamp: number;
}

// Risk Management
export interface RiskMetrics {
  portfolioVaR: number;
  componentVaR: Record<string, number>;
  concentrationRisk: number;
  leverageRatio: number;
  correlationMatrix: Record<string, Record<string, number>>;
  stressTestResults: Record<string, number>;
  riskLimits: RiskLimits;
  breachedLimits: string[];
}

export interface RiskLimits {
  maxPositionSize: number;
  maxPortfolioVaR: number;
  maxConcentration: number;
  maxLeverage: number;
  maxDrawdown: number;
  maxDailyLoss: number;
}

export interface RiskEvent {
  id: string;
  timestamp: string;
  eventType: string;
  riskType: string;
  symbol?: string;
  currentValue: number;
  limitValue: number;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  description: string;
  actionTaken?: string;
  resolvedAt?: string;
}

// Agent Types
export interface Agent {
  name: string;
  type: string;
  status: 'ACTIVE' | 'INACTIVE' | 'ERROR';
  lastUpdate: string;
  performance: AgentPerformance;
  configuration: Record<string, any>;
}

export interface AgentPerformance {
  totalSignals: number;
  accurateSignals: number;
  accuracy: number;
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  averageWin: number;
  averageLoss: number;
  profitFactor: number;
}

export interface AgentSignal {
  id: string;
  agentName: string;
  symbol: string;
  timestamp: string;
  signalType: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  strength: number;
  reasoning: Record<string, any>;
  featuresUsed: Record<string, any>;
  predictionHorizon?: number;
  targetPrice?: number;
  stopLoss?: number;
  takeProfit?: number;
}

// News and Sentiment
export interface NewsItem {
  id: string;
  headline: string;
  summary?: string;
  url: string;
  source: string;
  timestamp: string;
  symbol?: string;
  sentimentScore: number;
  sentimentLabel: 'positive' | 'negative' | 'neutral';
  relevanceScore: number;
  impactScore: number;
  keywords: string[];
  entities: Record<string, any>;
}

// System Status
export interface SystemStatus {
  status: 'HEALTHY' | 'DEGRADED' | 'DOWN';
  components: Record<string, ComponentStatus>;
  timestamp: string;
  uptime: number;
}

export interface ComponentStatus {
  status: 'HEALTHY' | 'DEGRADED' | 'DOWN';
  latency?: number;
  errorRate?: number;
  lastError?: string;
  lastCheck: string;
}

// WebSocket Message Types
export interface WebSocketMessage {
  type: string;
  timestamp: number;
  data: any;
}

export interface ConnectionInfo {
  clientId: string;
  connected: boolean;
  reconnectCount: number;
  lastMessage?: number;
}

// UI State Types
export interface PanelLayout {
  id: string;
  type: string;
  position: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  minimized: boolean;
  maximized: boolean;
  configuration: Record<string, any>;
}

export interface GridLayout {
  columns: number;
  rows: number;
  panels: PanelLayout[];
}

// Command Line Types
export interface Command {
  input: string;
  output: string;
  timestamp: string;
  error?: boolean;
}

export interface CommandHistory {
  commands: Command[];
  currentIndex: number;
}

// Chart Types
export interface ChartConfiguration {
  symbol: string;
  timeframe: string;
  indicators: string[];
  overlays: string[];
  theme: 'dark' | 'light';
  autoScale: boolean;
}

export interface ChartData {
  candles: OHLCV[];
  volume: Array<{ timestamp: number; value: number }>;
  indicators: Record<string, Array<{ timestamp: number; value: number }>>;
}

// Watchlist Types
export interface WatchlistItem {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  bid?: number;
  ask?: number;
  timestamp: number;
}

export interface Watchlist {
  id: string;
  name: string;
  symbols: string[];
  items: WatchlistItem[];
  lastUpdated: string;
}

// Settings Types
export interface UserSettings {
  theme: string;
  layout: GridLayout;
  watchlists: Watchlist[];
  notifications: NotificationSettings;
  trading: TradingSettings;
  display: DisplaySettings;
}

export interface NotificationSettings {
  enabled: boolean;
  riskAlerts: boolean;
  tradeExecutions: boolean;
  priceAlerts: boolean;
  systemAlerts: boolean;
  sound: boolean;
}

export interface TradingSettings {
  confirmOrders: boolean;
  defaultQuantity: number;
  defaultTimeInForce: string;
  riskChecks: boolean;
  paperTrading: boolean;
}

export interface DisplaySettings {
  fontSize: number;
  gridSize: number;
  showMilliseconds: boolean;
  priceDecimalPlaces: number;
  percentDecimalPlaces: number;
  flashUpdates: boolean;
}

// Error Types
export interface ApiError {
  message: string;
  code: string;
  details?: Record<string, any>;
}

// Utility Types
export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

export interface AsyncState<T> {
  data: T | null;
  loading: LoadingState;
  error: string | null;
  lastUpdated?: number;
}