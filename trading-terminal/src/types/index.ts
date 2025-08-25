export interface MarketData {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  high: number;
  low: number;
  open: number;
  vwap: number;
  timestamp: string;
}

export interface Position {
  symbol: string;
  quantity: number;
  avg_cost: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_percent: number;
  market_value: number;
}

export interface Order {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  order_type: 'MARKET' | 'LIMIT';
  status: 'PENDING' | 'FILLED' | 'CANCELLED' | 'REJECTED';
  timestamp: string;
  agent?: string;
}

export interface AgentSignal {
  agent_name: string;
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  reasoning: string;
  timestamp: string;
  features_used: string[];
  expected_return: number;
  risk_score: number;
}

export interface AgentPerformance {
  agent_name: string;
  total_pnl: number;
  win_rate: number;
  sharpe_ratio: number;
  max_drawdown: number;
  trades_count: number;
  avg_holding_period: number;
  accuracy: number;
}

export interface RiskMetrics {
  portfolio_var: number;
  portfolio_cvar: number;
  max_drawdown: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  correlation_matrix: { [key: string]: { [key: string]: number } };
  sector_exposure: { [key: string]: number };
  concentration_risk: number;
}

export interface NewsItem {
  title: string;
  content: string;
  sentiment_score: number;
  source: string;
  symbols_mentioned: string[];
  timestamp: string;
  relevance_score: number;
  impact_prediction: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL';
}

export interface SystemStatus {
  agents_active: { [key: string]: boolean };
  data_feeds_connected: { [key: string]: boolean };
  system_latency: number;
  orders_per_second: number;
  memory_usage: number;
  cpu_usage: number;
  uptime: string;
  last_error?: string;
}

export interface TrainingStatus {
  is_training: boolean;
  current_epoch: number;
  total_epochs: number;
  training_loss: number;
  validation_loss: number;
  model_accuracy: number;
  eta_completion: string;
  best_model_version: string;
}