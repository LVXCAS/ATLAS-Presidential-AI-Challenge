-- Bloomberg Trading System TimescaleDB Schema
-- High-performance time-series database for trading data

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Market data tables
CREATE TABLE IF NOT EXISTS market_data (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(15,4),
    high DECIMAL(15,4),
    low DECIMAL(15,4),
    close DECIMAL(15,4),
    volume BIGINT,
    vwap DECIMAL(15,4),
    trade_count INTEGER,
    source VARCHAR(20) DEFAULT 'alpaca'
);

-- Convert to hypertable (TimescaleDB)
SELECT create_hypertable('market_data', 'timestamp', 'symbol', 4, if_not_exists => TRUE);

-- Level 2 order book data
CREATE TABLE IF NOT EXISTS order_book (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    side VARCHAR(4) NOT NULL, -- 'BID' or 'ASK'
    price DECIMAL(15,4) NOT NULL,
    size INTEGER NOT NULL,
    level_num INTEGER NOT NULL, -- 1-20 for L2 data
    exchange VARCHAR(10),
    source VARCHAR(20) DEFAULT 'alpaca'
);

SELECT create_hypertable('order_book', 'timestamp', 'symbol', 4, if_not_exists => TRUE);

-- Real-time tick data
CREATE TABLE IF NOT EXISTS tick_data (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    price DECIMAL(15,4) NOT NULL,
    size INTEGER NOT NULL,
    side VARCHAR(4), -- 'BUY', 'SELL', or NULL
    exchange VARCHAR(10),
    conditions TEXT[], -- Trade conditions
    source VARCHAR(20) DEFAULT 'alpaca'
);

SELECT create_hypertable('tick_data', 'timestamp', 'symbol', 4, if_not_exists => TRUE);

-- Options data
CREATE TABLE IF NOT EXISTS options_data (
    symbol VARCHAR(50) NOT NULL, -- AAPL240315C00150000
    underlying VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    bid DECIMAL(15,4),
    ask DECIMAL(15,4),
    last DECIMAL(15,4),
    volume BIGINT,
    open_interest BIGINT,
    delta DECIMAL(8,6),
    gamma DECIMAL(8,6),
    theta DECIMAL(8,6),
    vega DECIMAL(8,6),
    rho DECIMAL(8,6),
    implied_volatility DECIMAL(8,6),
    strike_price DECIMAL(15,4),
    expiry_date DATE,
    option_type VARCHAR(4), -- 'CALL' or 'PUT'
    source VARCHAR(20) DEFAULT 'polygon'
);

SELECT create_hypertable('options_data', 'timestamp', 'symbol', 4, if_not_exists => TRUE);

-- News and sentiment data
CREATE TABLE IF NOT EXISTS news_sentiment (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20),
    timestamp TIMESTAMPTZ NOT NULL,
    headline TEXT NOT NULL,
    summary TEXT,
    url VARCHAR(500),
    source VARCHAR(50),
    sentiment_score DECIMAL(5,4), -- -1 to 1
    sentiment_label VARCHAR(20), -- 'positive', 'negative', 'neutral'
    relevance_score DECIMAL(5,4), -- 0 to 1
    impact_score DECIMAL(5,4), -- 0 to 1
    keywords TEXT[],
    entities JSONB
);

SELECT create_hypertable('news_sentiment', 'timestamp', if_not_exists => TRUE);

-- Economic indicators
CREATE TABLE IF NOT EXISTS economic_data (
    indicator_name VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    value DECIMAL(15,6),
    previous_value DECIMAL(15,6),
    forecast_value DECIMAL(15,6),
    importance VARCHAR(10), -- 'LOW', 'MEDIUM', 'HIGH'
    country VARCHAR(3), -- ISO country code
    currency VARCHAR(3),
    frequency VARCHAR(20), -- 'DAILY', 'WEEKLY', 'MONTHLY', etc.
    source VARCHAR(50)
);

SELECT create_hypertable('economic_data', 'timestamp', if_not_exists => TRUE);

-- Feature store for ML features
CREATE TABLE IF NOT EXISTS features (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value DECIMAL(15,8),
    feature_metadata JSONB,
    version INTEGER DEFAULT 1
);

SELECT create_hypertable('features', 'timestamp', 'symbol', 4, if_not_exists => TRUE);

-- Agent signals and predictions
CREATE TABLE IF NOT EXISTS agent_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    signal_type VARCHAR(50), -- 'BUY', 'SELL', 'HOLD'
    confidence DECIMAL(5,4), -- 0 to 1
    strength DECIMAL(5,4), -- Signal strength
    reasoning JSONB, -- Explanatory factors
    features_used JSONB, -- Features that contributed to signal
    prediction_horizon INTEGER, -- Minutes ahead
    target_price DECIMAL(15,4),
    stop_loss DECIMAL(15,4),
    take_profit DECIMAL(15,4)
);

SELECT create_hypertable('agent_signals', 'timestamp', 'symbol', 4, if_not_exists => TRUE);

-- Trading positions
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    position_type VARCHAR(10), -- 'LONG', 'SHORT'
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(15,4) NOT NULL,
    current_price DECIMAL(15,4),
    unrealized_pnl DECIMAL(15,4),
    realized_pnl DECIMAL(15,4) DEFAULT 0,
    agent_name VARCHAR(100),
    strategy_name VARCHAR(100),
    risk_metrics JSONB
);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_order_id VARCHAR(100) UNIQUE,
    broker_order_id VARCHAR(100),
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    order_type VARCHAR(20), -- 'MARKET', 'LIMIT', 'STOP', etc.
    side VARCHAR(10), -- 'BUY', 'SELL'
    quantity INTEGER NOT NULL,
    price DECIMAL(15,4),
    stop_price DECIMAL(15,4),
    time_in_force VARCHAR(10), -- 'DAY', 'GTC', 'IOC', etc.
    status VARCHAR(20), -- 'NEW', 'FILLED', 'CANCELLED', etc.
    filled_quantity INTEGER DEFAULT 0,
    average_fill_price DECIMAL(15,4),
    commission DECIMAL(10,4) DEFAULT 0,
    agent_name VARCHAR(100),
    strategy_name VARCHAR(100),
    rejection_reason TEXT
);

-- Trades table (filled orders)
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID REFERENCES orders(id),
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(15,4) NOT NULL,
    commission DECIMAL(10,4) DEFAULT 0,
    pnl DECIMAL(15,4),
    agent_name VARCHAR(100),
    strategy_name VARCHAR(100)
);

SELECT create_hypertable('trades', 'timestamp', 'symbol', 4, if_not_exists => TRUE);

-- Portfolio performance metrics
CREATE TABLE IF NOT EXISTS portfolio_metrics (
    timestamp TIMESTAMPTZ NOT NULL,
    total_value DECIMAL(15,4),
    cash_balance DECIMAL(15,4),
    equity_value DECIMAL(15,4),
    buying_power DECIMAL(15,4),
    day_pnl DECIMAL(15,4),
    total_pnl DECIMAL(15,4),
    positions_count INTEGER,
    long_exposure DECIMAL(15,4),
    short_exposure DECIMAL(15,4),
    net_exposure DECIMAL(15,4),
    gross_exposure DECIMAL(15,4),
    beta DECIMAL(8,6),
    sharpe_ratio DECIMAL(8,6),
    max_drawdown DECIMAL(8,6),
    var_95 DECIMAL(15,4), -- Value at Risk 95%
    expected_shortfall DECIMAL(15,4)
);

SELECT create_hypertable('portfolio_metrics', 'timestamp', if_not_exists => TRUE);

-- Risk metrics and alerts
CREATE TABLE IF NOT EXISTS risk_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(50), -- 'BREACH', 'WARNING', 'INFO'
    risk_type VARCHAR(50), -- 'POSITION_LIMIT', 'VAR_BREACH', 'DRAWDOWN', etc.
    symbol VARCHAR(20),
    current_value DECIMAL(15,4),
    limit_value DECIMAL(15,4),
    severity VARCHAR(10), -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    description TEXT,
    action_taken VARCHAR(100),
    resolved_at TIMESTAMPTZ
);

SELECT create_hypertable('risk_events', 'timestamp', if_not_exists => TRUE);

-- System performance metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    timestamp TIMESTAMPTZ NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6),
    unit VARCHAR(20),
    component VARCHAR(50), -- 'API', 'DATABASE', 'REDIS', 'AGENT', etc.
    tags JSONB
);

SELECT create_hypertable('system_metrics', 'timestamp', if_not_exists => TRUE);

-- Create indexes for better query performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_symbol_time ON market_data (symbol, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tick_data_symbol_time ON tick_data (symbol, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_order_book_symbol_time ON order_book (symbol, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_options_underlying_time ON options_data (underlying, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_news_symbol_time ON news_sentiment (symbol, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_features_symbol_name_time ON features (symbol, feature_name, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_agent_symbol_time ON agent_signals (agent_name, symbol, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_symbol_status ON orders (symbol, status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_symbol_time ON trades (symbol, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_risk_events_type_time ON risk_events (risk_type, timestamp DESC);

-- Create views for common queries
CREATE OR REPLACE VIEW latest_market_data AS
SELECT DISTINCT ON (symbol) 
    symbol, timestamp, open, high, low, close, volume, vwap
FROM market_data 
ORDER BY symbol, timestamp DESC;

CREATE OR REPLACE VIEW active_positions AS
SELECT symbol, 
       SUM(CASE WHEN position_type = 'LONG' THEN quantity ELSE -quantity END) as net_position,
       AVG(entry_price) as avg_entry_price,
       SUM(unrealized_pnl) as total_unrealized_pnl
FROM positions 
GROUP BY symbol
HAVING SUM(CASE WHEN position_type = 'LONG' THEN quantity ELSE -quantity END) != 0;

CREATE OR REPLACE VIEW latest_signals AS
SELECT DISTINCT ON (agent_name, symbol)
    agent_name, symbol, timestamp, signal_type, confidence, strength, reasoning
FROM agent_signals
ORDER BY agent_name, symbol, timestamp DESC;

-- Set up data retention policies (keep 1 year of tick data, 5 years of daily data)
SELECT add_retention_policy('tick_data', INTERVAL '1 year', if_not_exists => TRUE);
SELECT add_retention_policy('order_book', INTERVAL '6 months', if_not_exists => TRUE);
SELECT add_retention_policy('system_metrics', INTERVAL '3 months', if_not_exists => TRUE);
SELECT add_retention_policy('news_sentiment', INTERVAL '2 years', if_not_exists => TRUE);

-- Create continuous aggregates for performance
CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_hourly
WITH (timescaledb.continuous) AS
SELECT symbol,
       time_bucket('1 hour', timestamp) AS bucket,
       first(open, timestamp) as open,
       max(high) as high,
       min(low) as low,
       last(close, timestamp) as close,
       sum(volume) as volume,
       avg(vwap) as vwap
FROM market_data
GROUP BY symbol, bucket;

-- Refresh policy for continuous aggregates
SELECT add_continuous_aggregate_policy('market_data_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '30 minutes',
    if_not_exists => TRUE);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_user;