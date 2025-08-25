-- LangGraph Adaptive Multi-Strategy AI Trading System Database Schema
-- Comprehensive production-ready schema with time-series optimization
-- Supports 50,000+ symbols across global markets with high-frequency data

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- ============================================================================
-- MARKET DATA TABLES (Time-Series Optimized)
-- ============================================================================

-- High-frequency market data (1-minute bars and ticks)
CREATE TABLE IF NOT EXISTS market_data_hf (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    timeframe VARCHAR(10) NOT NULL, -- '1m', '5m', '15m', '1h', '1d'
    open DECIMAL(15,8) NOT NULL,
    high DECIMAL(15,8) NOT NULL,
    low DECIMAL(15,8) NOT NULL,
    close DECIMAL(15,8) NOT NULL,
    volume BIGINT NOT NULL,
    vwap DECIMAL(15,8),
    trades_count INTEGER,
    bid DECIMAL(15,8),
    ask DECIMAL(15,8),
    bid_size BIGINT,
    ask_size BIGINT,
    spread DECIMAL(15,8),
    provider VARCHAR(20) NOT NULL,
    quality_score DECIMAL(5,4) DEFAULT 1.0,
    data_flags INTEGER DEFAULT 0, -- Bitfield for data quality flags
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, exchange, timeframe)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('market_data_hf', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Daily aggregated market data for long-term analysis
CREATE TABLE IF NOT EXISTS market_data_daily (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(15,8) NOT NULL,
    high DECIMAL(15,8) NOT NULL,
    low DECIMAL(15,8) NOT NULL,
    close DECIMAL(15,8) NOT NULL,
    volume BIGINT NOT NULL,
    vwap DECIMAL(15,8),
    trades_count INTEGER,
    avg_spread DECIMAL(15,8),
    volatility DECIMAL(10,6),
    provider VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (date, symbol, exchange)
);

-- Options chain data
CREATE TABLE IF NOT EXISTS options_data (
    id BIGSERIAL,
    underlying_symbol VARCHAR(20) NOT NULL,
    option_symbol VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    expiration_date DATE NOT NULL,
    strike DECIMAL(15,8) NOT NULL,
    option_type CHAR(1) NOT NULL, -- 'C' or 'P'
    bid DECIMAL(15,8),
    ask DECIMAL(15,8),
    last DECIMAL(15,8),
    volume INTEGER,
    open_interest INTEGER,
    implied_volatility DECIMAL(10,6),
    delta DECIMAL(8,6),
    gamma DECIMAL(8,6),
    theta DECIMAL(8,6),
    vega DECIMAL(8,6),
    rho DECIMAL(8,6),
    provider VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, option_symbol)
);

SELECT create_hypertable('options_data', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Forex and cryptocurrency data
CREATE TABLE IF NOT EXISTS forex_crypto_data (
    id BIGSERIAL,
    pair VARCHAR(20) NOT NULL, -- 'EUR/USD', 'BTC/USD', etc.
    exchange VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    open DECIMAL(15,8) NOT NULL,
    high DECIMAL(15,8) NOT NULL,
    low DECIMAL(15,8) NOT NULL,
    close DECIMAL(15,8) NOT NULL,
    volume DECIMAL(20,8),
    quote_volume DECIMAL(20,8),
    provider VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, pair, exchange, timeframe)
);

SELECT create_hypertable('forex_crypto_data', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Time-series optimized indexes for market data
CREATE INDEX IF NOT EXISTS idx_market_data_hf_symbol_time ON market_data_hf(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_hf_exchange_time ON market_data_hf(exchange, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_hf_timeframe ON market_data_hf(timeframe, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_hf_provider ON market_data_hf(provider);
CREATE INDEX IF NOT EXISTS idx_market_data_hf_quality ON market_data_hf(quality_score) WHERE quality_score < 0.9;
CREATE INDEX IF NOT EXISTS idx_market_data_hf_composite ON market_data_hf(symbol, exchange, timeframe, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_market_data_daily_symbol ON market_data_daily(symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_daily_exchange ON market_data_daily(exchange, date DESC);

CREATE INDEX IF NOT EXISTS idx_options_underlying ON options_data(underlying_symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_options_expiration ON options_data(expiration_date, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_options_strike_type ON options_data(strike, option_type, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_forex_crypto_pair ON forex_crypto_data(pair, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_forex_crypto_exchange ON forex_crypto_data(exchange, timestamp DESC);

-- ============================================================================
-- SIGNAL GENERATION AND FUSION TABLES
-- ============================================================================

-- Raw signals from individual agents
CREATE TABLE IF NOT EXISTS signals (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    agent_name VARCHAR(50) NOT NULL,
    signal_type VARCHAR(30) NOT NULL,
    value DECIMAL(10,6) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    strength DECIMAL(5,4) NOT NULL,
    direction VARCHAR(10) NOT NULL, -- 'BUY', 'SELL', 'HOLD'
    entry_price DECIMAL(15,8),
    target_price DECIMAL(15,8),
    stop_loss DECIMAL(15,8),
    position_size DECIMAL(15,6),
    holding_period INTEGER, -- Expected holding period in minutes
    top_3_reasons JSONB NOT NULL,
    technical_indicators JSONB,
    fibonacci_levels JSONB,
    sentiment_data JSONB,
    timestamp TIMESTAMPTZ NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    market_regime VARCHAR(20),
    backtest_performance JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, agent_name, signal_type)
);

SELECT create_hypertable('signals', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Fused signals from Portfolio Allocator
CREATE TABLE IF NOT EXISTS fused_signals (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    final_signal DECIMAL(10,6) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    contributing_agents JSONB NOT NULL,
    signal_weights JSONB NOT NULL,
    conflict_resolution JSONB,
    top_3_reasons JSONB NOT NULL,
    fibonacci_confluence JSONB,
    cross_market_arbitrage JSONB,
    risk_adjusted_size DECIMAL(15,6),
    expected_return DECIMAL(10,6),
    expected_risk DECIMAL(10,6),
    model_version VARCHAR(20) NOT NULL,
    fusion_method VARCHAR(30) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('fused_signals', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Signal performance tracking
CREATE TABLE IF NOT EXISTS signal_performance (
    id BIGSERIAL PRIMARY KEY,
    signal_id BIGINT NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    agent_name VARCHAR(50) NOT NULL,
    signal_timestamp TIMESTAMPTZ NOT NULL,
    evaluation_timestamp TIMESTAMPTZ NOT NULL,
    actual_return DECIMAL(10,6),
    predicted_return DECIMAL(10,6),
    accuracy_score DECIMAL(5,4),
    profit_factor DECIMAL(8,4),
    hit_rate DECIMAL(5,4),
    max_favorable_excursion DECIMAL(10,6),
    max_adverse_excursion DECIMAL(10,6),
    holding_period_actual INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for signals
CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON signals(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_agent_time ON signals(agent_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_type_time ON signals(signal_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_direction ON signals(direction, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_confidence ON signals(confidence DESC, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_model_version ON signals(model_version);

CREATE INDEX IF NOT EXISTS idx_fused_signals_symbol ON fused_signals(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_fused_signals_confidence ON fused_signals(confidence DESC, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_fused_signals_direction ON fused_signals(direction, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_signal_performance_agent ON signal_performance(agent_name, evaluation_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signal_performance_symbol ON signal_performance(symbol, evaluation_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signal_performance_accuracy ON signal_performance(accuracy_score DESC);

-- ============================================================================
-- TRADING AND EXECUTION TABLES
-- ============================================================================

-- Orders table for order lifecycle management
CREATE TABLE IF NOT EXISTS orders (
    id BIGSERIAL,
    order_id VARCHAR(50) NOT NULL,
    parent_order_id VARCHAR(50), -- For child orders from algo execution
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL, -- 'BUY', 'SELL', 'SHORT', 'COVER'
    order_type VARCHAR(20) NOT NULL, -- 'MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT'
    quantity DECIMAL(15,6) NOT NULL,
    filled_quantity DECIMAL(15,6) DEFAULT 0,
    price DECIMAL(15,8),
    stop_price DECIMAL(15,8),
    time_in_force VARCHAR(10) DEFAULT 'DAY', -- 'DAY', 'GTC', 'IOC', 'FOK'
    status VARCHAR(20) NOT NULL, -- 'PENDING', 'SUBMITTED', 'PARTIAL', 'FILLED', 'CANCELLED', 'REJECTED'
    strategy VARCHAR(30) NOT NULL,
    agent_name VARCHAR(50) NOT NULL,
    signal_id BIGINT,
    submitted_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    filled_at TIMESTAMPTZ,
    cancelled_at TIMESTAMPTZ,
    broker VARCHAR(20) NOT NULL,
    broker_order_id VARCHAR(100),
    execution_algo VARCHAR(30), -- 'TWAP', 'VWAP', 'IS', 'ARRIVAL'
    algo_params JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (submitted_at, order_id)
);

SELECT create_hypertable('orders', 'submitted_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Trades/fills table
CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL,
    trade_id VARCHAR(50) NOT NULL,
    order_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(15,6) NOT NULL,
    price DECIMAL(15,8) NOT NULL,
    executed_at TIMESTAMPTZ NOT NULL,
    strategy VARCHAR(30) NOT NULL,
    agent_name VARCHAR(50) NOT NULL,
    gross_pnl DECIMAL(15,6),
    net_pnl DECIMAL(15,6),
    commission DECIMAL(10,6),
    fees DECIMAL(10,6),
    market_impact DECIMAL(10,6),
    slippage DECIMAL(10,6),
    execution_quality_score DECIMAL(5,4),
    broker VARCHAR(20) NOT NULL,
    broker_trade_id VARCHAR(100),
    settlement_date DATE,
    currency VARCHAR(10) DEFAULT 'USD',
    fx_rate DECIMAL(15,8) DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (executed_at, trade_id)
);

SELECT create_hypertable('trades', 'executed_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Positions table for real-time position tracking
CREATE TABLE IF NOT EXISTS positions (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    strategy VARCHAR(30) NOT NULL,
    agent_name VARCHAR(50) NOT NULL,
    quantity DECIMAL(15,6) NOT NULL,
    avg_cost DECIMAL(15,8) NOT NULL,
    market_value DECIMAL(15,6) NOT NULL,
    unrealized_pnl DECIMAL(15,6) NOT NULL,
    realized_pnl DECIMAL(15,6) DEFAULT 0,
    first_trade_at TIMESTAMPTZ NOT NULL,
    last_trade_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, exchange, strategy, agent_name)
);

-- Portfolio snapshots for historical tracking
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    total_value DECIMAL(15,6) NOT NULL,
    cash DECIMAL(15,6) NOT NULL,
    long_value DECIMAL(15,6) NOT NULL,
    short_value DECIMAL(15,6) NOT NULL,
    gross_exposure DECIMAL(15,6) NOT NULL,
    net_exposure DECIMAL(15,6) NOT NULL,
    leverage DECIMAL(8,4) NOT NULL,
    num_positions INTEGER NOT NULL,
    largest_position_pct DECIMAL(5,4),
    sector_exposures JSONB,
    geographic_exposures JSONB,
    strategy_allocations JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp)
);

SELECT create_hypertable('portfolio_snapshots', 'timestamp',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Indexes for trading tables
CREATE INDEX IF NOT EXISTS idx_orders_symbol_time ON orders(symbol, submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_strategy_time ON orders(strategy, submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status, submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_broker ON orders(broker, submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_signal_id ON orders(signal_id) WHERE signal_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, executed_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_strategy_time ON trades(strategy, executed_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_agent_time ON trades(agent_name, executed_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_pnl ON trades(net_pnl DESC, executed_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_order_id ON trades(order_id);

CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy);
CREATE INDEX IF NOT EXISTS idx_positions_agent ON positions(agent_name);
CREATE INDEX IF NOT EXISTS idx_positions_pnl ON positions(unrealized_pnl DESC);

CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_time ON portfolio_snapshots(timestamp DESC);

-- ============================================================================
-- PERFORMANCE AND RISK MANAGEMENT TABLES
-- ============================================================================

-- Model performance tracking with detailed metrics
CREATE TABLE IF NOT EXISTS model_performance (
    id BIGSERIAL,
    model_name VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    strategy VARCHAR(30) NOT NULL,
    agent_name VARCHAR(50) NOT NULL,
    evaluation_period VARCHAR(20) NOT NULL, -- 'daily', 'weekly', 'monthly'
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    total_return DECIMAL(10,6),
    annualized_return DECIMAL(10,6),
    volatility DECIMAL(10,6),
    sharpe_ratio DECIMAL(8,4),
    sortino_ratio DECIMAL(8,4),
    calmar_ratio DECIMAL(8,4),
    information_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    max_drawdown_duration INTEGER, -- Days
    win_rate DECIMAL(5,4),
    profit_factor DECIMAL(8,4),
    avg_win DECIMAL(10,6),
    avg_loss DECIMAL(10,6),
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    largest_win DECIMAL(10,6),
    largest_loss DECIMAL(10,6),
    consecutive_wins INTEGER,
    consecutive_losses INTEGER,
    alpha DECIMAL(8,4),
    beta DECIMAL(8,4),
    r_squared DECIMAL(5,4),
    tracking_error DECIMAL(8,4),
    var_95 DECIMAL(10,6),
    cvar_95 DECIMAL(10,6),
    skewness DECIMAL(8,4),
    kurtosis DECIMAL(8,4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (end_date, model_name, version, strategy, evaluation_period)
);

-- Real-time risk metrics
CREATE TABLE IF NOT EXISTS risk_metrics (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    portfolio_value DECIMAL(15,6) NOT NULL,
    cash DECIMAL(15,6) NOT NULL,
    gross_exposure DECIMAL(15,6) NOT NULL,
    net_exposure DECIMAL(15,6) NOT NULL,
    leverage DECIMAL(8,4) NOT NULL,
    var_1d_95 DECIMAL(15,6),
    var_1d_99 DECIMAL(15,6),
    var_5d_95 DECIMAL(15,6),
    var_5d_99 DECIMAL(15,6),
    expected_shortfall_95 DECIMAL(15,6),
    expected_shortfall_99 DECIMAL(15,6),
    max_position_size DECIMAL(15,6),
    max_position_pct DECIMAL(5,4),
    sector_concentration DECIMAL(5,4),
    geographic_concentration DECIMAL(5,4),
    correlation_risk DECIMAL(8,4),
    liquidity_risk DECIMAL(8,4),
    currency_exposure JSONB,
    factor_exposures JSONB,
    stress_test_results JSONB,
    risk_limits_status JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp)
);

SELECT create_hypertable('risk_metrics', 'timestamp',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Risk limit breaches and alerts
CREATE TABLE IF NOT EXISTS risk_alerts (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    alert_type VARCHAR(30) NOT NULL, -- 'LIMIT_BREACH', 'VAR_EXCEEDED', 'DRAWDOWN', etc.
    severity VARCHAR(10) NOT NULL, -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    symbol VARCHAR(20),
    strategy VARCHAR(30),
    agent_name VARCHAR(50),
    current_value DECIMAL(15,6),
    limit_value DECIMAL(15,6),
    breach_percentage DECIMAL(8,4),
    description TEXT NOT NULL,
    action_taken VARCHAR(100),
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Factor exposures tracking
CREATE TABLE IF NOT EXISTS factor_exposures (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    factor_name VARCHAR(30) NOT NULL, -- 'MOMENTUM', 'VALUE', 'GROWTH', 'QUALITY', etc.
    exposure DECIMAL(10,6) NOT NULL,
    target_exposure DECIMAL(10,6),
    deviation DECIMAL(10,6),
    contribution_to_risk DECIMAL(10,6),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, factor_name)
);

SELECT create_hypertable('factor_exposures', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Indexes for performance and risk tables
CREATE INDEX IF NOT EXISTS idx_model_performance_model ON model_performance(model_name, version, end_date DESC);
CREATE INDEX IF NOT EXISTS idx_model_performance_strategy ON model_performance(strategy, end_date DESC);
CREATE INDEX IF NOT EXISTS idx_model_performance_sharpe ON model_performance(sharpe_ratio DESC, end_date DESC);
CREATE INDEX IF NOT EXISTS idx_model_performance_return ON model_performance(total_return DESC, end_date DESC);

CREATE INDEX IF NOT EXISTS idx_risk_metrics_time ON risk_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_leverage ON risk_metrics(leverage DESC, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_var ON risk_metrics(var_1d_95 DESC, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_risk_alerts_time ON risk_alerts(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_risk_alerts_severity ON risk_alerts(severity, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_risk_alerts_type ON risk_alerts(alert_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_risk_alerts_unresolved ON risk_alerts(timestamp DESC) WHERE resolved_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_factor_exposures_factor ON factor_exposures(factor_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_factor_exposures_deviation ON factor_exposures(ABS(deviation) DESC, timestamp DESC);

-- ============================================================================
-- NEWS, SENTIMENT, AND ALTERNATIVE DATA TABLES
-- ============================================================================

-- News articles and sentiment analysis
CREATE TABLE IF NOT EXISTS news_articles (
    id BIGSERIAL,
    article_id VARCHAR(100) NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    summary TEXT,
    source VARCHAR(50) NOT NULL,
    author VARCHAR(100),
    published_at TIMESTAMPTZ NOT NULL,
    url TEXT,
    symbols JSONB, -- Array of related symbols
    categories JSONB, -- Array of categories
    sentiment_score DECIMAL(5,4), -- -1 to 1
    sentiment_confidence DECIMAL(5,4),
    sentiment_breakdown JSONB, -- Detailed sentiment analysis
    market_impact_score DECIMAL(5,4),
    event_type VARCHAR(30), -- 'EARNINGS', 'FDA_APPROVAL', 'MERGER', etc.
    language VARCHAR(10) DEFAULT 'en',
    processed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (published_at, article_id)
);

SELECT create_hypertable('news_articles', 'published_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Social media sentiment data
CREATE TABLE IF NOT EXISTS social_sentiment (
    id BIGSERIAL,
    platform VARCHAR(20) NOT NULL, -- 'twitter', 'reddit', 'stocktwits'
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    sentiment_score DECIMAL(5,4) NOT NULL,
    volume INTEGER NOT NULL, -- Number of mentions
    bullish_mentions INTEGER,
    bearish_mentions INTEGER,
    neutral_mentions INTEGER,
    trending_score DECIMAL(5,4),
    viral_posts JSONB,
    top_keywords JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, platform, symbol)
);

SELECT create_hypertable('social_sentiment', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Alternative data sources
CREATE TABLE IF NOT EXISTS alternative_data (
    id BIGSERIAL,
    data_type VARCHAR(30) NOT NULL, -- 'SATELLITE', 'CREDIT_CARD', 'WEATHER', etc.
    symbol VARCHAR(20),
    sector VARCHAR(30),
    timestamp TIMESTAMPTZ NOT NULL,
    data_value DECIMAL(15,6),
    data_payload JSONB NOT NULL,
    confidence_score DECIMAL(5,4),
    provider VARCHAR(50) NOT NULL,
    processing_version VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, data_type, provider, COALESCE(symbol, sector, 'GLOBAL'))
);

SELECT create_hypertable('alternative_data', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- ============================================================================
-- SYSTEM MONITORING AND AUDIT TABLES
-- ============================================================================

-- Comprehensive system logs
CREATE TABLE IF NOT EXISTS system_logs (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    level VARCHAR(10) NOT NULL, -- 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'
    component VARCHAR(50) NOT NULL,
    agent_name VARCHAR(50),
    message TEXT NOT NULL,
    metadata JSONB,
    trace_id VARCHAR(50), -- For distributed tracing
    user_id VARCHAR(50),
    session_id VARCHAR(50),
    request_id VARCHAR(50),
    execution_time_ms INTEGER,
    memory_usage_mb INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, component, level)
);

SELECT create_hypertable('system_logs', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Agent performance and health metrics
CREATE TABLE IF NOT EXISTS agent_metrics (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    agent_name VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL, -- 'ACTIVE', 'IDLE', 'ERROR', 'MAINTENANCE'
    cpu_usage DECIMAL(5,2),
    memory_usage_mb INTEGER,
    messages_processed INTEGER,
    avg_response_time_ms INTEGER,
    error_count INTEGER,
    last_heartbeat TIMESTAMPTZ,
    version VARCHAR(20),
    configuration JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, agent_name)
);

SELECT create_hypertable('agent_metrics', 'timestamp',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Audit trail for all system actions
CREATE TABLE IF NOT EXISTS audit_trail (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    action_type VARCHAR(30) NOT NULL, -- 'TRADE', 'SIGNAL', 'CONFIG_CHANGE', etc.
    entity_type VARCHAR(30) NOT NULL, -- 'ORDER', 'POSITION', 'MODEL', etc.
    entity_id VARCHAR(100) NOT NULL,
    agent_name VARCHAR(50),
    user_id VARCHAR(50),
    before_state JSONB,
    after_state JSONB,
    action_metadata JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- BACKTESTING AND SIMULATION TABLES
-- ============================================================================

-- Backtest runs and results
CREATE TABLE IF NOT EXISTS backtest_runs (
    id BIGSERIAL PRIMARY KEY,
    run_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    strategy_config JSONB NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DECIMAL(15,6) NOT NULL,
    benchmark VARCHAR(20),
    status VARCHAR(20) NOT NULL, -- 'RUNNING', 'COMPLETED', 'FAILED'
    total_return DECIMAL(10,6),
    annualized_return DECIMAL(10,6),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    total_trades INTEGER,
    win_rate DECIMAL(5,4),
    execution_time_seconds INTEGER,
    created_by VARCHAR(50),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Backtest trade details
CREATE TABLE IF NOT EXISTS backtest_trades (
    id BIGSERIAL,
    run_id VARCHAR(50) NOT NULL,
    trade_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(15,6) NOT NULL,
    entry_price DECIMAL(15,8) NOT NULL,
    exit_price DECIMAL(15,8),
    entry_date DATE NOT NULL,
    exit_date DATE,
    strategy VARCHAR(30) NOT NULL,
    pnl DECIMAL(15,6),
    pnl_pct DECIMAL(8,4),
    holding_period INTEGER,
    commission DECIMAL(10,6),
    slippage DECIMAL(10,6),
    signal_strength DECIMAL(5,4),
    market_regime VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (entry_date, run_id, trade_id)
);

-- ============================================================================
-- INDEXES FOR MONITORING AND ALTERNATIVE DATA
-- ============================================================================

-- News and sentiment indexes
CREATE INDEX IF NOT EXISTS idx_news_articles_symbols ON news_articles USING GIN(symbols);
CREATE INDEX IF NOT EXISTS idx_news_articles_source ON news_articles(source, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_articles_sentiment ON news_articles(sentiment_score DESC, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_articles_impact ON news_articles(market_impact_score DESC, published_at DESC);

CREATE INDEX IF NOT EXISTS idx_social_sentiment_symbol ON social_sentiment(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_social_sentiment_platform ON social_sentiment(platform, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_social_sentiment_trending ON social_sentiment(trending_score DESC, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_alternative_data_type ON alternative_data(data_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alternative_data_symbol ON alternative_data(symbol, timestamp DESC) WHERE symbol IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_alternative_data_provider ON alternative_data(provider, timestamp DESC);

-- System monitoring indexes
CREATE INDEX IF NOT EXISTS idx_system_logs_level_time ON system_logs(level, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_logs_component_time ON system_logs(component, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_logs_agent_time ON system_logs(agent_name, timestamp DESC) WHERE agent_name IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_system_logs_trace ON system_logs(trace_id) WHERE trace_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_agent_metrics_agent ON agent_metrics(agent_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_agent_metrics_status ON agent_metrics(status, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_agent_metrics_errors ON agent_metrics(error_count DESC, timestamp DESC) WHERE error_count > 0;

CREATE INDEX IF NOT EXISTS idx_audit_trail_time ON audit_trail(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_trail_action ON audit_trail(action_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_trail_entity ON audit_trail(entity_type, entity_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_trail_agent ON audit_trail(agent_name, timestamp DESC) WHERE agent_name IS NOT NULL;

-- Backtest indexes
CREATE INDEX IF NOT EXISTS idx_backtest_runs_status ON backtest_runs(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_performance ON backtest_runs(sharpe_ratio DESC, total_return DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_run ON backtest_trades(run_id, entry_date DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_symbol ON backtest_trades(symbol, entry_date DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_pnl ON backtest_trades(pnl DESC, entry_date DESC);

-- ============================================================================
-- DATA RETENTION AND ARCHIVAL POLICIES
-- ============================================================================

-- Retention policies for time-series data
SELECT add_retention_policy('market_data_hf', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('options_data', INTERVAL '1 year', if_not_exists => TRUE);
SELECT add_retention_policy('forex_crypto_data', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('signals', INTERVAL '1 year', if_not_exists => TRUE);
SELECT add_retention_policy('fused_signals', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('orders', INTERVAL '7 years', if_not_exists => TRUE); -- Regulatory requirement
SELECT add_retention_policy('trades', INTERVAL '7 years', if_not_exists => TRUE); -- Regulatory requirement
SELECT add_retention_policy('portfolio_snapshots', INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('risk_metrics', INTERVAL '3 years', if_not_exists => TRUE);
SELECT add_retention_policy('news_articles', INTERVAL '1 year', if_not_exists => TRUE);
SELECT add_retention_policy('social_sentiment', INTERVAL '6 months', if_not_exists => TRUE);
SELECT add_retention_policy('alternative_data', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('system_logs', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('agent_metrics', INTERVAL '1 year', if_not_exists => TRUE);
SELECT add_retention_policy('factor_exposures', INTERVAL '3 years', if_not_exists => TRUE);

-- Compression policies for older data
SELECT add_compression_policy('market_data_hf', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('options_data', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('forex_crypto_data', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('signals', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('fused_signals', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('orders', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_compression_policy('trades', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_compression_policy('portfolio_snapshots', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('risk_metrics', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('news_articles', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('social_sentiment', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('alternative_data', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('system_logs', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('agent_metrics', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('factor_exposures', INTERVAL '90 days', if_not_exists => TRUE);

-- ============================================================================
-- INITIAL DATA AND CONFIGURATION
-- ============================================================================

-- Create initial model performance records
INSERT INTO model_performance (
    model_name, version, strategy, agent_name, evaluation_period,
    start_date, end_date, total_return, sharpe_ratio, max_drawdown, win_rate, total_trades
) VALUES 
    ('momentum_v1', '1.0.0', 'momentum', 'momentum_agent', 'daily', CURRENT_DATE - INTERVAL '30 days', CURRENT_DATE, 0.0, 0.0, 0.0, 0.0, 0),
    ('mean_reversion_v1', '1.0.0', 'mean_reversion', 'mean_reversion_agent', 'daily', CURRENT_DATE - INTERVAL '30 days', CURRENT_DATE, 0.0, 0.0, 0.0, 0.0, 0),
    ('sentiment_v1', '1.0.0', 'sentiment', 'sentiment_agent', 'daily', CURRENT_DATE - INTERVAL '30 days', CURRENT_DATE, 0.0, 0.0, 0.0, 0.0, 0),
    ('options_v1', '1.0.0', 'options', 'options_agent', 'daily', CURRENT_DATE - INTERVAL '30 days', CURRENT_DATE, 0.0, 0.0, 0.0, 0.0, 0),
    ('portfolio_fusion_v1', '1.0.0', 'fusion', 'portfolio_allocator', 'daily', CURRENT_DATE - INTERVAL '30 days', CURRENT_DATE, 0.0, 0.0, 0.0, 0.0, 0)
ON CONFLICT DO NOTHING;

-- Create initial system log entry
INSERT INTO system_logs (timestamp, level, component, message, metadata)
VALUES (NOW(), 'INFO', 'database', 'Database schema initialized successfully', 
        '{"version": "1.0.0", "tables_created": 25, "indexes_created": 75, "retention_policies": 15}')
ON CONFLICT DO NOTHING;