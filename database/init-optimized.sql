-- High-Performance Database Schema for Hive Trade v0.2
-- Optimized for real-time trading operations

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- Create optimized tables with proper indexing

-- Users table (minimal for trading focus)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    risk_profile JSONB DEFAULT '{}'::JSONB
);

-- Create index for user lookups
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_active ON users(is_active) WHERE is_active = TRUE;

-- Symbols table (trading instruments)
CREATE TABLE IF NOT EXISTS symbols (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(255),
    asset_type VARCHAR(50) NOT NULL, -- 'stock', 'crypto', 'forex', 'option'
    exchange VARCHAR(100),
    is_tradable BOOLEAN DEFAULT TRUE,
    min_quantity DECIMAL(20,8) DEFAULT 0.00000001,
    tick_size DECIMAL(20,8) DEFAULT 0.01,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::JSONB
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_symbols_symbol ON symbols(symbol);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_symbols_tradable ON symbols(is_tradable) WHERE is_tradable = TRUE;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_symbols_asset_type ON symbols(asset_type);

-- Orders table (optimized for high-frequency trading)
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    symbol_id INTEGER NOT NULL REFERENCES symbols(id),
    order_type VARCHAR(20) NOT NULL, -- 'market', 'limit', 'stop', 'stop_limit'
    side VARCHAR(10) NOT NULL, -- 'buy', 'sell'
    quantity DECIMAL(20,8) NOT NULL CHECK (quantity > 0),
    price DECIMAL(20,8), -- NULL for market orders
    stop_price DECIMAL(20,8), -- For stop orders
    time_in_force VARCHAR(10) DEFAULT 'GTC', -- 'GTC', 'IOC', 'FOK', 'DAY'
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'partially_filled', 'filled', 'cancelled', 'rejected'
    filled_quantity DECIMAL(20,8) DEFAULT 0,
    avg_fill_price DECIMAL(20,8),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    executed_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    client_order_id VARCHAR(100),
    broker_order_id VARCHAR(100),
    execution_time_ms INTEGER, -- Execution time in milliseconds
    fees DECIMAL(20,8) DEFAULT 0,
    metadata JSONB DEFAULT '{}'::JSONB
);

-- High-performance indexes for orders
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_user_created ON orders(user_id, created_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_symbol_status_created ON orders(symbol_id, status, created_at DESC) 
    WHERE status IN ('pending', 'partially_filled');
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_status_updated ON orders(status, updated_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_client_order_id ON orders(client_order_id) WHERE client_order_id IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_broker_order_id ON orders(broker_order_id) WHERE broker_order_id IS NOT NULL;

-- Partitioned trades table for high-volume data
CREATE TABLE IF NOT EXISTS trades (
    id UUID DEFAULT uuid_generate_v4(),
    order_id UUID NOT NULL REFERENCES orders(id),
    user_id UUID NOT NULL REFERENCES users(id),
    symbol_id INTEGER NOT NULL REFERENCES symbols(id),
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    value DECIMAL(20,8) GENERATED ALWAYS AS (quantity * price) STORED,
    fees DECIMAL(20,8) DEFAULT 0,
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    execution_time_ms INTEGER,
    trade_id VARCHAR(100), -- Broker trade ID
    metadata JSONB DEFAULT '{}'::JSONB,
    PRIMARY KEY (id, executed_at)
) PARTITION BY RANGE (executed_at);

-- Create monthly partitions for trades (last 6 months + next 6 months)
DO $$
DECLARE
    start_date DATE;
    end_date DATE;
    partition_name TEXT;
BEGIN
    -- Create partitions for the last 6 months and next 6 months
    FOR i IN -6..6 LOOP
        start_date := DATE_TRUNC('month', CURRENT_DATE) + (i || ' months')::INTERVAL;
        end_date := start_date + '1 month'::INTERVAL;
        partition_name := 'trades_' || TO_CHAR(start_date, 'YYYY_MM');
        
        EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF trades 
                       FOR VALUES FROM (%L) TO (%L)',
                       partition_name, start_date, end_date);
                       
        -- Create indexes on each partition
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%I_symbol_executed ON %I(symbol_id, executed_at DESC)',
                       partition_name, partition_name);
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%I_user_executed ON %I(user_id, executed_at DESC)',
                       partition_name, partition_name);
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%I_order_id ON %I(order_id)',
                       partition_name, partition_name);
    END LOOP;
END $$;

-- Positions table (real-time position tracking)
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    symbol_id INTEGER NOT NULL REFERENCES symbols(id),
    quantity DECIMAL(20,8) NOT NULL DEFAULT 0,
    avg_cost DECIMAL(20,8) NOT NULL DEFAULT 0,
    market_value DECIMAL(20,8),
    unrealized_pnl DECIMAL(20,8),
    realized_pnl DECIMAL(20,8) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_trade_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::JSONB,
    UNIQUE(user_id, symbol_id)
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_user_symbol ON positions(user_id, symbol_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_updated ON positions(updated_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_nonzero ON positions(user_id) WHERE quantity != 0;

-- Market data table (high-frequency price data)
CREATE TABLE IF NOT EXISTS market_data (
    id BIGSERIAL,
    symbol_id INTEGER NOT NULL REFERENCES symbols(id),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open_price DECIMAL(20,8),
    high_price DECIMAL(20,8),
    low_price DECIMAL(20,8),
    close_price DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8),
    bid_price DECIMAL(20,8),
    ask_price DECIMAL(20,8),
    bid_size DECIMAL(20,8),
    ask_size DECIMAL(20,8),
    vwap DECIMAL(20,8), -- Volume-weighted average price
    trade_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create daily partitions for market data (last 30 days + next 7 days)
DO $$
DECLARE
    start_date DATE;
    end_date DATE;
    partition_name TEXT;
BEGIN
    FOR i IN -30..7 LOOP
        start_date := CURRENT_DATE + (i || ' days')::INTERVAL;
        end_date := start_date + '1 day'::INTERVAL;
        partition_name := 'market_data_' || TO_CHAR(start_date, 'YYYY_MM_DD');
        
        EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF market_data 
                       FOR VALUES FROM (%L) TO (%L)',
                       partition_name, start_date, end_date);
                       
        -- Create indexes on each partition
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%I_symbol_timestamp ON %I(symbol_id, timestamp DESC)',
                       partition_name, partition_name);
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%I_timestamp ON %I(timestamp DESC)',
                       partition_name, partition_name);
    END LOOP;
END $$;

-- Portfolio metrics (aggregated performance data)
CREATE TABLE IF NOT EXISTS portfolio_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    total_value DECIMAL(20,8) NOT NULL DEFAULT 0,
    cash_balance DECIMAL(20,8) NOT NULL DEFAULT 0,
    positions_value DECIMAL(20,8) NOT NULL DEFAULT 0,
    total_pnl DECIMAL(20,8) NOT NULL DEFAULT 0,
    daily_pnl DECIMAL(20,8) NOT NULL DEFAULT 0,
    buying_power DECIMAL(20,8) NOT NULL DEFAULT 0,
    margin_used DECIMAL(20,8) NOT NULL DEFAULT 0,
    day_trade_count INTEGER DEFAULT 0,
    metrics JSONB DEFAULT '{}'::JSONB
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolio_metrics_user_timestamp ON portfolio_metrics(user_id, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolio_metrics_timestamp ON portfolio_metrics(timestamp DESC);

-- Risk metrics (real-time risk monitoring)
CREATE TABLE IF NOT EXISTS risk_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    portfolio_var DECIMAL(20,8), -- Value at Risk
    portfolio_beta DECIMAL(10,4),
    max_drawdown DECIMAL(10,4),
    sharpe_ratio DECIMAL(10,4),
    position_concentration DECIMAL(10,4), -- Largest position as % of portfolio
    leverage DECIMAL(10,4),
    margin_ratio DECIMAL(10,4),
    day_trading_buying_power DECIMAL(20,8),
    risk_score INTEGER CHECK (risk_score BETWEEN 1 AND 100),
    alerts JSONB DEFAULT '[]'::JSONB,
    metrics JSONB DEFAULT '{}'::JSONB
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_risk_metrics_user_calculated ON risk_metrics(user_id, calculated_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_risk_metrics_risk_score ON risk_metrics(risk_score);

-- AI agent performance tracking
CREATE TABLE IF NOT EXISTS ai_agent_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_name VARCHAR(100) NOT NULL,
    agent_version VARCHAR(50),
    user_id UUID REFERENCES users(id),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    trades_count INTEGER DEFAULT 0,
    win_rate DECIMAL(5,4), -- Between 0 and 1
    total_pnl DECIMAL(20,8),
    max_drawdown DECIMAL(10,4),
    avg_trade_duration INTEGER, -- In minutes
    confidence_score DECIMAL(5,4),
    strategy_parameters JSONB DEFAULT '{}'::JSONB,
    performance_metrics JSONB DEFAULT '{}'::JSONB
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ai_agent_performance_agent_timestamp ON ai_agent_performance(agent_name, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ai_agent_performance_user_timestamp ON ai_agent_performance(user_id, timestamp DESC);

-- System logs (application and trading events)
CREATE TABLE IF NOT EXISTS system_logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    level VARCHAR(20) NOT NULL, -- 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    logger_name VARCHAR(255),
    message TEXT NOT NULL,
    user_id UUID REFERENCES users(id),
    order_id UUID,
    trade_id UUID,
    symbol_id INTEGER REFERENCES symbols(id),
    execution_time_ms INTEGER,
    metadata JSONB DEFAULT '{}'::JSONB
);

-- Only keep logs for 30 days to manage storage
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_logs_level_timestamp ON system_logs(level, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_logs_user_timestamp ON system_logs(user_id, timestamp DESC) WHERE user_id IS NOT NULL;

-- Create a function to automatically clean old logs
CREATE OR REPLACE FUNCTION cleanup_old_logs() RETURNS void AS $$
BEGIN
    DELETE FROM system_logs WHERE timestamp < NOW() - INTERVAL '30 days';
    DELETE FROM market_data WHERE timestamp < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;

-- Performance monitoring views
CREATE OR REPLACE VIEW trading_performance_summary AS
SELECT 
    DATE_TRUNC('minute', t.executed_at) as minute,
    s.symbol,
    COUNT(*) as trade_count,
    AVG(t.execution_time_ms) as avg_execution_time,
    MAX(t.execution_time_ms) as max_execution_time,
    SUM(t.quantity * t.price) as volume,
    AVG(t.price) as avg_price
FROM trades t
JOIN symbols s ON t.symbol_id = s.id
WHERE t.executed_at > NOW() - INTERVAL '1 hour'
GROUP BY minute, s.symbol, t.symbol_id
ORDER BY minute DESC, volume DESC;

CREATE OR REPLACE VIEW active_orders_summary AS
SELECT 
    s.symbol,
    o.side,
    COUNT(*) as order_count,
    SUM(o.quantity - o.filled_quantity) as total_quantity,
    AVG(o.price) as avg_price,
    MIN(o.price) as min_price,
    MAX(o.price) as max_price
FROM orders o
JOIN symbols s ON o.symbol_id = s.id
WHERE o.status IN ('pending', 'partially_filled')
GROUP BY s.symbol, o.side, o.symbol_id
ORDER BY s.symbol, o.side;

CREATE OR REPLACE VIEW portfolio_summary AS
SELECT 
    u.username,
    COUNT(DISTINCT p.symbol_id) as positions_count,
    SUM(p.market_value) as total_market_value,
    SUM(p.unrealized_pnl) as total_unrealized_pnl,
    SUM(p.realized_pnl) as total_realized_pnl,
    pm.total_value as portfolio_value,
    pm.daily_pnl
FROM users u
LEFT JOIN positions p ON u.id = p.user_id
LEFT JOIN LATERAL (
    SELECT total_value, daily_pnl 
    FROM portfolio_metrics 
    WHERE user_id = u.id 
    ORDER BY timestamp DESC 
    LIMIT 1
) pm ON true
WHERE p.quantity != 0 OR pm.total_value IS NOT NULL
GROUP BY u.id, u.username, pm.total_value, pm.daily_pnl;

-- Create triggers for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert some initial data
INSERT INTO symbols (symbol, name, asset_type, exchange, tick_size) VALUES
('AAPL', 'Apple Inc.', 'stock', 'NASDAQ', 0.01),
('GOOGL', 'Alphabet Inc.', 'stock', 'NASDAQ', 0.01),
('TSLA', 'Tesla Inc.', 'stock', 'NASDAQ', 0.01),
('MSFT', 'Microsoft Corporation', 'stock', 'NASDAQ', 0.01),
('NVDA', 'NVIDIA Corporation', 'stock', 'NASDAQ', 0.01),
('BTC/USD', 'Bitcoin', 'crypto', 'Alpaca', 0.01),
('ETH/USD', 'Ethereum', 'crypto', 'Alpaca', 0.01),
('SPY', 'SPDR S&P 500 ETF', 'etf', 'NYSE', 0.01)
ON CONFLICT (symbol) DO NOTHING;

-- Create a test user
INSERT INTO users (username, email) VALUES 
('demo_trader', 'demo@hive-trade.local')
ON CONFLICT (username) DO NOTHING;

-- Performance optimization: Create statistics targets
ALTER TABLE orders ALTER COLUMN symbol_id SET STATISTICS 1000;
ALTER TABLE trades ALTER COLUMN symbol_id SET STATISTICS 1000;
ALTER TABLE market_data ALTER COLUMN symbol_id SET STATISTICS 1000;

-- Enable parallel query execution for large tables
ALTER TABLE trades SET (parallel_workers = 4);
ALTER TABLE market_data SET (parallel_workers = 4);
ALTER TABLE system_logs SET (parallel_workers = 2);

-- Analyze tables for optimal query planning
ANALYZE;

-- Create a function to get current trading statistics
CREATE OR REPLACE FUNCTION get_trading_stats()
RETURNS TABLE (
    active_orders BIGINT,
    filled_orders_today BIGINT,
    trades_today BIGINT,
    volume_today DECIMAL,
    avg_execution_time_ms DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        (SELECT COUNT(*) FROM orders WHERE status IN ('pending', 'partially_filled')) as active_orders,
        (SELECT COUNT(*) FROM orders WHERE status = 'filled' AND DATE(executed_at) = CURRENT_DATE) as filled_orders_today,
        (SELECT COUNT(*) FROM trades WHERE DATE(executed_at) = CURRENT_DATE) as trades_today,
        (SELECT COALESCE(SUM(quantity * price), 0) FROM trades WHERE DATE(executed_at) = CURRENT_DATE) as volume_today,
        (SELECT COALESCE(AVG(execution_time_ms), 0) FROM trades WHERE DATE(executed_at) = CURRENT_DATE) as avg_execution_time_ms;
END;
$$ LANGUAGE plpgsql;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'High-performance database schema for Hive Trade v0.2 initialized successfully!';
    RAISE NOTICE 'Database optimized for real-time trading operations.';
END $$;