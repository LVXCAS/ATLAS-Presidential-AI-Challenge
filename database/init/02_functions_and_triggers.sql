-- LangGraph Trading System - Database Functions and Triggers
-- Advanced database functions for performance optimization and data integrity

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

-- Function to calculate portfolio metrics
CREATE OR REPLACE FUNCTION calculate_portfolio_metrics(
    p_timestamp TIMESTAMPTZ DEFAULT NOW()
) RETURNS TABLE (
    total_value DECIMAL(15,6),
    cash DECIMAL(15,6),
    long_value DECIMAL(15,6),
    short_value DECIMAL(15,6),
    gross_exposure DECIMAL(15,6),
    net_exposure DECIMAL(15,6),
    leverage DECIMAL(8,4),
    num_positions INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COALESCE(SUM(p.market_value), 0) + COALESCE(cash_balance.amount, 0) as total_value,
        COALESCE(cash_balance.amount, 0) as cash,
        COALESCE(SUM(CASE WHEN p.quantity > 0 THEN p.market_value ELSE 0 END), 0) as long_value,
        COALESCE(SUM(CASE WHEN p.quantity < 0 THEN ABS(p.market_value) ELSE 0 END), 0) as short_value,
        COALESCE(SUM(ABS(p.market_value)), 0) as gross_exposure,
        COALESCE(SUM(p.market_value), 0) as net_exposure,
        CASE 
            WHEN COALESCE(cash_balance.amount, 0) > 0 
            THEN COALESCE(SUM(ABS(p.market_value)), 0) / cash_balance.amount
            ELSE 0 
        END as leverage,
        COUNT(p.id)::INTEGER as num_positions
    FROM positions p
    CROSS JOIN (
        SELECT 1000000.0 as amount -- Default cash balance, should be updated from broker
    ) cash_balance
    WHERE p.quantity != 0;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate signal performance metrics
CREATE OR REPLACE FUNCTION calculate_signal_performance(
    p_agent_name VARCHAR(50),
    p_lookback_days INTEGER DEFAULT 30
) RETURNS TABLE (
    agent_name VARCHAR(50),
    total_signals INTEGER,
    avg_accuracy DECIMAL(5,4),
    avg_return DECIMAL(10,6),
    sharpe_ratio DECIMAL(8,4),
    hit_rate DECIMAL(5,4)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p_agent_name,
        COUNT(*)::INTEGER as total_signals,
        AVG(sp.accuracy_score) as avg_accuracy,
        AVG(sp.actual_return) as avg_return,
        CASE 
            WHEN STDDEV(sp.actual_return) > 0 
            THEN AVG(sp.actual_return) / STDDEV(sp.actual_return)
            ELSE 0 
        END as sharpe_ratio,
        AVG(CASE WHEN sp.actual_return > 0 THEN 1.0 ELSE 0.0 END) as hit_rate
    FROM signal_performance sp
    WHERE sp.agent_name = p_agent_name
      AND sp.evaluation_timestamp >= NOW() - (p_lookback_days || ' days')::INTERVAL;
END;
$$ LANGUAGE plpgsql;

-- Function to detect anomalies in market data
CREATE OR REPLACE FUNCTION detect_market_data_anomalies(
    p_symbol VARCHAR(20),
    p_lookback_hours INTEGER DEFAULT 24
) RETURNS TABLE (
    symbol VARCHAR(20),
    timestamp TIMESTAMPTZ,
    anomaly_type VARCHAR(30),
    severity VARCHAR(10),
    description TEXT
) AS $$
DECLARE
    avg_volume BIGINT;
    avg_volatility DECIMAL(10,6);
    price_threshold DECIMAL(15,8);
BEGIN
    -- Calculate baseline metrics
    SELECT 
        AVG(volume),
        STDDEV((high - low) / NULLIF(close, 0)),
        AVG(close) * 0.1 -- 10% price movement threshold
    INTO avg_volume, avg_volatility, price_threshold
    FROM market_data_hf
    WHERE symbol = p_symbol
      AND timestamp >= NOW() - (p_lookback_hours || ' hours')::INTERVAL;

    RETURN QUERY
    SELECT 
        md.symbol,
        md.timestamp,
        CASE 
            WHEN md.volume > avg_volume * 5 THEN 'VOLUME_SPIKE'
            WHEN (md.high - md.low) / NULLIF(md.close, 0) > avg_volatility * 3 THEN 'VOLATILITY_SPIKE'
            WHEN ABS(md.close - LAG(md.close) OVER (ORDER BY md.timestamp)) > price_threshold THEN 'PRICE_GAP'
            ELSE 'UNKNOWN'
        END as anomaly_type,
        CASE 
            WHEN md.volume > avg_volume * 10 OR (md.high - md.low) / NULLIF(md.close, 0) > avg_volatility * 5 THEN 'HIGH'
            WHEN md.volume > avg_volume * 3 OR (md.high - md.low) / NULLIF(md.close, 0) > avg_volatility * 2 THEN 'MEDIUM'
            ELSE 'LOW'
        END as severity,
        CASE 
            WHEN md.volume > avg_volume * 5 THEN 'Volume is ' || ROUND((md.volume::DECIMAL / avg_volume), 2) || 'x normal'
            WHEN (md.high - md.low) / NULLIF(md.close, 0) > avg_volatility * 3 THEN 'Volatility is ' || ROUND(((md.high - md.low) / NULLIF(md.close, 0) / avg_volatility), 2) || 'x normal'
            ELSE 'Unusual price movement detected'
        END as description
    FROM market_data_hf md
    WHERE md.symbol = p_symbol
      AND md.timestamp >= NOW() - INTERVAL '1 hour'
      AND (
          md.volume > avg_volume * 2 OR
          (md.high - md.low) / NULLIF(md.close, 0) > avg_volatility * 2 OR
          ABS(md.close - LAG(md.close) OVER (ORDER BY md.timestamp)) > price_threshold * 0.5
      );
END;
$$ LANGUAGE plpgsql;

-- Function to calculate Value at Risk (VaR)
CREATE OR REPLACE FUNCTION calculate_var(
    p_confidence_level DECIMAL(5,4) DEFAULT 0.95,
    p_lookback_days INTEGER DEFAULT 252
) RETURNS TABLE (
    var_1d DECIMAL(15,6),
    var_5d DECIMAL(15,6),
    expected_shortfall DECIMAL(15,6)
) AS $$
DECLARE
    portfolio_returns DECIMAL(10,6)[];
    var_percentile DECIMAL(5,4);
BEGIN
    var_percentile := 1.0 - p_confidence_level;
    
    -- Get portfolio returns for the lookback period
    SELECT ARRAY_AGG(daily_return ORDER BY date)
    INTO portfolio_returns
    FROM (
        SELECT 
            date,
            (total_value - LAG(total_value) OVER (ORDER BY date)) / NULLIF(LAG(total_value) OVER (ORDER BY date), 0) as daily_return
        FROM (
            SELECT 
                DATE(timestamp) as date,
                AVG(total_value) as total_value
            FROM portfolio_snapshots
            WHERE timestamp >= NOW() - (p_lookback_days || ' days')::INTERVAL
            GROUP BY DATE(timestamp)
            ORDER BY date
        ) daily_values
    ) returns_data
    WHERE daily_return IS NOT NULL;
    
    RETURN QUERY
    SELECT 
        PERCENTILE_CONT(var_percentile) WITHIN GROUP (ORDER BY unnest(portfolio_returns)) as var_1d,
        PERCENTILE_CONT(var_percentile) WITHIN GROUP (ORDER BY unnest(portfolio_returns)) * SQRT(5) as var_5d,
        AVG(unnest(portfolio_returns)) FILTER (WHERE unnest(portfolio_returns) <= PERCENTILE_CONT(var_percentile) WITHIN GROUP (ORDER BY unnest(portfolio_returns))) as expected_shortfall;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS FOR DATA INTEGRITY AND AUTOMATION
-- ============================================================================

-- Trigger function to update portfolio snapshots
CREATE OR REPLACE FUNCTION update_portfolio_snapshot()
RETURNS TRIGGER AS $$
BEGIN
    -- Insert new portfolio snapshot when positions change
    INSERT INTO portfolio_snapshots (
        timestamp, total_value, cash, long_value, short_value,
        gross_exposure, net_exposure, leverage, num_positions
    )
    SELECT 
        NOW(),
        pm.total_value,
        pm.cash,
        pm.long_value,
        pm.short_value,
        pm.gross_exposure,
        pm.net_exposure,
        pm.leverage,
        pm.num_positions
    FROM calculate_portfolio_metrics() pm;
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Create trigger on positions table
DROP TRIGGER IF EXISTS trigger_update_portfolio_snapshot ON positions;
CREATE TRIGGER trigger_update_portfolio_snapshot
    AFTER INSERT OR UPDATE OR DELETE ON positions
    FOR EACH STATEMENT
    EXECUTE FUNCTION update_portfolio_snapshot();

-- Trigger function to validate trade data
CREATE OR REPLACE FUNCTION validate_trade_data()
RETURNS TRIGGER AS $$
BEGIN
    -- Validate price is positive
    IF NEW.price <= 0 THEN
        RAISE EXCEPTION 'Trade price must be positive: %', NEW.price;
    END IF;
    
    -- Validate quantity is not zero
    IF NEW.quantity = 0 THEN
        RAISE EXCEPTION 'Trade quantity cannot be zero';
    END IF;
    
    -- Validate side matches quantity sign
    IF (NEW.side IN ('BUY', 'COVER') AND NEW.quantity < 0) OR 
       (NEW.side IN ('SELL', 'SHORT') AND NEW.quantity > 0) THEN
        RAISE EXCEPTION 'Trade side % does not match quantity sign %', NEW.side, NEW.quantity;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger on trades table
DROP TRIGGER IF EXISTS trigger_validate_trade_data ON trades;
CREATE TRIGGER trigger_validate_trade_data
    BEFORE INSERT OR UPDATE ON trades
    FOR EACH ROW
    EXECUTE FUNCTION validate_trade_data();

-- Trigger function to update position from trades
CREATE OR REPLACE FUNCTION update_position_from_trade()
RETURNS TRIGGER AS $$
DECLARE
    existing_position RECORD;
    new_quantity DECIMAL(15,6);
    new_avg_cost DECIMAL(15,8);
BEGIN
    -- Get existing position
    SELECT * INTO existing_position
    FROM positions
    WHERE symbol = NEW.symbol 
      AND exchange = NEW.exchange 
      AND strategy = NEW.strategy 
      AND agent_name = NEW.agent_name;
    
    IF existing_position IS NULL THEN
        -- Create new position
        INSERT INTO positions (
            symbol, exchange, strategy, agent_name, quantity, avg_cost,
            market_value, unrealized_pnl, first_trade_at, last_trade_at
        ) VALUES (
            NEW.symbol, NEW.exchange, NEW.strategy, NEW.agent_name,
            NEW.quantity, NEW.price,
            NEW.quantity * NEW.price, 0,
            NEW.executed_at, NEW.executed_at
        );
    ELSE
        -- Update existing position
        new_quantity := existing_position.quantity + NEW.quantity;
        
        IF new_quantity = 0 THEN
            -- Position closed
            DELETE FROM positions WHERE id = existing_position.id;
        ELSE
            -- Calculate new average cost
            IF SIGN(existing_position.quantity) = SIGN(NEW.quantity) THEN
                -- Adding to position
                new_avg_cost := (existing_position.avg_cost * ABS(existing_position.quantity) + NEW.price * ABS(NEW.quantity)) / ABS(new_quantity);
            ELSE
                -- Reducing position, keep existing avg cost
                new_avg_cost := existing_position.avg_cost;
            END IF;
            
            UPDATE positions SET
                quantity = new_quantity,
                avg_cost = new_avg_cost,
                market_value = new_quantity * NEW.price, -- This should be updated with real-time prices
                last_trade_at = NEW.executed_at,
                updated_at = NOW()
            WHERE id = existing_position.id;
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger on trades table
DROP TRIGGER IF EXISTS trigger_update_position_from_trade ON trades;
CREATE TRIGGER trigger_update_position_from_trade
    AFTER INSERT ON trades
    FOR EACH ROW
    EXECUTE FUNCTION update_position_from_trade();

-- Trigger function to log audit trail
CREATE OR REPLACE FUNCTION log_audit_trail()
RETURNS TRIGGER AS $$
DECLARE
    action_type VARCHAR(30);
    entity_type VARCHAR(30);
    entity_id VARCHAR(100);
BEGIN
    -- Determine action type
    IF TG_OP = 'INSERT' THEN
        action_type := 'CREATE';
    ELSIF TG_OP = 'UPDATE' THEN
        action_type := 'UPDATE';
    ELSIF TG_OP = 'DELETE' THEN
        action_type := 'DELETE';
    END IF;
    
    -- Determine entity type and ID based on table
    CASE TG_TABLE_NAME
        WHEN 'trades' THEN
            entity_type := 'TRADE';
            entity_id := COALESCE(NEW.trade_id, OLD.trade_id);
        WHEN 'orders' THEN
            entity_type := 'ORDER';
            entity_id := COALESCE(NEW.order_id, OLD.order_id);
        WHEN 'positions' THEN
            entity_type := 'POSITION';
            entity_id := COALESCE(NEW.symbol || '_' || NEW.strategy, OLD.symbol || '_' || OLD.strategy);
        WHEN 'signals' THEN
            entity_type := 'SIGNAL';
            entity_id := COALESCE(NEW.id::TEXT, OLD.id::TEXT);
        ELSE
            entity_type := UPPER(TG_TABLE_NAME);
            entity_id := COALESCE(NEW.id::TEXT, OLD.id::TEXT);
    END CASE;
    
    -- Insert audit record
    INSERT INTO audit_trail (
        timestamp, action_type, entity_type, entity_id,
        before_state, after_state
    ) VALUES (
        NOW(), action_type, entity_type, entity_id,
        CASE WHEN TG_OP != 'INSERT' THEN row_to_json(OLD) ELSE NULL END,
        CASE WHEN TG_OP != 'DELETE' THEN row_to_json(NEW) ELSE NULL END
    );
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Create audit triggers on key tables
DROP TRIGGER IF EXISTS trigger_audit_trades ON trades;
CREATE TRIGGER trigger_audit_trades
    AFTER INSERT OR UPDATE OR DELETE ON trades
    FOR EACH ROW
    EXECUTE FUNCTION log_audit_trail();

DROP TRIGGER IF EXISTS trigger_audit_orders ON orders;
CREATE TRIGGER trigger_audit_orders
    AFTER INSERT OR UPDATE OR DELETE ON orders
    FOR EACH ROW
    EXECUTE FUNCTION log_audit_trail();

DROP TRIGGER IF EXISTS trigger_audit_positions ON positions;
CREATE TRIGGER trigger_audit_positions
    AFTER INSERT OR UPDATE OR DELETE ON positions
    FOR EACH ROW
    EXECUTE FUNCTION log_audit_trail();

-- ============================================================================
-- MATERIALIZED VIEWS FOR PERFORMANCE
-- ============================================================================

-- Daily performance summary view
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_performance_summary AS
SELECT 
    DATE(t.executed_at) as trade_date,
    t.strategy,
    t.agent_name,
    COUNT(*) as total_trades,
    SUM(t.net_pnl) as daily_pnl,
    AVG(t.net_pnl) as avg_trade_pnl,
    SUM(CASE WHEN t.net_pnl > 0 THEN 1 ELSE 0 END)::DECIMAL / COUNT(*) as win_rate,
    MAX(t.net_pnl) as best_trade,
    MIN(t.net_pnl) as worst_trade,
    SUM(ABS(t.quantity * t.price)) as volume_traded
FROM trades t
WHERE t.executed_at >= CURRENT_DATE - INTERVAL '1 year'
GROUP BY DATE(t.executed_at), t.strategy, t.agent_name;

CREATE UNIQUE INDEX idx_daily_performance_summary ON daily_performance_summary(trade_date, strategy, agent_name);

-- Real-time signal strength view
CREATE MATERIALIZED VIEW IF NOT EXISTS signal_strength_summary AS
SELECT 
    s.symbol,
    s.agent_name,
    COUNT(*) as signal_count,
    AVG(s.confidence) as avg_confidence,
    AVG(s.strength) as avg_strength,
    COUNT(*) FILTER (WHERE s.direction = 'BUY') as buy_signals,
    COUNT(*) FILTER (WHERE s.direction = 'SELL') as sell_signals,
    MAX(s.timestamp) as last_signal_time
FROM signals s
WHERE s.timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY s.symbol, s.agent_name;

CREATE UNIQUE INDEX idx_signal_strength_summary ON signal_strength_summary(symbol, agent_name);

-- ============================================================================
-- REFRESH FUNCTIONS FOR MATERIALIZED VIEWS
-- ============================================================================

-- Function to refresh all materialized views
CREATE OR REPLACE FUNCTION refresh_materialized_views()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY daily_performance_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY signal_strength_summary;
    
    INSERT INTO system_logs (timestamp, level, component, message)
    VALUES (NOW(), 'INFO', 'database', 'Materialized views refreshed successfully');
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PERFORMANCE MONITORING FUNCTIONS
-- ============================================================================

-- Function to get database performance metrics
CREATE OR REPLACE FUNCTION get_database_performance_metrics()
RETURNS TABLE (
    metric_name VARCHAR(50),
    metric_value DECIMAL(15,6),
    unit VARCHAR(20)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'total_connections'::VARCHAR(50),
        (SELECT COUNT(*) FROM pg_stat_activity)::DECIMAL(15,6),
        'count'::VARCHAR(20)
    UNION ALL
    SELECT 
        'active_connections'::VARCHAR(50),
        (SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active')::DECIMAL(15,6),
        'count'::VARCHAR(20)
    UNION ALL
    SELECT 
        'database_size'::VARCHAR(50),
        (SELECT pg_database_size(current_database()))::DECIMAL(15,6),
        'bytes'::VARCHAR(20)
    UNION ALL
    SELECT 
        'cache_hit_ratio'::VARCHAR(50),
        (SELECT ROUND(100.0 * sum(blks_hit) / (sum(blks_hit) + sum(blks_read)), 2) 
         FROM pg_stat_database WHERE datname = current_database())::DECIMAL(15,6),
        'percent'::VARCHAR(20);
END;
$$ LANGUAGE plpgsql;