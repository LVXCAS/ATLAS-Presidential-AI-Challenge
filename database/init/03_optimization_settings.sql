-- LangGraph Trading System - Database Optimization Settings
-- PostgreSQL configuration optimizations for high-frequency trading workloads

-- ============================================================================
-- PERFORMANCE OPTIMIZATION SETTINGS
-- ============================================================================

-- Memory settings for high-performance workloads
-- These should be adjusted based on available system memory
ALTER SYSTEM SET shared_buffers = '4GB';  -- 25% of RAM for dedicated DB server
ALTER SYSTEM SET effective_cache_size = '12GB';  -- 75% of RAM
ALTER SYSTEM SET work_mem = '256MB';  -- Per-operation memory
ALTER SYSTEM SET maintenance_work_mem = '1GB';  -- For maintenance operations

-- Connection settings
ALTER SYSTEM SET max_connections = 200;  -- Adjust based on application needs
ALTER SYSTEM SET max_prepared_transactions = 100;

-- Write-ahead logging (WAL) settings for high throughput
ALTER SYSTEM SET wal_level = 'replica';
ALTER SYSTEM SET max_wal_size = '4GB';
ALTER SYSTEM SET min_wal_size = '1GB';
ALTER SYSTEM SET wal_buffers = '64MB';
ALTER SYSTEM SET wal_writer_delay = '10ms';
ALTER SYSTEM SET commit_delay = '10';  -- Microseconds
ALTER SYSTEM SET commit_siblings = 10;

-- Checkpoint settings for consistent performance
ALTER SYSTEM SET checkpoint_timeout = '15min';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET checkpoint_warning = '30s';

-- Background writer settings
ALTER SYSTEM SET bgwriter_delay = '10ms';
ALTER SYSTEM SET bgwriter_lru_maxpages = 1000;
ALTER SYSTEM SET bgwriter_lru_multiplier = 10.0;

-- Vacuum and autovacuum settings for time-series data
ALTER SYSTEM SET autovacuum = on;
ALTER SYSTEM SET autovacuum_max_workers = 6;
ALTER SYSTEM SET autovacuum_naptime = '10s';  -- More frequent for high-insert workload
ALTER SYSTEM SET autovacuum_vacuum_threshold = 1000;
ALTER SYSTEM SET autovacuum_vacuum_scale_factor = 0.05;  -- Vacuum when 5% of table changes
ALTER SYSTEM SET autovacuum_analyze_threshold = 500;
ALTER SYSTEM SET autovacuum_analyze_scale_factor = 0.02;  -- Analyze when 2% changes

-- Query planner settings
ALTER SYSTEM SET random_page_cost = 1.1;  -- SSD-optimized
ALTER SYSTEM SET seq_page_cost = 1.0;
ALTER SYSTEM SET cpu_tuple_cost = 0.01;
ALTER SYSTEM SET cpu_index_tuple_cost = 0.005;
ALTER SYSTEM SET cpu_operator_cost = 0.0025;

-- Parallel query settings
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
ALTER SYSTEM SET max_parallel_workers = 8;
ALTER SYSTEM SET max_worker_processes = 16;
ALTER SYSTEM SET parallel_tuple_cost = 0.1;
ALTER SYSTEM SET parallel_setup_cost = 1000.0;

-- Logging settings for monitoring
ALTER SYSTEM SET log_destination = 'csvlog';
ALTER SYSTEM SET logging_collector = on;
ALTER SYSTEM SET log_directory = 'pg_log';
ALTER SYSTEM SET log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log';
ALTER SYSTEM SET log_rotation_age = '1d';
ALTER SYSTEM SET log_rotation_size = '100MB';
ALTER SYSTEM SET log_min_duration_statement = '1s';  -- Log slow queries
ALTER SYSTEM SET log_checkpoints = on;
ALTER SYSTEM SET log_connections = on;
ALTER SYSTEM SET log_disconnections = on;
ALTER SYSTEM SET log_lock_waits = on;
ALTER SYSTEM SET log_temp_files = 0;  -- Log all temp files
ALTER SYSTEM SET log_autovacuum_min_duration = '1s';

-- Statistics settings
ALTER SYSTEM SET track_activities = on;
ALTER SYSTEM SET track_counts = on;
ALTER SYSTEM SET track_io_timing = on;
ALTER SYSTEM SET track_functions = 'all';
ALTER SYSTEM SET log_statement_stats = off;
ALTER SYSTEM SET log_parser_stats = off;
ALTER SYSTEM SET log_planner_stats = off;
ALTER SYSTEM SET log_executor_stats = off;

-- ============================================================================
-- TABLE-SPECIFIC OPTIMIZATIONS
-- ============================================================================

-- Optimize market data tables for time-series workloads
ALTER TABLE market_data_hf SET (
    fillfactor = 90,  -- Leave room for updates
    autovacuum_vacuum_scale_factor = 0.01,  -- More aggressive vacuuming
    autovacuum_analyze_scale_factor = 0.005,
    autovacuum_vacuum_cost_delay = 5,
    autovacuum_vacuum_cost_limit = 2000
);

ALTER TABLE market_data_daily SET (
    fillfactor = 95,  -- Less frequent updates
    autovacuum_vacuum_scale_factor = 0.02,
    autovacuum_analyze_scale_factor = 0.01
);

-- Optimize signals table for high insert rate
ALTER TABLE signals SET (
    fillfactor = 90,
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_analyze_scale_factor = 0.005,
    autovacuum_vacuum_cost_delay = 5
);

-- Optimize trades table for regulatory compliance
ALTER TABLE trades SET (
    fillfactor = 100,  -- No updates expected
    autovacuum_vacuum_scale_factor = 0.05,
    autovacuum_analyze_scale_factor = 0.02
);

-- Optimize orders table for frequent updates
ALTER TABLE orders SET (
    fillfactor = 80,  -- Frequent status updates
    autovacuum_vacuum_scale_factor = 0.02,
    autovacuum_analyze_scale_factor = 0.01,
    autovacuum_vacuum_cost_delay = 2
);

-- Optimize system logs for high volume
ALTER TABLE system_logs SET (
    fillfactor = 100,  -- Insert-only
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_analyze_scale_factor = 0.005
);

-- ============================================================================
-- PARTITIONING STRATEGIES
-- ============================================================================

-- Create partitioned tables for very high-volume data
-- This is for future implementation when data volume grows

-- Example: Partition market_data_hf by symbol for better performance
-- CREATE TABLE market_data_hf_aapl PARTITION OF market_data_hf
-- FOR VALUES IN ('AAPL');

-- ============================================================================
-- MONITORING AND MAINTENANCE PROCEDURES
-- ============================================================================

-- Create a function to analyze table statistics
CREATE OR REPLACE FUNCTION analyze_table_statistics()
RETURNS TABLE (
    table_name TEXT,
    row_count BIGINT,
    table_size TEXT,
    index_size TEXT,
    total_size TEXT,
    last_vacuum TIMESTAMPTZ,
    last_analyze TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        schemaname||'.'||tablename as table_name,
        n_tup_ins + n_tup_upd as row_count,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as table_size,
        pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) as index_size,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) + pg_indexes_size(schemaname||'.'||tablename)) as total_size,
        last_vacuum,
        last_analyze
    FROM pg_stat_user_tables
    WHERE schemaname = 'public'
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
END;
$$ LANGUAGE plpgsql;

-- Create a function to get slow queries
CREATE OR REPLACE FUNCTION get_slow_queries(
    min_duration INTERVAL DEFAULT '1 second'
)
RETURNS TABLE (
    query TEXT,
    calls BIGINT,
    total_time DOUBLE PRECISION,
    mean_time DOUBLE PRECISION,
    rows BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pss.query,
        pss.calls,
        pss.total_exec_time as total_time,
        pss.mean_exec_time as mean_time,
        pss.rows
    FROM pg_stat_statements pss
    WHERE pss.mean_exec_time > EXTRACT(EPOCH FROM min_duration) * 1000
    ORDER BY pss.mean_exec_time DESC
    LIMIT 20;
END;
$$ LANGUAGE plpgsql;

-- Create a function to monitor index usage
CREATE OR REPLACE FUNCTION monitor_index_usage()
RETURNS TABLE (
    table_name TEXT,
    index_name TEXT,
    index_scans BIGINT,
    tuples_read BIGINT,
    tuples_fetched BIGINT,
    size TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        schemaname||'.'||tablename as table_name,
        indexrelname as index_name,
        idx_scan as index_scans,
        idx_tup_read as tuples_read,
        idx_tup_fetch as tuples_fetched,
        pg_size_pretty(pg_relation_size(indexrelid)) as size
    FROM pg_stat_user_indexes
    WHERE schemaname = 'public'
    ORDER BY idx_scan DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- AUTOMATED MAINTENANCE JOBS
-- ============================================================================

-- Create a function for daily maintenance
CREATE OR REPLACE FUNCTION daily_maintenance()
RETURNS VOID AS $$
BEGIN
    -- Refresh materialized views
    PERFORM refresh_materialized_views();
    
    -- Update table statistics for query planner
    ANALYZE market_data_hf;
    ANALYZE signals;
    ANALYZE trades;
    ANALYZE orders;
    ANALYZE positions;
    
    -- Clean up old system logs (older than retention policy)
    DELETE FROM system_logs 
    WHERE timestamp < NOW() - INTERVAL '90 days';
    
    -- Log maintenance completion
    INSERT INTO system_logs (timestamp, level, component, message)
    VALUES (NOW(), 'INFO', 'maintenance', 'Daily maintenance completed successfully');
    
EXCEPTION WHEN OTHERS THEN
    INSERT INTO system_logs (timestamp, level, component, message, metadata)
    VALUES (NOW(), 'ERROR', 'maintenance', 'Daily maintenance failed', 
            json_build_object('error', SQLERRM));
    RAISE;
END;
$$ LANGUAGE plpgsql;

-- Create a function for weekly maintenance
CREATE OR REPLACE FUNCTION weekly_maintenance()
RETURNS VOID AS $$
BEGIN
    -- Reindex heavily used tables
    REINDEX TABLE CONCURRENTLY market_data_hf;
    REINDEX TABLE CONCURRENTLY signals;
    REINDEX TABLE CONCURRENTLY trades;
    
    -- Update all table statistics
    ANALYZE;
    
    -- Vacuum analyze all tables
    VACUUM ANALYZE;
    
    -- Log maintenance completion
    INSERT INTO system_logs (timestamp, level, component, message)
    VALUES (NOW(), 'INFO', 'maintenance', 'Weekly maintenance completed successfully');
    
EXCEPTION WHEN OTHERS THEN
    INSERT INTO system_logs (timestamp, level, component, message, metadata)
    VALUES (NOW(), 'ERROR', 'maintenance', 'Weekly maintenance failed', 
            json_build_object('error', SQLERRM));
    RAISE;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SECURITY SETTINGS
-- ============================================================================

-- Create read-only user for reporting
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'trading_readonly') THEN
        CREATE ROLE trading_readonly LOGIN PASSWORD 'readonly_secure_password_change_me';
    END IF;
END
$$;

-- Grant read-only permissions
GRANT CONNECT ON DATABASE postgres TO trading_readonly;
GRANT USAGE ON SCHEMA public TO trading_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO trading_readonly;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO trading_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO trading_readonly;

-- Create application user with limited permissions
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'trading_app') THEN
        CREATE ROLE trading_app LOGIN PASSWORD 'app_secure_password_change_me';
    END IF;
END
$$;

-- Grant application permissions
GRANT CONNECT ON DATABASE postgres TO trading_app;
GRANT USAGE ON SCHEMA public TO trading_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO trading_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO trading_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE ON TABLES TO trading_app;

-- Restrict DELETE permissions to specific tables
GRANT DELETE ON system_logs TO trading_app;
GRANT DELETE ON positions TO trading_app;  -- For position closing

-- ============================================================================
-- MONITORING VIEWS
-- ============================================================================

-- Create view for real-time system health
CREATE OR REPLACE VIEW system_health AS
SELECT 
    'database_connections' as metric,
    COUNT(*)::TEXT as value,
    'active: ' || COUNT(*) FILTER (WHERE state = 'active')::TEXT as details
FROM pg_stat_activity
UNION ALL
SELECT 
    'database_size' as metric,
    pg_size_pretty(pg_database_size(current_database())) as value,
    'tables: ' || COUNT(*)::TEXT as details
FROM information_schema.tables 
WHERE table_schema = 'public'
UNION ALL
SELECT 
    'cache_hit_ratio' as metric,
    ROUND(100.0 * sum(blks_hit) / (sum(blks_hit) + sum(blks_read)), 2)::TEXT || '%' as value,
    'buffer_hits: ' || sum(blks_hit)::TEXT as details
FROM pg_stat_database 
WHERE datname = current_database();

-- Create view for trading system metrics
CREATE OR REPLACE VIEW trading_metrics AS
SELECT 
    'total_positions' as metric,
    COUNT(*)::TEXT as value,
    'long: ' || COUNT(*) FILTER (WHERE quantity > 0)::TEXT || 
    ', short: ' || COUNT(*) FILTER (WHERE quantity < 0)::TEXT as details
FROM positions
UNION ALL
SELECT 
    'daily_trades' as metric,
    COUNT(*)::TEXT as value,
    'volume: ' || COALESCE(pg_size_pretty(SUM(ABS(quantity * price))::BIGINT), '0') as details
FROM trades
WHERE executed_at >= CURRENT_DATE
UNION ALL
SELECT 
    'active_signals' as metric,
    COUNT(*)::TEXT as value,
    'last_hour: ' || COUNT(*) FILTER (WHERE timestamp >= NOW() - INTERVAL '1 hour')::TEXT as details
FROM signals
WHERE timestamp >= NOW() - INTERVAL '24 hours';

-- Log optimization completion
INSERT INTO system_logs (timestamp, level, component, message, metadata)
VALUES (NOW(), 'INFO', 'database', 'Database optimization settings applied', 
        json_build_object('optimization_version', '1.0.0', 'settings_count', 50))
ON CONFLICT DO NOTHING;