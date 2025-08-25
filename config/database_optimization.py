"""
Database optimization configurations for real-time trading performance.
"""

import asyncio
from typing import Dict, Any, Optional
from sqlalchemy import create_engine, text, pool
from sqlalchemy.pool import QueuePool
import redis.asyncio as redis
from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class DatabaseOptimizer:
    """Advanced database optimization for high-frequency trading."""
    
    def __init__(self):
        self.postgres_optimizations = {
            # Connection Pool Settings
            'pool_size': 50,              # Increased from default 5
            'max_overflow': 100,          # Increased from default 10
            'pool_pre_ping': True,        # Verify connections before use
            'pool_recycle': 3600,         # Recycle connections every hour
            'pool_timeout': 30,           # Connection timeout
            
            # PostgreSQL Specific Optimizations
            'connect_args': {
                # Connection-level optimizations
                'application_name': 'hive_trade_core',
                'connect_timeout': 10,
                'command_timeout': 30,
                'server_settings': {
                    # Memory and performance settings
                    'shared_preload_libraries': 'pg_stat_statements',
                    'max_connections': '200',
                    'shared_buffers': '256MB',
                    'effective_cache_size': '1GB',
                    'maintenance_work_mem': '64MB',
                    'checkpoint_completion_target': '0.9',
                    'wal_buffers': '16MB',
                    'default_statistics_target': '100',
                    'random_page_cost': '1.1',
                    'effective_io_concurrency': '200',
                    'work_mem': '4MB',
                    
                    # Logging and monitoring
                    'log_min_duration_statement': '100',  # Log slow queries
                    'log_checkpoints': 'on',
                    'log_connections': 'on',
                    'log_disconnections': 'on',
                    'log_statement': 'mod',  # Log DDL/DML statements
                    
                    # Write-ahead logging optimizations
                    'wal_level': 'replica',
                    'max_wal_senders': '3',
                    'wal_keep_segments': '32',
                    'checkpoint_segments': '32',
                    
                    # Query optimization
                    'enable_seqscan': 'on',
                    'enable_indexscan': 'on',
                    'enable_bitmapscan': 'on',
                    'enable_hashjoin': 'on',
                    'enable_mergejoin': 'on',
                    'enable_nestloop': 'on',
                }
            }
        }
        
        self.redis_optimizations = {
            # Connection Pool Settings
            'max_connections': 100,       # Increased pool size
            'connection_kwargs': {
                'socket_timeout': 2.0,    # Fast timeout for trading
                'socket_connect_timeout': 2.0,
                'socket_keepalive': True,
                'socket_keepalive_options': {
                    1: 1,    # TCP_KEEPIDLE
                    2: 3,    # TCP_KEEPINTVL  
                    3: 5     # TCP_KEEPCNT
                },
                'health_check_interval': 10,
                'decode_responses': True,
            },
            
            # Redis Server Optimizations (applied via CONFIG SET)
            'server_config': {
                'maxmemory-policy': 'allkeys-lru',  # Evict least recently used
                'maxclients': '10000',              # Allow many connections
                'timeout': '300',                   # Client timeout
                'tcp-keepalive': '60',              # TCP keepalive
                'save': '900 1 300 10 60 10000',   # Persistence settings
                'stop-writes-on-bgsave-error': 'no',
                'rdbcompression': 'yes',
                'rdbchecksum': 'yes',
                'databases': '16',
                
                # Memory optimizations
                'maxmemory': '512mb',
                'hash-max-ziplist-entries': '512',
                'hash-max-ziplist-value': '64',
                'list-max-ziplist-size': '-2',
                'set-max-intset-entries': '512',
                'zset-max-ziplist-entries': '128',
                'zset-max-ziplist-value': '64',
            }
        }
        
        self.monitoring_queries = {
            'postgres_connections': """
                SELECT 
                    state,
                    count(*) as connections,
                    avg(extract(epoch from now() - query_start)) as avg_duration
                FROM pg_stat_activity 
                WHERE pid <> pg_backend_pid()
                GROUP BY state
            """,
            
            'postgres_slow_queries': """
                SELECT 
                    query,
                    calls,
                    total_time,
                    mean_time,
                    rows
                FROM pg_stat_statements 
                WHERE mean_time > 100 
                ORDER BY mean_time DESC 
                LIMIT 10
            """,
            
            'postgres_locks': """
                SELECT 
                    mode,
                    locktype,
                    count(*) as lock_count
                FROM pg_locks 
                GROUP BY mode, locktype
                ORDER BY lock_count DESC
            """,
            
            'postgres_cache_hit_ratio': """
                SELECT 
                    schemaname,
                    tablename,
                    heap_blks_read,
                    heap_blks_hit,
                    round(heap_blks_hit::numeric / (heap_blks_hit + heap_blks_read) * 100, 2) as cache_hit_ratio
                FROM pg_statio_user_tables 
                WHERE heap_blks_read > 0
                ORDER BY cache_hit_ratio ASC
            """
        }
    
    def get_optimized_postgres_engine(self) -> Any:
        """Create an optimized PostgreSQL engine for high-frequency trading."""
        
        engine = create_engine(
            settings.database.url,
            poolclass=QueuePool,
            echo=settings.debug,
            **self.postgres_optimizations
        )
        
        logger.info("Created optimized PostgreSQL engine with enhanced connection pool")
        return engine
    
    async def get_optimized_redis_pool(self) -> redis.ConnectionPool:
        """Create an optimized Redis connection pool."""
        
        redis_url = f"redis://:{settings.redis.password or ''}@{settings.redis.host}:{settings.redis.port}/{settings.redis.db}"
        
        pool = redis.ConnectionPool.from_url(
            redis_url,
            **self.redis_optimizations
        )
        
        # Apply server-side optimizations
        await self._apply_redis_server_config(pool)
        
        logger.info("Created optimized Redis connection pool with enhanced settings")
        return pool
    
    async def _apply_redis_server_config(self, pool: redis.ConnectionPool):
        """Apply server-side Redis optimizations."""
        client = redis.Redis(connection_pool=pool)
        
        try:
            for key, value in self.redis_optimizations['server_config'].items():
                try:
                    await client.config_set(key, value)
                    logger.debug(f"Applied Redis config: {key} = {value}")
                except Exception as e:
                    logger.warning(f"Could not set Redis config {key}: {e}")
            
            # Verify critical settings
            info = await client.info()
            logger.info(f"Redis version: {info.get('redis_version')}")
            logger.info(f"Redis max memory: {info.get('maxmemory_human')}")
            logger.info(f"Redis connected clients: {info.get('connected_clients')}")
            
        finally:
            await client.close()
    
    async def create_database_indexes(self, engine):
        """Create optimized indexes for trading operations."""
        
        index_queries = [
            # Orders table indexes
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_symbol_status_created 
            ON orders(symbol, status, created_at DESC) 
            WHERE status IN ('pending', 'partially_filled')
            """,
            
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_user_created 
            ON orders(user_id, created_at DESC)
            """,
            
            # Trades table indexes
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_symbol_timestamp 
            ON trades(symbol, executed_at DESC)
            """,
            
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_order_id 
            ON trades(order_id)
            """,
            
            # Market data indexes
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_symbol_timestamp 
            ON market_data(symbol, timestamp DESC)
            """,
            
            # Positions table indexes
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_user_symbol 
            ON positions(user_id, symbol, updated_at DESC)
            """,
            
            # Portfolio metrics indexes
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolio_metrics_user_timestamp 
            ON portfolio_metrics(user_id, timestamp DESC)
            """,
            
            # Risk metrics indexes
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_risk_metrics_user_timestamp 
            ON risk_metrics(user_id, calculated_at DESC)
            """,
        ]
        
        with engine.connect() as conn:
            for query in index_queries:
                try:
                    conn.execute(text(query))
                    logger.info(f"Created index: {query.split('idx_')[1].split(' ')[0]}")
                except Exception as e:
                    logger.warning(f"Could not create index: {e}")
    
    async def optimize_postgres_settings(self, engine):
        """Apply runtime PostgreSQL optimizations."""
        
        optimization_queries = [
            # Enable parallel query execution
            "SET max_parallel_workers_per_gather = 4",
            "SET max_parallel_workers = 8",
            "SET parallel_tuple_cost = 0.01",
            
            # Optimize query planner
            "SET random_page_cost = 1.1",  # For SSD storage
            "SET seq_page_cost = 1.0",
            "SET cpu_tuple_cost = 0.01",
            "SET cpu_index_tuple_cost = 0.005",
            "SET cpu_operator_cost = 0.0025",
            
            # Memory settings for current session
            "SET work_mem = '16MB'",        # Higher for complex queries
            "SET maintenance_work_mem = '256MB',",
            
            # Enable statement statistics
            "CREATE EXTENSION IF NOT EXISTS pg_stat_statements",
            
            # Enable parallel index creation
            "SET maintenance_work_mem = '1GB'",  # For index creation
        ]
        
        with engine.connect() as conn:
            for query in optimization_queries:
                try:
                    conn.execute(text(query))
                    logger.debug(f"Applied optimization: {query[:50]}...")
                except Exception as e:
                    logger.warning(f"Could not apply optimization: {e}")
    
    async def monitor_database_performance(self, engine) -> Dict[str, Any]:
        """Monitor database performance metrics."""
        
        metrics = {}
        
        with engine.connect() as conn:
            for metric_name, query in self.monitoring_queries.items():
                try:
                    result = conn.execute(text(query))
                    metrics[metric_name] = [dict(row) for row in result]
                except Exception as e:
                    logger.error(f"Failed to collect {metric_name}: {e}")
                    metrics[metric_name] = []
        
        return metrics
    
    async def analyze_query_performance(self, engine) -> Dict[str, Any]:
        """Analyze query performance and suggest optimizations."""
        
        analysis = {
            'slow_queries': [],
            'missing_indexes': [],
            'cache_performance': {},
            'recommendations': []
        }
        
        with engine.connect() as conn:
            # Analyze slow queries
            slow_query = """
                SELECT 
                    substring(query, 1, 100) as query_snippet,
                    calls,
                    total_time,
                    mean_time,
                    stddev_time,
                    rows,
                    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
                FROM pg_stat_statements 
                WHERE mean_time > 50
                ORDER BY mean_time DESC 
                LIMIT 10
            """
            
            try:
                result = conn.execute(text(slow_query))
                analysis['slow_queries'] = [dict(row) for row in result]
            except Exception as e:
                logger.warning(f"Could not analyze slow queries: {e}")
            
            # Check cache hit ratios
            cache_query = """
                SELECT 
                    'database' as type,
                    sum(heap_blks_read) as reads,
                    sum(heap_blks_hit) as hits,
                    sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) * 100 as hit_ratio
                FROM pg_statio_user_tables
            """
            
            try:
                result = conn.execute(text(cache_query))
                cache_data = dict(result.fetchone() or {})
                analysis['cache_performance'] = cache_data
                
                if cache_data.get('hit_ratio', 0) < 95:
                    analysis['recommendations'].append(
                        f"Cache hit ratio is {cache_data.get('hit_ratio', 0):.2f}%. Consider increasing shared_buffers."
                    )
            except Exception as e:
                logger.warning(f"Could not analyze cache performance: {e}")
        
        return analysis
    
    async def create_performance_views(self, engine):
        """Create database views for performance monitoring."""
        
        view_queries = [
            # Trading performance view
            """
            CREATE OR REPLACE VIEW trading_performance_summary AS
            SELECT 
                DATE_TRUNC('minute', executed_at) as minute,
                symbol,
                COUNT(*) as trade_count,
                AVG(execution_time_ms) as avg_execution_time,
                MAX(execution_time_ms) as max_execution_time,
                SUM(quantity * price) as volume
            FROM trades 
            WHERE executed_at > NOW() - INTERVAL '1 hour'
            GROUP BY minute, symbol
            ORDER BY minute DESC, volume DESC
            """,
            
            # System health view
            """
            CREATE OR REPLACE VIEW system_health_summary AS
            SELECT 
                'postgres' as component,
                CASE WHEN pg_is_in_recovery() THEN 'replica' ELSE 'primary' END as role,
                (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                (SELECT count(*) FROM pg_stat_activity WHERE state = 'idle in transaction') as idle_in_transaction,
                pg_database_size(current_database()) as database_size_bytes,
                (SELECT sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) * 100 
                 FROM pg_statio_user_tables) as cache_hit_ratio
            """,
            
            # Order book depth view
            """
            CREATE OR REPLACE VIEW order_book_depth AS
            SELECT 
                symbol,
                side,
                COUNT(*) as order_count,
                SUM(quantity - filled_quantity) as total_quantity,
                AVG(price) as avg_price,
                MIN(price) as min_price,
                MAX(price) as max_price
            FROM orders 
            WHERE status IN ('pending', 'partially_filled')
            GROUP BY symbol, side
            ORDER BY symbol, side
            """
        ]
        
        with engine.connect() as conn:
            for query in view_queries:
                try:
                    conn.execute(text(query))
                    view_name = query.split('VIEW ')[1].split(' AS')[0]
                    logger.info(f"Created performance view: {view_name}")
                except Exception as e:
                    logger.warning(f"Could not create performance view: {e}")


# Global optimizer instance
db_optimizer = DatabaseOptimizer()


async def apply_database_optimizations():
    """Apply all database optimizations."""
    logger.info("🔧 Applying database optimizations for real-time trading...")
    
    try:
        # Create optimized engines
        postgres_engine = db_optimizer.get_optimized_postgres_engine()
        redis_pool = await db_optimizer.get_optimized_redis_pool()
        
        # Apply PostgreSQL optimizations
        await db_optimizer.optimize_postgres_settings(postgres_engine)
        await db_optimizer.create_database_indexes(postgres_engine)
        await db_optimizer.create_performance_views(postgres_engine)
        
        logger.info("✅ Database optimizations applied successfully")
        
        return {
            'postgres_engine': postgres_engine,
            'redis_pool': redis_pool,
            'status': 'optimized'
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to apply database optimizations: {e}")
        raise


async def monitor_database_health():
    """Continuous database health monitoring."""
    postgres_engine = db_optimizer.get_optimized_postgres_engine()
    
    while True:
        try:
            # Collect performance metrics
            metrics = await db_optimizer.monitor_database_performance(postgres_engine)
            
            # Analyze query performance
            analysis = await db_optimizer.analyze_query_performance(postgres_engine)
            
            # Log critical metrics
            if metrics.get('postgres_connections'):
                active_conn = sum(c['connections'] for c in metrics['postgres_connections'] if c['state'] == 'active')
                logger.info(f"📊 Active PostgreSQL connections: {active_conn}")
            
            if analysis.get('cache_performance'):
                hit_ratio = analysis['cache_performance'].get('hit_ratio', 0)
                logger.info(f"📊 Database cache hit ratio: {hit_ratio:.2f}%")
            
            # Sleep before next check
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"Database monitoring error: {e}")
            await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(apply_database_optimizations())