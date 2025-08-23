"""
Database management for Bloomberg Terminal API
TimescaleDB connection and session management with SQLAlchemy 2.0.
"""

import asyncio
import logging
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool
from sqlalchemy import text, MetaData

from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Global database components
async_engine = None
async_session_factory = None


class Base(DeclarativeBase):
    """Base class for all database models."""
    metadata = MetaData()


async def init_database() -> None:
    """Initialize database connection and create tables if needed."""
    global async_engine, async_session_factory
    
    logger.info("Initializing database connection...")
    
    # Create async engine with optimized settings
    async_engine = create_async_engine(
        settings.get_database_url(),
        echo=settings.database.echo,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_pre_ping=settings.database.pool_pre_ping,
        # Disable pooling for async to avoid connection issues
        poolclass=NullPool,
        # Connection arguments for better performance
        connect_args={
            "server_settings": {
                "application_name": "bloomberg_terminal_api",
                "jit": "off",  # Disable JIT for faster startup
            },
            "command_timeout": 60,
        }
    )
    
    # Create session factory
    async_session_factory = async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=True,
        autocommit=False
    )
    
    try:
        # Test connection
        async with async_engine.begin() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar()
            logger.info(f"Connected to database: {version}")
            
        # Ensure TimescaleDB extension is enabled
        await _ensure_timescaledb()
        
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def _ensure_timescaledb() -> None:
    """Ensure TimescaleDB extension is properly configured."""
    try:
        async with get_async_session() as session:
            # Check if TimescaleDB is installed
            result = await session.execute(
                text("SELECT COUNT(*) FROM pg_extension WHERE extname = 'timescaledb'")
            )
            if result.scalar() == 0:
                logger.warning("TimescaleDB extension not found. Creating...")
                await session.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
                await session.commit()
            
            # Check TimescaleDB version
            result = await session.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"))
            version = result.scalar()
            logger.info(f"TimescaleDB version: {version}")
            
    except Exception as e:
        logger.error(f"TimescaleDB setup failed: {e}")
        # Continue without TimescaleDB if it fails
        pass


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session with proper error handling."""
    if not async_session_factory:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    async with async_session_factory() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_database_health() -> bool:
    """Check database connection health."""
    try:
        async with get_async_session() as session:
            await session.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


async def close_database() -> None:
    """Close database connections."""
    global async_engine
    
    if async_engine:
        logger.info("Closing database connections...")
        await async_engine.dispose()
        async_engine = None
        logger.info("Database connections closed")


class DatabaseService:
    """High-level database service for common operations."""
    
    @staticmethod
    async def execute_query(query: str, params: dict = None) -> list:
        """Execute a raw SQL query and return results."""
        async with get_async_session() as session:
            result = await session.execute(text(query), params or {})
            return result.fetchall()
    
    @staticmethod
    async def execute_insert(query: str, params: dict = None) -> Optional[int]:
        """Execute an insert query and return the inserted ID."""
        async with get_async_session() as session:
            result = await session.execute(text(query), params or {})
            await session.commit()
            return result.lastrowid
    
    @staticmethod
    async def execute_update(query: str, params: dict = None) -> int:
        """Execute an update query and return the number of affected rows."""
        async with get_async_session() as session:
            result = await session.execute(text(query), params or {})
            await session.commit()
            return result.rowcount
    
    @staticmethod
    async def get_latest_market_data(symbol: str, limit: int = 100) -> list:
        """Get latest market data for a symbol."""
        query = """
        SELECT symbol, timestamp, open, high, low, close, volume, vwap
        FROM market_data 
        WHERE symbol = :symbol 
        ORDER BY timestamp DESC 
        LIMIT :limit
        """
        return await DatabaseService.execute_query(query, {"symbol": symbol, "limit": limit})
    
    @staticmethod
    async def get_active_positions() -> list:
        """Get all active positions."""
        query = """
        SELECT symbol, 
               SUM(CASE WHEN position_type = 'LONG' THEN quantity ELSE -quantity END) as net_position,
               AVG(entry_price) as avg_entry_price,
               SUM(unrealized_pnl) as total_unrealized_pnl
        FROM positions 
        GROUP BY symbol
        HAVING SUM(CASE WHEN position_type = 'LONG' THEN quantity ELSE -quantity END) != 0
        """
        return await DatabaseService.execute_query(query)
    
    @staticmethod
    async def get_portfolio_metrics() -> dict:
        """Get latest portfolio metrics."""
        query = """
        SELECT * FROM portfolio_metrics 
        ORDER BY timestamp DESC 
        LIMIT 1
        """
        result = await DatabaseService.execute_query(query)
        return dict(result[0]) if result else {}
    
    @staticmethod
    async def get_risk_events(limit: int = 50) -> list:
        """Get recent risk events."""
        query = """
        SELECT * FROM risk_events 
        ORDER BY timestamp DESC 
        LIMIT :limit
        """
        return await DatabaseService.execute_query(query, {"limit": limit})
    
    @staticmethod
    async def insert_market_data(data: dict) -> None:
        """Insert market data efficiently."""
        query = """
        INSERT INTO market_data (symbol, timestamp, open, high, low, close, volume, vwap, source)
        VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume, :vwap, :source)
        ON CONFLICT (symbol, timestamp) DO UPDATE SET
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            vwap = EXCLUDED.vwap
        """
        await DatabaseService.execute_insert(query, data)
    
    @staticmethod
    async def insert_tick_data(data: dict) -> None:
        """Insert tick data efficiently."""
        query = """
        INSERT INTO tick_data (symbol, timestamp, price, size, side, exchange, source)
        VALUES (:symbol, :timestamp, :price, :size, :side, :exchange, :source)
        """
        await DatabaseService.execute_insert(query, data)
    
    @staticmethod
    async def insert_agent_signal(data: dict) -> str:
        """Insert agent signal and return the ID."""
        query = """
        INSERT INTO agent_signals (
            agent_name, symbol, timestamp, signal_type, confidence, strength,
            reasoning, features_used, prediction_horizon, target_price, stop_loss, take_profit
        ) VALUES (
            :agent_name, :symbol, :timestamp, :signal_type, :confidence, :strength,
            :reasoning, :features_used, :prediction_horizon, :target_price, :stop_loss, :take_profit
        ) RETURNING id
        """
        result = await DatabaseService.execute_query(query, data)
        return str(result[0][0]) if result else None
    
    @staticmethod
    async def update_position(symbol: str, data: dict) -> int:
        """Update position data."""
        query = """
        UPDATE positions SET
            current_price = :current_price,
            unrealized_pnl = :unrealized_pnl,
            risk_metrics = :risk_metrics
        WHERE symbol = :symbol
        """
        data["symbol"] = symbol
        return await DatabaseService.execute_update(query, data)
    
    @staticmethod
    async def get_symbol_watchlist() -> list:
        """Get symbols currently being watched."""
        query = """
        SELECT DISTINCT symbol 
        FROM market_data 
        WHERE timestamp > NOW() - INTERVAL '1 hour'
        ORDER BY symbol
        """
        result = await DatabaseService.execute_query(query)
        return [row[0] for row in result]