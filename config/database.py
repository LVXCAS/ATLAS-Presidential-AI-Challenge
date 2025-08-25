"""
Database connection and management for the LangGraph Trading System.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

import redis.asyncio as redis
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config.logging_config import get_logger
from config.settings import settings
from config.database_optimization import db_optimizer

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


class DatabaseManager:
    """Database connection manager for PostgreSQL and Redis with optimizations."""
    
    def __init__(self):
        self.postgres_engine = None
        self.postgres_async_engine = None
        self.postgres_session_factory = None
        self.postgres_async_session_factory = None
        self.redis_pool = None
        self._initialized = False
        self._optimized = False
    
    async def initialize(self):
        """Initialize database connections."""
        if self._initialized:
            return
        
        try:
            # Initialize PostgreSQL connections
            await self._init_postgres()
            
            # Initialize Redis connection
            await self._init_redis()
            
            # Test connections
            await self._test_connections()
            
            self._initialized = True
            logger.info("Database connections initialized successfully")
            
            # Apply production optimizations
            await self._apply_optimizations()
            
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            raise
    
    async def _init_postgres(self):
        """Initialize optimized PostgreSQL connections."""
        # Use optimized engine for production performance
        self.postgres_engine = db_optimizer.get_optimized_postgres_engine()
        
        # Session factory with optimized settings
        self.postgres_session_factory = sessionmaker(
            bind=self.postgres_engine,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False
        )
    
    async def _init_redis(self):
        """Initialize optimized Redis connection pool."""
        # Use optimized Redis pool for production performance
        self.redis_pool = await db_optimizer.get_optimized_redis_pool()
    
    async def _test_connections(self):
        """Test database connections."""
        # Test PostgreSQL
        with self.postgres_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
        
        # Test Redis
        redis_client = redis.Redis(connection_pool=self.redis_pool)
        await redis_client.ping()
        await redis_client.close()
    
    def get_postgres_session(self):
        """Get a PostgreSQL session."""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        
        return self.postgres_session_factory()
    
    @asynccontextmanager
    async def get_redis_client(self) -> AsyncGenerator[redis.Redis, None]:
        """Get a Redis client."""
        if not self._initialized:
            await self.initialize()
        
        client = redis.Redis(connection_pool=self.redis_pool)
        try:
            yield client
        finally:
            await client.close()
    
    async def close(self):
        """Close all database connections."""
        if self.postgres_engine:
            self.postgres_engine.dispose()
        
        if self.redis_pool:
            await self.redis_pool.disconnect()
        
        self._initialized = False
        self._optimized = False
        logger.info("Database connections closed")
    
    async def _apply_optimizations(self):
        """Apply database optimizations for production performance."""
        if self._optimized:
            return
            
        try:
            logger.info("🔧 Applying production database optimizations...")
            
            # Apply PostgreSQL optimizations
            await db_optimizer.optimize_postgres_settings(self.postgres_engine)
            await db_optimizer.create_database_indexes(self.postgres_engine)
            await db_optimizer.create_performance_views(self.postgres_engine)
            
            self._optimized = True
            logger.info("✅ Database optimizations applied successfully")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not apply all database optimizations: {e}")
    
    async def get_performance_metrics(self) -> dict:
        """Get database performance metrics."""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
            
        return await db_optimizer.monitor_database_performance(self.postgres_engine)
    
    async def analyze_performance(self) -> dict:
        """Analyze database performance and get recommendations."""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
            
        return await db_optimizer.analyze_query_performance(self.postgres_engine)


# Global database manager instance
db_manager = DatabaseManager()


# Convenience functions
def get_postgres_session():
    """Get a PostgreSQL session."""
    return db_manager.get_postgres_session()


async def get_redis_client() -> AsyncGenerator[redis.Redis, None]:
    """Get a Redis client."""
    async with db_manager.get_redis_client() as client:
        yield client


async def init_database():
    """Initialize database connections."""
    await db_manager.initialize()


async def close_database():
    """Close database connections."""
    await db_manager.close()


# Health check functions
async def check_postgres_health() -> bool:
    """Check PostgreSQL connection health."""
    try:
        with db_manager.postgres_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {e}")
        return False


async def check_redis_health() -> bool:
    """Check Redis connection health."""
    try:
        async with db_manager.get_redis_client() as client:
            await client.ping()
            return True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False


async def check_database_health() -> dict:
    """Check overall database health."""
    postgres_healthy = await check_postgres_health()
    redis_healthy = await check_redis_health()
    
    return {
        "postgres": postgres_healthy,
        "redis": redis_healthy,
        "overall": postgres_healthy and redis_healthy
    }


def get_database_config() -> dict:
    """Get database configuration for agents.
    
    Returns:
        dict: Database configuration settings
    """
    return {
        "postgres_url": db_manager.postgres_url,
        "redis_host": db_manager.redis_host,
        "redis_port": db_manager.redis_port,
        "redis_password": db_manager.redis_password,
        "redis_db": db_manager.redis_db,
        "debug": db_manager.debug
    }
