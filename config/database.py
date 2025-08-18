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

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


class DatabaseManager:
    """Database connection manager for PostgreSQL and Redis."""
    
    def __init__(self):
        self.postgres_engine = None
        self.postgres_async_engine = None
        self.postgres_session_factory = None
        self.postgres_async_session_factory = None
        self.redis_pool = None
        self._initialized = False
    
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
            
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            raise
    
    async def _init_postgres(self):
        """Initialize PostgreSQL connections."""
        # Synchronous engine for migrations and setup
        self.postgres_engine = create_engine(
            settings.database.url,
            echo=settings.debug
        )
        
        # Session factory
        self.postgres_session_factory = sessionmaker(
            bind=self.postgres_engine,
            expire_on_commit=False
        )
    
    async def _init_redis(self):
        """Initialize Redis connection pool."""
        redis_url = f"redis://:{settings.redis.password or ''}@{settings.redis.host}:{settings.redis.port}/{settings.redis.db}"
        
        self.redis_pool = redis.ConnectionPool.from_url(
            redis_url,
            max_connections=10,  # Default value
            socket_timeout=5,    # Default value
            decode_responses=True
        )
    
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
        logger.info("Database connections closed")


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