"""
Mock database implementation for development without external dependencies.
"""

import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from config.logging_config import get_logger

logger = get_logger(__name__)


class MockPostgresSession:
    """Mock PostgreSQL session for development."""
    
    def __init__(self):
        self.data = {}
    
    def execute(self, query):
        """Mock execute method."""
        logger.debug(f"Mock SQL execution: {query}")
        return MockResult(1)
    
    def commit(self):
        """Mock commit method."""
        logger.debug("Mock commit")
    
    def rollback(self):
        """Mock rollback method."""
        logger.debug("Mock rollback")
    
    def close(self):
        """Mock close method."""
        logger.debug("Mock session close")


class MockResult:
    """Mock SQL result."""
    
    def __init__(self, value):
        self.value = value
    
    def scalar(self):
        return self.value


class MockRedisClient:
    """Mock Redis client for development."""
    
    def __init__(self):
        self.data = {}
    
    async def ping(self):
        """Mock ping method."""
        logger.debug("Mock Redis ping")
        return True
    
    async def get(self, key):
        """Mock get method."""
        return self.data.get(key)
    
    async def set(self, key, value):
        """Mock set method."""
        self.data[key] = value
        return True
    
    async def close(self):
        """Mock close method."""
        logger.debug("Mock Redis close")


class MockDatabaseManager:
    """Mock database manager for development."""
    
    def __init__(self):
        self._initialized = False
        self.postgres_session = None
        self.redis_client = None
    
    async def initialize(self):
        """Initialize mock connections."""
        if self._initialized:
            return
        
        logger.info("Initializing mock database connections...")
        
        # Mock initialization
        self.postgres_session = MockPostgresSession()
        self.redis_client = MockRedisClient()
        
        self._initialized = True
        logger.info("Mock database connections initialized successfully")
    
    def get_postgres_session(self):
        """Get a mock PostgreSQL session."""
        if not self._initialized:
            raise RuntimeError("Mock database not initialized")
        
        return self.postgres_session
    
    @asynccontextmanager
    async def get_redis_client(self):
        """Get a mock Redis client."""
        if not self._initialized:
            await self.initialize()
        
        try:
            yield self.redis_client
        finally:
            pass  # No cleanup needed for mock
    
    async def close(self):
        """Close mock connections."""
        if self.redis_client:
            await self.redis_client.close()
        
        if self.postgres_session:
            self.postgres_session.close()
        
        self._initialized = False
        logger.info("Mock database connections closed")


# Global mock database manager instance
mock_db_manager = MockDatabaseManager()


# Mock convenience functions
def get_postgres_session():
    """Get a mock PostgreSQL session."""
    return mock_db_manager.get_postgres_session()


async def get_redis_client():
    """Get a mock Redis client."""
    async with mock_db_manager.get_redis_client() as client:
        yield client


async def init_database():
    """Initialize mock database connections."""
    await mock_db_manager.initialize()


async def close_database():
    """Close mock database connections."""
    await mock_db_manager.close()


# Mock health check functions
async def check_postgres_health() -> bool:
    """Check mock PostgreSQL connection health."""
    try:
        session = mock_db_manager.get_postgres_session()
        result = session.execute("SELECT 1")
        return result.scalar() == 1
    except Exception as e:
        logger.error(f"Mock PostgreSQL health check failed: {e}")
        return False


async def check_redis_health() -> bool:
    """Check mock Redis connection health."""
    try:
        async with mock_db_manager.get_redis_client() as client:
            await client.ping()
            return True
    except Exception as e:
        logger.error(f"Mock Redis health check failed: {e}")
        return False


async def check_database_health() -> dict:
    """Check overall mock database health."""
    postgres_healthy = await check_postgres_health()
    redis_healthy = await check_redis_health()
    
    return {
        "postgres": postgres_healthy,
        "redis": redis_healthy,
        "overall": postgres_healthy and redis_healthy
    }