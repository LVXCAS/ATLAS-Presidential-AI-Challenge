"""Environment-specific configurations."""

import os
from typing import Type, Union
from .base import BaseConfig


class DevelopmentConfig(BaseConfig):
    """Development environment configuration."""
    
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    
    # Use paper trading by default
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"
    
    # Relaxed risk limits for testing
    MAX_POSITION_SIZE: float = 0.05  # 5% for development
    MAX_DAILY_LOSS: float = 0.01     # 1% max daily loss
    
    # Faster cache expiration for development
    CACHE_TTL: int = 300  # 5 minutes


class TestingConfig(BaseConfig):
    """Testing environment configuration."""
    
    ENVIRONMENT: str = "testing"
    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"
    
    # Use test database
    DATABASE_URL: str = "sqlite:///test_quant_research.db"
    REDIS_URL: str = "redis://localhost:6379/1"
    
    # Conservative settings for tests
    MAX_WORKERS: int = 1
    BATCH_SIZE: int = 100
    CACHE_TTL: int = 60  # 1 minute


class ProductionConfig(BaseConfig):
    """Production environment configuration."""
    
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Live trading (be careful!)
    ALPACA_BASE_URL: str = "https://api.alpaca.markets"
    
    # Strict risk management
    MAX_POSITION_SIZE: float = 0.02  # 2% max position size
    MAX_DAILY_LOSS: float = 0.005    # 0.5% max daily loss
    
    # Production optimizations
    MAX_WORKERS: int = 8
    BATCH_SIZE: int = 5000
    CACHE_TTL: int = 7200  # 2 hours
    
    # Enable alerts
    EMAIL_ALERTS: bool = True


class ResearchConfig(BaseConfig):
    """Research environment for backtesting and analysis."""
    
    ENVIRONMENT: str = "research"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # No live trading in research
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"
    
    # Generous limits for research
    MAX_POSITION_SIZE: float = 1.0   # No limit for backtesting
    MAX_DAILY_LOSS: float = 1.0      # No limit for backtesting
    
    # High performance for large datasets
    MAX_WORKERS: int = 16
    BATCH_SIZE: int = 10000
    CACHE_TTL: int = 86400  # 24 hours


def get_config() -> Type[BaseConfig]:
    """Get configuration based on environment variable."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    config_map = {
        "development": DevelopmentConfig,
        "testing": TestingConfig,
        "production": ProductionConfig,
        "research": ResearchConfig,
    }
    
    config_class = config_map.get(env, DevelopmentConfig)
    return config_class()


# Global settings instance
settings = get_config()