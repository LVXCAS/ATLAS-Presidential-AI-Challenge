"""Base configuration settings."""

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any


class BaseConfig(BaseSettings):
    """Base configuration with common settings."""
    
    # Environment
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Project paths
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    MODELS_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "models")
    RESULTS_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "results")
    LOGS_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    
    # Database
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")
    REDIS_URL: Optional[str] = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # API Keys
    ALPACA_API_KEY: Optional[str] = Field(default=None, env="ALPACA_API_KEY")
    ALPACA_SECRET_KEY: Optional[str] = Field(default=None, env="ALPACA_SECRET_KEY")
    ALPACA_BASE_URL: str = Field(default="https://paper-api.alpaca.markets", env="ALPACA_BASE_URL")
    
    POLYGON_API_KEY: Optional[str] = Field(default=None, env="POLYGON_API_KEY")
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    FRED_API_KEY: Optional[str] = Field(default=None, env="FRED_API_KEY")
    
    # Performance settings
    MAX_WORKERS: int = Field(default=4, env="MAX_WORKERS")
    BATCH_SIZE: int = Field(default=1000, env="BATCH_SIZE")
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # seconds
    
    # Risk management
    MAX_POSITION_SIZE: float = Field(default=0.1, env="MAX_POSITION_SIZE")  # 10% of portfolio
    MAX_DAILY_LOSS: float = Field(default=0.02, env="MAX_DAILY_LOSS")  # 2% max daily loss
    
    # Notifications
    SLACK_WEBHOOK_URL: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")
    EMAIL_ALERTS: bool = Field(default=False, env="EMAIL_ALERTS")
    
    # Additional configuration fields
    PORTFOLIO_RISK_LIMIT: float = Field(default=0.15, env="PORTFOLIO_RISK_LIMIT")
    MODEL_REGISTRY_PATH: str = Field(default="models/registry", env="MODEL_REGISTRY_PATH")
    FEATURE_CACHE_SIZE: int = Field(default=100000, env="FEATURE_CACHE_SIZE")
    TRAINING_DATA_PATH: str = Field(default="data/training", env="TRAINING_DATA_PATH")
    DEFAULT_COMMISSION: float = Field(default=0.001, env="DEFAULT_COMMISSION")
    DEFAULT_SLIPPAGE: float = Field(default=0.0005, env="DEFAULT_SLIPPAGE")
    DEFAULT_INITIAL_CAPITAL: float = Field(default=100000.0, env="DEFAULT_INITIAL_CAPITAL")
    ENCRYPT_API_KEYS: bool = Field(default=True, env="ENCRYPT_API_KEYS")
    LOG_SENSITIVE_DATA: bool = Field(default=False, env="LOG_SENSITIVE_DATA")
    ENABLE_PROFILING: bool = Field(default=False, env="ENABLE_PROFILING")
    DEBUG_MODE: bool = Field(default=False, env="DEBUG_MODE")
    MOCK_TRADING: bool = Field(default=True, env="MOCK_TRADING")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    def __post_init__(self):
        """Create necessary directories."""
        for dir_path in [self.DATA_DIR, self.MODELS_DIR, self.RESULTS_DIR, self.LOGS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)