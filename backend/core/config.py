"""
Core Configuration for Bloomberg Terminal API
Environment-based configuration management with Pydantic settings.
"""

import os
from typing import List, Optional
from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseModel):
    """Database configuration."""
    url: str = Field(default="postgresql+asyncpg://trading_user:trading_password@localhost:5432/bloomberg_trading")
    pool_size: int = Field(default=20)
    max_overflow: int = Field(default=0)
    pool_pre_ping: bool = Field(default=True)
    echo: bool = Field(default=False)


class RedisSettings(BaseModel):
    """Redis configuration."""
    url: str = Field(default="redis://localhost:6379/0")
    encoding: str = Field(default="utf-8")
    decode_responses: bool = Field(default=True)
    socket_keepalive: bool = Field(default=True)
    socket_keepalive_options: dict = Field(default_factory=lambda: {"TCP_KEEPIDLE": 1, "TCP_KEEPINTVL": 3, "TCP_KEEPCNT": 5})
    max_connections: int = Field(default=100)


class TradingSettings(BaseModel):
    """Trading configuration."""
    paper_trading: bool = Field(default=True)
    max_position_size: float = Field(default=10000.0)
    max_portfolio_value: float = Field(default=1000000.0)
    risk_limit: float = Field(default=0.02)  # 2% max daily loss
    max_drawdown: float = Field(default=0.10)  # 10% max drawdown


class AlpacaSettings(BaseModel):
    """Alpaca broker configuration."""
    api_key: Optional[str] = Field(default=None)
    secret_key: Optional[str] = Field(default=None)
    base_url: str = Field(default="https://paper-api.alpaca.markets")  # Paper trading by default
    data_url: str = Field(default="https://data.alpaca.markets")


class PolygonSettings(BaseModel):
    """Polygon data provider configuration."""
    api_key: Optional[str] = Field(default=None)
    base_url: str = Field(default="https://api.polygon.io")


class WebSocketSettings(BaseModel):
    """WebSocket configuration."""
    port: int = Field(default=8001)
    max_connections: int = Field(default=1000)
    ping_interval: int = Field(default=20)
    ping_timeout: int = Field(default=10)


class MonitoringSettings(BaseModel):
    """Monitoring and observability configuration."""
    prometheus_port: int = Field(default=9090)
    log_level: str = Field(default="INFO")
    structured_logging: bool = Field(default=True)
    metrics_enabled: bool = Field(default=True)


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application
    app_name: str = Field(default="Bloomberg Terminal API")
    version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    environment: str = Field(default="development")
    
    # API
    api_port: int = Field(default=8000)
    allowed_origins: List[str] = Field(default=["http://localhost:3000"])
    api_key: Optional[str] = Field(default=None)
    
    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    alpaca: AlpacaSettings = Field(default_factory=AlpacaSettings)
    polygon: PolygonSettings = Field(default_factory=PolygonSettings)
    websocket: WebSocketSettings = Field(default_factory=WebSocketSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    # Performance settings
    max_workers: int = Field(default=4)
    request_timeout: int = Field(default=30)
    
    # Security
    secret_key: str = Field(default="bloomberg_terminal_secret_key_change_in_production")
    access_token_expire_minutes: int = Field(default=30)
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False
        
        # Environment variable mapping
        fields = {
            "database": {"env": "DATABASE"},
            "redis": {"env": "REDIS"},
            "trading": {"env": "TRADING"},
            "alpaca": {"env": "ALPACA"},
            "polygon": {"env": "POLYGON"},
            "websocket": {"env": "WEBSOCKET"},
            "monitoring": {"env": "MONITORING"}
        }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Override with environment variables
        self._load_from_environment()
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Database
        if os.getenv("DATABASE_URL"):
            self.database.url = os.getenv("DATABASE_URL")
        
        # Redis
        if os.getenv("REDIS_URL"):
            self.redis.url = os.getenv("REDIS_URL")
        
        # Alpaca
        if os.getenv("ALPACA_API_KEY"):
            self.alpaca.api_key = os.getenv("ALPACA_API_KEY")
        if os.getenv("ALPACA_SECRET_KEY"):
            self.alpaca.secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        # Polygon
        if os.getenv("POLYGON_API_KEY"):
            self.polygon.api_key = os.getenv("POLYGON_API_KEY")
        
        # Environment-specific overrides
        if self.environment == "production":
            self.debug = False
            self.database.echo = False
            self.trading.paper_trading = False
            self.alpaca.base_url = "https://api.alpaca.markets"
            self.monitoring.log_level = "WARNING"
        elif self.environment == "development":
            self.debug = True
            self.database.echo = True
            self.monitoring.log_level = "DEBUG"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    def get_database_url(self) -> str:
        """Get database URL with proper async driver."""
        return self.database.url
    
    def get_redis_url(self) -> str:
        """Get Redis URL."""
        return self.redis.url


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()