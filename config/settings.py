"""
Configuration settings for the LangGraph Trading System.
"""

from typing import Dict, List, Optional
from pydantic import BaseSettings, Field
from pydantic_settings import BaseSettings as PydanticBaseSettings


class DatabaseSettings(PydanticBaseSettings):
    """Database configuration settings."""
    
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    database: str = Field(default="trading_system", env="DB_NAME")
    username: str = Field(default="postgres", env="DB_USER")
    password: str = Field(default="", env="DB_PASSWORD")
    
    @property
    def url(self) -> str:
        """Get database connection URL."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisSettings(PydanticBaseSettings):
    """Redis configuration settings."""
    
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")


class BrokerSettings(PydanticBaseSettings):
    """Broker API configuration settings."""
    
    # Alpaca
    alpaca_api_key: str = Field(default="", env="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(default="", env="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field(default="https://paper-api.alpaca.markets", env="ALPACA_BASE_URL")
    
    # Polygon
    polygon_api_key: str = Field(default="", env="POLYGON_API_KEY")
    
    # Interactive Brokers
    ib_host: str = Field(default="localhost", env="IB_HOST")
    ib_port: int = Field(default=7497, env="IB_PORT")
    ib_client_id: int = Field(default=1, env="IB_CLIENT_ID")


class AISettings(PydanticBaseSettings):
    """AI/ML model configuration settings."""
    
    # OpenAI
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    
    # Google Gemini
    google_api_key: str = Field(default="", env="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-pro", env="GEMINI_MODEL")
    
    # DeepSeek
    deepseek_api_key: str = Field(default="", env="DEEPSEEK_API_KEY")
    deepseek_model: str = Field(default="deepseek-chat", env="DEEPSEEK_MODEL")


class RiskSettings(PydanticBaseSettings):
    """Risk management configuration settings."""
    
    max_daily_loss_pct: float = Field(default=0.10, description="Maximum daily loss percentage")
    max_position_size_pct: float = Field(default=0.05, description="Maximum position size percentage")
    max_portfolio_leverage: float = Field(default=2.0, description="Maximum portfolio leverage")
    var_confidence_level: float = Field(default=0.95, description="VaR confidence level")
    emergency_stop_loss_pct: float = Field(default=0.15, description="Emergency stop loss percentage")


class TradingSettings(PydanticBaseSettings):
    """Trading system configuration settings."""
    
    # Trading modes
    paper_trading: bool = Field(default=True, env="PAPER_TRADING")
    live_trading: bool = Field(default=False, env="LIVE_TRADING")
    
    # Portfolio settings
    initial_capital: float = Field(default=10000.0, env="INITIAL_CAPITAL")
    target_symbols: List[str] = Field(default_factory=lambda: ["SPY", "QQQ", "IWM"])
    
    # Strategy settings
    momentum_weight: float = Field(default=0.25, description="Momentum strategy weight")
    mean_reversion_weight: float = Field(default=0.25, description="Mean reversion strategy weight")
    sentiment_weight: float = Field(default=0.20, description="Sentiment strategy weight")
    options_weight: float = Field(default=0.15, description="Options strategy weight")
    short_selling_weight: float = Field(default=0.10, description="Short selling strategy weight")
    long_term_weight: float = Field(default=0.05, description="Long term strategy weight")


class Settings(PydanticBaseSettings):
    """Main application settings."""
    
    # Application info
    app_name: str = "LangGraph Trading System"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    brokers: BrokerSettings = Field(default_factory=BrokerSettings)
    ai: AISettings = Field(default_factory=AISettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()