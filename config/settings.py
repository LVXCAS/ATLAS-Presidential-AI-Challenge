"""
Configuration settings for the LangGraph Trading System.
"""

from typing import Dict, List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    host: str = Field(default="localhost", alias="DB_HOST")
    port: int = Field(default=5432, alias="DB_PORT")
    database: str = Field(default="trading_system", alias="DB_NAME")
    username: str = Field(default="postgres", alias="DB_USER")
    password: str = Field(default="", alias="DB_PASSWORD")

    @property
    def url(self) -> str:
        """Get database connection URL."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"


class RedisSettings(BaseSettings):
    """Redis configuration settings."""

    host: str = Field(default="localhost", alias="REDIS_HOST")
    port: int = Field(default=6379, alias="REDIS_PORT")
    password: Optional[str] = Field(default=None, alias="REDIS_PASSWORD")
    db: int = Field(default=0, alias="REDIS_DB")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"


class BrokerSettings(BaseSettings):
    """Broker API configuration settings."""

    # Alpaca
    alpaca_api_key: str = Field(default="", alias="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(default="", alias="ALPACA_SECRET_KEY")
    alpaca_paper_base_url: str = Field(
        default="https://paper-api.alpaca.markets", alias="ALPACA_PAPER_BASE_URL"
    )
    alpaca_live_base_url: str = Field(
        default="https://api.alpaca.markets", alias="ALPACA_LIVE_BASE_URL"
    )

    # Polygon
    polygon_api_key: str = Field(default="", alias="POLYGON_API_KEY")

    # Satellite Data Provider
    satellite_api_key: str = Field(default="", alias="SATELLITE_API_KEY")
    satellite_api_base_url: str = Field(
        default="https://api.satellite-data.com/v1", alias="SATELLITE_API_BASE_URL"
    )

    # Interactive Brokers
    ib_host: str = Field(default="localhost", alias="IB_HOST")
    ib_port: int = Field(default=7497, alias="IB_PORT")
    ib_client_id: int = Field(default=1, alias="IB_CLIENT_ID")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"


class AISettings(BaseSettings):
    """AI/ML model configuration settings."""

    # OpenAI
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", alias="OPENAI_MODEL")

    # Google Gemini
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-pro", alias="GEMINI_MODEL")

    # DeepSeek
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"


class RiskSettings(BaseSettings):
    """Risk management configuration settings."""

    max_daily_loss_pct: float = Field(
        default=0.10, description="Maximum daily loss percentage"
    )
    max_position_size_pct: float = Field(
        default=0.05, description="Maximum position size percentage"
    )
    max_portfolio_leverage: float = Field(
        default=2.0, description="Maximum portfolio leverage"
    )
    var_confidence_level: float = Field(
        default=0.95, description="VaR confidence level"
    )
    emergency_stop_loss_pct: float = Field(
        default=0.15, description="Emergency stop loss percentage"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"


class TradingSettings(BaseSettings):
    """Trading system configuration settings."""

    # Trading modes
    paper_trading: bool = Field(default=True, alias="PAPER_TRADING")
    live_trading: bool = Field(default=False, alias="LIVE_TRADING")

    # Portfolio settings
    initial_capital: float = Field(default=10000.0, alias="INITIAL_CAPITAL")
    target_symbols: List[str] = Field(default_factory=lambda: ["SPY", "QQQ", "IWM"])

    # Strategy settings
    momentum_weight: float = Field(default=0.25, description="Momentum strategy weight")
    mean_reversion_weight: float = Field(
        default=0.25, description="Mean reversion strategy weight"
    )
    sentiment_weight: float = Field(
        default=0.20, description="Sentiment strategy weight"
    )
    options_weight: float = Field(default=0.15, description="Options strategy weight")
    short_selling_weight: float = Field(
        default=0.10, description="Short selling strategy weight"
    )
    long_term_weight: float = Field(
        default=0.05, description="Long term strategy weight"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"


class Settings(BaseSettings):
    """Main application settings."""

    # Application info
    app_name: str = "LangGraph Trading System"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, alias="DEBUG")

    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    brokers: BrokerSettings = Field(default_factory=BrokerSettings)
    ai: AISettings = Field(default_factory=AISettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, alias="LOG_FILE")

    # Convenience properties for broker access
    @property
    def ALPACA_API_KEY(self) -> str:
        return self.brokers.alpaca_api_key
    
    @property
    def ALPACA_SECRET_KEY(self) -> str:
        return self.brokers.alpaca_secret_key
    
    @property
    def ALPACA_PAPER_BASE_URL(self) -> str:
        return self.brokers.alpaca_paper_base_url
    
    @property
    def ALPACA_LIVE_BASE_URL(self) -> str:
        return self.brokers.alpaca_live_base_url

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
