"""Data configuration settings."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, timedelta


class DataSourceConfig(BaseModel):
    """Configuration for data sources."""
    
    name: str
    enabled: bool = True
    rate_limit: Optional[int] = None  # requests per minute
    timeout: int = 30  # seconds
    retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds


class MarketDataConfig(BaseModel):
    """Market data configuration."""
    
    # Data sources
    sources: Dict[str, DataSourceConfig] = {
        "alpaca": DataSourceConfig(
            name="alpaca",
            rate_limit=200,
            timeout=30
        ),
        "polygon": DataSourceConfig(
            name="polygon", 
            rate_limit=5,
            timeout=30
        ),
        "yahoo": DataSourceConfig(
            name="yahoo",
            rate_limit=2000,
            timeout=30
        ),
        "alpha_vantage": DataSourceConfig(
            name="alpha_vantage",
            rate_limit=5,
            timeout=30
        ),
        "fred": DataSourceConfig(
            name="fred",
            rate_limit=120,
            timeout=30
        )
    }
    
    # Default symbols to track
    default_symbols: List[str] = [
        "SPY", "QQQ", "IWM", "DIA",  # ETFs
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",  # Tech
        "JPM", "BAC", "WFC", "GS",  # Finance
        "XOM", "CVX", "COP",  # Energy
        "JNJ", "PFE", "UNH",  # Healthcare
    ]
    
    # Crypto symbols
    crypto_symbols: List[str] = [
        "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"
    ]
    
    # Data frequencies
    frequencies: List[str] = ["1min", "5min", "15min", "1hour", "1day"]
    default_frequency: str = "1min"
    
    # Historical data lookback
    max_lookback_days: int = 365 * 5  # 5 years
    default_lookback_days: int = 252  # 1 trading year


class DatabaseConfig(BaseModel):
    """Database storage configuration."""
    
    # Table configurations
    tables: Dict[str, Dict] = {
        "bars": {
            "partition_by": "date",
            "index_columns": ["symbol", "timestamp"],
            "compression": "lz4"
        },
        "trades": {
            "partition_by": "date", 
            "index_columns": ["symbol", "timestamp"],
            "compression": "lz4"
        },
        "quotes": {
            "partition_by": "date",
            "index_columns": ["symbol", "timestamp"], 
            "compression": "lz4"
        },
        "fundamentals": {
            "partition_by": "quarter",
            "index_columns": ["symbol", "report_date"],
            "compression": "zstd"
        }
    }
    
    # Retention policies
    retention_days: Dict[str, int] = {
        "1min": 90,      # 3 months of minute data
        "5min": 365,     # 1 year of 5min data
        "15min": 730,    # 2 years of 15min data
        "1hour": 1825,   # 5 years of hourly data
        "1day": 7300,    # 20 years of daily data
    }
    
    # Performance settings
    batch_insert_size: int = 10000
    connection_pool_size: int = 10
    max_overflow: int = 20


class DataConfig(BaseModel):
    """Main data configuration."""
    
    market: MarketDataConfig = MarketDataConfig()
    database: DatabaseConfig = DatabaseConfig()
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    cache_max_size: int = 1000000  # 1M items
    
    # Data validation
    validate_data: bool = True
    remove_outliers: bool = True
    outlier_threshold: float = 5.0  # standard deviations
    
    # Real-time data
    realtime_enabled: bool = True
    websocket_reconnect_delay: int = 5  # seconds
    max_reconnect_attempts: int = 10