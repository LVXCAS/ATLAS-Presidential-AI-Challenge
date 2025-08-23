"""
Historical data loader with multiple data sources and caching capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available - using mock data")

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Historical data loader supporting multiple data sources:
    - Yahoo Finance
    - CSV files
    - Database connections
    - Mock data generation
    """
    
    def __init__(self, cache_dir: str = "data/cache", config: Dict[str, Any] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        
        # Data source priorities
        self.data_sources = self.config.get('data_sources', ['yfinance', 'csv', 'mock'])
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.cache_ttl_hours = self.config.get('cache_ttl_hours', 24)
        
        logger.info(f"DataLoader initialized with sources: {self.data_sources}")
    
    async def load_historical_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = '1d',
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Load historical data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval ('1d', '1h', '5m', etc.)
            force_refresh: Force refresh of cached data
        
        Returns:
            Dictionary mapping symbol to DataFrame with OHLCV data
        """
        results = {}
        
        for symbol in symbols:
            try:
                data = await self.load_symbol_data(
                    symbol, start_date, end_date, interval, force_refresh
                )
                if data is not None and not data.empty:
                    results[symbol] = data
                else:
                    logger.warning(f"No data loaded for {symbol}")
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
        
        logger.info(f"Loaded data for {len(results)} symbols")
        return results
    
    async def load_symbol_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = '1d',
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """Load historical data for a single symbol."""
        
        # Check cache first
        if self.cache_enabled and not force_refresh:
            cached_data = self._load_from_cache(symbol, start_date, end_date, interval)
            if cached_data is not None:
                logger.debug(f"Loaded {symbol} data from cache")
                return cached_data
        
        # Try each data source in order
        for source in self.data_sources:
            try:
                if source == 'yfinance':
                    data = await self._load_from_yfinance(symbol, start_date, end_date, interval)
                elif source == 'csv':
                    data = await self._load_from_csv(symbol, start_date, end_date, interval)
                elif source == 'mock':
                    data = await self._generate_mock_data(symbol, start_date, end_date, interval)
                else:
                    logger.warning(f"Unknown data source: {source}")
                    continue
                
                if data is not None and not data.empty:
                    # Cache the data
                    if self.cache_enabled:
                        self._save_to_cache(symbol, start_date, end_date, interval, data)
                    
                    logger.debug(f"Loaded {len(data)} bars for {symbol} from {source}")
                    return data
                
            except Exception as e:
                logger.warning(f"Failed to load {symbol} from {source}: {e}")
                continue
        
        logger.error(f"Failed to load data for {symbol} from all sources")
        return None
    
    async def _load_from_yfinance(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Load data from Yahoo Finance."""
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date.date(),
                end=end_date.date(),
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                return None
            
            # Standardize column names
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in data.columns:
                    if col == 'volume':
                        data[col] = 0  # Some assets don't have volume
                    else:
                        logger.error(f"Missing required column {col} for {symbol}")
                        return None
            
            # Clean data
            data = data.dropna()
            data = data[data['close'] > 0]  # Remove invalid prices
            
            return data[required_columns]
            
        except Exception as e:
            logger.error(f"Error loading {symbol} from yfinance: {e}")
            return None
    
    async def _load_from_csv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Load data from CSV files."""
        csv_path = self.cache_dir / f"{symbol}_{interval}.csv"
        
        if not csv_path.exists():
            return None
        
        try:
            data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            
            # Filter by date range
            data = data[(data.index >= start_date) & (data.index <= end_date)]
            
            if data.empty:
                return None
            
            # Standardize column names
            data.columns = data.columns.str.lower()
            
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                logger.error(f"CSV file {csv_path} missing required columns")
                return None
            
            return data[required_columns]
            
        except Exception as e:
            logger.error(f"Error loading {symbol} from CSV: {e}")
            return None
    
    async def _generate_mock_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> pd.DataFrame:
        """Generate realistic mock market data."""
        
        # Determine time delta based on interval
        if interval == '1d':
            freq = 'D'
            delta = timedelta(days=1)
        elif interval == '1h':
            freq = 'H'
            delta = timedelta(hours=1)
        elif interval == '5m':
            freq = '5min'
            delta = timedelta(minutes=5)
        else:
            freq = 'D'
            delta = timedelta(days=1)
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        if len(dates) == 0:
            return pd.DataFrame()
        
        # Set random seed for reproducible data
        np.random.seed(hash(symbol) % 2**32)
        
        # Market parameters
        initial_price = 100.0 + (hash(symbol) % 100)  # $100-200 base
        annual_return = 0.08  # 8% annual return
        annual_volatility = 0.25  # 25% annual volatility
        
        # Adjust for interval
        periods_per_year = 252 if interval == '1d' else 252 * 24 if interval == '1h' else 252 * 24 * 12
        dt = 1 / periods_per_year
        drift = (annual_return - 0.5 * annual_volatility**2) * dt
        vol = annual_volatility * np.sqrt(dt)
        
        # Generate random walks
        n_periods = len(dates)
        returns = np.random.normal(drift, vol, n_periods)
        
        # Add some trend and mean reversion
        trend = np.linspace(-0.1, 0.1, n_periods) * dt
        returns += trend
        
        # Calculate prices using geometric Brownian motion
        log_returns = returns
        log_prices = np.log(initial_price) + np.cumsum(log_returns)
        close_prices = np.exp(log_prices)
        
        # Generate OHLC from close prices
        data = []
        for i, (date, close) in enumerate(zip(dates, close_prices)):
            if i == 0:
                prev_close = initial_price
            else:
                prev_close = close_prices[i-1]
            
            # Generate intraday price action
            daily_range = abs(close - prev_close) * np.random.uniform(1.5, 3.0)
            
            high = max(prev_close, close) + daily_range * np.random.uniform(0, 0.5)
            low = min(prev_close, close) - daily_range * np.random.uniform(0, 0.5)
            open_price = prev_close + (close - prev_close) * np.random.uniform(-0.2, 0.2)
            
            # Ensure OHLC relationships are valid
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Generate volume based on price movement
            price_change_pct = abs(close - prev_close) / prev_close
            base_volume = 100000
            volume = int(base_volume * (1 + price_change_pct * 5) * np.random.uniform(0.5, 2.0))
            
            data.append({
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        
        logger.debug(f"Generated {len(df)} bars of mock data for {symbol}")
        return df
    
    def _get_cache_path(self, symbol: str, start_date: datetime, end_date: datetime, interval: str) -> Path:
        """Generate cache file path."""
        date_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        filename = f"{symbol}_{interval}_{date_str}.json"
        return self.cache_dir / filename
    
    def _load_from_cache(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Load data from cache if available and not expired."""
        cache_path = self._get_cache_path(symbol, start_date, end_date, interval)
        
        if not cache_path.exists():
            return None
        
        # Check if cache is expired
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age > timedelta(hours=self.cache_ttl_hours):
            logger.debug(f"Cache expired for {symbol}")
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            df = pd.DataFrame(cache_data['data'])
            df.index = pd.to_datetime(df.index)
            
            return df
            
        except Exception as e:
            logger.warning(f"Error loading cache for {symbol}: {e}")
            return None
    
    def _save_to_cache(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
        data: pd.DataFrame
    ):
        """Save data to cache."""
        cache_path = self._get_cache_path(symbol, start_date, end_date, interval)
        
        try:
            cache_data = {
                'symbol': symbol,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'interval': interval,
                'cached_at': datetime.now().isoformat(),
                'data': data.to_dict('index')
            }
            
            # Convert timestamps to strings for JSON serialization
            serializable_data = {}
            for timestamp, values in cache_data['data'].items():
                serializable_data[timestamp.isoformat()] = values
            cache_data['data'] = serializable_data
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.debug(f"Saved {symbol} data to cache")
            
        except Exception as e:
            logger.warning(f"Error saving cache for {symbol}: {e}")
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached data."""
        if symbol:
            # Clear cache for specific symbol
            pattern = f"{symbol}_*.json"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
            logger.info(f"Cleared cache for {symbol}")
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Cleared all cached data")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data."""
        cache_files = list(self.cache_dir.glob("*.json"))
        
        info = {
            'cache_dir': str(self.cache_dir),
            'total_files': len(cache_files),
            'cache_size_mb': sum(f.stat().st_size for f in cache_files) / 1024 / 1024,
            'files': []
        }
        
        for cache_file in cache_files:
            try:
                stat = cache_file.stat()
                info['files'].append({
                    'filename': cache_file.name,
                    'size_mb': stat.st_size / 1024 / 1024,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'age_hours': (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).total_seconds() / 3600
                })
            except Exception as e:
                logger.warning(f"Error getting info for {cache_file}: {e}")
        
        return info


async def create_data_feed(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    interval: str = '1d',
    data_loader: Optional[DataLoader] = None
) -> callable:
    """
    Create a data feed function for backtesting.
    
    Returns a function that takes a datetime and returns market data for that timestamp.
    """
    if data_loader is None:
        data_loader = DataLoader()
    
    # Load all historical data
    historical_data = await data_loader.load_historical_data(
        symbols, start_date, end_date, interval
    )
    
    def data_feed(timestamp: datetime) -> Dict[str, Dict[str, float]]:
        """Data feed function that returns market data for a specific timestamp."""
        result = {}
        
        for symbol, data in historical_data.items():
            # Find the closest data point to the requested timestamp
            if timestamp in data.index:
                row = data.loc[timestamp]
            else:
                # Find the closest timestamp before the requested time
                available_times = data.index[data.index <= timestamp]
                if len(available_times) == 0:
                    continue
                closest_time = available_times[-1]
                row = data.loc[closest_time]
            
            result[symbol] = {
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            }
        
        return result
    
    return data_feed