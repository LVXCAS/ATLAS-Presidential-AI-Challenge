"""
Real Data Connector - Connects New Agents to Real Market Data

This replaces simulated data in new agents with real data from:
- Alpaca API (best for US stocks, real-time)
- Polygon API (premium institutional data)
- OpenBB Platform (28+ professional providers)
- Yahoo Finance (free fallback)

Usage:
    from agents.real_data_connector import fetch_real_market_data

    # Instead of simulated data:
    data = await fetch_real_market_data(symbol="AAPL", days=252)
"""

import asyncio
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if env vars already set

# Import existing data providers
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logging.warning("yfinance not available - install with: pip install yfinance")

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logging.warning("alpaca_trade_api not available")

try:
    from polygon import RESTClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    logging.warning("polygon-api-client not available")

logger = logging.getLogger(__name__)

# OpenBB will be imported lazily when needed
OPENBB_AVAILABLE = None  # Unknown until first use
_openbb_provider = None


class RealDataConnector:
    """Fetches real market data from configured sources"""

    def __init__(self):
        """Initialize with API keys from environment"""
        self.alpaca_key = os.getenv('ALPACA_API_KEY')
        self.alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
        self.alpaca_base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        self.polygon_key = os.getenv('POLYGON_API_KEY')

        # Initialize clients
        self.alpaca_client = None
        self.polygon_client = None

        if ALPACA_AVAILABLE and self.alpaca_key:
            self.alpaca_client = tradeapi.REST(
                self.alpaca_key,
                self.alpaca_secret,
                self.alpaca_base_url,
                api_version='v2'
            )
            logger.info("✅ Alpaca client initialized")

        if POLYGON_AVAILABLE and self.polygon_key:
            self.polygon_client = RESTClient(self.polygon_key)
            logger.info("✅ Polygon client initialized")

    async def fetch_market_data(
        self,
        symbol: str,
        days: int = 252,
        timeframe: str = '1Day'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch real market data for a symbol

        Tries sources in order:
        1. Alpaca (if configured) - Best for US stocks, real-time
        2. Polygon (if configured) - Premium institutional data
        3. OpenBB (if available) - 28+ professional providers
        4. Yahoo Finance (free fallback) - Always available

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            days: Number of days of historical data
            timeframe: '1Min', '5Min', '1Hour', '1Day'

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Try Alpaca first (best for US stocks)
        if self.alpaca_client:
            try:
                data = await self._fetch_from_alpaca(symbol, start_date, end_date, timeframe)
                if data is not None and len(data) > 0:
                    logger.info(f"✅ Fetched {len(data)} bars from Alpaca for {symbol}")
                    return data
            except Exception as e:
                logger.warning(f"Alpaca fetch failed for {symbol}: {e}")

        # Try Polygon (premium data)
        if self.polygon_client:
            try:
                data = await self._fetch_from_polygon(symbol, start_date, end_date)
                if data is not None and len(data) > 0:
                    logger.info(f"✅ Fetched {len(data)} bars from Polygon for {symbol}")
                    return data
            except Exception as e:
                logger.warning(f"Polygon fetch failed for {symbol}: {e}")

        # Try OpenBB (28+ professional providers)
        if OPENBB_AVAILABLE:
            try:
                data = await self._fetch_from_openbb(symbol, days, timeframe)
                if data is not None and len(data) > 0:
                    logger.info(f"✅ Fetched {len(data)} bars from OpenBB for {symbol}")
                    return data
            except Exception as e:
                logger.warning(f"OpenBB fetch failed for {symbol}: {e}")

        # Fallback to Yahoo Finance (free)
        if YFINANCE_AVAILABLE:
            try:
                data = await self._fetch_from_yahoo(symbol, days)
                if data is not None and len(data) > 0:
                    logger.info(f"✅ Fetched {len(data)} bars from Yahoo Finance for {symbol}")
                    return data
            except Exception as e:
                logger.warning(f"Yahoo Finance fetch failed for {symbol}: {e}")

        logger.error(f"❌ All data sources failed for {symbol}")
        return None

    async def _fetch_from_alpaca(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Alpaca"""

        # Map timeframe string to Alpaca TimeFrame
        timeframe_map = {
            '1Min': TimeFrame.Minute,
            '5Min': TimeFrame(5, TimeFrame.Minute),
            '1Hour': TimeFrame.Hour,
            '1Day': TimeFrame.Day
        }

        tf = timeframe_map.get(timeframe, TimeFrame.Day)

        # Fetch bars (Alpaca requires YYYY-MM-DD format, not full ISO)
        bars = self.alpaca_client.get_bars(
            symbol,
            tf,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            adjustment='raw'
        ).df

        if bars is None or len(bars) == 0:
            return None

        # Standardize column names
        bars = bars.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })

        # Reset index to get timestamp as column
        bars = bars.reset_index()
        bars = bars.rename(columns={'timestamp': 'date'})

        return bars[['date', 'open', 'high', 'low', 'close', 'volume']]

    async def _fetch_from_polygon(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Polygon"""

        # Convert dates to YYYY-MM-DD format
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        # Fetch aggregates (daily bars)
        aggs = []
        for agg in self.polygon_client.list_aggs(
            ticker=symbol,
            multiplier=1,
            timespan='day',
            from_=start_str,
            to=end_str,
            limit=50000
        ):
            aggs.append({
                'date': datetime.fromtimestamp(agg.timestamp / 1000),
                'open': agg.open,
                'high': agg.high,
                'low': agg.low,
                'close': agg.close,
                'volume': agg.volume
            })

        if not aggs:
            return None

        df = pd.DataFrame(aggs)
        return df

    async def _fetch_from_openbb(
        self,
        symbol: str,
        days: int,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data from OpenBB Platform (28+ providers)"""

        # Lazy load OpenBB on first use
        global OPENBB_AVAILABLE, _openbb_provider
        if OPENBB_AVAILABLE is None:
            try:
                from agents.openbb_data_provider import openbb_provider
                _openbb_provider = openbb_provider
                OPENBB_AVAILABLE = True
                logger.info("✅ OpenBB provider loaded successfully")
            except Exception as e:
                OPENBB_AVAILABLE = False
                logger.debug(f"OpenBB not available: {e}")
                return None

        if not OPENBB_AVAILABLE:
            return None

        # Map timeframe to period
        period_map = {
            '1Min': f"{days}d",
            '5Min': f"{days}d",
            '1Hour': f"{days}d",
            '1Day': f"{days}d"
        }

        # Map timeframe to interval
        interval_map = {
            '1Min': '1m',
            '5Min': '5m',
            '1Hour': '1h',
            '1Day': '1d'
        }

        period = period_map.get(timeframe, f"{days}d")
        interval = interval_map.get(timeframe, '1d')

        # Fetch using OpenBB provider
        df = await _openbb_provider.get_equity_data(
            symbol=symbol,
            period=period,
            interval=interval
        )

        if df is None or len(df) == 0:
            return None

        # Standardize column names (OpenBB uses capitalized names)
        df = df.reset_index()

        # Try different column name variations
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['date', 'timestamp']:
                column_mapping[col] = 'date'
            elif col_lower == 'open':
                column_mapping[col] = 'open'
            elif col_lower == 'high':
                column_mapping[col] = 'high'
            elif col_lower == 'low':
                column_mapping[col] = 'low'
            elif col_lower == 'close':
                column_mapping[col] = 'close'
            elif col_lower == 'volume':
                column_mapping[col] = 'volume'

        df = df.rename(columns=column_mapping)

        # Ensure we have the required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if all(col in df.columns for col in required_cols):
            return df[required_cols]
        else:
            logger.warning(f"OpenBB data missing required columns. Got: {df.columns.tolist()}")
            return None

    async def _fetch_from_yahoo(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance"""

        # Download data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{days}d")

        if df is None or len(df) == 0:
            return None

        # Standardize column names
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        return df[['date', 'open', 'high', 'low', 'close', 'volume']]

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current/latest price for a symbol"""

        # Try Alpaca first (real-time for paper trading)
        if self.alpaca_client:
            try:
                quote = self.alpaca_client.get_latest_trade(symbol)
                return float(quote.price)
            except:
                pass

        # Try Polygon
        if self.polygon_client:
            try:
                quote = self.polygon_client.get_last_trade(symbol)
                return float(quote.price)
            except:
                pass

        # Fallback to Yahoo Finance
        if YFINANCE_AVAILABLE:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                return float(info.get('currentPrice', info.get('regularMarketPrice', 0)))
            except:
                pass

        return None

    def get_status(self) -> Dict:
        """Get status of all data sources"""
        # Check OpenBB availability by attempting lazy load
        openbb_status = OPENBB_AVAILABLE
        if openbb_status is None:
            try:
                from agents.openbb_data_provider import openbb_provider
                openbb_status = openbb_provider.available
                logger.debug(f"OpenBB availability check: {openbb_status}")
            except Exception as e:
                logger.debug(f"OpenBB import failed: {e}")
                openbb_status = False

        return {
            'alpaca': {
                'configured': self.alpaca_client is not None,
                'api_key': bool(self.alpaca_key),
                'status': 'active' if self.alpaca_client else 'not configured'
            },
            'polygon': {
                'configured': self.polygon_client is not None,
                'api_key': bool(self.polygon_key),
                'status': 'active' if self.polygon_client else 'not configured'
            },
            'openbb': {
                'configured': openbb_status,
                'status': 'active (28+ providers)' if openbb_status else 'not installed'
            },
            'yahoo': {
                'configured': YFINANCE_AVAILABLE,
                'status': 'active' if YFINANCE_AVAILABLE else 'not installed'
            }
        }


# Global instance
_real_data_connector = None

def get_real_data_connector() -> RealDataConnector:
    """Get global RealDataConnector instance"""
    global _real_data_connector
    if _real_data_connector is None:
        _real_data_connector = RealDataConnector()
    return _real_data_connector


# Convenience function
async def fetch_real_market_data(
    symbol: str,
    days: int = 252,
    timeframe: str = '1Day'
) -> Optional[pd.DataFrame]:
    """
    Fetch real market data for a symbol

    This is the main function to use in your agents!

    Example:
        # In your agent:
        from agents.real_data_connector import fetch_real_market_data

        # Get 1 year of daily data
        df = await fetch_real_market_data("AAPL", days=252)

        # Get 10 days of 5-minute bars
        df = await fetch_real_market_data("SPY", days=10, timeframe="5Min")
    """
    connector = get_real_data_connector()
    return await connector.fetch_market_data(symbol, days, timeframe)


async def get_current_price(symbol: str) -> Optional[float]:
    """Get current price for a symbol"""
    connector = get_real_data_connector()
    return await connector.get_current_price(symbol)


def check_data_sources():
    """Check status of all data sources"""
    connector = get_real_data_connector()
    status = connector.get_status()

    print("\n" + "=" * 60)
    print("DATA SOURCE STATUS")
    print("=" * 60)

    for source, info in status.items():
        status_symbol = "[OK]" if info['configured'] else "[X]"
        print(f"{status_symbol} {source.upper()}: {info['status']}")

    print("=" * 60 + "\n")

    return status


# Test function
async def test_real_data():
    """Test fetching real data"""
    print("Testing Real Data Connector...")
    print("=" * 60)

    # Check sources
    check_data_sources()

    # Test fetch
    symbols = ['AAPL', 'SPY', 'TSLA']

    for symbol in symbols:
        print(f"\nFetching {symbol}...")
        df = await fetch_real_market_data(symbol, days=5)

        if df is not None:
            print(f"[OK] Success! Got {len(df)} bars")
            print(f"   Latest close: ${df['close'].iloc[-1]:.2f}")
            print(f"   Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
        else:
            print(f"[FAIL] Failed to fetch data for {symbol}")

    # Test current price
    print(f"\nCurrent Prices:")
    for symbol in symbols:
        price = await get_current_price(symbol)
        if price:
            print(f"  {symbol}: ${price:.2f}")


if __name__ == "__main__":
    asyncio.run(test_real_data())
