#!/usr/bin/env python3
"""
POLYGON FUTURES DATA FETCHER
Alternative to Alpaca - uses Polygon.io API for futures data

Supports:
- MES (Micro E-mini S&P 500) via SPY proxy
- MNQ (Micro E-mini Nasdaq-100) via QQQ proxy
- Real-time quotes
- Historical OHLCV data
- FREE tier available (limited requests)
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv
import requests

load_dotenv()


class PolygonFuturesFetcher:
    """
    Fetch futures data from Polygon.io

    Uses SPY/QQQ as proxies for MES/MNQ
    """

    def __init__(self):
        """Initialize Polygon connection"""

        self.api_key = os.getenv('POLYGON_API_KEY')

        if not self.api_key:
            print("[WARNING] No Polygon API key found. Set POLYGON_API_KEY in .env")
            self.api_key = None
            return

        self.base_url = 'https://api.polygon.io'
        print(f"[POLYGON] Connected with API key")

    def get_bars(self, symbol: str, timeframe: str = '15Min', limit: int = 500) -> Optional[pd.DataFrame]:
        """
        Get historical candles for futures

        Args:
            symbol: Futures symbol ('MES', 'MNQ')
            timeframe: Candle size ('1Min', '5Min', '15Min', '1Hour', '1Day')
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """

        if not self.api_key:
            print("[ERROR] Polygon API key not set")
            return None

        try:
            # Map futures to stock proxies
            proxy_map = {
                'MES': 'SPY',
                'MNQ': 'QQQ'
            }

            proxy_symbol = proxy_map.get(symbol, symbol)

            print(f"[POLYGON] Fetching {proxy_symbol} data (proxy for {symbol})...")

            # Map timeframes to Polygon multiplier/timespan
            timeframe_map = {
                '1Min': (1, 'minute'),
                '5Min': (5, 'minute'),
                '15Min': (15, 'minute'),
                '1Hour': (1, 'hour'),
                '1Day': (1, 'day')
            }

            if timeframe not in timeframe_map:
                print(f"[ERROR] Unsupported timeframe: {timeframe}")
                return None

            multiplier, timespan = timeframe_map[timeframe]

            # Calculate date range
            end = datetime.now()

            if timespan == 'minute':
                start = end - timedelta(days=5)  # 5 days for minute data
            elif timespan == 'hour':
                start = end - timedelta(days=30)  # 30 days for hour data
            else:
                start = end - timedelta(days=365)  # 1 year for daily

            # Format dates for Polygon API (YYYY-MM-DD format)
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')

            # Build Polygon API request
            url = f"{self.base_url}/v2/aggs/ticker/{proxy_symbol}/range/{multiplier}/{timespan}/{start_str}/{end_str}"

            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'limit': limit,
                'apiKey': self.api_key
            }

            # Make request
            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                print(f"[ERROR] Polygon API returned {response.status_code}: {response.text}")
                return None

            data = response.json()

            if data.get('status') != 'OK' or not data.get('results'):
                print(f"[WARNING] No data returned from Polygon for {symbol}")
                return None

            # Convert to DataFrame
            results = data['results']

            df = pd.DataFrame(results)

            # Rename columns to match our standard format
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            })

            # Select and order columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df.set_index('timestamp', inplace=True)

            # Scale to futures levels
            if symbol == 'MES':
                # SPY ~$450, MES ~$4500 (10x)
                scale_factor = 10.0
                df['open'] *= scale_factor
                df['high'] *= scale_factor
                df['low'] *= scale_factor
                df['close'] *= scale_factor

            elif symbol == 'MNQ':
                # QQQ ~$400, MNQ ~$16000 (40x)
                scale_factor = 40.0
                df['open'] *= scale_factor
                df['high'] *= scale_factor
                df['low'] *= scale_factor
                df['close'] *= scale_factor

            print(f"[POLYGON] Fetched {len(df)} candles")

            return df

        except Exception as e:
            print(f"[ERROR] Polygon fetch failed: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for futures contract"""

        if not self.api_key:
            return None

        try:
            # Map to proxy
            proxy_map = {
                'MES': 'SPY',
                'MNQ': 'QQQ'
            }

            proxy_symbol = proxy_map.get(symbol, symbol)

            # Get latest trade via Polygon
            url = f"{self.base_url}/v2/last/trade/{proxy_symbol}"

            params = {'apiKey': self.api_key}

            response = requests.get(url, params=params, timeout=5)

            if response.status_code != 200:
                return None

            data = response.json()

            if data.get('status') == 'OK' and data.get('results'):
                price = data['results']['p']  # Last trade price

                # Scale to futures levels
                if symbol == 'MES':
                    price *= 10.0
                elif symbol == 'MNQ':
                    price *= 40.0

                return price

        except Exception as e:
            print(f"[ERROR] Getting price from Polygon: {e}")
            return None


# Wrapper class that tries Alpaca first, falls back to Polygon
class HybridFuturesFetcher:
    """
    Smart futures data fetcher that tries Alpaca first, falls back to Polygon
    """

    def __init__(self, paper_trading: bool = True):
        """Initialize both fetchers"""

        print("[HYBRID FETCHER] Initializing...")

        # Try Alpaca first
        try:
            from data.futures_data_fetcher import FuturesDataFetcher
            self.alpaca = FuturesDataFetcher(paper_trading=paper_trading)
        except Exception as e:
            print(f"[WARNING] Alpaca init failed: {e}")
            self.alpaca = None

        # Initialize Polygon as backup
        self.polygon = PolygonFuturesFetcher()

        print("[HYBRID FETCHER] Ready (Alpaca primary, Polygon backup)")

    def get_bars(self, symbol: str, timeframe: str = '15Min', limit: int = 500) -> Optional[pd.DataFrame]:
        """
        Get bars - try Alpaca first, fall back to Polygon
        """

        # Try Alpaca first
        if self.alpaca:
            try:
                df = self.alpaca.get_bars(symbol, timeframe, limit)
                if df is not None and not df.empty:
                    print(f"[HYBRID] Using Alpaca data for {symbol}")
                    return df
            except Exception as e:
                print(f"[HYBRID] Alpaca failed ({e}), trying Polygon...")

        # Fall back to Polygon
        if self.polygon:
            df = self.polygon.get_bars(symbol, timeframe, limit)
            if df is not None and not df.empty:
                print(f"[HYBRID] Using Polygon data for {symbol}")
                return df

        print(f"[HYBRID] Both sources failed for {symbol}")
        return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price - try Alpaca first, fall back to Polygon"""

        # Try Alpaca
        if self.alpaca:
            try:
                price = self.alpaca.get_current_price(symbol)
                if price:
                    return price
            except:
                pass

        # Fall back to Polygon
        if self.polygon:
            return self.polygon.get_current_price(symbol)

        return None


def demo():
    """Demo Polygon futures fetcher"""

    print("\n" + "="*70)
    print("POLYGON FUTURES DATA FETCHER DEMO")
    print("="*70)

    # Test Polygon directly
    fetcher = PolygonFuturesFetcher()

    if not fetcher.api_key:
        print("\n[SETUP REQUIRED]")
        print("Add to .env file:")
        print("  POLYGON_API_KEY=your_key_here")
        print("\nGet key from: https://polygon.io")
        return

    # Test MES data
    print("\n[TEST 1] Fetching MES 15-min data via Polygon...")
    df = fetcher.get_bars('MES', '15Min', limit=100)

    if df is not None and not df.empty:
        print(f"[OK] Fetched {len(df)} candles")
        print(f"  Latest close: ${df['close'].iloc[-1]:.2f}")
        print(f"  Range: {df.index[0]} to {df.index[-1]}")
        print("\nLast 5 candles:")
        print(df.tail())
    else:
        print("[ERROR] Failed to fetch data")

    # Test MNQ
    print("\n[TEST 2] Fetching MNQ 15-min data...")
    df_mnq = fetcher.get_bars('MNQ', '15Min', limit=100)

    if df_mnq is not None and not df_mnq.empty:
        print(f"[OK] Fetched {len(df_mnq)} candles")
        print(f"  Latest close: ${df_mnq['close'].iloc[-1]:.2f}")

    # Test current prices
    print("\n[TEST 3] Getting current prices...")
    mes_price = fetcher.get_current_price('MES')
    mnq_price = fetcher.get_current_price('MNQ')

    if mes_price:
        print(f"[OK] MES current: ${mes_price:.2f}")
    if mnq_price:
        print(f"[OK] MNQ current: ${mnq_price:.2f}")

    # Test hybrid fetcher
    print("\n" + "="*70)
    print("TESTING HYBRID FETCHER (Alpaca + Polygon)")
    print("="*70)

    hybrid = HybridFuturesFetcher(paper_trading=True)

    print("\n[TEST 4] Fetching via hybrid fetcher...")
    df_hybrid = hybrid.get_bars('MES', '15Min', limit=50)

    if df_hybrid is not None and not df_hybrid.empty:
        print(f"[OK] Hybrid fetcher returned {len(df_hybrid)} candles")
        print(f"  Latest: ${df_hybrid['close'].iloc[-1]:.2f}")

    print("\n" + "="*70)
    print("Polygon futures fetcher ready!")
    print("="*70)


if __name__ == "__main__":
    demo()
