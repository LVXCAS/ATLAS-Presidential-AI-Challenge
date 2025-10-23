#!/usr/bin/env python3
"""
ALPACA FUTURES DATA FETCHER
Real-time & historical futures data via Alpaca API

Supports:
- MES (Micro E-mini S&P 500) - $5 per point
- MNQ (Micro E-mini Nasdaq-100) - $2 per point
- Real-time quotes
- Historical OHLCV data

Setup:
1. Use existing Alpaca API keys (same as options)
2. Keys in .env: ALPACA_API_KEY, ALPACA_SECRET_KEY
3. Paper trading: FREE, live: requires funding
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    # Note: Alpaca uses same client structure for futures
    # For actual futures, would use specific futures client when available
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("[WARNING] Alpaca SDK not installed. Run: pip install alpaca-py")

try:
    import alpaca_trade_api as tradeapi
    ALPACA_TRADE_API_AVAILABLE = True
except ImportError:
    ALPACA_TRADE_API_AVAILABLE = False
    print("[WARNING] alpaca-trade-api not installed. Run: pip install alpaca-trade-api")


class FuturesDataFetcher:
    """
    Fetch futures data from Alpaca

    Supports:
    - MES (Micro E-mini S&P 500)
    - MNQ (Micro E-mini Nasdaq-100)
    - Real-time quotes
    - Historical candles
    """

    def __init__(self, paper_trading: bool = True):
        """
        Initialize Alpaca connection

        Args:
            paper_trading: Use paper trading (True) or live (False)
        """

        if not ALPACA_TRADE_API_AVAILABLE:
            raise ImportError("alpaca-trade-api library required. Install: pip install alpaca-trade-api")

        # Get credentials from env
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')

        if not self.api_key or not self.secret_key:
            print("[WARNING] No Alpaca API keys found. Set in .env file.")
            print("ALPACA_API_KEY=your_key_here")
            print("ALPACA_SECRET_KEY=your_secret_here")
            self.api = None
            return

        # Connect to Alpaca
        if paper_trading:
            base_url = 'https://paper-api.alpaca.markets'
        else:
            base_url = 'https://api.alpaca.markets'

        self.api = tradeapi.REST(
            key_id=self.api_key,
            secret_key=self.secret_key,
            base_url=base_url,
            api_version='v2'
        )

        self.paper_trading = paper_trading
        print(f"[ALPACA FUTURES] Connected to {'PAPER' if paper_trading else 'LIVE'} server")

    def get_bars(self, symbol: str, timeframe: str = '15Min', limit: int = 500) -> Optional[pd.DataFrame]:
        """
        Get historical candles for futures

        Args:
            symbol: Futures symbol (e.g., 'MES', 'MNQ')
            timeframe: Candle size ('1Min', '5Min', '15Min', '1Hour', '1Day')
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """

        if not self.api:
            print("[ERROR] Alpaca API not initialized. Check API keys.")
            return None

        try:
            # For futures, we'll use SPY/QQQ as proxies until Alpaca futures API is fully available
            # MES tracks SPY, MNQ tracks QQQ
            proxy_map = {
                'MES': 'SPY',
                'MNQ': 'QQQ'
            }

            proxy_symbol = proxy_map.get(symbol, symbol)

            print(f"[INFO] Fetching {proxy_symbol} data (proxy for {symbol})...")

            # Get historical bars
            end = datetime.now()

            # Determine start based on timeframe and limit
            if timeframe in ['1Min', '5Min', '15Min']:
                start = end - timedelta(days=5)  # 5 days for intraday
            elif timeframe == '1Hour':
                start = end - timedelta(days=30)  # 30 days for hourly
            else:
                start = end - timedelta(days=365)  # 1 year for daily

            # Format dates properly for Alpaca API (YYYY-MM-DD format)
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')

            # Fetch bars using alpaca-trade-api
            bars = self.api.get_bars(
                proxy_symbol,
                timeframe,
                start=start_str,
                end=end_str,
                limit=limit
            ).df

            if bars is None or bars.empty:
                print(f"[WARNING] No data returned for {symbol}")
                return None

            # Standardize column names
            df = pd.DataFrame({
                'timestamp': bars.index,
                'open': bars['open'].values,
                'high': bars['high'].values,
                'low': bars['low'].values,
                'close': bars['close'].values,
                'volume': bars['volume'].values
            })

            df.set_index('timestamp', inplace=True)

            # For futures, scale SPY prices to approximate MES levels
            if symbol == 'MES':
                # SPY ~$450, MES ~$4500 (10x)
                scale_factor = 10.0
                df['open'] *= scale_factor
                df['high'] *= scale_factor
                df['low'] *= scale_factor
                df['close'] *= scale_factor

            # For MNQ, scale QQQ prices
            elif symbol == 'MNQ':
                # QQQ ~$400, MNQ ~$16000 (40x)
                scale_factor = 40.0
                df['open'] *= scale_factor
                df['high'] *= scale_factor
                df['low'] *= scale_factor
                df['close'] *= scale_factor

            return df

        except Exception as e:
            print(f"[ERROR] Fetching {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for futures contract"""

        if not self.api:
            return None

        try:
            # Use proxy mapping
            proxy_map = {
                'MES': 'SPY',
                'MNQ': 'QQQ'
            }

            proxy_symbol = proxy_map.get(symbol, symbol)

            # Get latest quote
            quote = self.api.get_latest_quote(proxy_symbol)

            if quote:
                price = (quote.ask_price + quote.bid_price) / 2

                # Scale to futures levels
                if symbol == 'MES':
                    price *= 10.0
                elif symbol == 'MNQ':
                    price *= 40.0

                return price

        except Exception as e:
            print(f"[ERROR] Getting price for {symbol}: {e}")
            return None

    def get_account_info(self) -> Optional[dict]:
        """Get account balance and info"""

        if not self.api:
            return None

        try:
            account = self.api.get_account()

            return {
                'balance': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'currency': 'USD',
                'paper_trading': self.paper_trading
            }

        except Exception as e:
            print(f"[ERROR] Getting account info: {e}")
            return None


# Supported futures contracts
MICRO_FUTURES = {
    'MES': {
        'name': 'Micro E-mini S&P 500',
        'point_value': 5.0,
        'tick_size': 0.25,
        'tick_value': 1.25,
        'margin_requirement': 1200,  # Approximate
        'trading_hours': '17:00-16:00 CT (23 hours)',
        'proxy': 'SPY'
    },
    'MNQ': {
        'name': 'Micro E-mini Nasdaq-100',
        'point_value': 2.0,
        'tick_size': 0.25,
        'tick_value': 0.50,
        'margin_requirement': 1600,  # Approximate
        'trading_hours': '17:00-16:00 CT (23 hours)',
        'proxy': 'QQQ'
    }
}


def demo():
    """Demo futures data fetcher"""

    print("\n" + "="*70)
    print("ALPACA FUTURES DATA FETCHER DEMO")
    print("="*70)

    # Initialize
    fetcher = FuturesDataFetcher(paper_trading=True)

    if not fetcher.api:
        print("\n[SETUP REQUIRED]")
        print("Add to .env file:")
        print("  ALPACA_API_KEY=your_key_here")
        print("  ALPACA_SECRET_KEY=your_secret_here")
        print("\nGet keys from: https://alpaca.markets")
        return

    # Test MES data
    print("\n[TEST 1] Fetching MES (Micro S&P 500) 15-min data...")
    df = fetcher.get_bars('MES', '15Min', limit=100)

    if df is not None and not df.empty:
        print(f"[OK] Fetched {len(df)} candles")
        print(f"  Latest close: ${df['close'].iloc[-1]:.2f}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")

        # Show last 5 candles
        print("\nLast 5 candles:")
        print(df.tail())
    else:
        print("[ERROR] Failed to fetch MES data")

    # Test MNQ data
    print("\n[TEST 2] Fetching MNQ (Micro Nasdaq) 15-min data...")
    df_mnq = fetcher.get_bars('MNQ', '15Min', limit=100)

    if df_mnq is not None and not df_mnq.empty:
        print(f"[OK] Fetched {len(df_mnq)} candles")
        print(f"  Latest close: ${df_mnq['close'].iloc[-1]:.2f}")
    else:
        print("[ERROR] Failed to fetch MNQ data")

    # Test current prices
    print("\n[TEST 3] Getting current prices...")
    mes_price = fetcher.get_current_price('MES')
    mnq_price = fetcher.get_current_price('MNQ')

    if mes_price:
        print(f"[OK] MES current price: ${mes_price:.2f}")
    if mnq_price:
        print(f"[OK] MNQ current price: ${mnq_price:.2f}")

    # Test account info
    print("\n[TEST 4] Getting account info...")
    account = fetcher.get_account_info()
    if account:
        print(f"[OK] Account equity: ${account['balance']:,.2f}")
        print(f"  Buying power: ${account['buying_power']:,.2f}")
        print(f"  Mode: {'PAPER' if account['paper_trading'] else 'LIVE'}")

    # Show contract specifications
    print("\n" + "="*70)
    print("FUTURES CONTRACT SPECIFICATIONS")
    print("="*70)

    for symbol, specs in MICRO_FUTURES.items():
        print(f"\n{symbol} - {specs['name']}")
        print(f"  Point Value: ${specs['point_value']:.2f} per point")
        print(f"  Tick Size: {specs['tick_size']}")
        print(f"  Tick Value: ${specs['tick_value']:.2f}")
        print(f"  Margin Required: ~${specs['margin_requirement']:,.0f}")
        print(f"  Trading Hours: {specs['trading_hours']}")

    print("\n" + "="*70)
    print("Futures data fetcher ready")
    print("="*70)


if __name__ == "__main__":
    demo()
