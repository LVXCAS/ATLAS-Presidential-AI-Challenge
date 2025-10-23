#!/usr/bin/env python3
"""
OANDA DATA FETCHER
Real-time & historical forex data via OANDA API

FREE practice account: https://www.oanda.com/us-en/trading/
Setup:
1. Create practice account (free, instant)
2. Get API key from account dashboard
3. Add to .env file: OANDA_API_KEY=your_key_here
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    import v20
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False
    print("[WARNING] v20 library not installed. Run: pip install v20")


class OandaDataFetcher:
    """
    Fetch forex data from OANDA

    Supports 70+ currency pairs
    Real-time quotes
    Historical candles
    FREE practice account
    """

    def __init__(self, api_key: Optional[str] = None, account_id: Optional[str] = None, practice=True):
        """
        Initialize OANDA connection

        Args:
            api_key: OANDA API key (or set OANDA_API_KEY env var)
            account_id: OANDA account ID (or set OANDA_ACCOUNT_ID env var)
            practice: Use practice server (True) or live (False)
        """

        if not OANDA_AVAILABLE:
            raise ImportError("v20 library required. Install: pip install v20")

        # Get credentials from env if not provided
        self.api_key = api_key or os.getenv('OANDA_API_KEY')
        self.account_id = account_id or os.getenv('OANDA_ACCOUNT_ID')

        if not self.api_key:
            print("[WARNING] No OANDA_API_KEY found. Set in .env or pass to constructor.")
            print("Get free practice account: https://www.oanda.com/us-en/trading/")
            self.api = None
            return

        # Connect to OANDA
        hostname = 'api-fxpractice.oanda.com' if practice else 'api-fxtrade.oanda.com'
        self.api = v20.Context(
            hostname=hostname,
            port=443,
            token=self.api_key
        )

        self.practice = practice
        print(f"[OANDA] Connected to {'PRACTICE' if practice else 'LIVE'} server")

    def get_bars(self, symbol: str, timeframe: str = 'H1', limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Get historical candles for forex pair

        Args:
            symbol: Forex pair (e.g., 'EUR_USD', 'GBP_USD')
            timeframe: Candle size ('M1', 'M5', 'M15', 'H1', 'H4', 'D')
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """

        if not self.api:
            print("[ERROR] OANDA API not initialized. Check API key.")
            return None

        try:
            # Map common timeframe names
            timeframe_map = {
                '1m': 'M1', '5m': 'M5', '15m': 'M15',
                '1h': 'H1', '4h': 'H4', '1d': 'D',
                '1Min': 'M1', '5Min': 'M5', '15Min': 'M15',
                '1Hour': 'H1', '4Hour': 'H4', '1Day': 'D'
            }
            granularity = timeframe_map.get(timeframe, timeframe)

            # Fetch candles
            params = {
                "count": limit,
                "granularity": granularity,
                "price": "M"  # Mid prices
            }

            response = self.api.instrument.candles(symbol, **params)

            if response.status != 200:
                print(f"[ERROR] OANDA API error: {response.status}")
                return None

            # Convert to DataFrame
            candles = response.body['candles']

            if not candles:
                print(f"[WARNING] No data returned for {symbol}")
                return None

            data = []
            for candle in candles:
                if not candle.complete:
                    continue  # Skip incomplete candles

                data.append({
                    'timestamp': pd.to_datetime(candle.time),
                    'open': float(candle.mid.o),
                    'high': float(candle.mid.h),
                    'low': float(candle.mid.l),
                    'close': float(candle.mid.c),
                    'volume': int(candle.volume)
                })

            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            print(f"[ERROR] Fetching {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current bid/ask for forex pair"""

        if not self.api:
            return None

        try:
            response = self.api.pricing.get(
                self.account_id,
                instruments=symbol
            )

            if response.status == 200:
                price = response.body['prices'][0]
                # Return mid price
                bid = float(price.bids[0].price)
                ask = float(price.asks[0].price)
                return (bid + ask) / 2

        except Exception as e:
            print(f"[ERROR] Getting price for {symbol}: {e}")
            return None

    def get_account_info(self) -> Optional[dict]:
        """Get account balance and info"""

        if not self.api or not self.account_id:
            return None

        try:
            response = self.api.account.get(self.account_id)

            if response.status == 200:
                account = response.body['account']
                return {
                    'balance': float(account.balance),
                    'currency': account.currency,
                    'unrealized_pl': float(account.unrealizedPL),
                    'margin_available': float(account.marginAvailable),
                    'open_trades': int(account.openTradeCount)
                }

        except Exception as e:
            print(f"[ERROR] Getting account info: {e}")
            return None


# List of major forex pairs
MAJOR_PAIRS = [
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF',
    'AUD_USD', 'USD_CAD', 'NZD_USD'
]

MINOR_PAIRS = [
    'EUR_GBP', 'EUR_JPY', 'GBP_JPY', 'EUR_AUD',
    'GBP_AUD', 'EUR_CAD', 'GBP_CAD'
]

EXOTIC_PAIRS = [
    'USD_TRY', 'USD_ZAR', 'USD_MXN', 'USD_SGD'
]


def demo():
    """Demo OANDA data fetcher"""

    print("\n" + "="*70)
    print("OANDA DATA FETCHER DEMO")
    print("="*70)

    # Initialize (will fail gracefully if no API key)
    fetcher = OandaDataFetcher(practice=True)

    if not fetcher.api:
        print("\n[SETUP REQUIRED]")
        print("1. Go to: https://www.oanda.com/us-en/trading/")
        print("2. Create FREE practice account (instant approval)")
        print("3. Get API key from account dashboard")
        print("4. Add to .env file:")
        print("   OANDA_API_KEY=your_key_here")
        print("   OANDA_ACCOUNT_ID=your_account_id")
        print("\nThen run this script again.")
        return

    # Test fetching data
    print("\n[TEST 1] Fetching EUR/USD 1-hour data...")
    df = fetcher.get_bars('EUR_USD', '1h', limit=50)

    if df is not None and not df.empty:
        print(f"[OK] Fetched {len(df)} candles")
        print(f"  Latest close: {df['close'].iloc[-1]:.5f}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")

        # Show last 5 candles
        print("\nLast 5 candles:")
        print(df.tail())
    else:
        print("[ERROR] Failed to fetch data")

    # Test current price
    print("\n[TEST 2] Getting current EUR/USD price...")
    price = fetcher.get_current_price('EUR_USD')
    if price:
        print(f"[OK] Current price: {price:.5f}")
    else:
        print("[ERROR] Failed to get price")

    # Test account info
    print("\n[TEST 3] Getting account info...")
    account = fetcher.get_account_info()
    if account:
        print(f"[OK] Account balance: ${account['balance']:,.2f} {account['currency']}")
        print(f"  Open trades: {account['open_trades']}")
    else:
        print("[ERROR] Failed to get account info")

    print("\n" + "="*70)
    print("OANDA integration ready")
    print("="*70)


if __name__ == "__main__":
    demo()
