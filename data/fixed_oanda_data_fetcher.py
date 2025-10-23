#!/usr/bin/env python3
"""
FIXED OANDA DATA FETCHER
Real-time & historical forex data via OANDA REST API

FIXES:
- Replaces v20 library with direct REST API calls
- Uses requests library with 5-second timeouts
- No more hanging on API calls
- Handles timeouts gracefully

FREE practice account: https://www.oanda.com/us-en/trading/
Setup:
1. Create practice account (free, instant)
2. Get API key from account dashboard
3. Add to .env file: OANDA_API_KEY=your_key_here
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Default OANDA credentials
DEFAULT_API_KEY = "0bff5dc7375409bb8747deebab8988a1-d8b26324102c95d6f2b6f641bc330a7c"
DEFAULT_ACCOUNT_ID = os.getenv('OANDA_ACCOUNT_ID', '101-004-29328895-001')


class FixedOandaDataFetcher:
    """
    Fetch forex data from OANDA using direct REST API calls

    Supports 70+ currency pairs
    Real-time quotes
    Historical candles
    FREE practice account
    NO MORE HANGING - Uses 5-second timeouts
    """

    def __init__(self, api_key: Optional[str] = None, account_id: Optional[str] = None,
                 practice=True, timeout: int = 5):
        """
        Initialize OANDA REST API connection

        Args:
            api_key: OANDA API key (or set OANDA_API_KEY env var)
            account_id: OANDA account ID (or set OANDA_ACCOUNT_ID env var)
            practice: Use practice server (True) or live (False)
            timeout: Request timeout in seconds (default: 5)
        """

        # Get credentials from env if not provided
        self.api_key = api_key or os.getenv('OANDA_API_KEY') or DEFAULT_API_KEY
        self.account_id = account_id or os.getenv('OANDA_ACCOUNT_ID') or DEFAULT_ACCOUNT_ID
        self.timeout = timeout

        if not self.api_key:
            print("[WARNING] No OANDA_API_KEY found. Set in .env or pass to constructor.")
            print("Get free practice account: https://www.oanda.com/us-en/trading/")
            self.base_url = None
            return

        # Set base URL
        if practice:
            self.base_url = 'https://api-fxpractice.oanda.com/v3'
        else:
            self.base_url = 'https://api-fxtrade.oanda.com/v3'

        # Set headers
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        self.practice = practice
        print(f"[OANDA] Connected to {'PRACTICE' if practice else 'LIVE'} server")
        print(f"[OANDA] Using direct REST API with {timeout}s timeout")

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

        if not self.base_url:
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

            # Build request URL
            url = f"{self.base_url}/instruments/{symbol}/candles"
            params = {
                'count': limit,
                'granularity': granularity,
                'price': 'M'  # Mid prices
            }

            # Make request with timeout
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=self.timeout
            )

            if response.status_code != 200:
                print(f"[ERROR] OANDA API error: {response.status_code}")
                print(f"Response: {response.text}")
                return None

            # Parse response
            data = response.json()
            candles = data.get('candles', [])

            if not candles:
                print(f"[WARNING] No data returned for {symbol}")
                return None

            # Convert to DataFrame
            rows = []
            for candle in candles:
                if not candle.get('complete', False):
                    continue  # Skip incomplete candles

                mid = candle.get('mid', {})
                rows.append({
                    'timestamp': pd.to_datetime(candle['time']),
                    'open': float(mid.get('o', 0)),
                    'high': float(mid.get('h', 0)),
                    'low': float(mid.get('l', 0)),
                    'close': float(mid.get('c', 0)),
                    'volume': int(candle.get('volume', 0))
                })

            df = pd.DataFrame(rows)
            df.set_index('timestamp', inplace=True)

            return df

        except requests.Timeout:
            print(f"[TIMEOUT] Request for {symbol} exceeded {self.timeout}s")
            return None
        except Exception as e:
            print(f"[ERROR] Fetching {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current bid/ask for forex pair"""

        if not self.base_url:
            return None

        try:
            url = f"{self.base_url}/accounts/{self.account_id}/pricing"
            params = {'instruments': symbol}

            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=self.timeout
            )

            if response.status_code != 200:
                print(f"[ERROR] Pricing API error: {response.status_code}")
                return None

            data = response.json()
            prices = data.get('prices', [])

            if prices:
                price_data = prices[0]
                bids = price_data.get('bids', [])
                asks = price_data.get('asks', [])

                if bids and asks:
                    bid = float(bids[0].get('price', 0))
                    ask = float(asks[0].get('price', 0))
                    return (bid + ask) / 2

            return None

        except requests.Timeout:
            print(f"[TIMEOUT] Price request for {symbol} exceeded {self.timeout}s")
            return None
        except Exception as e:
            print(f"[ERROR] Getting price for {symbol}: {e}")
            return None

    def get_account_info(self) -> Optional[dict]:
        """Get account balance and info"""

        if not self.base_url or not self.account_id:
            return None

        try:
            url = f"{self.base_url}/accounts/{self.account_id}"

            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout
            )

            if response.status_code != 200:
                print(f"[ERROR] Account API error: {response.status_code}")
                return None

            data = response.json()
            account = data.get('account', {})

            return {
                'balance': float(account.get('balance', 0)),
                'currency': account.get('currency', 'USD'),
                'unrealized_pl': float(account.get('unrealizedPL', 0)),
                'margin_available': float(account.get('marginAvailable', 0)),
                'open_trades': int(account.get('openTradeCount', 0))
            }

        except requests.Timeout:
            print(f"[TIMEOUT] Account info request exceeded {self.timeout}s")
            return None
        except Exception as e:
            print(f"[ERROR] Getting account info: {e}")
            return None

    def get_open_trades(self) -> List[Dict]:
        """Get all open trades"""

        if not self.base_url or not self.account_id:
            return []

        try:
            url = f"{self.base_url}/accounts/{self.account_id}/openTrades"

            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout
            )

            if response.status_code != 200:
                print(f"[ERROR] Open trades API error: {response.status_code}")
                return []

            data = response.json()
            trades = data.get('trades', [])

            result = []
            for trade in trades:
                result.append({
                    'trade_id': trade.get('id'),
                    'instrument': trade.get('instrument'),
                    'units': int(trade.get('currentUnits', 0)),
                    'price': float(trade.get('price', 0)),
                    'unrealized_pl': float(trade.get('unrealizedPL', 0)),
                    'open_time': trade.get('openTime')
                })

            return result

        except requests.Timeout:
            print(f"[TIMEOUT] Open trades request exceeded {self.timeout}s")
            return []
        except Exception as e:
            print(f"[ERROR] Getting open trades: {e}")
            return []


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
    """Demo FIXED OANDA data fetcher"""

    print("\n" + "="*70)
    print("FIXED OANDA DATA FETCHER DEMO")
    print("="*70)
    print("Using direct REST API calls with 5-second timeout")
    print("="*70)

    # Initialize
    fetcher = FixedOandaDataFetcher(practice=True)

    if not fetcher.base_url:
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
        print(f"  Margin available: ${account['margin_available']:,.2f}")
    else:
        print("[ERROR] Failed to get account info")

    # Test open trades
    print("\n[TEST 4] Getting open trades...")
    trades = fetcher.get_open_trades()
    if trades is not None:
        print(f"[OK] Found {len(trades)} open trades")
        for trade in trades:
            print(f"  {trade['instrument']}: {trade['units']} units, P&L: ${trade['unrealized_pl']:.2f}")
    else:
        print("[ERROR] Failed to get open trades")

    print("\n" + "="*70)
    print("OANDA integration ready with timeout protection")
    print("No more hanging issues!")
    print("="*70)


if __name__ == "__main__":
    demo()
