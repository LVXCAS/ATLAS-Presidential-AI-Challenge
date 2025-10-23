#!/usr/bin/env python3
"""
Sequential API Manager - Exhausts one API before moving to the next
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import pandas as pd

# Suppress yfinance errors
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.WARNING)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

try:
    from polygon import RESTClient as PolygonRESTClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class SequentialAPIManager:
    """
    Manages multiple API sources sequentially.
    Uses all calls on one API, then moves to next when rate limited.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # API usage tracking
        self.api_errors = {
            'yahoo': 0,
            'alpaca': 0,
            'polygon': 0,
            'finnhub': 0,
            'twelvedata': 0
        }

        self.api_success = {
            'yahoo': 0,
            'alpaca': 0,
            'polygon': 0,
            'finnhub': 0,
            'twelvedata': 0
        }

        self.api_last_error = {
            'yahoo': None,
            'alpaca': None,
            'polygon': None,
            'finnhub': None,
            'twelvedata': None
        }

        # Sequential priority order
        self.api_priority = ['yahoo', 'alpaca', 'polygon', 'finnhub', 'twelvedata']
        self.current_api_index = 0

        # Rate limit thresholds before switching API
        self.error_threshold = 3  # Switch after 3 consecutive errors

        # Initialize available APIs
        self.available_apis = []
        self._check_available_apis()

        print("\n" + "="*60)
        print("SEQUENTIAL API MANAGER INITIALIZED")
        print("="*60)
        print(f"Available APIs: {', '.join(self.available_apis)}")
        print(f"Priority order: {' > '.join(self.available_apis)}")
        print("="*60)

    def _check_available_apis(self):
        """Check which APIs are available"""

        # Yahoo Finance (always try first - free)
        if YFINANCE_AVAILABLE:
            self.available_apis.append('yahoo')

        # Alpaca
        if ALPACA_AVAILABLE and self.config.get('alpaca_key'):
            self.available_apis.append('alpaca')

        # Polygon
        if POLYGON_AVAILABLE and self.config.get('polygon_key'):
            self.available_apis.append('polygon')

        # Finnhub
        if REQUESTS_AVAILABLE and self.config.get('finnhub_key'):
            self.available_apis.append('finnhub')

        # TwelveData
        if REQUESTS_AVAILABLE and self.config.get('twelvedata_key'):
            self.available_apis.append('twelvedata')

    def get_current_api(self) -> str:
        """Get currently active API"""
        if not self.available_apis:
            return None

        if self.current_api_index >= len(self.available_apis):
            self.current_api_index = 0  # Reset to first API

        return self.available_apis[self.current_api_index]

    def switch_to_next_api(self):
        """Switch to next API in sequence"""
        old_api = self.get_current_api()
        self.current_api_index = (self.current_api_index + 1) % len(self.available_apis)
        new_api = self.get_current_api()

        print(f"\n{'='*60}")
        print(f"API SWITCH: {old_api.upper()} > {new_api.upper()}")
        print(f"{'='*60}")
        print(f"Reason: {old_api} exceeded error threshold ({self.api_errors[old_api]} errors)")
        print(f"Now using: {new_api}")
        print(f"{'='*60}\n")

    def record_success(self, api_name: str):
        """Record successful API call"""
        self.api_success[api_name] = self.api_success.get(api_name, 0) + 1
        self.api_errors[api_name] = 0  # Reset error count on success

    def record_error(self, api_name: str, error_msg: str = None):
        """Record API error and switch if threshold exceeded"""
        self.api_errors[api_name] = self.api_errors.get(api_name, 0) + 1
        self.api_last_error[api_name] = datetime.now()

        # Check if we should switch APIs
        if self.api_errors[api_name] >= self.error_threshold:
            self.switch_to_next_api()

    def get_market_data(self, symbol: str, period: str = "10d") -> Optional[Dict]:
        """
        Get market data using sequential API fallback.
        Tries current API first, falls back to next on error.
        """

        # Try each API in sequence until one works
        for attempt in range(len(self.available_apis)):
            current_api = self.get_current_api()

            if current_api == 'yahoo':
                data = self._get_yahoo_data(symbol, period)
            elif current_api == 'alpaca':
                data = self._get_alpaca_data(symbol, period)
            elif current_api == 'polygon':
                data = self._get_polygon_data(symbol, period)
            elif current_api == 'finnhub':
                data = self._get_finnhub_data(symbol)
            elif current_api == 'twelvedata':
                data = self._get_twelvedata_data(symbol)
            else:
                data = None

            if data:
                self.record_success(current_api)
                return data
            else:
                self.record_error(current_api, "Failed to fetch data")
                # Loop will try next API

        # All APIs failed
        print(f"❌ ALL APIs FAILED for {symbol}")
        return None

    def _get_yahoo_data(self, symbol: str, period: str = "10d") -> Optional[Dict]:
        """Get data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if hist.empty or len(hist) < 5:
                # Try shorter period
                hist = ticker.history(period="5d")
                if hist.empty or len(hist) < 3:
                    return None

            # Extract market data
            current_price = float(hist["Close"].iloc[-1])
            prev_price = float(hist["Close"].iloc[-5]) if len(hist) >= 5 else current_price
            price_momentum = (current_price - prev_price) / prev_price if prev_price > 0 else 0
            current_volume = float(hist["Volume"].iloc[-1])
            avg_volume = hist["Volume"].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            returns = hist["Close"].pct_change().dropna()
            realized_vol = returns.std() * 100 * (252 ** 0.5) if len(returns) > 0 else 20.0

            return {
                "symbol": symbol,
                "current_price": current_price,
                "price_momentum": price_momentum,
                "realized_vol": realized_vol,
                "volume_ratio": volume_ratio,
                "price_position": 0.5,
                "timestamp": datetime.now(),
                "source": "yahoo"
            }

        except Exception as e:
            # Check for rate limiting
            if "Too Many Requests" in str(e) or "Rate limit" in str(e):
                print(f"⚠️  Yahoo Finance RATE LIMITED: {e}")
            return None

    def _get_alpaca_data(self, symbol: str, period: str = "10d") -> Optional[Dict]:
        """Get data from Alpaca"""
        try:
            api_key = self.config.get('alpaca_key')
            secret_key = self.config.get('alpaca_secret')
            base_url = self.config.get('alpaca_base_url', 'https://paper-api.alpaca.markets')

            if not api_key or not secret_key:
                return None

            api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')

            # Get bars data
            bars = api.get_bars(
                symbol,
                '1Day',
                limit=10
            ).df

            if bars.empty:
                return None

            current_price = float(bars['close'].iloc[-1])
            prev_price = float(bars['close'].iloc[-5]) if len(bars) >= 5 else current_price
            price_momentum = (current_price - prev_price) / prev_price if prev_price > 0 else 0
            current_volume = float(bars['volume'].iloc[-1])
            avg_volume = bars['volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            returns = bars['close'].pct_change().dropna()
            realized_vol = returns.std() * 100 * (252 ** 0.5) if len(returns) > 0 else 20.0

            return {
                "symbol": symbol,
                "current_price": current_price,
                "price_momentum": price_momentum,
                "realized_vol": realized_vol,
                "volume_ratio": volume_ratio,
                "price_position": 0.5,
                "timestamp": datetime.now(),
                "source": "alpaca"
            }

        except Exception as e:
            print(f"Alpaca error for {symbol}: {e}")
            return None

    def _get_polygon_data(self, symbol: str, period: str = "10d") -> Optional[Dict]:
        """Get data from Polygon"""
        try:
            api_key = self.config.get('polygon_key')
            if not api_key:
                return None

            client = PolygonRESTClient(api_key)

            # Get aggregates
            from datetime import date, timedelta
            end_date = date.today()
            start_date = end_date - timedelta(days=10)

            aggs = client.get_aggs(
                symbol,
                1,
                'day',
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            if not aggs:
                return None

            # Convert to dataframe
            df = pd.DataFrame(aggs)

            current_price = float(df['close'].iloc[-1])
            prev_price = float(df['close'].iloc[-5]) if len(df) >= 5 else current_price
            price_momentum = (current_price - prev_price) / prev_price if prev_price > 0 else 0
            current_volume = float(df['volume'].iloc[-1])
            avg_volume = df['volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            returns = df['close'].pct_change().dropna()
            realized_vol = returns.std() * 100 * (252 ** 0.5) if len(returns) > 0 else 20.0

            return {
                "symbol": symbol,
                "current_price": current_price,
                "price_momentum": price_momentum,
                "realized_vol": realized_vol,
                "volume_ratio": volume_ratio,
                "price_position": 0.5,
                "timestamp": datetime.now(),
                "source": "polygon"
            }

        except Exception as e:
            print(f"Polygon error for {symbol}: {e}")
            return None

    def _get_finnhub_data(self, symbol: str) -> Optional[Dict]:
        """Get data from Finnhub"""
        try:
            api_key = self.config.get('finnhub_key')
            if not api_key:
                return None

            # Get quote
            url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return None

            data = response.json()

            current_price = float(data.get('c', 0))
            open_price = float(data.get('o', 0))

            if current_price == 0:
                return None

            # Calculate basic metrics
            price_momentum = (current_price - open_price) / open_price if open_price > 0 else 0

            return {
                "symbol": symbol,
                "current_price": current_price,
                "price_momentum": price_momentum,
                "realized_vol": 25.0,  # Default estimate
                "volume_ratio": 1.0,   # Not available
                "price_position": 0.5,
                "timestamp": datetime.now(),
                "source": "finnhub"
            }

        except Exception as e:
            print(f"Finnhub error for {symbol}: {e}")
            return None

    def _get_twelvedata_data(self, symbol: str) -> Optional[Dict]:
        """Get data from TwelveData"""
        try:
            api_key = self.config.get('twelvedata_key')
            if not api_key:
                return None

            # Get quote
            url = f"https://api.twelvedata.com/quote?symbol={symbol}&apikey={api_key}"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return None

            data = response.json()

            current_price = float(data.get('close', 0))
            open_price = float(data.get('open', 0))

            if current_price == 0:
                return None

            price_momentum = (current_price - open_price) / open_price if open_price > 0 else 0

            return {
                "symbol": symbol,
                "current_price": current_price,
                "price_momentum": price_momentum,
                "realized_vol": 25.0,  # Default estimate
                "volume_ratio": 1.0,   # Not available
                "price_position": 0.5,
                "timestamp": datetime.now(),
                "source": "twelvedata"
            }

        except Exception as e:
            print(f"TwelveData error for {symbol}: {e}")
            return None

    def get_stats(self) -> Dict:
        """Get API usage statistics"""
        return {
            'current_api': self.get_current_api(),
            'api_success': self.api_success,
            'api_errors': self.api_errors,
            'available_apis': self.available_apis
        }


# Global instance
_api_manager = None

def get_api_manager(config: Dict = None) -> SequentialAPIManager:
    """Get or create global API manager"""
    global _api_manager
    if _api_manager is None:
        import os
        from dotenv import load_dotenv
        load_dotenv()

        default_config = {
            'alpaca_key': os.getenv('ALPACA_API_KEY'),
            'alpaca_secret': os.getenv('ALPACA_SECRET_KEY'),
            'alpaca_base_url': os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
            'polygon_key': os.getenv('POLYGON_API_KEY'),
            'finnhub_key': os.getenv('FINNHUB_API_KEY'),
            'twelvedata_key': os.getenv('TWELVEDATA_API_KEY'),
        }

        _api_manager = SequentialAPIManager(config or default_config)

    return _api_manager
