#!/usr/bin/env python3
"""
Multi-API Data Provider
Robust market data integration with multiple API fallbacks to avoid rate limits
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import asyncio
import requests
import json
import time
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import alpha_vantage
    from alpha_vantage.timeseries import TimeSeries
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv('.env')
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
    TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
except:
    ALPHA_VANTAGE_API_KEY = None
    POLYGON_API_KEY = None
    TWELVEDATA_API_KEY = None
    FINNHUB_API_KEY = None

class MultiAPIDataProvider:
    """Multi-source market data provider with automatic fallbacks"""

    def __init__(self):
        self.session = requests.Session()
        self.data_cache = {}
        self.cache_expiry = 300  # 5 minutes
        self.api_usage = {}
        self.api_limits = {
            'yfinance': {'calls_per_minute': 2000, 'daily_limit': None},  # Very generous for Yahoo
            'alpha_vantage': {'calls_per_minute': 5, 'daily_limit': 500},
            'polygon': {'calls_per_minute': 5, 'daily_limit': 1000},
            'twelvedata': {'calls_per_minute': 8, 'daily_limit': 800},
            'finnhub': {'calls_per_minute': 30, 'daily_limit': None}  # Reduced from 60
        }

        # Initialize API availability
        self.available_apis = []
        self._check_api_availability()

        print(f"Multi-API Data Provider initialized with {len(self.available_apis)} available sources")

    def _check_api_availability(self):
        """Check which APIs are available"""
        if YFINANCE_AVAILABLE:
            self.available_apis.append('yfinance')
            print("+ Yahoo Finance available (primary)")

        if ALPHA_VANTAGE_API_KEY and ALPHA_VANTAGE_AVAILABLE:
            self.available_apis.append('alpha_vantage')
            print("+ Alpha Vantage available")

        if POLYGON_API_KEY:
            self.available_apis.append('polygon')
            print("+ Polygon available")

        if TWELVEDATA_API_KEY:
            self.available_apis.append('twelvedata')
            print("+ TwelveData available")

        if FINNHUB_API_KEY:
            self.available_apis.append('finnhub')
            print("+ Finnhub available (fallback only)")

    def _check_rate_limit(self, api_name: str) -> bool:
        """Check if API is within rate limits"""
        if api_name not in self.api_usage:
            self.api_usage[api_name] = []

        now = time.time()
        # Remove old timestamps
        self.api_usage[api_name] = [t for t in self.api_usage[api_name] if now - t < 60]

        limit = self.api_limits[api_name]['calls_per_minute']
        if len(self.api_usage[api_name]) >= limit:
            return False

        return True

    def _record_api_call(self, api_name: str):
        """Record an API call for rate limiting"""
        if api_name not in self.api_usage:
            self.api_usage[api_name] = []
        self.api_usage[api_name].append(time.time())

    async def get_comprehensive_analysis(self, symbol: str, period: str = "60d") -> Dict:
        """Get comprehensive market data with automatic fallbacks"""
        cache_key = f"{symbol}_{period}"

        # Check cache first
        if cache_key in self.data_cache:
            cache_time, data = self.data_cache[cache_key]
            if time.time() - cache_time < self.cache_expiry:
                return data

        # Try APIs in order of preference
        for api_name in self.available_apis:
            if not self._check_rate_limit(api_name):
                print(f"Rate limit reached for {api_name}, trying next API...")
                continue

            try:
                data = await self._get_data_from_api(api_name, symbol, period)
                if data:
                    self._record_api_call(api_name)
                    self.data_cache[cache_key] = (time.time(), data)
                    print(f"Successfully got data from {api_name} for {symbol}")
                    return data
            except Exception as e:
                print(f"Error with {api_name} for {symbol}: {e}")
                continue

        print(f"All APIs failed for {symbol}, returning fallback data")
        return self._get_fallback_data(symbol)

    async def _get_data_from_api(self, api_name: str, symbol: str, period: str) -> Optional[Dict]:
        """Get data from specific API"""
        if api_name == 'yfinance':
            return await self._get_yfinance_data(symbol, period)
        elif api_name == 'alpha_vantage':
            return await self._get_alpha_vantage_data(symbol, period)
        elif api_name == 'polygon':
            return await self._get_polygon_data(symbol, period)
        elif api_name == 'twelvedata':
            return await self._get_twelvedata_data(symbol, period)
        elif api_name == 'finnhub':
            return await self._get_finnhub_data(symbol, period)
        return None

    async def _get_yfinance_data(self, symbol: str, period: str) -> Optional[Dict]:
        """Get data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if hist.empty:
                return None

            current_price = float(hist['Close'].iloc[-1])
            volume = hist['Volume'].mean()

            # Calculate basic technical indicators
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * 100 * (252 ** 0.5)

            # Simple RSI calculation
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Basic momentum
            price_5d_ago = hist['Close'].iloc[-6] if len(hist) >= 6 else current_price
            momentum_5d = ((current_price - price_5d_ago) / price_5d_ago) * 100

            return {
                'current_price': current_price,
                'technical_indicators': {
                    'rsi': rsi.iloc[-1] if not rsi.empty else 50,
                    'volume_sma': volume,
                    'volume_ratio': hist['Volume'].iloc[-1] / volume if volume > 0 else 1.0
                },
                'volatility_analysis': {
                    'realized_vol_20d': volatility,
                    'vol_percentile': 0.5,
                    'vol_trend': 'STABLE',
                    'vol_regime': 'NORMAL'
                },
                'momentum_analysis': {
                    'price_momentum_5d': momentum_5d,
                    'momentum_strength': 'MODERATE' if abs(momentum_5d) > 2 else 'WEAK',
                    'acceleration': 0
                },
                'signals': {
                    'overall_signal': 'BULLISH' if momentum_5d > 1 else 'BEARISH' if momentum_5d < -1 else 'NEUTRAL',
                    'signal_strength': min(abs(momentum_5d) / 5.0, 1.0),
                    'confidence': 0.6 if abs(momentum_5d) > 2 else 0.4,
                    'bullish_factors': ['positive_momentum'] if momentum_5d > 1 else [],
                    'bearish_factors': ['negative_momentum'] if momentum_5d < -1 else []
                },
                'support_resistance': {
                    'nearest_support': None,
                    'nearest_resistance': None,
                    'support_levels': [],
                    'resistance_levels': []
                }
            }
        except Exception as e:
            print(f"Yahoo Finance error: {e}")
            return None

    async def _get_alpha_vantage_data(self, symbol: str, period: str) -> Optional[Dict]:
        """Get data from Alpha Vantage"""
        try:
            if not ALPHA_VANTAGE_API_KEY:
                return None

            ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
            data, meta_data = ts.get_daily(symbol=symbol, outputsize='compact')

            if data.empty:
                return None

            current_price = float(data['4. close'].iloc[-1])

            # Basic analysis similar to Yahoo Finance
            returns = data['4. close'].pct_change().dropna()
            volatility = returns.std() * 100 * (252 ** 0.5)

            return {
                'current_price': current_price,
                'technical_indicators': {'rsi': 50, 'volume_ratio': 1.0},
                'volatility_analysis': {'realized_vol_20d': volatility},
                'momentum_analysis': {'price_momentum_5d': 0},
                'signals': {'overall_signal': 'NEUTRAL', 'signal_strength': 0.0, 'confidence': 0.5},
                'support_resistance': {}
            }
        except Exception as e:
            print(f"Alpha Vantage error: {e}")
            return None

    async def _get_polygon_data(self, symbol: str, period: str) -> Optional[Dict]:
        """Get data from Polygon"""
        try:
            if not POLYGON_API_KEY:
                return None

            # Use basic REST API call for Polygon
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
            params = {'apikey': POLYGON_API_KEY}

            response = self.session.get(url, params=params)
            if response.status_code != 200:
                return None

            data = response.json()
            if not data.get('results'):
                return None

            latest = data['results'][-1]
            current_price = latest['c']  # close price

            return {
                'current_price': current_price,
                'technical_indicators': {'rsi': 50, 'volume_ratio': 1.0},
                'volatility_analysis': {'realized_vol_20d': 20.0},
                'momentum_analysis': {'price_momentum_5d': 0},
                'signals': {'overall_signal': 'NEUTRAL', 'signal_strength': 0.0, 'confidence': 0.5},
                'support_resistance': {}
            }
        except Exception as e:
            print(f"Polygon error: {e}")
            return None

    async def _get_twelvedata_data(self, symbol: str, period: str) -> Optional[Dict]:
        """Get data from TwelveData"""
        try:
            if not TWELVEDATA_API_KEY:
                return None

            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': symbol,
                'interval': '1day',
                'outputsize': '30',
                'apikey': TWELVEDATA_API_KEY
            }

            response = self.session.get(url, params=params)
            if response.status_code != 200:
                return None

            data = response.json()
            if 'values' not in data or not data['values']:
                return None

            current_price = float(data['values'][0]['close'])

            return {
                'current_price': current_price,
                'technical_indicators': {'rsi': 50, 'volume_ratio': 1.0},
                'volatility_analysis': {'realized_vol_20d': 20.0},
                'momentum_analysis': {'price_momentum_5d': 0},
                'signals': {'overall_signal': 'NEUTRAL', 'signal_strength': 0.0, 'confidence': 0.5},
                'support_resistance': {}
            }
        except Exception as e:
            print(f"TwelveData error: {e}")
            return None

    async def _get_finnhub_data(self, symbol: str, period: str) -> Optional[Dict]:
        """Get data from Finnhub (fallback only)"""
        try:
            if not FINNHUB_API_KEY:
                return None

            # Only use for basic quote
            url = f"https://finnhub.io/api/v1/quote"
            params = {'symbol': symbol, 'token': FINNHUB_API_KEY}

            response = self.session.get(url, params=params)
            if response.status_code != 200:
                return None

            data = response.json()
            current_price = data.get('c', 0)  # current price

            if current_price <= 0:
                return None

            return {
                'current_price': current_price,
                'technical_indicators': {'rsi': 50, 'volume_ratio': 1.0},
                'volatility_analysis': {'realized_vol_20d': 20.0},
                'momentum_analysis': {'price_momentum_5d': 0},
                'signals': {'overall_signal': 'NEUTRAL', 'signal_strength': 0.0, 'confidence': 0.5},
                'support_resistance': {}
            }
        except Exception as e:
            print(f"Finnhub error: {e}")
            return None

    def _get_fallback_data(self, symbol: str) -> Dict:
        """Get fallback data when all APIs fail"""
        return {
            'current_price': 100.0,  # Placeholder
            'technical_indicators': {'rsi': 50, 'volume_ratio': 1.0},
            'volatility_analysis': {'realized_vol_20d': 20.0},
            'momentum_analysis': {'price_momentum_5d': 0},
            'signals': {'overall_signal': 'NEUTRAL', 'signal_strength': 0.0, 'confidence': 0.3},
            'support_resistance': {}
        }

# Create global instance
multi_api_provider = MultiAPIDataProvider()