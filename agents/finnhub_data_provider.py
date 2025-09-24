#!/usr/bin/env python3
"""
Finnhub Data Provider
Professional market data integration with Finnhub API
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

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv('.env')
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
except:
    FINNHUB_API_KEY = None

class FinnhubDataProvider:
    """Professional market data provider using Finnhub API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or FINNHUB_API_KEY or "d32sc4pr01qtm631ej60d32sc4pr01qtm631ej6g"  # Use env var or default
        self.base_url = "https://finnhub.io/api/v1"
        self.session = requests.Session()
        
        # Rate limiting - Finnhub free tier allows 60 calls/minute
        self.rate_limit = 30  # Conservative: 30 requests per minute
        self.request_times = []
        
        # Cache for market data
        self.data_cache = {}
        self.cache_expiry = 300  # 5 minutes
        
        print(f"+ Finnhub Data Provider initialized with API key: ...{self.api_key[-6:]}")
    
    def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        now = time.time()
        
        # Remove old timestamps
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # Check if we're within rate limit
        if len(self.request_times) >= self.rate_limit:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                print(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
        
        self.request_times.append(now)
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with error handling and rate limiting"""
        self._check_rate_limit()
        
        if params is None:
            params = {}
        
        params['token'] = self.api_key
        
        try:
            response = self.session.get(f"{self.base_url}/{endpoint}", params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"- Finnhub API request failed: {e}")
            return {}
        except json.JSONDecodeError as e:
            print(f"- Failed to parse Finnhub response: {e}")
            return {}
    
    async def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote for a symbol"""
        cache_key = f"quote_{symbol}"
        
        # Check cache
        if cache_key in self.data_cache:
            cached_data, timestamp = self.data_cache[cache_key]
            if time.time() - timestamp < 60:  # 1-minute cache for quotes
                return cached_data
        
        try:
            data = self._make_request("quote", {"symbol": symbol})
            
            if data and 'c' in data:  # 'c' is current price
                quote_data = {
                    'symbol': symbol,
                    'current_price': data.get('c', 0),
                    'change': data.get('d', 0),
                    'percent_change': data.get('dp', 0),
                    'high': data.get('h', 0),
                    'low': data.get('l', 0),
                    'open': data.get('o', 0),
                    'previous_close': data.get('pc', 0),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'finnhub'
                }
                
                # Cache the result
                self.data_cache[cache_key] = (quote_data, time.time())
                
                return quote_data
            else:
                print(f"- No quote data for {symbol}")
                return {}
                
        except Exception as e:
            print(f"- Error getting quote for {symbol}: {e}")
            return {}
    
    async def get_historical_data(self, symbol: str, resolution: str = "D", 
                                days_back: int = 365) -> pd.DataFrame:
        """Get historical price data"""
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Convert to Unix timestamps
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        cache_key = f"candles_{symbol}_{resolution}_{days_back}"
        
        # Check cache
        if cache_key in self.data_cache:
            cached_data, timestamp = self.data_cache[cache_key]
            if time.time() - timestamp < self.cache_expiry:
                return cached_data
        
        try:
            # Try Finnhub first (but will likely fail on free tier)
            data = self._make_request("stock/candle", {
                "symbol": symbol,
                "resolution": resolution,
                "from": start_timestamp,
                "to": end_timestamp
            })
            
            # If Finnhub fails (403 error), fall back to Yahoo Finance
            if not data or data.get('s') != 'ok':
                print(f"- Finnhub historical data not available for {symbol}, using Yahoo Finance")
                return await self._get_yahoo_historical_data(symbol, days_back)
            
            if data and data.get('s') == 'ok' and 't' in data:
                # Convert to DataFrame
                df = pd.DataFrame({
                    'timestamp': data['t'],
                    'open': data['o'],
                    'high': data['h'],
                    'low': data['l'],
                    'close': data['c'],
                    'volume': data['v']
                })
                
                # Convert timestamp to datetime
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('datetime', inplace=True)
                df.drop('timestamp', axis=1, inplace=True)
                
                # Sort by date
                df.sort_index(inplace=True)
                
                # Cache the result
                self.data_cache[cache_key] = (df, time.time())
                
                print(f"+ Retrieved {len(df)} days of historical data for {symbol}")
                return df
            else:
                print(f"- No historical data for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"- Error getting historical data for {symbol}: {e}")
            # Final fallback to Yahoo Finance
            return await self._get_yahoo_historical_data(symbol, days_back)
    
    async def _get_yahoo_historical_data(self, symbol: str, days_back: int = 365) -> pd.DataFrame:
        """Fallback to Yahoo Finance for historical data"""
        try:
            import yfinance as yf
            
            # Download data
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                print(f"- No Yahoo Finance data for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names to match Finnhub format
            hist = hist.reset_index()
            hist.columns = [col[0] if isinstance(col, tuple) else col for col in hist.columns]
            
            # Ensure we have the right columns and standardize column names
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if all(col in hist.columns for col in required_cols):
                hist = hist[required_cols].copy()
                
                # Standardize column names to lowercase for consistency
                hist.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                hist['timestamp'] = pd.to_datetime(hist['date']).astype(int) // 10**9
                
                print(f"+ Yahoo Finance data loaded for {symbol} ({len(hist)} days)")
                return hist
            else:
                print(f"- Yahoo Finance data missing required columns for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"- Yahoo Finance fallback failed for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_company_profile(self, symbol: str) -> Dict:
        """Get company profile information"""
        cache_key = f"profile_{symbol}"
        
        # Check cache (longer cache for company info)
        if cache_key in self.data_cache:
            cached_data, timestamp = self.data_cache[cache_key]
            if time.time() - timestamp < 3600:  # 1-hour cache
                return cached_data
        
        try:
            data = self._make_request("stock/profile2", {"symbol": symbol})
            
            if data:
                profile_data = {
                    'symbol': symbol,
                    'name': data.get('name', ''),
                    'country': data.get('country', ''),
                    'currency': data.get('currency', ''),
                    'exchange': data.get('exchange', ''),
                    'industry': data.get('finnhubIndustry', ''),
                    'market_cap': data.get('marketCapitalization', 0),
                    'shares_outstanding': data.get('shareOutstanding', 0),
                    'logo': data.get('logo', ''),
                    'weburl': data.get('weburl', ''),
                    'source': 'finnhub'
                }
                
                # Cache the result
                self.data_cache[cache_key] = (profile_data, time.time())
                
                return profile_data
            else:
                return {}
                
        except Exception as e:
            print(f"- Error getting company profile for {symbol}: {e}")
            return {}
    
    async def get_financial_metrics(self, symbol: str) -> Dict:
        """Get key financial metrics"""
        cache_key = f"metrics_{symbol}"
        
        # Check cache
        if cache_key in self.data_cache:
            cached_data, timestamp = self.data_cache[cache_key]
            if time.time() - timestamp < 3600:  # 1-hour cache
                return cached_data
        
        try:
            data = self._make_request("stock/metric", {"symbol": symbol, "metric": "all"})
            
            if data and 'metric' in data:
                metrics = data['metric']
                
                metrics_data = {
                    'symbol': symbol,
                    'pe_ratio': metrics.get('peBasicExclExtraTTM', 0),
                    'pb_ratio': metrics.get('pbQuarterly', 0),
                    'ps_ratio': metrics.get('psQuarterly', 0),
                    'roe': metrics.get('roeRfy', 0),
                    'roa': metrics.get('roaRfy', 0),
                    'debt_to_equity': metrics.get('totalDebt/totalEquityQuarterly', 0),
                    'current_ratio': metrics.get('currentRatioQuarterly', 0),
                    'gross_margin': metrics.get('grossMarginTTM', 0),
                    'net_margin': metrics.get('netProfitMarginTTM', 0),
                    'beta': metrics.get('beta', 1.0),
                    '52w_high': metrics.get('52WeekHigh', 0),
                    '52w_low': metrics.get('52WeekLow', 0),
                    'source': 'finnhub'
                }
                
                # Cache the result
                self.data_cache[cache_key] = (metrics_data, time.time())
                
                return metrics_data
            else:
                return {}
                
        except Exception as e:
            print(f"- Error getting financial metrics for {symbol}: {e}")
            return {}
    
    async def get_earnings_calendar(self, from_date: str = None, to_date: str = None) -> List[Dict]:
        """Get earnings calendar"""
        if from_date is None:
            from_date = datetime.now().strftime('%Y-%m-%d')
        if to_date is None:
            to_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        
        try:
            data = self._make_request("calendar/earnings", {
                "from": from_date,
                "to": to_date
            })
            
            if data and 'earningsCalendar' in data:
                earnings_list = []
                
                for earning in data['earningsCalendar']:
                    earnings_list.append({
                        'symbol': earning.get('symbol', ''),
                        'date': earning.get('date', ''),
                        'eps_estimate': earning.get('epsEstimate', 0),
                        'eps_actual': earning.get('epsActual', 0),
                        'revenue_estimate': earning.get('revenueEstimate', 0),
                        'revenue_actual': earning.get('revenueActual', 0),
                        'quarter': earning.get('quarter', 0),
                        'year': earning.get('year', 0)
                    })
                
                return earnings_list
            else:
                return []
                
        except Exception as e:
            print(f"- Error getting earnings calendar: {e}")
            return []
    
    async def get_news(self, symbol: str = None, category: str = "general", 
                      min_id: int = 0) -> List[Dict]:
        """Get market news"""
        try:
            params = {
                "category": category,
                "minId": min_id
            }
            
            if symbol:
                # Company-specific news
                endpoint = "company-news"
                # Add date range for company news
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)
                params.update({
                    "symbol": symbol,
                    "from": start_date.strftime('%Y-%m-%d'),
                    "to": end_date.strftime('%Y-%m-%d')
                })
            else:
                # General market news
                endpoint = "news"
            
            data = self._make_request(endpoint, params)
            
            if data and isinstance(data, list):
                news_list = []
                
                for article in data[:20]:  # Limit to 20 articles
                    news_list.append({
                        'id': article.get('id', 0),
                        'headline': article.get('headline', ''),
                        'summary': article.get('summary', ''),
                        'source': article.get('source', ''),
                        'url': article.get('url', ''),
                        'datetime': datetime.fromtimestamp(article.get('datetime', 0)).isoformat() if article.get('datetime') else '',
                        'related': article.get('related', ''),
                        'image': article.get('image', ''),
                        'category': category,
                        'symbol': symbol
                    })
                
                return news_list
            else:
                return []
                
        except Exception as e:
            print(f"- Error getting news: {e}")
            return []
    
    async def get_technical_indicators(self, symbol: str, resolution: str = "D", 
                                     indicator: str = "rsi", days_back: int = 100) -> Dict:
        """Get technical indicators"""
        
        # First get historical data
        df = await self.get_historical_data(symbol, resolution, days_back)
        
        if df.empty:
            return {}
        
        try:
            # Calculate common technical indicators
            indicators = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
            
            # RSI calculation
            if len(df) >= 14:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                indicators['rsi'] = float(rsi.iloc[-1]) if not rsi.empty else 50
            
            # Moving averages
            if len(df) >= 20:
                indicators['sma_20'] = float(df['close'].rolling(20).mean().iloc[-1])
            if len(df) >= 50:
                indicators['sma_50'] = float(df['close'].rolling(50).mean().iloc[-1])
            if len(df) >= 200:
                indicators['sma_200'] = float(df['close'].rolling(200).mean().iloc[-1])
            
            # EMA
            if len(df) >= 12:
                indicators['ema_12'] = float(df['close'].ewm(span=12).mean().iloc[-1])
            if len(df) >= 26:
                indicators['ema_26'] = float(df['close'].ewm(span=26).mean().iloc[-1])
            
            # MACD
            if 'ema_12' in indicators and 'ema_26' in indicators:
                indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            
            # Bollinger Bands
            if len(df) >= 20:
                sma_20 = df['close'].rolling(20).mean()
                std_20 = df['close'].rolling(20).std()
                indicators['bb_upper'] = float((sma_20 + (std_20 * 2)).iloc[-1])
                indicators['bb_lower'] = float((sma_20 - (std_20 * 2)).iloc[-1])
                indicators['bb_middle'] = float(sma_20.iloc[-1])
            
            # Current price position
            current_price = df['close'].iloc[-1]
            indicators['current_price'] = float(current_price)
            
            # Price momentum
            if len(df) >= 10:
                indicators['momentum_10d'] = float((current_price / df['close'].iloc[-10] - 1) * 100)
            
            return indicators
            
        except Exception as e:
            print(f"- Error calculating technical indicators for {symbol}: {e}")
            return {}
    
    async def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get quotes for multiple symbols efficiently"""
        results = {}
        
        print(f"Getting quotes for {len(symbols)} symbols...")
        
        # Process in batches to respect rate limits
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            # Process batch
            tasks = [self.get_quote(symbol) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            for symbol, result in zip(batch, batch_results):
                if isinstance(result, dict) and result:
                    results[symbol] = result
                else:
                    results[symbol] = {'error': str(result) if isinstance(result, Exception) else 'No data'}
            
            # Small delay between batches
            if i + batch_size < len(symbols):
                await asyncio.sleep(1)
        
        print(f"+ Retrieved quotes for {len([r for r in results.values() if 'error' not in r])}/{len(symbols)} symbols")
        return results
    
    async def get_market_summary(self) -> Dict:
        """Get overall market summary"""
        try:
            # Get quotes for major indices
            indices = ['SPY', 'QQQ', 'IWM', 'DIA', 'VIX']
            quotes = await self.get_multiple_quotes(indices)
            
            market_summary = {
                'timestamp': datetime.now().isoformat(),
                'indices': quotes,
                'market_status': 'open' if 9.5 <= datetime.now().hour <= 16 else 'closed',
                'source': 'finnhub'
            }
            
            return market_summary
            
        except Exception as e:
            print(f"- Error getting market summary: {e}")
            return {}

# Create global instance (can be initialized with custom API key)
finnhub_provider = FinnhubDataProvider()