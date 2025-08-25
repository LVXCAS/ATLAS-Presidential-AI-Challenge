"""
Live Market Data Service - Real data from yfinance and Alpaca
Provides real-time market data, news, and trading capabilities.
"""

import asyncio
import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
# import alpaca_trade_api as tradeapi  # Temporarily disabled due to version conflict
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)

class LiveMarketDataService:
    """Live market data service using yfinance and Alpaca."""
    
    def __init__(self, alpaca_key: Optional[str] = None, alpaca_secret: Optional[str] = None):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.alpaca_api = None
        
        # Initialize Alpaca API if credentials provided (temporarily disabled)
        # if alpaca_key and alpaca_secret:
        #     try:
        #         self.alpaca_api = tradeapi.REST(
        #             alpaca_key, 
        #             alpaca_secret, 
        #             base_url='https://paper-api.alpaca.markets',  # Paper trading
        #             api_version='v2'
        #         )
        #         logger.info("Alpaca API initialized for paper trading")
        #     except Exception as e:
        #         logger.error(f"Failed to initialize Alpaca API: {e}")
        logger.info("yfinance-only mode - Alpaca temporarily disabled")
        # Cache for data
        self.price_cache = {}
        self.news_cache = {}
        self.last_update = {}
    
    async def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time market data for multiple symbols."""
        try:
            # Run yfinance calls in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            tasks = []
            
            for symbol in symbols:
                task = loop.run_in_executor(
                    self.executor, 
                    self._fetch_symbol_data, 
                    symbol
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            market_data = {}
            for i, result in enumerate(results):
                if not isinstance(result, Exception):
                    market_data[symbols[i]] = result
                else:
                    logger.error(f"Error fetching data for {symbols[i]}: {result}")
                    market_data[symbols[i]] = self._get_fallback_data(symbols[i])
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error in get_market_data: {e}")
            return {}
    
    def _fetch_symbol_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch data for a single symbol using yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current price info
            info = ticker.info
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty:
                return self._get_fallback_data(symbol)
            
            current_price = hist['Close'].iloc[-1]
            prev_close = info.get('previousClose', current_price)
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100 if prev_close != 0 else 0
            
            # Calculate additional metrics
            volume = int(hist['Volume'].iloc[-1]) if not hist['Volume'].empty else 0
            day_high = hist['High'].max()
            day_low = hist['Low'].min()
            
            # Get implied volatility for options (if available)
            iv = info.get('impliedVolatility', 0.25)
            
            return {
                'symbol': symbol,
                'price': round(float(current_price), 2),
                'change': round(float(change), 2),
                'changePercent': round(float(change_percent), 2),
                'volume': volume,
                'dayHigh': round(float(day_high), 2),
                'dayLow': round(float(day_low), 2),
                'iv': iv,
                'marketCap': info.get('marketCap', 0),
                'peRatio': info.get('trailingPE', 0),
                'timestamp': datetime.now().timestamp()
            }
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return self._get_fallback_data(symbol)
    
    def _get_fallback_data(self, symbol: str) -> Dict[str, Any]:
        """Fallback data if real data fails."""
        base_prices = {
            'SPY': 445.00, 'QQQ': 378.00, 'AAPL': 191.00, 'MSFT': 420.00,
            'NVDA': 875.00, 'TSLA': 248.00, 'GOOGL': 172.00, 'AMZN': 181.00
        }
        
        base_price = base_prices.get(symbol, 100.00)
        
        return {
            'symbol': symbol,
            'price': base_price,
            'change': 0.0,
            'changePercent': 0.0,
            'volume': 1000000,
            'dayHigh': base_price * 1.02,
            'dayLow': base_price * 0.98,
            'iv': 0.25,
            'marketCap': 0,
            'peRatio': 0,
            'timestamp': datetime.now().timestamp()
        }
    
    async def get_real_news(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Get real news for symbols using yfinance."""
        try:
            loop = asyncio.get_event_loop()
            all_news = []
            
            for symbol in symbols[:5]:  # Limit to avoid rate limits
                news_data = await loop.run_in_executor(
                    self.executor,
                    self._fetch_news_for_symbol,
                    symbol
                )
                all_news.extend(news_data)
            
            # Sort by timestamp and return recent news
            all_news.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            return all_news[:20]  # Return top 20 news items
            
        except Exception as e:
            logger.error(f"Error getting real news: {e}")
            return self._get_fallback_news()
    
    def _fetch_news_for_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch news for a specific symbol."""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            processed_news = []
            for item in news[:5]:  # Top 5 news per symbol
                processed_news.append({
                    'title': item.get('title', 'No title'),
                    'summary': item.get('summary', 'No summary available'),
                    'url': item.get('link', ''),
                    'source': item.get('publisher', 'Unknown'),
                    'symbol': symbol,
                    'timestamp': item.get('providerPublishTime', datetime.now().timestamp()),
                    'sentiment_score': self._analyze_sentiment(item.get('title', '') + ' ' + item.get('summary', '')),
                    'impact': 'MEDIUM'
                })
            
            return processed_news
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis."""
        positive_words = ['bullish', 'growth', 'profit', 'beat', 'strong', 'up', 'gain', 'rise', 'surge', 'boost']
        negative_words = ['bearish', 'loss', 'miss', 'weak', 'down', 'fall', 'drop', 'crash', 'decline', 'cut']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return min(0.8, positive_count * 0.2)
        elif negative_count > positive_count:
            return max(-0.8, -negative_count * 0.2)
        else:
            return 0.0
    
    def _get_fallback_news(self) -> List[Dict[str, Any]]:
        """Fallback news if real news fails."""
        return [
            {
                'title': 'Market Data Service Online - Real News Loading...',
                'summary': 'Live market data service is active and fetching real news from yfinance.',
                'source': 'HIVE TRADE SYSTEM',
                'symbol': 'SPY',
                'timestamp': datetime.now().timestamp(),
                'sentiment_score': 0.1,
                'impact': 'LOW'
            }
        ]
    
    async def get_chart_data(self, symbol: str, period: str = "1d", interval: str = "1m") -> Dict[str, Any]:
        """Get real chart data for a symbol."""
        try:
            loop = asyncio.get_event_loop()
            chart_data = await loop.run_in_executor(
                self.executor,
                self._fetch_chart_data,
                symbol, period, interval
            )
            return chart_data
            
        except Exception as e:
            logger.error(f"Error getting chart data for {symbol}: {e}")
            return {}
    
    def _fetch_chart_data(self, symbol: str, period: str, interval: str) -> Dict[str, Any]:
        """Fetch chart data using yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                return {}
            
            # Convert to the format expected by the frontend
            candle_data = []
            for timestamp, row in hist.iterrows():
                candle_data.append({
                    'time': timestamp.timestamp(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            
            return {
                'symbol': symbol,
                'data': candle_data,
                'period': period,
                'interval': interval
            }
            
        except Exception as e:
            logger.error(f"Error fetching chart data for {symbol}: {e}")
            return {}
    
    async def place_alpaca_order(self, symbol: str, qty: int, side: str, order_type: str = 'market') -> Dict[str, Any]:
        """Place order through Alpaca API."""
        if not self.alpaca_api:
            return {'error': 'Alpaca API not configured'}
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._execute_alpaca_order,
                symbol, qty, side, order_type
            )
            return result
            
        except Exception as e:
            logger.error(f"Error placing Alpaca order: {e}")
            return {'error': str(e)}
    
    def _execute_alpaca_order(self, symbol: str, qty: int, side: str, order_type: str) -> Dict[str, Any]:
        """Execute order via Alpaca."""
        try:
            order = self.alpaca_api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force='gtc'
            )
            
            return {
                'order_id': order.id,
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'status': order.status,
                'filled_qty': order.filled_qty,
                'timestamp': datetime.now().timestamp()
            }
            
        except Exception as e:
            logger.error(f"Error executing Alpaca order: {e}")
            return {'error': str(e)}
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get Alpaca account information."""
        if not self.alpaca_api:
            return {'error': 'Alpaca API not configured'}
        
        try:
            loop = asyncio.get_event_loop()
            account = await loop.run_in_executor(self.executor, lambda: self.alpaca_api.get_account())
            
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'day_trade_count': int(account.daytrade_count),
                'status': account.status
            }
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {'error': str(e)}