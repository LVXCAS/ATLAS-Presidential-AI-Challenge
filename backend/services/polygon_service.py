"""
Polygon.io Market Data Service
Real-time and historical stock data integration.
"""

import asyncio
import logging
import aiohttp
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class StockBar:
    """Stock price bar data."""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class StockQuote:
    """Real-time stock quote."""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: str


class PolygonService:
    """Polygon.io market data service."""
    
    def __init__(self):
        self.api_key = settings.polygon.api_key
        self.base_url = settings.polygon.base_url
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def get_stock_bars(self, symbol: str, timespan: str = "minute", 
                           multiplier: int = 5, from_date: str = None, 
                           to_date: str = None, limit: int = 50) -> List[StockBar]:
        """
        Get historical stock bars from Polygon.io
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            timespan: minute, hour, day, week, month, quarter, year
            multiplier: Size of the time window
            from_date: Start date (YYYY-MM-DD format)
            to_date: End date (YYYY-MM-DD format)  
            limit: Number of bars to return
        """
        session = None
        try:
            if not from_date:
                from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            if not to_date:
                to_date = datetime.now().strftime('%Y-%m-%d')
                
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
            params = {
                "adjusted": "true",
                "sort": "asc",
                "limit": limit,
                "apikey": self.api_key
            }
            
            session = aiohttp.ClientSession()
                
            async with session.get(url, params=params) as response:
                logger.info(f"Polygon API request: {url} - Status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Polygon response data: {data}")
                    
                    if data.get("status") in ["OK", "DELAYED"] and "results" in data:
                        bars = []
                        for bar in data["results"]:
                            bars.append(StockBar(
                                timestamp=datetime.fromtimestamp(bar["t"] / 1000).isoformat(),
                                open=bar["o"],
                                high=bar["h"],
                                low=bar["l"], 
                                close=bar["c"],
                                volume=bar["v"]
                            ))
                        logger.info(f"Found {len(bars)} bars for {symbol}")
                        return bars
                    else:
                        logger.warning(f"No data found for symbol {symbol}: {data}")
                        return []
                else:
                    text = await response.text()
                    logger.error(f"Polygon API error: {response.status} - {text}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching stock bars from Polygon: {e}")
            return []
        finally:
            if session:
                await session.close()
    
    async def get_real_time_quote(self, symbol: str) -> Optional[StockQuote]:
        """Get real-time quote for a symbol."""
        try:
            url = f"{self.base_url}/v2/last/trade/{symbol}"
            params = {"apikey": self.api_key}
            
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") in ["OK", "DELAYED"] and "results" in data:
                        result = data["results"]
                        
                        # Get previous close for change calculation
                        prev_close = await self._get_previous_close(symbol)
                        current_price = result["p"]
                        change = current_price - prev_close if prev_close else 0
                        change_percent = (change / prev_close * 100) if prev_close else 0
                        
                        return StockQuote(
                            symbol=symbol,
                            price=current_price,
                            change=change,
                            change_percent=change_percent,
                            volume=result.get("s", 0),
                            timestamp=datetime.fromtimestamp(result["t"] / 1000).isoformat()
                        )
                        
        except Exception as e:
            logger.error(f"Error fetching real-time quote: {e}")
            return None
    
    async def _get_previous_close(self, symbol: str) -> Optional[float]:
        """Get previous trading day close price."""
        try:
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/prev"
            params = {"apikey": self.api_key}
            
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") in ["OK", "DELAYED"] and "results" in data:
                        return data["results"][0]["c"]
                        
        except Exception as e:
            logger.error(f"Error fetching previous close: {e}")
            return None
    
    async def search_stocks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for stocks by symbol or name."""
        session = None
        try:
            url = f"{self.base_url}/v3/reference/tickers"
            params = {
                "search": query,
                "market": "stocks",
                "active": "true",
                "limit": limit,
                "apikey": self.api_key
            }
            
            session = aiohttp.ClientSession()
                
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") in ["OK", "DELAYED"] and "results" in data:
                        return [
                            {
                                "symbol": ticker["ticker"],
                                "name": ticker.get("name", ""),
                                "market": ticker.get("market", ""),
                                "type": ticker.get("type", "")
                            }
                            for ticker in data["results"]
                        ]
                else:
                    text = await response.text()
                    logger.error(f"Search API error: {response.status} - {text}")
                    return []
        except Exception as e:
            logger.error(f"Error searching stocks: {e}")
            return []
        finally:
            if session:
                await session.close()


# Global polygon service instance
polygon_service = PolygonService()