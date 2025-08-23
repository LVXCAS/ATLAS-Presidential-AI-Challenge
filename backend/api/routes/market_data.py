"""
Market Data API Routes for Bloomberg Terminal
Real-time and historical market data endpoints.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from core.redis_manager import get_redis_manager
from core.database import DatabaseService

router = APIRouter()


class SymbolRequest(BaseModel):
    symbols: List[str]


class MarketDataResponse(BaseModel):
    symbol: str
    price: float
    timestamp: float
    volume: int
    change: Optional[float] = None
    change_percent: Optional[float] = None


class QuoteResponse(BaseModel):
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    spread: float
    timestamp: float


class PriceHistoryResponse(BaseModel):
    symbol: str
    data: List[Dict[str, Any]]
    interval: str


@router.get("/prices/{symbol}", response_model=MarketDataResponse)
async def get_latest_price(symbol: str):
    """Get latest price for a symbol."""
    try:
        symbol = symbol.upper()
        redis_manager = get_redis_manager()
        redis_client = await redis_manager.get_client()
        
        # Get price data from Redis
        price_data = await redis_client.hgetall(f"price:{symbol}")
        
        if not price_data:
            raise HTTPException(status_code=404, detail=f"No price data found for {symbol}")
        
        # Calculate change if possible
        change = None
        change_percent = None
        
        # Get historical data for change calculation
        history = await redis_manager.get_price_history(symbol, limit=2)
        if len(history) >= 2:
            current_price = float(price_data["price"])
            previous_price = history[-2]["price"]
            change = current_price - previous_price
            change_percent = (change / previous_price) * 100 if previous_price != 0 else 0
        
        return MarketDataResponse(
            symbol=symbol,
            price=float(price_data["price"]),
            timestamp=float(price_data["timestamp"]),
            volume=int(price_data.get("volume", 0)),
            change=change,
            change_percent=change_percent
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching price data: {str(e)}")


@router.post("/prices/batch", response_model=List[MarketDataResponse])
async def get_batch_prices(request: SymbolRequest):
    """Get latest prices for multiple symbols."""
    try:
        redis_manager = get_redis_manager()
        redis_client = await redis_manager.get_client()
        
        results = []
        
        for symbol in request.symbols:
            symbol = symbol.upper()
            price_data = await redis_client.hgetall(f"price:{symbol}")
            
            if price_data:
                # Calculate change if possible
                change = None
                change_percent = None
                
                history = await redis_manager.get_price_history(symbol, limit=2)
                if len(history) >= 2:
                    current_price = float(price_data["price"])
                    previous_price = history[-2]["price"]
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100 if previous_price != 0 else 0
                
                results.append(MarketDataResponse(
                    symbol=symbol,
                    price=float(price_data["price"]),
                    timestamp=float(price_data["timestamp"]),
                    volume=int(price_data.get("volume", 0)),
                    change=change,
                    change_percent=change_percent
                ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching batch prices: {str(e)}")


@router.get("/quotes/{symbol}", response_model=QuoteResponse)
async def get_latest_quote(symbol: str):
    """Get latest quote (bid/ask) for a symbol."""
    try:
        symbol = symbol.upper()
        redis_manager = get_redis_manager()
        redis_client = await redis_manager.get_client()
        
        quote_data = await redis_client.hgetall(f"quote:{symbol}")
        
        if not quote_data:
            raise HTTPException(status_code=404, detail=f"No quote data found for {symbol}")
        
        return QuoteResponse(
            symbol=symbol,
            bid=float(quote_data["bid"]),
            ask=float(quote_data["ask"]),
            bid_size=int(quote_data["bid_size"]),
            ask_size=int(quote_data["ask_size"]),
            spread=float(quote_data["spread"]),
            timestamp=float(quote_data["timestamp"])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching quote data: {str(e)}")


@router.get("/history/{symbol}", response_model=PriceHistoryResponse)
async def get_price_history(
    symbol: str,
    interval: str = Query("1m", regex="^(1m|5m|15m|30m|1h|4h|1d)$"),
    limit: int = Query(100, ge=1, le=1000),
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
):
    """Get historical price data for a symbol."""
    try:
        symbol = symbol.upper()
        
        # First try Redis cache for recent data
        redis_manager = get_redis_manager()
        
        if not start_time and not end_time and limit <= 1000:
            # Use Redis for recent data
            history = await redis_manager.get_price_history(symbol, limit=limit)
            if history:
                return PriceHistoryResponse(
                    symbol=symbol,
                    data=history,
                    interval="tick"
                )
        
        # Fall back to database for historical data
        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM market_data
        WHERE symbol = :symbol
        """
        
        params = {"symbol": symbol}
        
        if start_time:
            query += " AND timestamp >= :start_time"
            params["start_time"] = start_time
        
        if end_time:
            query += " AND timestamp <= :end_time"
            params["end_time"] = end_time
        
        query += " ORDER BY timestamp DESC LIMIT :limit"
        params["limit"] = limit
        
        rows = await DatabaseService.execute_query(query, params)
        
        data = []
        for row in rows:
            data.append({
                "timestamp": int(row[0].timestamp() * 1000),  # Convert to milliseconds
                "open": float(row[1]) if row[1] else None,
                "high": float(row[2]) if row[2] else None,
                "low": float(row[3]) if row[3] else None,
                "close": float(row[4]) if row[4] else None,
                "volume": int(row[5]) if row[5] else 0
            })
        
        # Reverse to get chronological order
        data.reverse()
        
        return PriceHistoryResponse(
            symbol=symbol,
            data=data,
            interval=interval
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching historical data: {str(e)}")


@router.get("/indicators/{symbol}")
async def get_technical_indicators(symbol: str):
    """Get technical indicators for a symbol."""
    try:
        symbol = symbol.upper()
        redis_manager = get_redis_manager()
        redis_client = await redis_manager.get_client()
        
        indicators_data = await redis_client.hgetall(f"indicators:{symbol}")
        
        if not indicators_data:
            raise HTTPException(status_code=404, detail=f"No indicator data found for {symbol}")
        
        # Convert string values back to float
        indicators = {}
        for k, v in indicators_data.items():
            try:
                indicators[k] = float(v)
            except ValueError:
                indicators[k] = v
        
        return {
            "symbol": symbol,
            "indicators": indicators,
            "timestamp": indicators.get("timestamp")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching indicators: {str(e)}")


@router.get("/watchlist")
async def get_watchlist():
    """Get current market data watchlist."""
    try:
        symbols = await DatabaseService.get_symbol_watchlist()
        
        # Get current prices for all symbols
        redis_manager = get_redis_manager()
        redis_client = await redis_manager.get_client()
        
        watchlist_data = []
        
        for symbol in symbols:
            price_data = await redis_client.hgetall(f"price:{symbol}")
            if price_data:
                # Get quote data
                quote_data = await redis_client.hgetall(f"quote:{symbol}")
                
                # Calculate change
                change = None
                change_percent = None
                history = await redis_manager.get_price_history(symbol, limit=2)
                if len(history) >= 2:
                    current_price = float(price_data["price"])
                    previous_price = history[-2]["price"]
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100 if previous_price != 0 else 0
                
                watchlist_data.append({
                    "symbol": symbol,
                    "price": float(price_data["price"]),
                    "change": change,
                    "change_percent": change_percent,
                    "volume": int(price_data.get("volume", 0)),
                    "bid": float(quote_data.get("bid", 0)) if quote_data else None,
                    "ask": float(quote_data.get("ask", 0)) if quote_data else None,
                    "timestamp": float(price_data["timestamp"])
                })
        
        return {
            "watchlist": watchlist_data,
            "total_symbols": len(watchlist_data),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching watchlist: {str(e)}")


@router.post("/watchlist/add")
async def add_to_watchlist(request: SymbolRequest):
    """Add symbols to watchlist."""
    try:
        # This would typically update a user-specific watchlist
        # For now, we'll just confirm the symbols exist
        
        redis_manager = get_redis_manager()
        redis_client = await redis_manager.get_client()
        
        added_symbols = []
        
        for symbol in request.symbols:
            symbol = symbol.upper()
            # Check if we have data for this symbol
            price_data = await redis_client.hgetall(f"price:{symbol}")
            if price_data:
                added_symbols.append(symbol)
        
        return {
            "message": f"Added {len(added_symbols)} symbols to watchlist",
            "symbols": added_symbols
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding to watchlist: {str(e)}")


@router.get("/market-status")
async def get_market_status():
    """Get current market status."""
    try:
        now = datetime.now()
        
        # Simple market hours check (US Eastern Time)
        # This is simplified - in production, you'd use proper timezone handling
        hour = now.hour
        weekday = now.weekday()
        
        # US market hours: 9:30 AM - 4:00 PM ET (assuming server time)
        is_market_open = weekday < 5 and 9 <= hour < 16
        
        # Pre-market: 4:00 AM - 9:30 AM
        is_pre_market = weekday < 5 and 4 <= hour < 9
        
        # After-hours: 4:00 PM - 8:00 PM
        is_after_hours = weekday < 5 and 16 <= hour < 20
        
        status = "CLOSED"
        if is_market_open:
            status = "OPEN"
        elif is_pre_market:
            status = "PRE_MARKET"
        elif is_after_hours:
            status = "AFTER_HOURS"
        
        return {
            "status": status,
            "is_open": is_market_open,
            "timestamp": now.isoformat(),
            "next_open": None,  # Would calculate next market open
            "next_close": None  # Would calculate next market close
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting market status: {str(e)}")


@router.get("/stats")
async def get_market_stats():
    """Get market data statistics."""
    try:
        redis_manager = get_redis_manager()
        redis_client = await redis_manager.get_client()
        
        # Count active symbols
        price_keys = await redis_client.keys("price:*")
        active_symbols = len(price_keys)
        
        # Get some basic stats
        quote_keys = await redis_client.keys("quote:*")
        indicator_keys = await redis_client.keys("indicators:*")
        
        return {
            "active_symbols": active_symbols,
            "symbols_with_quotes": len(quote_keys),
            "symbols_with_indicators": len(indicator_keys),
            "data_sources": ["alpaca", "polygon"],
            "update_frequency": "real-time",
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting market stats: {str(e)}")