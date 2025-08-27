"""
Stock Data API Routes
Endpoints for fetching stock charts and quotes from Polygon.io
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import logging

from services.polygon_service import polygon_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/stocks", tags=["stocks"])


@router.get("/bars/{symbol}")
async def get_stock_bars(
    symbol: str,
    timespan: str = Query(default="minute", description="Time interval"),
    multiplier: int = Query(default=5, description="Multiplier for timespan"),
    from_date: Optional[str] = Query(default=None, description="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(default=None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(default=50, description="Number of bars")
):
    """Get historical stock price bars."""
    try:
        async with polygon_service as service:
            bars = await service.get_stock_bars(
                symbol=symbol.upper(),
                timespan=timespan,
                multiplier=multiplier,
                from_date=from_date,
                to_date=to_date,
                limit=limit
            )
            
            return {
                "symbol": symbol.upper(),
                "timespan": timespan,
                "multiplier": multiplier,
                "bars": [
                    {
                        "timestamp": bar.timestamp,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume
                    }
                    for bar in bars
                ]
            }
            
    except Exception as e:
        logger.error(f"Error fetching stock bars: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch stock data")


@router.get("/quote/{symbol}")
async def get_stock_quote(symbol: str):
    """Get real-time stock quote."""
    try:
        async with polygon_service as service:
            quote = await service.get_real_time_quote(symbol.upper())
            
            if quote:
                return {
                    "symbol": quote.symbol,
                    "price": quote.price,
                    "change": quote.change,
                    "change_percent": quote.change_percent,
                    "volume": quote.volume,
                    "timestamp": quote.timestamp
                }
            else:
                raise HTTPException(status_code=404, detail="Quote not found")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching stock quote: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch quote")


@router.get("/search")
async def search_stocks(
    query: str = Query(description="Search query (symbol or name)"),
    limit: int = Query(default=10, description="Max results")
):
    """Search for stocks by symbol or name."""
    try:
        async with polygon_service as service:
            results = await service.search_stocks(query, limit)
            
            return {
                "query": query,
                "results": results
            }
            
    except Exception as e:
        logger.error(f"Error searching stocks: {e}")
        raise HTTPException(status_code=500, detail="Failed to search stocks")