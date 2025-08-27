"""
Simple FastAPI server for stock data endpoints
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import asyncio
import logging

# Import our polygon service
from services.polygon_service import polygon_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Stock Data API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5175", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/stocks/bars/{symbol}")
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
        bars = await polygon_service.get_stock_bars(
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


@app.get("/api/stocks/quote/{symbol}")
async def get_stock_quote(symbol: str):
    """Get real-time stock quote."""
    try:
        quote = await polygon_service.get_real_time_quote(symbol.upper())
        
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


@app.get("/api/stocks/search")
async def search_stocks(
    query: str = Query(description="Search query (symbol or name)"),
    limit: int = Query(default=10, description="Max results")
):
    """Search for stocks by symbol or name."""
    try:
        results = await polygon_service.search_stocks(query, limit)
        
        return {
            "query": query,
            "results": results
        }
            
    except Exception as e:
        logger.error(f"Error searching stocks: {e}")
        raise HTTPException(status_code=500, detail="Failed to search stocks")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Stock Data API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002, reload=True)