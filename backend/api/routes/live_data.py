"""
Live Market Data API Routes
Real-time data from yfinance and Alpaca integration.
"""

import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from services.market_data_service_live import LiveMarketDataService
from core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

# Initialize live market data service
live_service = LiveMarketDataService(
    alpaca_key=settings.alpaca.api_key,
    alpaca_secret=settings.alpaca.secret_key
)

class OrderRequest(BaseModel):
    symbol: str
    quantity: int
    side: str  # 'buy' or 'sell'
    order_type: str = 'market'

@router.get("/market-data")
async def get_live_market_data(symbols: str = Query(..., description="Comma-separated symbols")):
    """Get real-time market data for symbols."""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        data = await live_service.get_market_data(symbol_list)
        return {"status": "success", "data": data}
    
    except Exception as e:
        logger.error(f"Error getting live market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/news")
async def get_live_news(symbols: str = Query("SPY,QQQ,AAPL,MSFT,NVDA", description="Comma-separated symbols")):
    """Get real news for symbols."""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        news = await live_service.get_real_news(symbol_list)
        return {"status": "success", "data": news}
    
    except Exception as e:
        logger.error(f"Error getting live news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chart/{symbol}")
async def get_chart_data(
    symbol: str,
    period: str = Query("1d", description="Period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max"),
    interval: str = Query("1m", description="Interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo")
):
    """Get real chart data for a symbol."""
    try:
        data = await live_service.get_chart_data(symbol.upper(), period, interval)
        return {"status": "success", "data": data}
    
    except Exception as e:
        logger.error(f"Error getting chart data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/orders")
async def place_order(order: OrderRequest):
    """Place order through Alpaca."""
    try:
        result = await live_service.place_alpaca_order(
            symbol=order.symbol.upper(),
            qty=order.quantity,
            side=order.side.lower(),
            order_type=order.order_type.lower()
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return {"status": "success", "data": result}
    
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/account")
async def get_account():
    """Get Alpaca account information."""
    try:
        account_info = await live_service.get_account_info()
        
        if 'error' in account_info:
            raise HTTPException(status_code=400, detail=account_info['error'])
        
        return {"status": "success", "data": account_info}
    
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_live_service_status():
    """Get live service status."""
    return {
        "status": "operational",
        "services": {
            "yfinance": "active",
            "alpaca": "configured" if live_service.alpaca_api else "not_configured"
        },
        "message": "Live market data service operational"
    }