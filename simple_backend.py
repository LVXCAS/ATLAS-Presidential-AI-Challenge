"""
HIVE TRADE Live Data Backend
Real market data with Alpaca API + yfinance + All US Market Tickers
"""

import yfinance as yf
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import time
import json
from typing import Optional
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="HIVE TRADE LIVE DATA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Comprehensive US Market Tickers
ALL_US_TICKERS = {
    # Major Indices
    'SPY': 'SPDR S&P 500 ETF',
    'QQQ': 'Invesco QQQ ETF',
    'IWM': 'iShares Russell 2000 ETF',
    'DIA': 'SPDR Dow Jones ETF',
    'VTI': 'Vanguard Total Stock ETF',
    
    # FAANG + Tech Giants
    'AAPL': 'Apple Inc',
    'MSFT': 'Microsoft Corp',
    'GOOGL': 'Alphabet Inc Class A',
    'GOOG': 'Alphabet Inc Class C',
    'AMZN': 'Amazon.com Inc',
    'META': 'Meta Platforms Inc',
    'TSLA': 'Tesla Inc',
    'NVDA': 'NVIDIA Corp',
    'NFLX': 'Netflix Inc',
    
    # Major Banks
    'JPM': 'JPMorgan Chase & Co',
    'BAC': 'Bank of America Corp',
    'WFC': 'Wells Fargo & Co',
    'GS': 'Goldman Sachs Group',
    'MS': 'Morgan Stanley',
    'C': 'Citigroup Inc',
    
    # Healthcare
    'JNJ': 'Johnson & Johnson',
    'PFE': 'Pfizer Inc',
    'UNH': 'UnitedHealth Group',
    'MRNA': 'Moderna Inc',
    'ABBV': 'AbbVie Inc',
    
    # Energy
    'XOM': 'Exxon Mobil Corp',
    'CVX': 'Chevron Corp',
    'COP': 'ConocoPhillips',
    'SLB': 'Schlumberger NV',
    
    # Consumer
    'KO': 'Coca-Cola Co',
    'PEP': 'PepsiCo Inc',
    'WMT': 'Walmart Inc',
    'HD': 'Home Depot Inc',
    'MCD': 'McDonalds Corp',
    'NKE': 'Nike Inc',
    'DIS': 'Walt Disney Co',
    
    # Industrials
    'BA': 'Boeing Co',
    'CAT': 'Caterpillar Inc',
    'GE': 'General Electric',
    'MMM': '3M Co',
    
    # Semiconductors
    'INTC': 'Intel Corp',
    'AMD': 'Advanced Micro Devices',
    'QCOM': 'Qualcomm Inc',
    'AVGO': 'Broadcom Inc',
    'MU': 'Micron Technology',
    
    # Electric Vehicles
    'F': 'Ford Motor Co',
    'GM': 'General Motors Co',
    'RIVN': 'Rivian Automotive',
    'LCID': 'Lucid Group Inc',
    
    # Crypto Related
    'COIN': 'Coinbase Global Inc',
    'MSTR': 'MicroStrategy Inc',
    'SQ': 'Block Inc',
    
    # Meme Stocks
    'GME': 'GameStop Corp',
    'AMC': 'AMC Entertainment',
    'BB': 'BlackBerry Ltd',
    'NOK': 'Nokia Corp',
    
    # REITs
    'O': 'Realty Income Corp',
    'SPG': 'Simon Property Group',
    
    # Utilities
    'NEE': 'NextEra Energy Inc',
    'DUK': 'Duke Energy Corp',
    
    # Communication
    'T': 'AT&T Inc',
    'VZ': 'Verizon Communications',
    'CMCSA': 'Comcast Corp'
}

executor = ThreadPoolExecutor(max_workers=8)

# Load comprehensive ticker database
def load_comprehensive_tickers():
    """Load comprehensive NYSE/NASDAQ ticker database"""
    try:
        with open('comprehensive_ticker_database.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Comprehensive ticker database not found, using default tickers")
        return ALL_US_TICKERS
    except Exception as e:
        print(f"Error loading comprehensive ticker database: {e}")
        return ALL_US_TICKERS

# Load comprehensive tickers
COMPREHENSIVE_TICKERS = load_comprehensive_tickers()
print(f"Loaded {len(COMPREHENSIVE_TICKERS)} comprehensive tickers")

# Alpaca API setup (Paper trading by default)
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET = os.getenv('ALPACA_SECRET_KEY')

alpaca_trading_client = None
alpaca_data_client = None

try:
    if ALPACA_API_KEY and ALPACA_SECRET:
        alpaca_trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET, paper=True)
        alpaca_data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET)
        print(f"Alpaca API: CONNECTED (Paper Trading) - Key: {ALPACA_API_KEY[:8]}...")
    else:
        pass  # Status now shown in startup
except Exception as e:
    print(f"Alpaca API: CONNECTION FAILED - {e}")

def fetch_market_data(symbols_str):
    """Fetch real market data using yfinance"""
    symbols = [s.strip().upper() for s in symbols_str.split(',')]
    market_data = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d", interval="1m")
            
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                prev_close = info.get('previousClose', current_price)
                change = current_price - prev_close
                change_percent = (change / prev_close) * 100 if prev_close != 0 else 0
                
                # Calculate VWAP approximation 
                high_today = float(hist['High'].iloc[-1])
                low_today = float(hist['Low'].iloc[-1])
                open_today = float(hist['Open'].iloc[-1])
                vwap = (high_today + low_today + current_price) / 3
                
                market_data[symbol] = {
                    'symbol': symbol,
                    'price': round(current_price, 2),
                    'change': round(change, 2),
                    'change_percent': round(change_percent, 2),
                    'volume': int(hist['Volume'].iloc[-1]),
                    'high': round(high_today, 2),
                    'low': round(low_today, 2),
                    'open': round(open_today, 2),
                    'vwap': round(vwap, 2),
                    'timestamp': str(int(time.time()))
                }
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            market_data[symbol] = {
                'symbol': symbol, 'price': 100.0, 'change': 0.0,
                'change_percent': 0.0, 'volume': 1000000, 'high': 102.0,
                'low': 98.0, 'open': 99.0, 'vwap': 100.0, 'timestamp': str(int(time.time()))
            }
    
    return market_data

def fetch_news(symbols_str):
    """Fetch real news using yfinance"""
    symbols = [s.strip().upper() for s in symbols_str.split(',')]
    all_news = []
    
    for symbol in symbols[:3]:  # Limit to avoid rate limits
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            for item in news[:3]:  # Top 3 per symbol
                all_news.append({
                    'title': item.get('title', 'No title'),
                    'summary': item.get('summary', 'No summary'),
                    'source': item.get('publisher', 'Unknown'),
                    'symbol': symbol,
                    'timestamp': item.get('providerPublishTime', 0),
                    'sentiment_score': 0.1,
                    'impact': 'MEDIUM'
                })
        except Exception as e:
            print(f"News error for {symbol}: {e}")
    
    return all_news

def fetch_chart_data(symbol, period, interval):
    """Fetch chart data using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        
        if hist.empty:
            return []
        
        chart_data = []
        for timestamp, row in hist.iterrows():
            chart_data.append({
                'time': timestamp.timestamp(),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': int(row['Volume'])
            })
        
        return chart_data
    except Exception as e:
        print(f"Chart error for {symbol}: {e}")
        return []

@app.get("/")
async def root():
    return {"status": "HIVE TRADE LIVE DATA API OPERATIONAL", "version": "1.0.0"}

@app.get("/api/v1/live/status")
async def get_status():
    return {
        "status": "operational",
        "services": {"yfinance": "active"},
        "message": "Live data service ready"
    }


@app.get("/api/v1/live/news")
async def get_live_news(symbols: str = Query("SPY,QQQ,AAPL")):
    loop = asyncio.get_event_loop()
    news = await loop.run_in_executor(executor, fetch_news, symbols)
    return {"status": "success", "data": news}

@app.get("/api/v1/live/chart/{symbol}")
async def get_chart_data(
    symbol: str,
    period: str = Query("1d"),
    interval: str = Query("1m")
):
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(executor, fetch_chart_data, symbol.upper(), period, interval)
    return {"status": "success", "data": {"symbol": symbol, "data": data}}

@app.get("/api/v1/live/tickers")
async def get_all_tickers():
    """Get all available US market tickers"""
    return {
        "status": "success",
        "data": ALL_US_TICKERS,
        "count": len(ALL_US_TICKERS)
    }

@app.get("/api/v1/live/market-data")
async def get_live_market_data(
    symbols: Optional[str] = None,
    sector: Optional[str] = None,
    category: Optional[str] = None, 
    limit: int = 20
):
    """Get live market data for multiple symbols with smart filtering"""
    try:
        # Determine which symbols to fetch
        if sector or category:
            # Filter symbols by sector/category from comprehensive database
            filtered_symbols = []
            print(f"Filtering by sector: {sector}, category: {category}")
            for symbol, ticker_info in COMPREHENSIVE_TICKERS.items():
                if sector and ticker_info.get('sector', '') != sector:
                    continue
                if category and ticker_info.get('category', '') != category:
                    continue
                filtered_symbols.append(symbol)
            
            print(f"Found {len(filtered_symbols)} symbols matching criteria: {filtered_symbols[:5]}")
            # Take first 'limit' symbols
            symbol_list = filtered_symbols[:limit]
        elif symbols and symbols.strip():
            # Use provided symbols
            symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
            symbol_list = symbol_list[:limit]  # Respect limit
        else:
            # Use default symbols
            default_symbols = "SPY,QQQ,AAPL,MSFT,NVDA,TSLA,META,GOOGL,AMZN,NFLX,JPM,BAC,AMD,INTC,COIN,GME,XLK,XLF,TLT,GLD"
            symbol_list = [s.strip().upper() for s in default_symbols.split(',')]
            symbol_list = symbol_list[:limit]  # Respect limit
        
        market_data = {}
        
        # Process symbols sequentially for now (to avoid asyncio issues)
        for symbol in symbol_list:
            try:
                result = fetch_single_quote(symbol)
                if isinstance(result, dict) and result:
                    market_data[symbol] = result
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
        
        return {
            "status": "success",
            "data": market_data,
            "timestamp": asyncio.get_event_loop().time(),
            "count": len(market_data)
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

def fetch_single_quote(symbol):
    """Fetch quote data for a single symbol (for parallel processing)"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Try to get real-time quote data first
        try:
            info = ticker.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            prev_close = info.get('regularMarketPreviousClose')
            
            if current_price and prev_close:
                change = current_price - prev_close
                change_percent = (change / prev_close * 100) if prev_close != 0 else 0
                
                return {
                    "symbol": symbol,
                    "price": float(current_price),
                    "change": float(change),
                    "change_percent": float(change_percent),
                    "volume": int(info.get('regularMarketVolume', 0)),
                    "high": float(info.get('regularMarketDayHigh', current_price)),
                    "low": float(info.get('regularMarketDayLow', current_price)),
                    "open": float(info.get('regularMarketOpen', current_price)),
                    "vwap": float((info.get('regularMarketDayHigh', current_price) + info.get('regularMarketDayLow', current_price) + current_price) / 3),
                    "timestamp": str(int(time.time()))
                }
        except:
            pass
        
        # Fallback to historical data if real-time fails
        history = ticker.history(period="2d", interval="1d")
        
        if not history.empty:
            current_price = float(history['Close'].iloc[-1])
            prev_close = float(history['Close'].iloc[-2]) if len(history) > 1 else current_price
            
            high = float(history['High'].iloc[-1])
            low = float(history['Low'].iloc[-1])
            open_price = float(history['Open'].iloc[-1])
            volume = int(history['Volume'].iloc[-1])
            
            change = current_price - prev_close
            change_percent = (change / prev_close * 100) if prev_close != 0 else 0
            
            # Calculate VWAP (approximation using current data)
            vwap = (high + low + current_price) / 3
            
            return {
                "symbol": symbol,
                "price": current_price,
                "change": change,
                "change_percent": change_percent,
                "volume": volume,
                "high": high,
                "low": low,
                "open": open_price,
                "vwap": vwap,
                "timestamp": str(int(time.time()))
            }
        else:
            return None
            
    except Exception as e:
        print(f"Error fetching quote for {symbol}: {e}")
        return None

@app.post("/api/v1/alpaca/order")
async def place_alpaca_order(
    symbol: str,
    qty: int,
    side: str,  # 'buy' or 'sell'
    order_type: str = 'market'
):
    """Place order through Alpaca API"""
    if not alpaca_trading_client:
        raise HTTPException(status_code=400, detail="Alpaca API not configured")
    
    try:
        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
        
        market_order_data = MarketOrderRequest(
            symbol=symbol.upper(),
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.GTC
        )
        
        order = alpaca_trading_client.submit_order(order_data=market_order_data)
        
        return {
            "status": "success",
            "data": {
                "order_id": str(order.id),
                "symbol": order.symbol,
                "qty": str(order.qty),
                "side": order.side.value,
                "status": order.status.value,
                "submitted_at": str(order.submitted_at)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Order failed: {str(e)}")

@app.get("/api/v1/alpaca/account")
async def get_alpaca_account():
    """Get Alpaca account info"""
    if not alpaca_trading_client:
        raise HTTPException(status_code=400, detail="Alpaca API not configured")
    
    try:
        account = alpaca_trading_client.get_account()
        return {
            "status": "success",
            "data": {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "day_trade_count": account.daytrade_count,
                "status": account.status.value,
                "trading_blocked": account.trading_blocked,
                "account_blocked": account.account_blocked
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Account info failed: {str(e)}")

@app.get("/api/v1/alpaca/positions")
async def get_alpaca_positions():
    """Get current positions"""
    if not alpaca_trading_client:
        raise HTTPException(status_code=400, detail="Alpaca API not configured")
    
    try:
        positions = alpaca_trading_client.get_all_positions()
        return {
            "status": "success",
            "data": [{
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "side": pos.side.value,
                "market_value": float(pos.market_value),
                "cost_basis": float(pos.cost_basis),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
                "current_price": float(pos.current_price)
            } for pos in positions]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Positions failed: {str(e)}")

@app.get("/api/stocks/search")
async def search_stocks(
    query: str = Query(..., description="Search query for stocks"),
    sector: str = Query(None, description="Filter by sector"),
    category: str = Query(None, description="Filter by category"),
    market: str = Query(None, description="Filter by market (stocks/ETF)"),
    limit: int = Query(20, description="Number of results to return")
):
    """Search for stocks by symbol or company name with filtering"""
    try:
        query = query.upper().strip()
        results = []
        
        # Search through comprehensive ticker database
        for symbol, ticker_info in COMPREHENSIVE_TICKERS.items():
            # Text matching
            name = ticker_info.get('name', '')
            if query in symbol or query.lower() in name.lower():
                # Apply filters
                if sector and ticker_info.get('sector', '') != sector:
                    continue
                if category and ticker_info.get('category', '') != category:
                    continue
                if market and ticker_info.get('market', '') != market:
                    continue
                    
                results.append({
                    "symbol": symbol,
                    "name": name,
                    "market": ticker_info.get('market', 'stocks'),
                    "sector": ticker_info.get('sector', ''),
                    "category": ticker_info.get('category', ''),
                    "exchange": ticker_info.get('exchange', 'NYSE/NASDAQ'),
                    "type": "ETF" if ticker_info.get('market') == 'ETF' else "CS"
                })
                
        # Sort by relevance (exact symbol matches first, then name matches)
        results.sort(key=lambda x: (
            0 if x['symbol'] == query else 1,  # Exact symbol match first
            len(x['name'])  # Shorter names first for similar relevance
        ))
        
        # Limit results
        return {"results": results[:limit]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stocks/quote/{symbol}")
async def get_stock_quote(symbol: str):
    """Get real-time quote for a stock using yfinance"""
    try:
        ticker = yf.Ticker(symbol.upper())
        history = ticker.history(period="2d")
        
        if not history.empty:
            current_price = float(history['Close'].iloc[-1])
            prev_close = float(history['Close'].iloc[-2]) if len(history) > 1 else current_price
            
            change = current_price - prev_close
            change_percent = (change / prev_close * 100) if prev_close != 0 else 0
            
            quote = {
                "symbol": symbol.upper(),
                "price": current_price,
                "change": change,
                "change_percent": change_percent,
                "volume": int(history['Volume'].iloc[-1]) if len(history) > 0 else 0,
                "timestamp": str(int(time.time())),
            }
            
            return quote
        else:
            raise HTTPException(status_code=404, detail=f"No quote data available for {symbol}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stocks/bars/{symbol}")
async def get_stock_bars(symbol: str, timespan: str = "day", limit: int = 30):
    """Get historical stock bars using yfinance"""
    try:
        ticker = yf.Ticker(symbol.upper())
        
        # Map timespan to yfinance period
        period_map = {"day": "30d", "week": "3mo", "month": "1y"}
        period = period_map.get(timespan, "30d")
        
        history = ticker.history(period=period)
        
        if not history.empty:
            bars = []
            for timestamp, row in history.iterrows():
                bars.append({
                    "timestamp": timestamp.isoformat(),
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": float(row['Close']),
                    "volume": int(row['Volume'])
                })
            
            # Limit results
            bars = bars[-limit:] if limit else bars
            
            return {"symbol": symbol.upper(), "bars": bars}
        else:
            raise HTTPException(status_code=404, detail=f"No bar data available for {symbol}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stocks/sectors")
async def get_available_sectors():
    """Get list of available sectors"""
    sectors = set()
    for ticker_info in COMPREHENSIVE_TICKERS.values():
        sector = ticker_info.get('sector', '')
        if sector:
            sectors.add(sector)
    
    return {"sectors": sorted(list(sectors))}

@app.get("/api/stocks/categories")
async def get_available_categories():
    """Get list of available categories"""
    categories = set()
    for ticker_info in COMPREHENSIVE_TICKERS.values():
        category = ticker_info.get('category', '')
        if category:
            categories.add(category)
    
    return {"categories": sorted(list(categories))}

@app.get("/api/stocks/browse")
async def browse_stocks(
    sector: str = Query(None, description="Filter by sector"),
    category: str = Query(None, description="Filter by category"), 
    market: str = Query(None, description="Filter by market (stocks/ETF)"),
    limit: int = Query(50, description="Number of results to return"),
    offset: int = Query(0, description="Pagination offset")
):
    """Browse stocks with filtering and pagination"""
    try:
        results = []
        
        for symbol, ticker_info in COMPREHENSIVE_TICKERS.items():
            # Apply filters
            if sector and ticker_info.get('sector', '') != sector:
                continue
            if category and ticker_info.get('category', '') != category:
                continue  
            if market and ticker_info.get('market', '') != market:
                continue
                
            results.append({
                "symbol": symbol,
                "name": ticker_info.get('name', ''),
                "market": ticker_info.get('market', 'stocks'),
                "sector": ticker_info.get('sector', ''),
                "category": ticker_info.get('category', ''),
                "exchange": ticker_info.get('exchange', 'NYSE/NASDAQ'),
                "type": "ETF" if ticker_info.get('market') == 'ETF' else "CS"
            })
        
        # Sort alphabetically by symbol
        results.sort(key=lambda x: x['symbol'])
        
        # Apply pagination
        total_count = len(results)
        paginated_results = results[offset:offset + limit]
        
        return {
            "results": paginated_results,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Global variables for Alpaca connection
alpaca_trading_client = None
alpaca_data_client = None
alpaca_config = {
    "api_key": None,
    "api_secret": None,
    "base_url": "paper",
    "status": "disconnected"
}

@app.post("/api/alpaca/configure")
async def configure_alpaca_api(request: dict):
    """Configure Alpaca API credentials"""
    global alpaca_trading_client, alpaca_data_client, alpaca_config
    
    try:
        api_key = request.get("api_key")
        api_secret = request.get("api_secret")
        base_url = request.get("base_url", "paper")
        
        if not api_key or not api_secret:
            return {"status": "error", "message": "API key and secret are required"}
        
        # Update configuration
        alpaca_config.update({
            "api_key": api_key,
            "api_secret": api_secret,
            "base_url": base_url
        })
        
        # Set environment variables for Alpaca clients
        import os
        os.environ["APCA_API_KEY_ID"] = api_key
        os.environ["APCA_API_SECRET_KEY"] = api_secret
        
        if base_url == "live":
            os.environ["APCA_API_BASE_URL"] = "https://api.alpaca.markets"
        else:
            os.environ["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"
        
        try:
            # Create new Alpaca clients
            alpaca_trading_client = TradingClient(api_key, api_secret, paper=(base_url == "paper"))
            alpaca_data_client = StockHistoricalDataClient(api_key, api_secret)
            
            # Test the connection by getting account info
            account = alpaca_trading_client.get_account()
            
            alpaca_config["status"] = "connected"
            print(f"Alpaca API Connected - Account: {account.account_number} ({base_url.upper()})")
            
            return {
                "status": "connected",
                "message": f"Successfully connected to Alpaca {base_url.upper()} environment",
                "account_number": account.account_number,
                "buying_power": str(account.buying_power),
                "cash": str(account.cash),
                "portfolio_value": str(account.portfolio_value)
            }
            
        except Exception as e:
            alpaca_config["status"] = "error"
            error_msg = str(e)
            print(f"Alpaca connection failed: {error_msg}")
            
            # Clear environment variables on failure
            if "APCA_API_KEY_ID" in os.environ:
                del os.environ["APCA_API_KEY_ID"]
            if "APCA_API_SECRET_KEY" in os.environ:
                del os.environ["APCA_API_SECRET_KEY"]
            if "APCA_API_BASE_URL" in os.environ:
                del os.environ["APCA_API_BASE_URL"]
            
            return {
                "status": "error",
                "message": f"Failed to connect to Alpaca API: {error_msg}"
            }
            
    except Exception as e:
        alpaca_config["status"] = "error"
        return {"status": "error", "message": f"Configuration error: {str(e)}"}

@app.get("/api/alpaca/status")
async def get_alpaca_status():
    """Get current Alpaca API connection status"""
    global alpaca_config, alpaca_trading_client
    
    status_info = {
        "status": alpaca_config["status"],
        "environment": alpaca_config["base_url"],
        "has_credentials": bool(alpaca_config["api_key"] and alpaca_config["api_secret"])
    }
    
    # If connected, get additional account info
    if alpaca_config["status"] == "connected" and alpaca_trading_client:
        try:
            account = alpaca_trading_client.get_account()
            status_info.update({
                "account_number": account.account_number,
                "buying_power": str(account.buying_power),
                "cash": str(account.cash),
                "portfolio_value": str(account.portfolio_value),
                "day_trade_count": getattr(account, 'day_trade_count', 0),
                "pattern_day_trader": getattr(account, 'pattern_day_trader', False)
            })
        except Exception as e:
            print(f"Failed to get account info: {e}")
            # Don't change connection status for account info errors
            status_info["account_info_error"] = str(e)
    
    return status_info

@app.post("/api/alpaca/disconnect")
async def disconnect_alpaca():
    """Disconnect from Alpaca API"""
    global alpaca_trading_client, alpaca_data_client, alpaca_config
    
    # Clear clients and config
    alpaca_trading_client = None
    alpaca_data_client = None
    alpaca_config = {
        "api_key": None,
        "api_secret": None,
        "base_url": "paper",
        "status": "disconnected"
    }
    
    # Clear environment variables
    import os
    for env_var in ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "APCA_API_BASE_URL"]:
        if env_var in os.environ:
            del os.environ[env_var]
    
    print("Alpaca API disconnected")
    return {"status": "disconnected", "message": "Disconnected from Alpaca API"}

if __name__ == "__main__":
    print("HIVE TRADE LIVE DATA SERVER STARTING...")
    print(f"Loaded {len(COMPREHENSIVE_TICKERS)} comprehensive tickers")
    print("yfinance integration: ACTIVE")
    
    # Check for Alpaca API configuration
    api_key = os.getenv("APCA_API_KEY_ID")
    if api_key:
        print(f"Alpaca API: CONFIGURED")
    else:
        print("Alpaca API: NOT CONFIGURED - Use ALPACA API tab to configure")
    
    print("CORS: Enabled for all origins")
    print("Server: http://localhost:8001")
    print("Status: http://localhost:8001/api/v1/live/status")
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")