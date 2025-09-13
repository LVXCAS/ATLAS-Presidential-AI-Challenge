#!/usr/bin/env python3
"""
Live Data Manager - Real-time market data integration for live trading
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import threading
import time
import queue
import warnings
warnings.filterwarnings('ignore')

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
    import polygon
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class LiveDataManager:
    """Manage real-time market data from multiple sources"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.data_feeds = {}
        self.callbacks = {}
        self.is_running = False
        self.data_queue = queue.Queue()
        self.price_cache = {}
        self.last_update = {}
        
        # Data sources configuration
        self.sources = {}
        
        # Initialize available sources
        self._initialize_sources()
    
    def _initialize_sources(self):
        """Initialize available data sources"""
        print("LIVE DATA MANAGER INITIALIZATION")
        print("=" * 45)
        
        # Check Alpaca
        if ALPACA_AVAILABLE and self.config.get('alpaca_key'):
            print("+ Alpaca API available")
            if 'alpaca' in self.sources:
                self.sources['alpaca']['available'] = True
        else:
            print("- Alpaca API: Not configured")
            
        # Check Polygon
        if POLYGON_AVAILABLE and self.config.get('polygon_key'):
            print("+ Polygon API available")
            if 'polygon' in self.sources:
                self.sources['polygon']['available'] = True
        else:
            print("- Polygon API: Not configured")
            
        # Yahoo Finance (free)
        if YFINANCE_AVAILABLE:
            print("+ Yahoo Finance available (free)")
            self.sources['yahoo'] = {'available': True}
        else:
            print("- Yahoo Finance: Not available")
            
        # Finnhub (free tier)
        if REQUESTS_AVAILABLE and self.config.get('finnhub_key'):
            print("+ Finnhub API available")
            self.sources['finnhub'] = {'available': True}
        else:
            print("- Finnhub API: Not configured")
            
        # TwelveData (free tier)
        if REQUESTS_AVAILABLE and self.config.get('twelvedata_key'):
            print("+ TwelveData API available")
            self.sources['twelvedata'] = {'available': True}
        else:
            print("- TwelveData API: Not configured")
        
        print(f"\nActive data sources: {len([s for s in self.sources.values() if s.get('available', False)])}")
    
    def _setup_alpaca(self):
        """Setup Alpaca data feed"""
        try:
            api_key = self.config.get('alpaca_key')
            secret_key = self.config.get('alpaca_secret')
            base_url = self.config.get('alpaca_base_url', 'https://paper-api.alpaca.markets')
            
            if not api_key or not secret_key:
                return None
                
            api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
            return {
                'api': api,
                'type': 'rest',
                'real_time': True,
                'cost': 'free',
                'rate_limit': 200  # requests per minute
            }
        except Exception as e:
            print(f"Alpaca setup error: {e}")
            return None
    
    def _setup_polygon(self):
        """Setup Polygon data feed"""
        try:
            api_key = self.config.get('polygon_key')
            if not api_key:
                return None
                
            client = polygon.RESTClient(api_key)
            return {
                'client': client,
                'type': 'rest',
                'real_time': True,
                'cost': 'paid',
                'rate_limit': 5  # requests per minute for free tier
            }
        except Exception as e:
            print(f"Polygon setup error: {e}")
            return None
    
    def _setup_yahoo(self):
        """Setup Yahoo Finance data feed"""
        return {
            'type': 'polling',
            'real_time': False,
            'cost': 'free',
            'rate_limit': 60,  # Conservative limit
            'delay': 15  # 15-minute delay
        }
    
    def _setup_finnhub(self):
        """Setup Finnhub data feed"""
        api_key = self.config.get('finnhub_key')
        if not api_key:
            return None
            
        return {
            'api_key': api_key,
            'base_url': 'https://finnhub.io/api/v1',
            'type': 'rest',
            'real_time': True,
            'cost': 'freemium',
            'rate_limit': 60  # 60 calls per minute for free
        }
    
    def _setup_twelvedata(self):
        """Setup TwelveData feed"""
        api_key = self.config.get('twelvedata_key')
        if not api_key:
            return None
            
        return {
            'api_key': api_key,
            'base_url': 'https://api.twelvedata.com',
            'type': 'rest',
            'real_time': True,
            'cost': 'freemium',
            'rate_limit': 800  # 800 calls per day for free
        }
    
    async def start_live_feed(self, symbols: List[str], callback: Callable = None):
        """Start live data feed for specified symbols"""
        print(f"\nSTARTING LIVE DATA FEED")
        print(f"Symbols: {', '.join(symbols)}")
        print("=" * 40)
        
        self.symbols = symbols
        self.is_running = True
        
        if callback:
            self.callbacks['default'] = callback
        
        # Start data feeds based on available sources
        tasks = []
        
        # Yahoo Finance as fallback (always available)
        if self.sources.get('yahoo', {}).get('available', False):
            tasks.append(self._start_yahoo_feed(symbols))
        
        # Finnhub real-time feed
        if self.sources.get('finnhub', {}).get('available', False):
            tasks.append(self._start_finnhub_feed(symbols))
        
        # TwelveData feed
        if self.sources.get('twelvedata', {}).get('available', False):
            tasks.append(self._start_twelvedata_feed(symbols))
        
        # Alpaca feed
        if self.sources.get('alpaca', {}).get('available', False):
            tasks.append(self._start_alpaca_feed(symbols))
        
        # Start data processing
        tasks.append(self._process_data_queue())
        
        if not tasks:
            print("No data sources available! Please configure API keys.")
            return False
        
        print(f"Starting {len(tasks)} data feed tasks...")
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("Stopping live data feed...")
            self.is_running = False
        
        return True
    
    async def _start_yahoo_feed(self, symbols: List[str]):
        """Start Yahoo Finance data feed (delayed but free)"""
        print("Starting Yahoo Finance feed (15-min delay)...")
        
        while self.is_running:
            try:
                for symbol in symbols:
                    # Get latest data
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="1d", interval="1m")
                    
                    if not hist.empty:
                        latest = hist.iloc[-1]
                        
                        # Fix MultiIndex if needed
                        if isinstance(hist.columns, pd.MultiIndex):
                            hist.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in hist.columns]
                            latest = hist.iloc[-1]
                        
                        market_data = {
                            'symbol': symbol,
                            'price': float(latest['Close']),
                            'open': float(latest['Open']),
                            'high': float(latest['High']),
                            'low': float(latest['Low']),
                            'volume': int(latest['Volume']),
                            'timestamp': datetime.now().isoformat(),
                            'source': 'yahoo',
                            'delay': 15  # minutes
                        }
                        
                        self.data_queue.put(market_data)
                        print(f"Yahoo: {symbol} @ ${market_data['price']:.2f}")
                    
                    await asyncio.sleep(2)  # Rate limiting
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                print(f"Yahoo feed error: {e}")
                await asyncio.sleep(30)
    
    async def _start_finnhub_feed(self, symbols: List[str]):
        """Start Finnhub real-time feed"""
        if not REQUESTS_AVAILABLE:
            return
            
        print("Starting Finnhub real-time feed...")
        api_key = self.config.get('finnhub_key')
        
        while self.is_running:
            try:
                for symbol in symbols:
                    # Get real-time quote
                    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}"
                    
                    import requests
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        market_data = {
                            'symbol': symbol,
                            'price': float(data.get('c', 0)),  # Current price
                            'open': float(data.get('o', 0)),   # Open
                            'high': float(data.get('h', 0)),   # High
                            'low': float(data.get('l', 0)),    # Low
                            'change': float(data.get('d', 0)), # Change
                            'change_percent': float(data.get('dp', 0)), # Change percent
                            'timestamp': datetime.now().isoformat(),
                            'source': 'finnhub',
                            'delay': 0  # Real-time
                        }
                        
                        self.data_queue.put(market_data)
                        print(f"Finnhub: {symbol} @ ${market_data['price']:.2f} ({market_data['change_percent']:+.1f}%)")
                    
                    await asyncio.sleep(1)  # Rate limiting (60 calls/minute)
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                print(f"Finnhub feed error: {e}")
                await asyncio.sleep(30)
    
    async def _start_twelvedata_feed(self, symbols: List[str]):
        """Start TwelveData real-time feed"""
        if not REQUESTS_AVAILABLE:
            return
            
        print("Starting TwelveData real-time feed...")
        api_key = self.config.get('twelvedata_key')
        
        while self.is_running:
            try:
                # Batch request for efficiency
                symbol_str = ','.join(symbols)
                url = f"https://api.twelvedata.com/price?symbol={symbol_str}&apikey={api_key}"
                
                import requests
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Handle single vs multiple symbols
                    if isinstance(data, dict) and 'price' in data:
                        # Single symbol
                        market_data = {
                            'symbol': symbols[0],
                            'price': float(data['price']),
                            'timestamp': datetime.now().isoformat(),
                            'source': 'twelvedata',
                            'delay': 0
                        }
                        self.data_queue.put(market_data)
                        print(f"TwelveData: {symbols[0]} @ ${market_data['price']:.2f}")
                    
                    elif isinstance(data, dict):
                        # Multiple symbols
                        for symbol, price_data in data.items():
                            if isinstance(price_data, dict) and 'price' in price_data:
                                market_data = {
                                    'symbol': symbol,
                                    'price': float(price_data['price']),
                                    'timestamp': datetime.now().isoformat(),
                                    'source': 'twelvedata',
                                    'delay': 0
                                }
                                self.data_queue.put(market_data)
                                print(f"TwelveData: {symbol} @ ${market_data['price']:.2f}")
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                print(f"TwelveData feed error: {e}")
                await asyncio.sleep(30)
    
    async def _start_alpaca_feed(self, symbols: List[str]):
        """Start Alpaca real-time feed"""
        if not ALPACA_AVAILABLE:
            return
            
        print("Starting Alpaca real-time feed...")
        
        try:
            # Setup Alpaca API
            alpaca_config = self._setup_alpaca()
            if not alpaca_config:
                return
            
            api = alpaca_config['api']
            
            while self.is_running:
                try:
                    for symbol in symbols:
                        # Get latest trade
                        latest_trade = api.get_latest_trade(symbol)
                        
                        if latest_trade:
                            market_data = {
                                'symbol': symbol,
                                'price': float(latest_trade.price),
                                'size': int(latest_trade.size),
                                'timestamp': latest_trade.timestamp.isoformat(),
                                'source': 'alpaca',
                                'delay': 0
                            }
                            
                            self.data_queue.put(market_data)
                            print(f"Alpaca: {symbol} @ ${market_data['price']:.2f}")
                        
                        await asyncio.sleep(0.5)  # Rate limiting
                    
                    await asyncio.sleep(5)  # Update every 5 seconds
                    
                except Exception as e:
                    print(f"Alpaca feed error: {e}")
                    await asyncio.sleep(30)
                    
        except Exception as e:
            print(f"Alpaca setup error: {e}")
    
    async def _process_data_queue(self):
        """Process incoming market data"""
        print("Starting data processing queue...")
        
        while self.is_running:
            try:
                # Process all queued data
                while not self.data_queue.empty():
                    market_data = self.data_queue.get_nowait()
                    
                    # Update price cache
                    symbol = market_data['symbol']
                    self.price_cache[symbol] = market_data
                    self.last_update[symbol] = datetime.now()
                    
                    # Call registered callbacks
                    for callback_name, callback in self.callbacks.items():
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(market_data)
                            else:
                                callback(market_data)
                        except Exception as e:
                            print(f"Callback error ({callback_name}): {e}")
                
                await asyncio.sleep(0.1)  # Process queue frequently
                
            except Exception as e:
                print(f"Data processing error: {e}")
                await asyncio.sleep(1)
    
    def register_callback(self, name: str, callback: Callable):
        """Register a callback for market data updates"""
        self.callbacks[name] = callback
        print(f"Registered callback: {name}")
    
    def get_latest_price(self, symbol: str) -> Optional[Dict]:
        """Get latest price for a symbol"""
        return self.price_cache.get(symbol)
    
    def get_all_prices(self) -> Dict:
        """Get all cached prices"""
        return self.price_cache.copy()
    
    async def get_historical_data(self, symbol: str, period: str = "1d") -> Optional[pd.DataFrame]:
        """Get historical data for backtesting"""
        try:
            if YFINANCE_AVAILABLE:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval="1m")
                
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in data.columns]
                
                return data
            
            return None
            
        except Exception as e:
            print(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def stop_feed(self):
        """Stop the live data feed"""
        print("Stopping live data feed...")
        self.is_running = False
    
    def get_feed_status(self) -> Dict:
        """Get status of data feeds"""
        status = {
            'is_running': self.is_running,
            'symbols': getattr(self, 'symbols', []),
            'active_sources': len([s for s in self.sources.values() if s.get('available', False)]),
            'cached_symbols': list(self.price_cache.keys()),
            'last_updates': {symbol: update.isoformat() for symbol, update in self.last_update.items()},
            'queue_size': self.data_queue.qsize()
        }
        
        return status

# Global instance
live_data_manager = LiveDataManager()

def setup_live_data(config: Dict = None):
    """Setup live data manager with configuration"""
    global live_data_manager
    live_data_manager = LiveDataManager(config)
    return live_data_manager

async def start_live_feed(symbols: List[str], callback: Callable = None):
    """Start live data feed"""
    return await live_data_manager.start_live_feed(symbols, callback)

def get_live_price(symbol: str) -> Optional[Dict]:
    """Get latest live price"""
    return live_data_manager.get_latest_price(symbol)

if __name__ == "__main__":
    async def test_live_data():
        # Test configuration
        config = {
            # Add your API keys here for real trading
            # 'finnhub_key': 'your_finnhub_key',
            # 'twelvedata_key': 'your_twelvedata_key',
            # 'alpaca_key': 'your_alpaca_key',
            # 'alpaca_secret': 'your_alpaca_secret',
        }
        
        # Setup data manager
        data_manager = setup_live_data(config)
        
        # Test callback
        async def price_callback(data):
            print(f"LIVE: {data['symbol']} @ ${data['price']:.2f} from {data['source']}")
        
        # Register callback
        data_manager.register_callback('test', price_callback)
        
        # Start live feed
        symbols = ['AAPL', 'SPY', 'QQQ']
        print(f"Starting live feed for: {symbols}")
        print("Press Ctrl+C to stop...")
        
        try:
            await start_live_feed(symbols, price_callback)
        except KeyboardInterrupt:
            print("\nStopping live feed...")
            data_manager.stop_feed()
    
    # Run test
    import asyncio
    asyncio.run(test_live_data())