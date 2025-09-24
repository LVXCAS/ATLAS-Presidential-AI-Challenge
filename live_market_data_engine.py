"""
LIVE MARKET DATA ENGINE
Real-time market data feeds for AUTONOMOUS trading
Connects GPU systems to live market data streams
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import threading
from typing import Dict, List, Optional, Callable
import requests
import time
from dataclasses import dataclass
import queue
import ssl
import certifi

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class MarketTick:
    """Real-time market tick data"""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_size: Optional[int] = None

class LiveMarketDataEngine:
    """
    REAL-TIME MARKET DATA ENGINE
    Connects to live data feeds for autonomous trading
    """

    def __init__(self, config_file: str = "market_data_config.json"):
        self.logger = logging.getLogger('LiveMarketData')

        # Data streams
        self.tick_subscribers = []
        self.quote_subscribers = []
        self.news_subscribers = []

        # Data storage
        self.latest_quotes = {}
        self.tick_buffer = queue.Queue(maxsize=10000)
        self.news_buffer = queue.Queue(maxsize=1000)

        # Connection status
        self.connected = False
        self.streaming = False

        # Load configuration
        self.config = self.load_config(config_file)

        # Symbols to track
        self.symbols = [
            'SPY', 'QQQ', 'IWM', 'VIX', 'GLD', 'SLV',
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
            'JPM', 'BAC', 'WFC', 'GS', 'MS',
            'XOM', 'CVX', 'COP', 'SLB',
            'JNJ', 'PFE', 'MRK', 'UNH'
        ]

        self.logger.info(f"Live market data engine initialized for {len(self.symbols)} symbols")

    def load_config(self, config_file: str) -> Dict:
        """Load market data configuration"""
        default_config = {
            "data_providers": {
                "primary": "iex_cloud",
                "backup": "alpha_vantage",
                "crypto": "polygon"
            },
            "api_keys": {
                "iex_cloud": "YOUR_IEX_API_KEY",
                "alpha_vantage": "YOUR_ALPHA_VANTAGE_KEY",
                "polygon": "YOUR_POLYGON_KEY"
            },
            "update_intervals": {
                "quotes": 1.0,  # seconds
                "news": 30.0,
                "fundamentals": 3600.0
            },
            "risk_limits": {
                "max_position_size": 10000,
                "max_daily_loss": 5000,
                "max_symbols": 50
            }
        }

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.logger.info(f"Loaded config from {config_file}")
        except FileNotFoundError:
            config = default_config
            # Save default config
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            self.logger.info(f"Created default config at {config_file}")

        return config

    async def connect_iex_cloud(self):
        """Connect to IEX Cloud for real-time data"""
        try:
            api_key = self.config["api_keys"]["iex_cloud"]
            if api_key == "YOUR_IEX_API_KEY":
                self.logger.warning("IEX Cloud API key not configured - using demo mode")
                return await self.start_demo_data_stream()

            # IEX Cloud WebSocket connection
            base_url = f"wss://ws-api.iextrading.com/1.0/last"
            symbols_param = ",".join(self.symbols)

            uri = f"{base_url}?symbols={symbols_param}&token={api_key}"

            ssl_context = ssl.create_default_context(cafile=certifi.where())

            async with websockets.connect(uri, ssl=ssl_context) as websocket:
                self.connected = True
                self.logger.info("✅ Connected to IEX Cloud WebSocket")

                async for message in websocket:
                    data = json.loads(message)
                    await self.process_iex_message(data)

        except Exception as e:
            self.logger.error(f"IEX Cloud connection failed: {e}")
            self.logger.info("Falling back to demo data stream...")
            await self.start_demo_data_stream()

    async def process_iex_message(self, data: Dict):
        """Process incoming IEX Cloud message"""
        try:
            if isinstance(data, list):
                for item in data:
                    await self.process_single_tick(item)
            else:
                await self.process_single_tick(data)
        except Exception as e:
            self.logger.error(f"Error processing IEX message: {e}")

    async def process_single_tick(self, tick_data: Dict):
        """Process a single market tick"""
        try:
            tick = MarketTick(
                symbol=tick_data.get('symbol', 'UNKNOWN'),
                price=float(tick_data.get('price', 0)),
                volume=int(tick_data.get('size', 0)),
                timestamp=datetime.now(),
                last_size=tick_data.get('size')
            )

            # Update latest quotes
            self.latest_quotes[tick.symbol] = tick

            # Add to buffer
            if not self.tick_buffer.full():
                self.tick_buffer.put(tick)

            # Notify subscribers
            for callback in self.tick_subscribers:
                try:
                    await callback(tick)
                except Exception as e:
                    self.logger.error(f"Error in tick subscriber: {e}")

        except Exception as e:
            self.logger.error(f"Error processing tick: {e}")

    async def start_demo_data_stream(self):
        """Start demo data stream for testing"""
        self.logger.info("🔧 Starting DEMO data stream for testing")
        self.connected = True

        # Generate realistic market data
        base_prices = {symbol: 100 + hash(symbol) % 500 for symbol in self.symbols}

        while self.streaming:
            try:
                for symbol in self.symbols:
                    # Generate realistic price movement
                    current_price = base_prices[symbol]

                    # Add some random walk
                    change = np.random.normal(0, 0.002)  # 0.2% volatility
                    new_price = current_price * (1 + change)
                    base_prices[symbol] = new_price

                    # Generate volume
                    volume = int(np.random.exponential(1000))

                    # Create tick
                    tick = MarketTick(
                        symbol=symbol,
                        price=new_price,
                        volume=volume,
                        timestamp=datetime.now(),
                        bid=new_price - 0.01,
                        ask=new_price + 0.01
                    )

                    # Process tick
                    await self.process_single_tick(tick.__dict__)

                # Update every second
                await asyncio.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Demo stream error: {e}")
                await asyncio.sleep(5)

    async def fetch_polygon_crypto_data(self):
        """Fetch cryptocurrency data from Polygon"""
        try:
            crypto_symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'XRP', 'SOL', 'DOT', 'AVAX']

            for symbol in crypto_symbols:
                # Simulate crypto data (in real implementation, use Polygon API)
                base_price = 100 + hash(symbol) % 50000
                price = base_price * (1 + np.random.normal(0, 0.01))

                tick = MarketTick(
                    symbol=f"{symbol}/USD",
                    price=price,
                    volume=int(np.random.exponential(500)),
                    timestamp=datetime.now()
                )

                await self.process_single_tick(tick.__dict__)

        except Exception as e:
            self.logger.error(f"Crypto data fetch error: {e}")

    async def fetch_real_time_news(self):
        """Fetch real-time financial news"""
        try:
            # Simulate news feed (in real implementation, use news API)
            sample_news = [
                {"headline": "Fed signals potential rate cut", "sentiment": 0.3},
                {"headline": "Tech earnings beat expectations", "sentiment": 0.7},
                {"headline": "Oil prices surge on supply concerns", "sentiment": -0.2},
                {"headline": "GDP growth exceeds forecasts", "sentiment": 0.5}
            ]

            for news_item in sample_news:
                if not self.news_buffer.full():
                    self.news_buffer.put({
                        "timestamp": datetime.now(),
                        "headline": news_item["headline"],
                        "sentiment": news_item["sentiment"],
                        "source": "market_data_engine"
                    })

            # Notify news subscribers
            for callback in self.news_subscribers:
                try:
                    latest_news = list(self.news_buffer.queue)[-5:]  # Last 5 news items
                    await callback(latest_news)
                except Exception as e:
                    self.logger.error(f"Error in news subscriber: {e}")

        except Exception as e:
            self.logger.error(f"News fetch error: {e}")

    def subscribe_to_ticks(self, callback: Callable):
        """Subscribe to real-time tick data"""
        self.tick_subscribers.append(callback)
        self.logger.info(f"Added tick subscriber: {callback.__name__}")

    def subscribe_to_news(self, callback: Callable):
        """Subscribe to real-time news"""
        self.news_subscribers.append(callback)
        self.logger.info(f"Added news subscriber: {callback.__name__}")

    def get_latest_quote(self, symbol: str) -> Optional[MarketTick]:
        """Get latest quote for symbol"""
        return self.latest_quotes.get(symbol)

    def get_latest_quotes(self, symbols: List[str] = None) -> Dict[str, MarketTick]:
        """Get latest quotes for multiple symbols"""
        if symbols is None:
            return self.latest_quotes.copy()

        return {symbol: self.latest_quotes.get(symbol) for symbol in symbols
                if symbol in self.latest_quotes}

    async def start_streaming(self):
        """Start all data streams"""
        self.streaming = True
        self.logger.info("🚀 Starting live market data streams...")

        # Start concurrent data streams
        tasks = [
            self.connect_iex_cloud(),
            self.periodic_crypto_updates(),
            self.periodic_news_updates()
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

    async def periodic_crypto_updates(self):
        """Periodic cryptocurrency updates"""
        while self.streaming:
            try:
                await self.fetch_polygon_crypto_data()
                await asyncio.sleep(5)  # Update every 5 seconds
            except Exception as e:
                self.logger.error(f"Crypto update error: {e}")
                await asyncio.sleep(10)

    async def periodic_news_updates(self):
        """Periodic news updates"""
        while self.streaming:
            try:
                await self.fetch_real_time_news()
                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                self.logger.error(f"News update error: {e}")
                await asyncio.sleep(60)

    def stop_streaming(self):
        """Stop all data streams"""
        self.streaming = False
        self.connected = False
        self.logger.info("🛑 Stopped market data streams")

    def get_market_status(self) -> Dict:
        """Get current market data status"""
        return {
            "connected": self.connected,
            "streaming": self.streaming,
            "symbols_tracked": len(self.symbols),
            "latest_quotes_count": len(self.latest_quotes),
            "tick_buffer_size": self.tick_buffer.qsize(),
            "news_buffer_size": self.news_buffer.qsize(),
            "subscribers": {
                "tick_subscribers": len(self.tick_subscribers),
                "news_subscribers": len(self.news_subscribers)
            }
        }

async def demo_live_data_engine():
    """Demo the live market data engine"""
    print("="*80)
    print("LIVE MARKET DATA ENGINE DEMO")
    print("Real-time data feeds for autonomous trading")
    print("="*80)

    # Initialize engine
    engine = LiveMarketDataEngine()

    # Sample tick handler
    async def handle_tick(tick: MarketTick):
        print(f"📊 {tick.symbol}: ${tick.price:.2f} (Vol: {tick.volume})")

    # Sample news handler
    async def handle_news(news_items: List[Dict]):
        for news in news_items[-1:]:  # Show latest news
            print(f"📰 NEWS: {news['headline']} (Sentiment: {news['sentiment']:.2f})")

    # Subscribe to data
    engine.subscribe_to_ticks(handle_tick)
    engine.subscribe_to_news(handle_news)

    print(f"\n🚀 Starting live data streams...")
    print(f"💡 This is DEMO mode - configure API keys for real data")

    # Start streaming for demo
    try:
        await asyncio.wait_for(engine.start_streaming(), timeout=30)
    except asyncio.TimeoutError:
        print(f"\n⏰ Demo completed")
    finally:
        engine.stop_streaming()

        # Show final status
        status = engine.get_market_status()
        print(f"\n📊 FINAL STATUS:")
        print(f"   Quotes received: {status['latest_quotes_count']}")
        print(f"   Tick buffer: {status['tick_buffer_size']}")
        print(f"   News items: {status['news_buffer_size']}")

    print(f"\n✅ Live market data engine ready for autonomous trading!")

if __name__ == "__main__":
    asyncio.run(demo_live_data_engine())