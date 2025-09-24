"""
Real-time Market Data Integration System

Professional-grade market data feeds with multiple providers, data quality
validation, and high-frequency streaming capabilities for institutional trading.
"""

import asyncio
import websocket
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from queue import Queue, PriorityQueue
import time
import requests
from abc import ABC, abstractmethod
import redis
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Market data providers
import yfinance as yf
import alpha_vantage
from polygon import RESTClient as PolygonClient
import twelvedata
from iexfinance.stocks import Stock
import quandl

# Data processing
from scipy import stats
import talib

class DataProvider(Enum):
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    IEX = "iex"
    TWELVE_DATA = "twelve_data"
    YAHOO_FINANCE = "yahoo_finance"
    QUANDL = "quandl"
    BLOOMBERG = "bloomberg"
    REUTERS = "reuters"

class DataType(Enum):
    QUOTES = "quotes"
    TRADES = "trades"
    BARS = "bars"
    LEVEL2 = "level2"
    NEWS = "news"
    OPTIONS = "options"
    FUNDAMENTALS = "fundamentals"

@dataclass
class MarketDataConfig:
    """Configuration for market data providers"""
    providers: List[DataProvider] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    data_types: List[DataType] = field(default_factory=list)
    update_frequency: int = 1000  # milliseconds
    enable_level2: bool = False
    enable_options: bool = False
    enable_news: bool = True
    quality_checks: bool = True
    backup_providers: bool = True
    max_latency_ms: int = 100

@dataclass
class Quote:
    """Real-time quote data"""
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last_price: float
    timestamp: datetime
    exchange: str = ""
    volume: int = 0
    provider: str = ""

@dataclass
class Trade:
    """Real-time trade data"""
    symbol: str
    price: float
    size: int
    timestamp: datetime
    exchange: str = ""
    conditions: List[str] = field(default_factory=list)
    provider: str = ""

@dataclass
class Bar:
    """OHLCV bar data"""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime
    vwap: float = 0.0
    trades_count: int = 0
    provider: str = ""

@dataclass
class Level2Data:
    """Order book level 2 data"""
    symbol: str
    bids: List[Tuple[float, int]]  # price, size
    asks: List[Tuple[float, int]]  # price, size
    timestamp: datetime
    exchange: str = ""
    provider: str = ""

class BaseDataProvider(ABC):
    """Abstract base class for market data providers"""

    def __init__(self, api_key: str = "", config: Dict[str, Any] = None):
        self.api_key = api_key
        self.config = config or {}
        self.is_connected = False
        self.subscribers = {}
        self.data_queue = Queue()
        self.error_count = 0
        self.last_heartbeat = datetime.now()

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to data provider"""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from provider"""
        pass

    @abstractmethod
    async def subscribe(self, symbols: List[str], data_types: List[DataType]):
        """Subscribe to market data"""
        pass

    @abstractmethod
    async def unsubscribe(self, symbols: List[str], data_types: List[DataType]):
        """Unsubscribe from market data"""
        pass

    def add_subscriber(self, callback: Callable):
        """Add data subscriber"""
        subscriber_id = f"sub_{len(self.subscribers)}"
        self.subscribers[subscriber_id] = callback
        return subscriber_id

    def remove_subscriber(self, subscriber_id: str):
        """Remove data subscriber"""
        if subscriber_id in self.subscribers:
            del self.subscribers[subscriber_id]

    def notify_subscribers(self, data: Any):
        """Notify all subscribers of new data"""
        for callback in self.subscribers.values():
            try:
                callback(data)
            except Exception as e:
                logging.error(f"Error notifying subscriber: {e}")

    def is_healthy(self) -> bool:
        """Check provider health"""
        return (
            self.is_connected and
            self.error_count < 5 and
            (datetime.now() - self.last_heartbeat).seconds < 60
        )

class AlphaVantageProvider(BaseDataProvider):
    """Alpha Vantage market data provider"""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.av = alpha_vantage.AlphaVantage(key=api_key)

    async def connect(self) -> bool:
        """Connect to Alpha Vantage"""
        try:
            # Test connection with a simple quote request
            data, _ = self.av.get_intraday('AAPL', interval='1min', outputsize='compact')
            self.is_connected = True
            self.last_heartbeat = datetime.now()
            logging.info("Connected to Alpha Vantage")
            return True
        except Exception as e:
            logging.error(f"Alpha Vantage connection failed: {e}")
            self.error_count += 1
            return False

    async def disconnect(self):
        """Disconnect from Alpha Vantage"""
        self.is_connected = False

    async def subscribe(self, symbols: List[str], data_types: List[DataType]):
        """Subscribe to Alpha Vantage data"""
        # Alpha Vantage doesn't have real-time streaming, so we'll poll
        for symbol in symbols:
            asyncio.create_task(self._poll_data(symbol, data_types))

    async def unsubscribe(self, symbols: List[str], data_types: List[DataType]):
        """Unsubscribe from Alpha Vantage data"""
        # Implementation would track and stop polling tasks
        pass

    async def _poll_data(self, symbol: str, data_types: List[DataType]):
        """Poll data for symbol"""
        while self.is_connected:
            try:
                if DataType.QUOTES in data_types:
                    quote_data = await self._get_quote(symbol)
                    if quote_data:
                        self.notify_subscribers(quote_data)

                if DataType.BARS in data_types:
                    bar_data = await self._get_bars(symbol)
                    if bar_data:
                        self.notify_subscribers(bar_data)

                await asyncio.sleep(5)  # Poll every 5 seconds

            except Exception as e:
                logging.error(f"Error polling Alpha Vantage data for {symbol}: {e}")
                self.error_count += 1
                await asyncio.sleep(10)

    async def _get_quote(self, symbol: str) -> Optional[Quote]:
        """Get real-time quote"""
        try:
            data, _ = self.av.get_quote_endpoint(symbol)
            if data:
                return Quote(
                    symbol=symbol,
                    bid=float(data.get('02. open', 0)),
                    ask=float(data.get('05. price', 0)),
                    bid_size=0,
                    ask_size=0,
                    last_price=float(data.get('05. price', 0)),
                    timestamp=datetime.now(timezone.utc),
                    provider="alpha_vantage"
                )
        except Exception as e:
            logging.error(f"Error getting Alpha Vantage quote for {symbol}: {e}")
            return None

    async def _get_bars(self, symbol: str) -> Optional[Bar]:
        """Get latest bar data"""
        try:
            data, _ = self.av.get_intraday(symbol, interval='1min', outputsize='compact')
            if data:
                latest_time = max(data.keys())
                latest_data = data[latest_time]

                return Bar(
                    symbol=symbol,
                    open=float(latest_data['1. open']),
                    high=float(latest_data['2. high']),
                    low=float(latest_data['3. low']),
                    close=float(latest_data['4. close']),
                    volume=int(latest_data['5. volume']),
                    timestamp=pd.to_datetime(latest_time).tz_localize('UTC'),
                    provider="alpha_vantage"
                )
        except Exception as e:
            logging.error(f"Error getting Alpha Vantage bars for {symbol}: {e}")
            return None

class PolygonProvider(BaseDataProvider):
    """Polygon.io market data provider with WebSocket streaming"""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = PolygonClient(api_key)
        self.ws = None
        self.ws_thread = None

    async def connect(self) -> bool:
        """Connect to Polygon.io WebSocket"""
        try:
            # Test REST API connection
            self.client.get_aggs("AAPL", 1, "day", "2023-01-01", "2023-01-02")

            # Start WebSocket connection
            await self._start_websocket()

            self.is_connected = True
            self.last_heartbeat = datetime.now()
            logging.info("Connected to Polygon.io")
            return True

        except Exception as e:
            logging.error(f"Polygon connection failed: {e}")
            self.error_count += 1
            return False

    async def disconnect(self):
        """Disconnect from Polygon.io"""
        if self.ws:
            self.ws.close()
        self.is_connected = False

    async def subscribe(self, symbols: List[str], data_types: List[DataType]):
        """Subscribe to Polygon.io streams"""
        if not self.ws:
            return

        # Build subscription message
        subscription = {"action": "subscribe"}

        if DataType.QUOTES in data_types:
            subscription["quotes"] = [f"Q.{symbol}" for symbol in symbols]

        if DataType.TRADES in data_types:
            subscription["trades"] = [f"T.{symbol}" for symbol in symbols]

        if DataType.BARS in data_types:
            subscription["bars"] = [f"A.{symbol}" for symbol in symbols]

        # Send subscription
        self.ws.send(json.dumps(subscription))

    async def unsubscribe(self, symbols: List[str], data_types: List[DataType]):
        """Unsubscribe from Polygon.io streams"""
        if not self.ws:
            return

        subscription = {"action": "unsubscribe"}

        if DataType.QUOTES in data_types:
            subscription["quotes"] = [f"Q.{symbol}" for symbol in symbols]

        if DataType.TRADES in data_types:
            subscription["trades"] = [f"T.{symbol}" for symbol in symbols]

        self.ws.send(json.dumps(subscription))

    async def _start_websocket(self):
        """Start WebSocket connection"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                for item in data:
                    self._process_message(item)
            except Exception as e:
                logging.error(f"Error processing Polygon message: {e}")

        def on_error(ws, error):
            logging.error(f"Polygon WebSocket error: {error}")
            self.error_count += 1

        def on_close(ws, close_status_code, close_msg):
            logging.info("Polygon WebSocket closed")
            self.is_connected = False

        def on_open(ws):
            # Authenticate
            auth_msg = {"action": "auth", "params": self.api_key}
            ws.send(json.dumps(auth_msg))

        # Create WebSocket connection
        self.ws = websocket.WebSocketApp(
            "wss://socket.polygon.io/stocks",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )

        # Start in separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def _process_message(self, message: Dict[str, Any]):
        """Process incoming WebSocket message"""
        try:
            event_type = message.get('ev')

            if event_type == 'Q':  # Quote
                quote = Quote(
                    symbol=message.get('sym', ''),
                    bid=message.get('bp', 0),
                    ask=message.get('ap', 0),
                    bid_size=message.get('bs', 0),
                    ask_size=message.get('as', 0),
                    last_price=message.get('ap', 0),
                    timestamp=datetime.fromtimestamp(message.get('t', 0) / 1000, tz=timezone.utc),
                    exchange=message.get('x', ''),
                    provider="polygon"
                )
                self.notify_subscribers(quote)

            elif event_type == 'T':  # Trade
                trade = Trade(
                    symbol=message.get('sym', ''),
                    price=message.get('p', 0),
                    size=message.get('s', 0),
                    timestamp=datetime.fromtimestamp(message.get('t', 0) / 1000, tz=timezone.utc),
                    exchange=message.get('x', ''),
                    provider="polygon"
                )
                self.notify_subscribers(trade)

            elif event_type == 'A':  # Aggregate/Bar
                bar = Bar(
                    symbol=message.get('sym', ''),
                    open=message.get('o', 0),
                    high=message.get('h', 0),
                    low=message.get('l', 0),
                    close=message.get('c', 0),
                    volume=message.get('v', 0),
                    timestamp=datetime.fromtimestamp(message.get('s', 0) / 1000, tz=timezone.utc),
                    vwap=message.get('vw', 0),
                    trades_count=message.get('n', 0),
                    provider="polygon"
                )
                self.notify_subscribers(bar)

            self.last_heartbeat = datetime.now()

        except Exception as e:
            logging.error(f"Error processing Polygon message: {e}")

class DataQualityValidator:
    """Validates market data quality and detects anomalies"""

    def __init__(self):
        self.price_history = {}
        self.volume_history = {}
        self.anomaly_threshold = 3.0  # Standard deviations

    def validate_quote(self, quote: Quote) -> bool:
        """Validate quote data quality"""
        try:
            # Basic sanity checks
            if quote.bid <= 0 or quote.ask <= 0:
                return False

            if quote.bid >= quote.ask:
                return False

            # Check for extreme price movements
            if self._is_price_anomaly(quote.symbol, quote.last_price):
                logging.warning(f"Price anomaly detected for {quote.symbol}: {quote.last_price}")
                return False

            # Update price history
            self._update_price_history(quote.symbol, quote.last_price)

            return True

        except Exception as e:
            logging.error(f"Error validating quote: {e}")
            return False

    def validate_trade(self, trade: Trade) -> bool:
        """Validate trade data quality"""
        try:
            # Basic sanity checks
            if trade.price <= 0 or trade.size <= 0:
                return False

            # Check for extreme price movements
            if self._is_price_anomaly(trade.symbol, trade.price):
                logging.warning(f"Trade price anomaly detected for {trade.symbol}: {trade.price}")
                return False

            # Check for extreme volume
            if self._is_volume_anomaly(trade.symbol, trade.size):
                logging.warning(f"Trade volume anomaly detected for {trade.symbol}: {trade.size}")
                return False

            # Update histories
            self._update_price_history(trade.symbol, trade.price)
            self._update_volume_history(trade.symbol, trade.size)

            return True

        except Exception as e:
            logging.error(f"Error validating trade: {e}")
            return False

    def _is_price_anomaly(self, symbol: str, price: float) -> bool:
        """Check if price is an anomaly"""
        if symbol not in self.price_history:
            return False

        prices = self.price_history[symbol]
        if len(prices) < 10:
            return False

        mean_price = np.mean(prices)
        std_price = np.std(prices)

        if std_price == 0:
            return False

        z_score = abs(price - mean_price) / std_price
        return z_score > self.anomaly_threshold

    def _is_volume_anomaly(self, symbol: str, volume: int) -> bool:
        """Check if volume is an anomaly"""
        if symbol not in self.volume_history:
            return False

        volumes = self.volume_history[symbol]
        if len(volumes) < 10:
            return False

        mean_volume = np.mean(volumes)
        std_volume = np.std(volumes)

        if std_volume == 0:
            return False

        z_score = abs(volume - mean_volume) / std_volume
        return z_score > self.anomaly_threshold

    def _update_price_history(self, symbol: str, price: float):
        """Update price history for symbol"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append(price)

        # Keep only last 100 prices
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]

    def _update_volume_history(self, symbol: str, volume: int):
        """Update volume history for symbol"""
        if symbol not in self.volume_history:
            self.volume_history[symbol] = []

        self.volume_history[symbol].append(volume)

        # Keep only last 100 volumes
        if len(self.volume_history[symbol]) > 100:
            self.volume_history[symbol] = self.volume_history[symbol][-100:]

class MarketDataManager:
    """Central manager for all market data providers and distribution"""

    def __init__(self, config: MarketDataConfig, redis_client: redis.Redis = None):
        self.config = config
        self.providers: Dict[DataProvider, BaseDataProvider] = {}
        self.validator = DataQualityValidator()
        self.redis_client = redis_client
        self.subscribers = {}
        self.is_running = False
        self.data_stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'messages_rejected': 0,
            'latency_ms': []
        }

    def add_provider(self, provider_type: DataProvider, provider: BaseDataProvider):
        """Add data provider"""
        self.providers[provider_type] = provider
        provider.add_subscriber(self._handle_market_data)
        logging.info(f"Added market data provider: {provider_type.value}")

    async def connect_all_providers(self) -> Dict[DataProvider, bool]:
        """Connect to all configured providers"""
        results = {}

        for provider_type, provider in self.providers.items():
            try:
                result = await provider.connect()
                results[provider_type] = result
                if result:
                    logging.info(f"Connected to {provider_type.value}")
                else:
                    logging.error(f"Failed to connect to {provider_type.value}")
            except Exception as e:
                logging.error(f"Error connecting to {provider_type.value}: {e}")
                results[provider_type] = False

        return results

    async def disconnect_all_providers(self):
        """Disconnect from all providers"""
        for provider in self.providers.values():
            try:
                await provider.disconnect()
            except Exception as e:
                logging.error(f"Error disconnecting provider: {e}")

    async def subscribe_to_data(self):
        """Subscribe to configured market data"""
        for provider in self.providers.values():
            if provider.is_connected:
                try:
                    await provider.subscribe(self.config.symbols, self.config.data_types)
                except Exception as e:
                    logging.error(f"Error subscribing to data: {e}")

    def add_subscriber(self, callback: Callable, data_types: List[DataType] = None) -> str:
        """Add market data subscriber"""
        subscriber_id = f"sub_{len(self.subscribers)}"
        self.subscribers[subscriber_id] = {
            'callback': callback,
            'data_types': data_types or list(DataType)
        }
        return subscriber_id

    def remove_subscriber(self, subscriber_id: str):
        """Remove market data subscriber"""
        if subscriber_id in self.subscribers:
            del self.subscribers[subscriber_id]

    def _handle_market_data(self, data: Any):
        """Handle incoming market data"""
        try:
            receive_time = datetime.now(timezone.utc)
            self.data_stats['messages_received'] += 1

            # Validate data quality
            if not self._validate_data(data):
                self.data_stats['messages_rejected'] += 1
                return

            # Calculate latency
            if hasattr(data, 'timestamp'):
                latency = (receive_time - data.timestamp).total_seconds() * 1000
                self.data_stats['latency_ms'].append(latency)

                # Keep only last 1000 latency measurements
                if len(self.data_stats['latency_ms']) > 1000:
                    self.data_stats['latency_ms'] = self.data_stats['latency_ms'][-1000:]

            # Store in Redis for real-time access
            if self.redis_client:
                self._store_in_redis(data)

            # Notify subscribers
            self._notify_subscribers(data)

            self.data_stats['messages_processed'] += 1

        except Exception as e:
            logging.error(f"Error handling market data: {e}")

    def _validate_data(self, data: Any) -> bool:
        """Validate incoming data"""
        if not self.config.quality_checks:
            return True

        try:
            if isinstance(data, Quote):
                return self.validator.validate_quote(data)
            elif isinstance(data, Trade):
                return self.validator.validate_trade(data)
            else:
                return True  # Accept other data types by default

        except Exception as e:
            logging.error(f"Error validating data: {e}")
            return False

    def _store_in_redis(self, data: Any):
        """Store data in Redis for real-time access"""
        try:
            key_prefix = f"market_data:{data.symbol}"

            if isinstance(data, Quote):
                self.redis_client.set(
                    f"{key_prefix}:quote",
                    json.dumps({
                        'bid': data.bid,
                        'ask': data.ask,
                        'bid_size': data.bid_size,
                        'ask_size': data.ask_size,
                        'last_price': data.last_price,
                        'timestamp': data.timestamp.isoformat(),
                        'provider': data.provider
                    }),
                    ex=60  # Expire after 60 seconds
                )

            elif isinstance(data, Trade):
                # Store latest trade
                self.redis_client.set(
                    f"{key_prefix}:trade",
                    json.dumps({
                        'price': data.price,
                        'size': data.size,
                        'timestamp': data.timestamp.isoformat(),
                        'provider': data.provider
                    }),
                    ex=60
                )

                # Add to trade stream
                self.redis_client.lpush(
                    f"{key_prefix}:trades",
                    json.dumps({
                        'price': data.price,
                        'size': data.size,
                        'timestamp': data.timestamp.isoformat()
                    })
                )
                # Keep only last 1000 trades
                self.redis_client.ltrim(f"{key_prefix}:trades", 0, 999)

        except Exception as e:
            logging.error(f"Error storing data in Redis: {e}")

    def _notify_subscribers(self, data: Any):
        """Notify subscribers of new data"""
        data_type = None

        if isinstance(data, Quote):
            data_type = DataType.QUOTES
        elif isinstance(data, Trade):
            data_type = DataType.TRADES
        elif isinstance(data, Bar):
            data_type = DataType.BARS

        for subscriber_info in self.subscribers.values():
            try:
                # Check if subscriber wants this data type
                if data_type and data_type not in subscriber_info['data_types']:
                    continue

                # Notify subscriber
                subscriber_info['callback'](data)

            except Exception as e:
                logging.error(f"Error notifying subscriber: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get market data statistics"""
        avg_latency = np.mean(self.data_stats['latency_ms']) if self.data_stats['latency_ms'] else 0

        return {
            'messages_received': self.data_stats['messages_received'],
            'messages_processed': self.data_stats['messages_processed'],
            'messages_rejected': self.data_stats['messages_rejected'],
            'rejection_rate': self.data_stats['messages_rejected'] / max(self.data_stats['messages_received'], 1),
            'average_latency_ms': avg_latency,
            'max_latency_ms': max(self.data_stats['latency_ms']) if self.data_stats['latency_ms'] else 0,
            'provider_health': {
                provider_type.value: provider.is_healthy()
                for provider_type, provider in self.providers.items()
            }
        }

    async def get_latest_quote(self, symbol: str) -> Optional[Quote]:
        """Get latest quote from Redis"""
        if not self.redis_client:
            return None

        try:
            data = self.redis_client.get(f"market_data:{symbol}:quote")
            if data:
                quote_data = json.loads(data)
                return Quote(
                    symbol=symbol,
                    bid=quote_data['bid'],
                    ask=quote_data['ask'],
                    bid_size=quote_data['bid_size'],
                    ask_size=quote_data['ask_size'],
                    last_price=quote_data['last_price'],
                    timestamp=datetime.fromisoformat(quote_data['timestamp']),
                    provider=quote_data['provider']
                )
        except Exception as e:
            logging.error(f"Error getting latest quote for {symbol}: {e}")

        return None

# Example usage and testing
async def example_market_data_setup():
    """Example of setting up real-time market data"""

    # Configure market data
    config = MarketDataConfig(
        providers=[DataProvider.ALPHA_VANTAGE, DataProvider.POLYGON],
        symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
        data_types=[DataType.QUOTES, DataType.TRADES, DataType.BARS],
        update_frequency=1000,
        quality_checks=True,
        backup_providers=True
    )

    # Setup Redis connection
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

    # Create market data manager
    manager = MarketDataManager(config, redis_client)

    # Add providers (replace with actual API keys)
    alpha_vantage = AlphaVantageProvider("your_alpha_vantage_key")
    polygon = PolygonProvider("your_polygon_key")

    manager.add_provider(DataProvider.ALPHA_VANTAGE, alpha_vantage)
    manager.add_provider(DataProvider.POLYGON, polygon)

    # Connect to providers
    connection_results = await manager.connect_all_providers()
    print(f"Provider connections: {connection_results}")

    # Subscribe to data
    await manager.subscribe_to_data()

    # Add a sample subscriber
    def handle_quote(data):
        if isinstance(data, Quote):
            print(f"Quote: {data.symbol} - Bid: {data.bid}, Ask: {data.ask}")

    manager.add_subscriber(handle_quote, [DataType.QUOTES])

    # Let it run for a while
    await asyncio.sleep(30)

    # Print statistics
    stats = manager.get_statistics()
    print(f"Market data statistics: {stats}")

    # Cleanup
    await manager.disconnect_all_providers()

if __name__ == "__main__":
    # Run example
    asyncio.run(example_market_data_setup())