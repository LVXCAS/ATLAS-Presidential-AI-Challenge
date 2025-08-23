"""
Market Data Service for Bloomberg Terminal
High-performance real-time market data streaming with Alpaca and Polygon integration.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import websockets
from websockets.exceptions import ConnectionClosed

import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream
from alpaca_trade_api.common import URL
import redis.asyncio as redis

from core.config import get_settings
from core.database import DatabaseService
from core.redis_manager import get_redis_manager

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class MarketDataPoint:
    """Structured market data point."""
    symbol: str
    timestamp: datetime
    price: float
    size: int = 0
    volume: int = 0
    side: Optional[str] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    close: Optional[float] = None
    vwap: Optional[float] = None
    source: str = "alpaca"


@dataclass
class QuoteData:
    """Level 1 quote data."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    spread: float
    source: str = "alpaca"


class MarketDataService:
    """High-performance market data streaming service."""
    
    def __init__(self):
        self.settings = settings
        self.redis_manager = get_redis_manager()
        self.subscribers: Dict[str, Set[Callable]] = {}
        self.active_symbols: Set[str] = set()
        self.stream: Optional[Stream] = None
        self.is_streaming = False
        self._tasks: List[asyncio.Task] = []
        
        # Performance metrics
        self.last_update_time = {}
        self.update_counts = {}
        self.latency_samples = {}
        
        # Initialize Alpaca API
        if settings.alpaca.api_key and settings.alpaca.secret_key:
            self.alpaca_api = tradeapi.REST(
                settings.alpaca.api_key,
                settings.alpaca.secret_key,
                settings.alpaca.base_url
            )
        else:
            self.alpaca_api = None
            logger.warning("Alpaca API credentials not configured")
    
    async def start_streaming(self) -> None:
        """Start real-time market data streaming."""
        if self.is_streaming:
            logger.warning("Market data streaming already active")
            return
        
        logger.info("Starting market data streaming...")
        
        try:
            # Initialize Redis connection
            await self.redis_manager.initialize()
            
            # Start Alpaca data stream
            if self.alpaca_api:
                await self._start_alpaca_stream()
            
            # Start background tasks
            self._tasks.extend([
                asyncio.create_task(self._market_hours_monitor()),
                asyncio.create_task(self._performance_monitor()),
                asyncio.create_task(self._data_quality_monitor())
            ])
            
            self.is_streaming = True
            logger.info("Market data streaming started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start market data streaming: {e}")
            await self.stop_streaming()
            raise
    
    async def stop_streaming(self) -> None:
        """Stop market data streaming."""
        logger.info("Stopping market data streaming...")
        
        self.is_streaming = False
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Close Alpaca stream
        if self.stream:
            await self.stream.stop_ws()
            self.stream = None
        
        logger.info("Market data streaming stopped")
    
    async def _start_alpaca_stream(self) -> None:
        """Initialize Alpaca WebSocket stream."""
        try:
            self.stream = Stream(
                settings.alpaca.api_key,
                settings.alpaca.secret_key,
                base_url=URL(settings.alpaca.base_url),
                data_feed='iex'  # Use IEX for real-time data
            )
            
            # Subscribe to trade updates
            self.stream.subscribe_trades(self._handle_trade_update, "*")
            
            # Subscribe to quote updates
            self.stream.subscribe_quotes(self._handle_quote_update, "*")
            
            # Subscribe to bars (1-minute aggregates)
            self.stream.subscribe_bars(self._handle_bar_update, "*")
            
            # Start the stream
            await self.stream.run()
            
        except Exception as e:
            logger.error(f"Failed to start Alpaca stream: {e}")
            raise
    
    async def _handle_trade_update(self, trade) -> None:
        """Handle real-time trade updates."""
        try:
            symbol = trade.symbol
            timestamp = trade.timestamp.replace(tzinfo=timezone.utc)
            
            # Create market data point
            data_point = MarketDataPoint(
                symbol=symbol,
                timestamp=timestamp,
                price=float(trade.price),
                size=int(trade.size),
                side="BUY" if hasattr(trade, 'takerSide') and trade.takerSide == "B" else "SELL",
                source="alpaca"
            )
            
            # Process the update
            await self._process_market_update(data_point)
            
        except Exception as e:
            logger.error(f"Error handling trade update: {e}")
    
    async def _handle_quote_update(self, quote) -> None:
        """Handle real-time quote updates."""
        try:
            symbol = quote.symbol
            timestamp = quote.timestamp.replace(tzinfo=timezone.utc)
            
            # Create quote data
            quote_data = QuoteData(
                symbol=symbol,
                timestamp=timestamp,
                bid=float(quote.bid_price),
                ask=float(quote.ask_price),
                bid_size=int(quote.bid_size),
                ask_size=int(quote.ask_size),
                spread=float(quote.ask_price) - float(quote.bid_price),
                source="alpaca"
            )
            
            # Store in Redis for fast access
            await self._cache_quote_data(quote_data)
            
            # Notify subscribers
            await self._notify_subscribers("quote", quote_data)
            
        except Exception as e:
            logger.error(f"Error handling quote update: {e}")
    
    async def _handle_bar_update(self, bar) -> None:
        """Handle real-time bar (OHLCV) updates."""
        try:
            symbol = bar.symbol
            timestamp = bar.timestamp.replace(tzinfo=timezone.utc)
            
            # Create market data point
            data_point = MarketDataPoint(
                symbol=symbol,
                timestamp=timestamp,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=int(bar.volume),
                vwap=float(bar.vwap) if hasattr(bar, 'vwap') and bar.vwap else None,
                source="alpaca"
            )
            
            # Process the update
            await self._process_market_update(data_point, is_bar=True)
            
        except Exception as e:
            logger.error(f"Error handling bar update: {e}")
    
    async def _process_market_update(self, data_point: MarketDataPoint, is_bar: bool = False) -> None:
        """Process and distribute market data updates."""
        try:
            symbol = data_point.symbol
            
            # Update performance metrics
            current_time = time.time()
            if symbol in self.last_update_time:
                latency = current_time - self.last_update_time[symbol]
                self.latency_samples.setdefault(symbol, []).append(latency)
                # Keep only last 100 samples
                if len(self.latency_samples[symbol]) > 100:
                    self.latency_samples[symbol] = self.latency_samples[symbol][-100:]
            
            self.last_update_time[symbol] = current_time
            self.update_counts[symbol] = self.update_counts.get(symbol, 0) + 1
            
            # Cache in Redis for ultra-fast access
            await self._cache_market_data(data_point)
            
            # Store in TimescaleDB (non-blocking)
            asyncio.create_task(self._store_market_data(data_point, is_bar))
            
            # Notify WebSocket subscribers
            await self._notify_subscribers("market_data", data_point)
            
            # Update derived indicators if needed
            if not is_bar:  # Only for tick data
                asyncio.create_task(self._update_technical_indicators(symbol, data_point))
            
        except Exception as e:
            logger.error(f"Error processing market update for {data_point.symbol}: {e}")
    
    async def _cache_market_data(self, data_point: MarketDataPoint) -> None:
        """Cache market data in Redis for fast access."""
        try:
            redis_client = await self.redis_manager.get_client()
            
            # Cache latest price
            key = f"price:{data_point.symbol}"
            data = {
                "price": data_point.price,
                "timestamp": data_point.timestamp.timestamp(),
                "volume": data_point.volume or 0,
                "size": data_point.size
            }
            
            await redis_client.hset(key, mapping={k: str(v) for k, v in data.items()})
            await redis_client.expire(key, 300)  # 5-minute expiry
            
            # Add to recent prices list (for charting)
            list_key = f"prices:{data_point.symbol}"
            price_data = json.dumps({
                "p": data_point.price,
                "t": int(data_point.timestamp.timestamp() * 1000),
                "v": data_point.volume or 0
            })
            
            await redis_client.lpush(list_key, price_data)
            await redis_client.ltrim(list_key, 0, 999)  # Keep last 1000 points
            await redis_client.expire(list_key, 3600)  # 1-hour expiry
            
        except Exception as e:
            logger.error(f"Error caching market data: {e}")
    
    async def _cache_quote_data(self, quote_data: QuoteData) -> None:
        """Cache quote data in Redis."""
        try:
            redis_client = await self.redis_manager.get_client()
            
            key = f"quote:{quote_data.symbol}"
            data = {
                "bid": quote_data.bid,
                "ask": quote_data.ask,
                "bid_size": quote_data.bid_size,
                "ask_size": quote_data.ask_size,
                "spread": quote_data.spread,
                "timestamp": quote_data.timestamp.timestamp()
            }
            
            await redis_client.hset(key, mapping={k: str(v) for k, v in data.items()})
            await redis_client.expire(key, 300)  # 5-minute expiry
            
        except Exception as e:
            logger.error(f"Error caching quote data: {e}")
    
    async def _store_market_data(self, data_point: MarketDataPoint, is_bar: bool) -> None:
        """Store market data in TimescaleDB."""
        try:
            if is_bar:
                # Store as OHLCV bar
                await DatabaseService.insert_market_data({
                    "symbol": data_point.symbol,
                    "timestamp": data_point.timestamp,
                    "open": data_point.open,
                    "high": data_point.high,
                    "low": data_point.low,
                    "close": data_point.close,
                    "volume": data_point.volume,
                    "vwap": data_point.vwap,
                    "source": data_point.source
                })
            else:
                # Store as tick data
                await DatabaseService.insert_tick_data({
                    "symbol": data_point.symbol,
                    "timestamp": data_point.timestamp,
                    "price": data_point.price,
                    "size": data_point.size,
                    "side": data_point.side,
                    "exchange": "IEX",
                    "source": data_point.source
                })
                
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
    
    async def _notify_subscribers(self, data_type: str, data: Any) -> None:
        """Notify all subscribers of market data updates."""
        if data_type in self.subscribers:
            for callback in self.subscribers[data_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Error notifying subscriber: {e}")
    
    async def _update_technical_indicators(self, symbol: str, data_point: MarketDataPoint) -> None:
        """Update technical indicators for the symbol."""
        try:
            # Get recent prices for indicator calculations
            redis_client = await self.redis_manager.get_client()
            prices_data = await redis_client.lrange(f"prices:{symbol}", 0, 49)  # Last 50 points
            
            if len(prices_data) < 20:  # Need at least 20 points for most indicators
                return
            
            prices = [json.loads(p)["p"] for p in prices_data]
            prices.reverse()  # Oldest first
            
            # Calculate basic indicators
            sma_20 = sum(prices[-20:]) / 20
            
            # RSI calculation (simplified)
            if len(prices) >= 14:
                gains = []
                losses = []
                for i in range(1, len(prices)):
                    change = prices[i] - prices[i-1]
                    gains.append(change if change > 0 else 0)
                    losses.append(-change if change < 0 else 0)
                
                avg_gain = sum(gains[-14:]) / 14
                avg_loss = sum(losses[-14:]) / 14
                
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100
            else:
                rsi = 50  # Default neutral RSI
            
            # Cache indicators
            indicators_key = f"indicators:{symbol}"
            indicators = {
                "sma_20": sma_20,
                "rsi": rsi,
                "last_price": data_point.price,
                "timestamp": data_point.timestamp.timestamp()
            }
            
            await redis_client.hset(indicators_key, mapping={k: str(v) for k, v in indicators.items()})
            await redis_client.expire(indicators_key, 300)
            
        except Exception as e:
            logger.error(f"Error updating technical indicators: {e}")
    
    async def _market_hours_monitor(self) -> None:
        """Monitor market hours and adjust streaming accordingly."""
        while self.is_streaming:
            try:
                # Check if market is open (simplified logic)
                now = datetime.now(timezone.utc)
                hour = now.hour
                weekday = now.weekday()
                
                # US market hours: 9:30 AM - 4:00 PM ET (14:30 - 21:00 UTC)
                market_open = (weekday < 5 and 14 <= hour < 21)
                
                if market_open and not self.active_symbols:
                    # Market opened, subscribe to default symbols
                    default_symbols = ["SPY", "QQQ", "AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN"]
                    await self.subscribe_to_symbols(default_symbols)
                    
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in market hours monitor: {e}")
                await asyncio.sleep(60)
    
    async def _performance_monitor(self) -> None:
        """Monitor streaming performance and log metrics."""
        while self.is_streaming:
            try:
                await asyncio.sleep(30)  # Report every 30 seconds
                
                if self.update_counts:
                    total_updates = sum(self.update_counts.values())
                    avg_latency = 0
                    
                    if self.latency_samples:
                        all_latencies = []
                        for samples in self.latency_samples.values():
                            all_latencies.extend(samples)
                        avg_latency = sum(all_latencies) / len(all_latencies) * 1000  # ms
                    
                    logger.info(
                        f"Market data performance: {total_updates} updates, "
                        f"{avg_latency:.1f}ms avg latency, "
                        f"{len(self.active_symbols)} symbols active"
                    )
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
    
    async def _data_quality_monitor(self) -> None:
        """Monitor data quality and detect anomalies."""
        while self.is_streaming:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                redis_client = await self.redis_manager.get_client()
                
                for symbol in self.active_symbols:
                    # Check for stale data
                    price_data = await redis_client.hget(f"price:{symbol}", "timestamp")
                    if price_data:
                        last_update = float(price_data)
                        if time.time() - last_update > 300:  # 5 minutes
                            logger.warning(f"Stale data detected for {symbol}")
                    
                    # Check for price spikes (simplified)
                    prices_data = await redis_client.lrange(f"prices:{symbol}", 0, 9)  # Last 10 points
                    if len(prices_data) >= 10:
                        prices = [json.loads(p)["p"] for p in prices_data]
                        avg_price = sum(prices) / len(prices)
                        latest_price = prices[0]
                        
                        if abs(latest_price - avg_price) / avg_price > 0.1:  # 10% spike
                            logger.warning(f"Price spike detected for {symbol}: {latest_price} vs avg {avg_price}")
                
            except Exception as e:
                logger.error(f"Error in data quality monitor: {e}")
    
    # Public API methods
    
    async def subscribe_to_symbols(self, symbols: List[str]) -> None:
        """Subscribe to market data for specified symbols."""
        try:
            for symbol in symbols:
                if symbol not in self.active_symbols:
                    self.active_symbols.add(symbol)
                    logger.info(f"Subscribed to market data for {symbol}")
            
            # Update stream subscriptions if needed
            if self.stream and symbols:
                # Add new symbols to existing subscriptions
                self.stream.subscribe_trades(self._handle_trade_update, *symbols)
                self.stream.subscribe_quotes(self._handle_quote_update, *symbols)
                self.stream.subscribe_bars(self._handle_bar_update, *symbols)
                
        except Exception as e:
            logger.error(f"Error subscribing to symbols: {e}")
    
    async def unsubscribe_from_symbols(self, symbols: List[str]) -> None:
        """Unsubscribe from market data for specified symbols."""
        try:
            for symbol in symbols:
                if symbol in self.active_symbols:
                    self.active_symbols.remove(symbol)
                    logger.info(f"Unsubscribed from market data for {symbol}")
                    
        except Exception as e:
            logger.error(f"Error unsubscribing from symbols: {e}")
    
    def subscribe_to_updates(self, data_type: str, callback: Callable) -> None:
        """Subscribe to market data updates."""
        if data_type not in self.subscribers:
            self.subscribers[data_type] = set()
        self.subscribers[data_type].add(callback)
    
    def unsubscribe_from_updates(self, data_type: str, callback: Callable) -> None:
        """Unsubscribe from market data updates."""
        if data_type in self.subscribers:
            self.subscribers[data_type].discard(callback)
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol."""
        try:
            redis_client = await self.redis_manager.get_client()
            price_data = await redis_client.hget(f"price:{symbol}", "price")
            return float(price_data) if price_data else None
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None
    
    async def get_latest_quote(self, symbol: str) -> Optional[Dict]:
        """Get the latest quote for a symbol."""
        try:
            redis_client = await self.redis_manager.get_client()
            quote_data = await redis_client.hgetall(f"quote:{symbol}")
            if quote_data:
                return {k: float(v) for k, v in quote_data.items() if k != "timestamp"}
            return None
        except Exception as e:
            logger.error(f"Error getting latest quote for {symbol}: {e}")
            return None
    
    def is_healthy(self) -> bool:
        """Check if the market data service is healthy."""
        return self.is_streaming and bool(self.active_symbols)