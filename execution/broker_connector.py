import asyncio
import logging
import json
import websockets
import aiohttp
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import deque
import time
import uuid
import os
from pathlib import Path
import ssl

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import APIError, TimeFrame
    from alpaca_trade_api.stream import Stream
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logging.warning("Alpaca Trade API not available. Install with: pip install alpaca-trade-api")

from event_bus import TradingEventBus, Event, Priority


class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class AssetType(Enum):
    STOCK = "us_equity"
    OPTION = "us_option"
    CRYPTO = "crypto"


class OrderStatus(Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    DONE_FOR_DAY = "done_for_day"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REPLACED = "replaced"
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"
    ACCEPTED = "accepted"
    PENDING_NEW = "pending_new"
    ACCEPTED_FOR_BIDDING = "accepted_for_bidding"
    STOPPED = "stopped"
    REJECTED = "rejected"
    SUSPENDED = "suspended"
    CALCULATED = "calculated"


class ConnectionStatus(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"


@dataclass
class OrderRequest:
    """Order request structure"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Union[int, float]
    order_type: OrderType
    asset_type: AssetType = AssetType.STOCK
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"
    extended_hours: bool = False
    client_order_id: Optional[str] = None
    order_class: Optional[str] = None
    take_profit: Optional[Dict[str, float]] = None
    stop_loss: Optional[Dict[str, float]] = None
    trail_price: Optional[float] = None
    trail_percent: Optional[float] = None
    
    # Options specific
    option_symbol: Optional[str] = None
    expiration_date: Optional[str] = None
    strike_price: Optional[float] = None
    option_type: Optional[str] = None  # "call" or "put"
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3
    
    def to_alpaca_dict(self) -> Dict[str, Any]:
        """Convert to Alpaca API format"""
        order_dict = {
            'symbol': self.symbol,
            'qty': str(self.quantity),
            'side': self.side.value,
            'type': self.order_type.value,
            'time_in_force': self.time_in_force,
            'extended_hours': self.extended_hours
        }
        
        if self.limit_price:
            order_dict['limit_price'] = str(self.limit_price)
        if self.stop_price:
            order_dict['stop_price'] = str(self.stop_price)
        if self.client_order_id:
            order_dict['client_order_id'] = self.client_order_id
        if self.trail_price:
            order_dict['trail_price'] = str(self.trail_price)
        if self.trail_percent:
            order_dict['trail_percent'] = str(self.trail_percent)
        
        # Bracket orders
        if self.order_class:
            order_dict['order_class'] = self.order_class
            if self.take_profit:
                order_dict['take_profit'] = self.take_profit
            if self.stop_loss:
                order_dict['stop_loss'] = self.stop_loss
        
        return order_dict


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    quantity: float
    side: str
    market_value: float
    avg_entry_price: float
    unrealized_pl: float
    unrealized_plpc: float
    current_price: Optional[float] = None
    asset_type: AssetType = AssetType.STOCK
    
    # Options specific
    expiration_date: Optional[str] = None
    strike_price: Optional[float] = None
    option_type: Optional[str] = None


@dataclass
class TradeExecution:
    """Trade execution data"""
    execution_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    asset_type: AssetType = AssetType.STOCK


class ConnectionManager:
    """Manages broker connection and reconnection logic"""
    
    def __init__(self, max_retries: int = 5, retry_delay: float = 5.0):
        self.logger = logging.getLogger(f"{__name__}.ConnectionManager")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.status = ConnectionStatus.DISCONNECTED
        self.retry_count = 0
        self.last_error = None
        
    async def connect_with_retry(self, connect_func: Callable) -> bool:
        """Connect with automatic retry logic"""
        self.retry_count = 0
        
        while self.retry_count < self.max_retries:
            try:
                self.status = ConnectionStatus.CONNECTING
                success = await connect_func()
                
                if success:
                    self.status = ConnectionStatus.CONNECTED
                    self.retry_count = 0
                    self.last_error = None
                    self.logger.info("Connection established successfully")
                    return True
                else:
                    raise Exception("Connection function returned False")
                    
            except Exception as e:
                self.retry_count += 1
                self.last_error = str(e)
                self.status = ConnectionStatus.ERROR
                
                self.logger.error(f"Connection attempt {self.retry_count}/{self.max_retries} failed: {e}")
                
                if self.retry_count < self.max_retries:
                    await asyncio.sleep(self.retry_delay * self.retry_count)  # Exponential backoff
                else:
                    self.status = ConnectionStatus.DISCONNECTED
                    self.logger.error("Max connection retries exceeded")
                    return False
        
        return False


class OrderQueue:
    """Queue for managing orders when connection is lost"""
    
    def __init__(self, max_size: int = 1000):
        self.queue = deque(maxlen=max_size)
        self.pending_orders: Dict[str, OrderRequest] = {}
        self.failed_orders: List[OrderRequest] = []
        self.lock = threading.RLock()
        
    def add_order(self, order: OrderRequest):
        """Add order to queue"""
        with self.lock:
            self.queue.append(order)
            self.pending_orders[order.order_id] = order
    
    def get_next_order(self) -> Optional[OrderRequest]:
        """Get next order from queue"""
        with self.lock:
            if self.queue:
                return self.queue.popleft()
            return None
    
    def remove_order(self, order_id: str):
        """Remove order from pending"""
        with self.lock:
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
    
    def mark_failed(self, order: OrderRequest):
        """Mark order as failed"""
        with self.lock:
            self.failed_orders.append(order)
            if order.order_id in self.pending_orders:
                del self.pending_orders[order.order_id]
    
    def get_queue_status(self) -> Dict[str, int]:
        """Get queue status"""
        with self.lock:
            return {
                'queued': len(self.queue),
                'pending': len(self.pending_orders),
                'failed': len(self.failed_orders)
            }
    
    def clear_failed_orders(self):
        """Clear failed orders list"""
        with self.lock:
            self.failed_orders.clear()


class AlpacaBrokerConnector:
    """Main Alpaca broker connector with full functionality"""
    
    def __init__(self, 
                 event_bus: TradingEventBus,
                 trading_mode: TradingMode = TradingMode.PAPER,
                 api_key: Optional[str] = None,
                 secret_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        
        self.logger = logging.getLogger(__name__)
        self.event_bus = event_bus
        self.trading_mode = trading_mode
        
        # API credentials
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        
        # URLs based on trading mode
        if base_url:
            self.base_url = base_url
        elif trading_mode == TradingMode.PAPER:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
        
        self.data_url = "https://data.alpaca.markets"
        
        # Alpaca API client
        self.api = None
        self.stream = None
        
        # Connection management
        self.connection_manager = ConnectionManager()
        self.order_queue = OrderQueue()
        
        # State tracking
        self.positions: Dict[str, Position] = {}
        self.active_orders: Dict[str, Dict] = {}
        self.account_info: Dict[str, Any] = {}
        
        # WebSocket management
        self.ws_connected = False
        self.ws_task = None
        
        # Statistics
        self.stats = {
            'orders_sent': 0,
            'orders_filled': 0,
            'orders_failed': 0,
            'connection_drops': 0,
            'reconnections': 0
        }
        
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Setup event bus handlers"""
        if self.event_bus:
            self.event_bus.subscribe("place_order", self._handle_place_order_event)
            self.event_bus.subscribe("cancel_order", self._handle_cancel_order_event)
            self.event_bus.subscribe("get_positions", self._handle_get_positions_event)
            self.event_bus.subscribe("get_account", self._handle_get_account_event)
    
    async def initialize(self) -> bool:
        """Initialize the broker connector"""
        if not ALPACA_AVAILABLE:
            self.logger.error("Alpaca Trade API not installed")
            return False
        
        if not self.api_key or not self.secret_key:
            self.logger.error("Alpaca API credentials not provided")
            return False
        
        try:
            # Initialize Alpaca API
            self.api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url,
                api_version='v2'
            )
            
            # Test connection
            success = await self.connection_manager.connect_with_retry(self._test_connection)
            
            if success:
                # Start order processing task
                asyncio.create_task(self._process_order_queue())
                
                # Initialize WebSocket stream
                await self._initialize_websocket()
                
                # Publish connection status
                await self._publish_connection_status(ConnectionStatus.CONNECTED)
                
                self.logger.info(f"Alpaca connector initialized in {self.trading_mode.value} mode")
                return True
            else:
                await self._publish_connection_status(ConnectionStatus.ERROR)
                return False
                
        except Exception as e:
            self.logger.error(f"Error initializing Alpaca connector: {e}")
            await self._publish_connection_status(ConnectionStatus.ERROR)
            return False
    
    async def _test_connection(self) -> bool:
        """Test API connection"""
        try:
            account = self.api.get_account()
            self.account_info = {
                'account_id': account.id,
                'status': account.status,
                'currency': account.currency,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked
            }
            
            self.logger.info(f"Connected to Alpaca {self.trading_mode.value} account: {account.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    async def _initialize_websocket(self):
        """Initialize WebSocket connection for real-time updates"""
        try:
            self.stream = Stream(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url,
                data_feed='iex' if self.trading_mode == TradingMode.PAPER else 'sip'
            )
            
            # Subscribe to trade updates
            self.stream.subscribe_trade_updates(self._handle_trade_update)
            
            # Start stream
            self.ws_task = asyncio.create_task(self._run_websocket_stream())
            self.logger.info("WebSocket stream initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing WebSocket: {e}")
    
    async def _run_websocket_stream(self):
        """Run the WebSocket stream"""
        try:
            self.stream.run()
            self.ws_connected = True
            
        except Exception as e:
            self.logger.error(f"WebSocket stream error: {e}")
            self.ws_connected = False
            
            # Attempt to reconnect
            await asyncio.sleep(5)
            if self.connection_manager.status == ConnectionStatus.CONNECTED:
                self.logger.info("Attempting WebSocket reconnection")
                await self._initialize_websocket()
    
    async def _handle_trade_update(self, data):
        """Handle trade updates from WebSocket"""
        try:
            order_id = data.order['id']
            status = data.order['status']
            
            # Update active orders
            if order_id in self.active_orders:
                self.active_orders[order_id]['status'] = status
            
            # Create execution data
            execution = None
            if hasattr(data, 'execution') and data.execution:
                execution = TradeExecution(
                    execution_id=str(uuid.uuid4()),
                    order_id=order_id,
                    symbol=data.order['symbol'],
                    side=OrderSide(data.order['side']),
                    quantity=float(data.execution['qty']),
                    price=float(data.execution['price']),
                    timestamp=datetime.now(),
                    commission=0.0  # Alpaca is commission-free
                )
            
            # Publish trade update
            await self.event_bus.publish(
                "trade_update",
                {
                    'order_id': order_id,
                    'symbol': data.order['symbol'],
                    'status': status,
                    'filled_qty': data.order.get('filled_qty', 0),
                    'execution': execution.to_dict() if execution else None,
                    'timestamp': datetime.now().isoformat()
                },
                priority=Priority.HIGH
            )
            
            # Update statistics
            if status == 'filled':
                self.stats['orders_filled'] += 1
                
                # Remove from order queue if present
                self.order_queue.remove_order(order_id)
            
            self.logger.info(f"Trade update: {data.order['symbol']} - {status}")
            
        except Exception as e:
            self.logger.error(f"Error handling trade update: {e}")
    
    async def place_order(self, order_request: OrderRequest) -> Optional[str]:
        """Place a trading order"""
        try:
            # Add to queue first
            self.order_queue.add_order(order_request)
            
            # If connected, try to place immediately
            if self.connection_manager.status == ConnectionStatus.CONNECTED:
                return await self._place_order_direct(order_request)
            else:
                self.logger.info(f"Order {order_request.order_id} queued - connection not available")
                return order_request.order_id
                
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            self.stats['orders_failed'] += 1
            return None
    
    async def _place_order_direct(self, order_request: OrderRequest) -> Optional[str]:
        """Place order directly with Alpaca"""
        try:
            order_dict = order_request.to_alpaca_dict()
            
            # Place order based on asset type
            if order_request.asset_type == AssetType.OPTION:
                # Options trading (if supported by your Alpaca account)
                order = self.api.submit_order(**order_dict)
            else:
                # Stock trading
                order = self.api.submit_order(**order_dict)
            
            # Store active order
            self.active_orders[order.id] = {
                'alpaca_id': order.id,
                'client_id': order_request.order_id,
                'symbol': order.symbol,
                'status': order.status,
                'created_at': order.created_at
            }
            
            # Remove from queue
            self.order_queue.remove_order(order_request.order_id)
            
            self.stats['orders_sent'] += 1
            
            # Publish order placed event
            await self.event_bus.publish(
                "order_placed",
                {
                    'order_id': order.id,
                    'client_order_id': order_request.order_id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'quantity': order.qty,
                    'order_type': order.order_type,
                    'status': order.status,
                    'created_at': order.created_at.isoformat()
                },
                priority=Priority.HIGH
            )
            
            self.logger.info(f"Order placed: {order.symbol} {order.side} {order.qty} @ {order.order_type}")
            return order.id
            
        except APIError as e:
            self.logger.error(f"Alpaca API error placing order: {e}")
            order_request.retry_count += 1
            
            if order_request.retry_count < order_request.max_retries:
                # Re-queue for retry
                self.order_queue.add_order(order_request)
            else:
                self.order_queue.mark_failed(order_request)
                self.stats['orders_failed'] += 1
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error placing order directly: {e}")
            self.order_queue.mark_failed(order_request)
            self.stats['orders_failed'] += 1
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            if self.connection_manager.status != ConnectionStatus.CONNECTED:
                self.logger.warning(f"Cannot cancel order {order_id} - not connected")
                return False
            
            self.api.cancel_order(order_id)
            
            # Remove from active orders
            if order_id in self.active_orders:
                del self.active_orders[order_id]
            
            # Publish cancellation event
            await self.event_bus.publish(
                "order_cancelled",
                {
                    'order_id': order_id,
                    'timestamp': datetime.now().isoformat()
                },
                priority=Priority.NORMAL
            )
            
            self.logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        try:
            if self.connection_manager.status != ConnectionStatus.CONNECTED:
                return self.positions
            
            positions = self.api.list_positions()
            
            self.positions = {}
            for pos in positions:
                position = Position(
                    symbol=pos.symbol,
                    quantity=float(pos.qty),
                    side=pos.side,
                    market_value=float(pos.market_value),
                    avg_entry_price=float(pos.avg_entry_price),
                    unrealized_pl=float(pos.unrealized_pl),
                    unrealized_plpc=float(pos.unrealized_plpc),
                    current_price=float(pos.current_price) if pos.current_price else None
                )
                self.positions[pos.symbol] = position
            
            return self.positions
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return self.positions
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            if self.connection_manager.status == ConnectionStatus.CONNECTED:
                await self._test_connection()  # Updates account_info
            
            return self.account_info
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return self.account_info
    
    async def get_market_data(self, symbol: str, timeframe: str = "1Min", limit: int = 100) -> Optional[pd.DataFrame]:
        """Get market data for a symbol"""
        try:
            if self.connection_manager.status != ConnectionStatus.CONNECTED:
                return None
            
            # Convert timeframe
            tf_map = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, TimeFrame.Unit.Minute),
                "15Min": TimeFrame(15, TimeFrame.Unit.Minute),
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day
            }
            
            timeframe_obj = tf_map.get(timeframe, TimeFrame.Minute)
            
            # Get historical data
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)  # Last 30 days
            
            bars = self.api.get_bars(
                symbol,
                timeframe_obj,
                start=start_time.isoformat(),
                end=end_time.isoformat(),
                limit=limit
            ).df
            
            if not bars.empty:
                # Convert to standard format
                bars = bars.rename(columns={
                    'open': 'open',
                    'high': 'high', 
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                })
                
            return bars
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    async def _process_order_queue(self):
        """Process queued orders"""
        while True:
            try:
                if (self.connection_manager.status == ConnectionStatus.CONNECTED and 
                    self.order_queue.queue):
                    
                    order = self.order_queue.get_next_order()
                    if order:
                        await self._place_order_direct(order)
                        await asyncio.sleep(0.1)  # Small delay between orders
                
                await asyncio.sleep(1)  # Check queue every second
                
            except Exception as e:
                self.logger.error(f"Error processing order queue: {e}")
                await asyncio.sleep(5)
    
    async def _publish_connection_status(self, status: ConnectionStatus):
        """Publish connection status update"""
        if self.event_bus:
            await self.event_bus.publish(
                "broker_connection_status",
                {
                    'status': status.value,
                    'trading_mode': self.trading_mode.value,
                    'timestamp': datetime.now().isoformat(),
                    'last_error': self.connection_manager.last_error,
                    'retry_count': self.connection_manager.retry_count
                },
                priority=Priority.HIGH if status == ConnectionStatus.ERROR else Priority.NORMAL
            )
    
    # Event handlers
    async def _handle_place_order_event(self, event: Event):
        """Handle place order events from EventBus"""
        try:
            data = event.data
            
            order_request = OrderRequest(
                order_id=data.get('order_id', str(uuid.uuid4())),
                symbol=data['symbol'],
                side=OrderSide(data['side']),
                quantity=data['quantity'],
                order_type=OrderType(data.get('order_type', 'market')),
                asset_type=AssetType(data.get('asset_type', 'us_equity')),
                limit_price=data.get('limit_price'),
                stop_price=data.get('stop_price'),
                time_in_force=data.get('time_in_force', 'day')
            )
            
            order_id = await self.place_order(order_request)
            
            if order_id:
                self.logger.info(f"Order placed via event: {order_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling place order event: {e}")
    
    async def _handle_cancel_order_event(self, event: Event):
        """Handle cancel order events"""
        try:
            data = event.data
            order_id = data['order_id']
            
            success = await self.cancel_order(order_id)
            
            if success:
                self.logger.info(f"Order cancelled via event: {order_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling cancel order event: {e}")
    
    async def _handle_get_positions_event(self, event: Event):
        """Handle get positions events"""
        try:
            positions = await self.get_positions()
            
            await self.event_bus.publish(
                "positions_response",
                {
                    'positions': {symbol: asdict(pos) for symbol, pos in positions.items()},
                    'timestamp': datetime.now().isoformat()
                },
                priority=Priority.NORMAL
            )
            
        except Exception as e:
            self.logger.error(f"Error handling get positions event: {e}")
    
    async def _handle_get_account_event(self, event: Event):
        """Handle get account events"""
        try:
            account_info = await self.get_account_info()
            
            await self.event_bus.publish(
                "account_response",
                {
                    'account': account_info,
                    'timestamp': datetime.now().isoformat()
                },
                priority=Priority.NORMAL
            )
            
        except Exception as e:
            self.logger.error(f"Error handling get account event: {e}")
    
    async def disconnect(self):
        """Disconnect from broker"""
        try:
            # Stop WebSocket stream
            if self.ws_task:
                self.ws_task.cancel()
                try:
                    await self.ws_task
                except asyncio.CancelledError:
                    pass
            
            if self.stream:
                self.stream.stop()
            
            self.connection_manager.status = ConnectionStatus.DISCONNECTED
            await self._publish_connection_status(ConnectionStatus.DISCONNECTED)
            
            self.logger.info("Broker connector disconnected")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting: {e}")
    
    def get_connector_stats(self) -> Dict[str, Any]:
        """Get connector statistics"""
        queue_stats = self.order_queue.get_queue_status()
        
        return {
            'connection_status': self.connection_manager.status.value,
            'trading_mode': self.trading_mode.value,
            'websocket_connected': self.ws_connected,
            'active_orders': len(self.active_orders),
            'positions': len(self.positions),
            'queue_stats': queue_stats,
            'statistics': self.stats,
            'account_info': {
                'buying_power': self.account_info.get('buying_power', 0),
                'cash': self.account_info.get('cash', 0),
                'portfolio_value': self.account_info.get('portfolio_value', 0)
            }
        }


# Utility functions for easy mode switching
def create_paper_trading_connector(event_bus: TradingEventBus, 
                                 api_key: Optional[str] = None, 
                                 secret_key: Optional[str] = None) -> AlpacaBrokerConnector:
    """Create paper trading connector"""
    return AlpacaBrokerConnector(
        event_bus=event_bus,
        trading_mode=TradingMode.PAPER,
        api_key=api_key,
        secret_key=secret_key
    )


def create_live_trading_connector(event_bus: TradingEventBus,
                                api_key: Optional[str] = None,
                                secret_key: Optional[str] = None) -> AlpacaBrokerConnector:
    """Create live trading connector"""
    return AlpacaBrokerConnector(
        event_bus=event_bus,
        trading_mode=TradingMode.LIVE,
        api_key=api_key,
        secret_key=secret_key
    )


# Example usage
async def main():
    """Example usage of the broker connector"""
    logging.basicConfig(level=logging.INFO)
    
    # Create event bus
    event_bus = TradingEventBus()
    await event_bus.start()
    
    try:
        # Create paper trading connector
        connector = create_paper_trading_connector(event_bus)
        
        # Initialize
        if await connector.initialize():
            # Get account info
            account = await connector.get_account_info()
            print(f"Account buying power: ${account.get('buying_power', 0):,.2f}")
            
            # Place a test order
            order_request = OrderRequest(
                order_id=str(uuid.uuid4()),
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=1,
                order_type=OrderType.MARKET
            )
            
            order_id = await connector.place_order(order_request)
            if order_id:
                print(f"Order placed: {order_id}")
            
            # Get positions
            positions = await connector.get_positions()
            print(f"Current positions: {len(positions)}")
            
            # Get connector stats
            stats = connector.get_connector_stats()
            print(f"Connector stats: {stats}")
            
        # Wait a bit to see trade updates
        await asyncio.sleep(10)
        
    finally:
        await connector.disconnect()
        await event_bus.stop()


if __name__ == "__main__":
    asyncio.run(main())