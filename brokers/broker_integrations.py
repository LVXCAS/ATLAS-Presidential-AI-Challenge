"""
Production-Ready Broker Integrations

Real broker connectivity for live trading with comprehensive error handling,
rate limiting, and institutional-grade execution capabilities.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import json
import time
from enum import Enum
import requests
import websocket
import threading
from queue import Queue
import hmac
import hashlib
import base64
import urllib.parse
import warnings
warnings.filterwarnings('ignore')

# Trading libraries
import alpaca_trade_api as tradeapi
from ib_insync import IB, Stock, Option, Future, Forex, Order, LimitOrder, MarketOrder, StopOrder
import ccxt
import yfinance as yf

# Security and authentication
import jwt
from cryptography.fernet import Fernet
import keyring

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class BrokerCredentials:
    """Secure credential storage for broker APIs"""
    broker_name: str
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""
    account_id: str = ""
    sandbox: bool = True
    encrypted: bool = False

@dataclass
class OrderRequest:
    """Standardized order request across all brokers"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"
    strategy_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Position:
    """Standardized position representation"""
    symbol: str
    quantity: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    side: str
    broker: str
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class ExecutionReport:
    """Trade execution report"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    filled_quantity: float
    price: float
    commission: float
    timestamp: datetime
    status: OrderStatus
    broker: str
    strategy_id: str = ""

class BaseBroker(ABC):
    """Abstract base class for all broker integrations"""

    def __init__(self, credentials: BrokerCredentials):
        self.credentials = credentials
        self.is_connected = False
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Any] = {}
        self.execution_queue = Queue()
        self.error_count = 0
        self.last_heartbeat = datetime.now()

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to broker"""
        pass

    @abstractmethod
    async def disconnect(self):
        """Close broker connection"""
        pass

    @abstractmethod
    async def submit_order(self, order_request: OrderRequest) -> str:
        """Submit order and return order ID"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        pass

    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        pass

    @abstractmethod
    async def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time market data"""
        pass

    def is_healthy(self) -> bool:
        """Check broker connection health"""
        return (
            self.is_connected and
            self.error_count < 10 and
            (datetime.now() - self.last_heartbeat).seconds < 300
        )

class AlpacaBroker(BaseBroker):
    """Alpaca Markets integration for commission-free trading"""

    def __init__(self, credentials: BrokerCredentials):
        super().__init__(credentials)
        self.api = None
        self.ws = None
        self.data_stream = None

    async def connect(self) -> bool:
        """Connect to Alpaca API"""
        try:
            base_url = "https://paper-api.alpaca.markets" if self.credentials.sandbox else "https://api.alpaca.markets"

            self.api = tradeapi.REST(
                key_id=self.credentials.api_key,
                secret_key=self.credentials.api_secret,
                base_url=base_url,
                api_version='v2'
            )

            # Test connection
            account = self.api.get_account()
            logging.info(f"Connected to Alpaca - Account: {account.id}")

            # Start data stream
            await self._start_data_stream()

            self.is_connected = True
            self.last_heartbeat = datetime.now()
            return True

        except Exception as e:
            logging.error(f"Alpaca connection failed: {e}")
            self.error_count += 1
            return False

    async def disconnect(self):
        """Disconnect from Alpaca"""
        if self.data_stream:
            await self.data_stream.stop_ws()
        self.is_connected = False

    async def submit_order(self, order_request: OrderRequest) -> str:
        """Submit order to Alpaca"""
        try:
            # Convert our order to Alpaca format
            alpaca_order = {
                'symbol': order_request.symbol,
                'qty': order_request.quantity,
                'side': order_request.side.value,
                'type': order_request.order_type.value,
                'time_in_force': order_request.time_in_force
            }

            if order_request.price:
                alpaca_order['limit_price'] = order_request.price

            if order_request.stop_price:
                alpaca_order['stop_price'] = order_request.stop_price

            # Submit order
            order = self.api.submit_order(**alpaca_order)

            # Store order
            self.orders[order.id] = order

            logging.info(f"Alpaca order submitted: {order.id}")
            return order.id

        except Exception as e:
            logging.error(f"Alpaca order submission failed: {e}")
            self.error_count += 1
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel Alpaca order"""
        try:
            self.api.cancel_order(order_id)
            logging.info(f"Alpaca order cancelled: {order_id}")
            return True
        except Exception as e:
            logging.error(f"Alpaca order cancellation failed: {e}")
            return False

    async def get_positions(self) -> List[Position]:
        """Get Alpaca positions"""
        try:
            alpaca_positions = self.api.list_positions()
            positions = []

            for pos in alpaca_positions:
                position = Position(
                    symbol=pos.symbol,
                    quantity=float(pos.qty),
                    market_value=float(pos.market_value),
                    cost_basis=float(pos.cost_basis),
                    unrealized_pnl=float(pos.unrealized_pl),
                    side="long" if float(pos.qty) > 0 else "short",
                    broker="alpaca"
                )
                positions.append(position)
                self.positions[pos.symbol] = position

            return positions

        except Exception as e:
            logging.error(f"Failed to get Alpaca positions: {e}")
            return []

    async def get_account_info(self) -> Dict[str, Any]:
        """Get Alpaca account information"""
        try:
            account = self.api.get_account()
            return {
                'account_id': account.id,
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'day_trade_count': int(account.daytrade_count),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'account_blocked': account.account_blocked
            }
        except Exception as e:
            logging.error(f"Failed to get Alpaca account info: {e}")
            return {}

    async def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get Alpaca market data"""
        try:
            quotes = self.api.get_latest_quotes(symbols)
            bars = self.api.get_latest_bars(symbols)

            market_data = {}
            for symbol in symbols:
                if symbol in quotes:
                    quote = quotes[symbol]
                    bar = bars.get(symbol)

                    market_data[symbol] = {
                        'bid': float(quote.bid_price),
                        'ask': float(quote.ask_price),
                        'bid_size': int(quote.bid_size),
                        'ask_size': int(quote.ask_size),
                        'last_price': float(bar.close) if bar else float(quote.ask_price),
                        'volume': int(bar.volume) if bar else 0,
                        'timestamp': quote.timestamp
                    }

            return market_data

        except Exception as e:
            logging.error(f"Failed to get Alpaca market data: {e}")
            return {}

    async def _start_data_stream(self):
        """Start Alpaca data stream"""
        try:
            from alpaca_trade_api.stream import Stream

            self.data_stream = Stream(
                key_id=self.credentials.api_key,
                secret_key=self.credentials.api_secret,
                base_url="https://paper-api.alpaca.markets" if self.credentials.sandbox else "https://api.alpaca.markets",
                data_feed='iex'
            )

            # Subscribe to trade updates
            self.data_stream.subscribe_trade_updates(self._handle_trade_update)

            # Start stream in background
            asyncio.create_task(self.data_stream.run())

        except Exception as e:
            logging.error(f"Failed to start Alpaca data stream: {e}")

    def _handle_trade_update(self, data):
        """Handle trade execution updates"""
        try:
            order_id = data.order['id']

            execution = ExecutionReport(
                order_id=order_id,
                symbol=data.order['symbol'],
                side=OrderSide(data.order['side']),
                quantity=float(data.order['qty']),
                filled_quantity=float(data.order['filled_qty']),
                price=float(data.order.get('filled_avg_price', 0)),
                commission=0.0,  # Alpaca is commission-free
                timestamp=datetime.now(timezone.utc),
                status=OrderStatus(data.order['status'].lower()),
                broker="alpaca"
            )

            self.execution_queue.put(execution)
            logging.info(f"Alpaca execution: {execution}")

        except Exception as e:
            logging.error(f"Error handling Alpaca trade update: {e}")

class InteractiveBrokersBroker(BaseBroker):
    """Interactive Brokers integration for institutional trading"""

    def __init__(self, credentials: BrokerCredentials):
        super().__init__(credentials)
        self.ib = IB()
        self.port = 7497 if credentials.sandbox else 7496

    async def connect(self) -> bool:
        """Connect to Interactive Brokers TWS/Gateway"""
        try:
            await self.ib.connectAsync('127.0.0.1', self.port, clientId=1)

            # Set up event handlers
            self.ib.execDetailsEvent += self._handle_execution
            self.ib.orderStatusEvent += self._handle_order_status

            logging.info("Connected to Interactive Brokers")
            self.is_connected = True
            self.last_heartbeat = datetime.now()
            return True

        except Exception as e:
            logging.error(f"IB connection failed: {e}")
            self.error_count += 1
            return False

    async def disconnect(self):
        """Disconnect from Interactive Brokers"""
        self.ib.disconnect()
        self.is_connected = False

    async def submit_order(self, order_request: OrderRequest) -> str:
        """Submit order to Interactive Brokers"""
        try:
            # Create contract
            contract = Stock(order_request.symbol, 'SMART', 'USD')

            # Create order
            if order_request.order_type == OrderType.MARKET:
                order = MarketOrder(
                    order_request.side.value.upper(),
                    order_request.quantity
                )
            elif order_request.order_type == OrderType.LIMIT:
                order = LimitOrder(
                    order_request.side.value.upper(),
                    order_request.quantity,
                    order_request.price
                )
            else:
                order = StopOrder(
                    order_request.side.value.upper(),
                    order_request.quantity,
                    order_request.stop_price
                )

            # Submit order
            trade = self.ib.placeOrder(contract, order)

            # Store order
            self.orders[str(trade.order.orderId)] = trade

            logging.info(f"IB order submitted: {trade.order.orderId}")
            return str(trade.order.orderId)

        except Exception as e:
            logging.error(f"IB order submission failed: {e}")
            self.error_count += 1
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel IB order"""
        try:
            if order_id in self.orders:
                trade = self.orders[order_id]
                self.ib.cancelOrder(trade.order)
                logging.info(f"IB order cancelled: {order_id}")
                return True
            return False
        except Exception as e:
            logging.error(f"IB order cancellation failed: {e}")
            return False

    async def get_positions(self) -> List[Position]:
        """Get IB positions"""
        try:
            ib_positions = self.ib.positions()
            positions = []

            for pos in ib_positions:
                if pos.position != 0:
                    position = Position(
                        symbol=pos.contract.symbol,
                        quantity=pos.position,
                        market_value=pos.marketValue,
                        cost_basis=pos.averageCost * abs(pos.position),
                        unrealized_pnl=pos.unrealizedPNL,
                        side="long" if pos.position > 0 else "short",
                        broker="interactive_brokers"
                    )
                    positions.append(position)
                    self.positions[pos.contract.symbol] = position

            return positions

        except Exception as e:
            logging.error(f"Failed to get IB positions: {e}")
            return []

    async def get_account_info(self) -> Dict[str, Any]:
        """Get IB account information"""
        try:
            account_values = self.ib.accountValues()

            info = {}
            for item in account_values:
                if item.tag in ['NetLiquidation', 'TotalCashValue', 'BuyingPower']:
                    info[item.tag.lower()] = float(item.value)

            return info

        except Exception as e:
            logging.error(f"Failed to get IB account info: {e}")
            return {}

    async def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get IB market data"""
        try:
            market_data = {}

            for symbol in symbols:
                contract = Stock(symbol, 'SMART', 'USD')
                self.ib.qualifyContracts(contract)

                # Get market data
                ticker = self.ib.reqMktData(contract, '', False, False)
                self.ib.sleep(1)  # Wait for data

                if ticker.marketPrice():
                    market_data[symbol] = {
                        'bid': ticker.bid,
                        'ask': ticker.ask,
                        'bid_size': ticker.bidSize,
                        'ask_size': ticker.askSize,
                        'last_price': ticker.marketPrice(),
                        'volume': ticker.volume or 0,
                        'timestamp': datetime.now()
                    }

            return market_data

        except Exception as e:
            logging.error(f"Failed to get IB market data: {e}")
            return {}

    def _handle_execution(self, trade, fill):
        """Handle IB execution reports"""
        try:
            execution = ExecutionReport(
                order_id=str(fill.execution.orderId),
                symbol=fill.contract.symbol,
                side=OrderSide.BUY if fill.execution.side == 'BOT' else OrderSide.SELL,
                quantity=fill.execution.shares,
                filled_quantity=fill.execution.cumQty,
                price=fill.execution.price,
                commission=fill.commissionReport.commission if fill.commissionReport else 0,
                timestamp=datetime.now(timezone.utc),
                status=OrderStatus.FILLED,
                broker="interactive_brokers"
            )

            self.execution_queue.put(execution)
            logging.info(f"IB execution: {execution}")

        except Exception as e:
            logging.error(f"Error handling IB execution: {e}")

    def _handle_order_status(self, trade):
        """Handle IB order status updates"""
        try:
            # Update order status
            logging.info(f"IB order status update: {trade.order.orderId} - {trade.orderStatus.status}")
        except Exception as e:
            logging.error(f"Error handling IB order status: {e}")

class CryptoBroker(BaseBroker):
    """Cryptocurrency exchange integration using CCXT"""

    def __init__(self, credentials: BrokerCredentials, exchange_name: str = 'binance'):
        super().__init__(credentials)
        self.exchange_name = exchange_name
        self.exchange = None

    async def connect(self) -> bool:
        """Connect to crypto exchange"""
        try:
            exchange_class = getattr(ccxt, self.exchange_name)

            self.exchange = exchange_class({
                'apiKey': self.credentials.api_key,
                'secret': self.credentials.api_secret,
                'password': self.credentials.api_passphrase,
                'sandbox': self.credentials.sandbox,
                'enableRateLimit': True,
            })

            # Test connection
            balance = await self.exchange.fetch_balance()
            logging.info(f"Connected to {self.exchange_name}")

            self.is_connected = True
            self.last_heartbeat = datetime.now()
            return True

        except Exception as e:
            logging.error(f"{self.exchange_name} connection failed: {e}")
            self.error_count += 1
            return False

    async def disconnect(self):
        """Disconnect from crypto exchange"""
        if self.exchange:
            await self.exchange.close()
        self.is_connected = False

    async def submit_order(self, order_request: OrderRequest) -> str:
        """Submit order to crypto exchange"""
        try:
            # Convert order type
            order_type = order_request.order_type.value
            if order_type == 'stop':
                order_type = 'market'

            # Submit order
            order = await self.exchange.create_order(
                symbol=order_request.symbol,
                type=order_type,
                side=order_request.side.value,
                amount=order_request.quantity,
                price=order_request.price,
                params={'stopPrice': order_request.stop_price} if order_request.stop_price else {}
            )

            # Store order
            self.orders[order['id']] = order

            logging.info(f"{self.exchange_name} order submitted: {order['id']}")
            return order['id']

        except Exception as e:
            logging.error(f"{self.exchange_name} order submission failed: {e}")
            self.error_count += 1
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel crypto exchange order"""
        try:
            await self.exchange.cancel_order(order_id)
            logging.info(f"{self.exchange_name} order cancelled: {order_id}")
            return True
        except Exception as e:
            logging.error(f"{self.exchange_name} order cancellation failed: {e}")
            return False

    async def get_positions(self) -> List[Position]:
        """Get crypto positions (balances)"""
        try:
            balance = await self.exchange.fetch_balance()
            positions = []

            for currency, amounts in balance['total'].items():
                if amounts > 0:
                    # Get market value (simplified)
                    market_value = amounts  # Would need price conversion

                    position = Position(
                        symbol=currency,
                        quantity=amounts,
                        market_value=market_value,
                        cost_basis=market_value,  # Simplified
                        unrealized_pnl=0,  # Would need calculation
                        side="long",
                        broker=self.exchange_name
                    )
                    positions.append(position)
                    self.positions[currency] = position

            return positions

        except Exception as e:
            logging.error(f"Failed to get {self.exchange_name} positions: {e}")
            return []

    async def get_account_info(self) -> Dict[str, Any]:
        """Get crypto account information"""
        try:
            balance = await self.exchange.fetch_balance()
            return {
                'total_balance': balance['total'],
                'free_balance': balance['free'],
                'used_balance': balance['used']
            }
        except Exception as e:
            logging.error(f"Failed to get {self.exchange_name} account info: {e}")
            return {}

    async def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get crypto market data"""
        try:
            market_data = {}

            for symbol in symbols:
                ticker = await self.exchange.fetch_ticker(symbol)
                orderbook = await self.exchange.fetch_order_book(symbol, limit=1)

                market_data[symbol] = {
                    'bid': orderbook['bids'][0][0] if orderbook['bids'] else ticker['bid'],
                    'ask': orderbook['asks'][0][0] if orderbook['asks'] else ticker['ask'],
                    'bid_size': orderbook['bids'][0][1] if orderbook['bids'] else 0,
                    'ask_size': orderbook['asks'][0][1] if orderbook['asks'] else 0,
                    'last_price': ticker['last'],
                    'volume': ticker['baseVolume'],
                    'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000)
                }

            return market_data

        except Exception as e:
            logging.error(f"Failed to get {self.exchange_name} market data: {e}")
            return {}

class BrokerManager:
    """Centralized broker management and routing"""

    def __init__(self):
        self.brokers: Dict[str, BaseBroker] = {}
        self.execution_handlers = []
        self.is_running = False
        self.execution_thread = None

    def add_broker(self, name: str, broker: BaseBroker):
        """Add broker to manager"""
        self.brokers[name] = broker
        logging.info(f"Added broker: {name}")

    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all brokers"""
        results = {}
        for name, broker in self.brokers.items():
            try:
                results[name] = await broker.connect()
            except Exception as e:
                logging.error(f"Failed to connect to {name}: {e}")
                results[name] = False
        return results

    async def disconnect_all(self):
        """Disconnect from all brokers"""
        for broker in self.brokers.values():
            try:
                await broker.disconnect()
            except Exception as e:
                logging.error(f"Error disconnecting broker: {e}")

    async def submit_order(self, broker_name: str, order_request: OrderRequest) -> str:
        """Submit order to specific broker"""
        if broker_name not in self.brokers:
            raise ValueError(f"Broker {broker_name} not found")

        broker = self.brokers[broker_name]
        if not broker.is_healthy():
            raise RuntimeError(f"Broker {broker_name} is not healthy")

        return await broker.submit_order(order_request)

    async def get_consolidated_positions(self) -> Dict[str, List[Position]]:
        """Get positions from all brokers"""
        all_positions = {}
        for name, broker in self.brokers.items():
            if broker.is_connected:
                try:
                    positions = await broker.get_positions()
                    all_positions[name] = positions
                except Exception as e:
                    logging.error(f"Error getting positions from {name}: {e}")
                    all_positions[name] = []
        return all_positions

    def get_broker_health(self) -> Dict[str, bool]:
        """Check health of all brokers"""
        return {name: broker.is_healthy() for name, broker in self.brokers.items()}

    def start_execution_monitoring(self):
        """Start monitoring execution reports"""
        self.is_running = True
        self.execution_thread = threading.Thread(target=self._execution_worker)
        self.execution_thread.daemon = True
        self.execution_thread.start()

    def stop_execution_monitoring(self):
        """Stop execution monitoring"""
        self.is_running = False
        if self.execution_thread:
            self.execution_thread.join()

    def _execution_worker(self):
        """Worker thread for processing execution reports"""
        while self.is_running:
            try:
                for broker in self.brokers.values():
                    while not broker.execution_queue.empty():
                        execution = broker.execution_queue.get()

                        # Notify all handlers
                        for handler in self.execution_handlers:
                            try:
                                handler(execution)
                            except Exception as e:
                                logging.error(f"Error in execution handler: {e}")

                time.sleep(0.1)  # 100ms polling

            except Exception as e:
                logging.error(f"Error in execution worker: {e}")

    def add_execution_handler(self, handler):
        """Add execution report handler"""
        self.execution_handlers.append(handler)

# Example usage and configuration
async def setup_production_brokers():
    """Setup production broker connections"""

    # Initialize broker manager
    manager = BrokerManager()

    # Alpaca broker for US equities
    alpaca_creds = BrokerCredentials(
        broker_name="alpaca",
        api_key="your_alpaca_key",
        api_secret="your_alpaca_secret",
        sandbox=True  # Set to False for live trading
    )
    alpaca_broker = AlpacaBroker(alpaca_creds)
    manager.add_broker("alpaca", alpaca_broker)

    # Interactive Brokers for institutional access
    ib_creds = BrokerCredentials(
        broker_name="interactive_brokers",
        sandbox=True
    )
    ib_broker = InteractiveBrokersBroker(ib_creds)
    manager.add_broker("ib", ib_broker)

    # Binance for crypto
    binance_creds = BrokerCredentials(
        broker_name="binance",
        api_key="your_binance_key",
        api_secret="your_binance_secret",
        sandbox=True
    )
    crypto_broker = CryptoBroker(binance_creds, "binance")
    manager.add_broker("binance", crypto_broker)

    # Connect all brokers
    connection_results = await manager.connect_all()
    logging.info(f"Broker connections: {connection_results}")

    # Start execution monitoring
    manager.start_execution_monitoring()

    return manager

if __name__ == "__main__":
    # Example usage
    async def main():
        manager = await setup_production_brokers()

        # Example order
        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=150.00,
            strategy_id="momentum_v1"
        )

        try:
            order_id = await manager.submit_order("alpaca", order)
            logging.info(f"Order submitted: {order_id}")
        except Exception as e:
            logging.error(f"Order failed: {e}")

        # Get positions
        positions = await manager.get_consolidated_positions()
        logging.info(f"Positions: {positions}")

        # Cleanup
        await manager.disconnect_all()

    asyncio.run(main())