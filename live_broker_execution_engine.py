"""
LIVE BROKER EXECUTION ENGINE
Real-time order execution for AUTONOMOUS trading
Connects GPU trading signals to live broker APIs
"""

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import threading
from typing import Dict, List, Optional, Union
import time
from dataclasses import dataclass, field
from enum import Enum
import requests
import queue

# Broker API imports
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False

try:
    import ccxt
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

@dataclass
class TradingOrder:
    """Trading order specification"""
    order_id: str
    symbol: str
    action: str  # BUY, SELL
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    strategy_source: str = "GPU_GENERATED"
    created_at: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: Optional[int] = None
    commission: float = 0.0

@dataclass
class Position:
    """Current position data"""
    symbol: str
    quantity: int
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0

class LiveBrokerExecutionEngine:
    """
    REAL-TIME BROKER EXECUTION ENGINE
    Executes trading signals through live broker APIs
    """

    def __init__(self, config_file: str = "broker_config.json"):
        self.logger = logging.getLogger('LiveExecution')

        # Order management
        self.pending_orders = {}
        self.executed_orders = {}
        self.positions = {}

        # Execution queues
        self.order_queue = asyncio.Queue()
        self.execution_results = queue.Queue()

        # Connection status
        self.brokers_connected = {}
        self.execution_active = False

        # Load configuration
        self.config = self.load_config(config_file)

        # Risk management
        self.daily_pnl = 0.0
        self.max_daily_loss = self.config.get("risk_limits", {}).get("max_daily_loss", 5000)
        self.max_position_size = self.config.get("risk_limits", {}).get("max_position_size", 10000)

        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.total_commission = 0.0
        self.starting_cash = self.config.get("starting_capital", 100000.0)

        self.logger.info("Live broker execution engine initialized")

    def load_config(self, config_file: str) -> Dict:
        """Load broker configuration"""
        default_config = {
            "brokers": {
                "alpaca": {
                    "enabled": True,
                    "paper_trading": True,
                    "api_key": "YOUR_ALPACA_API_KEY",
                    "secret_key": "YOUR_ALPACA_SECRET_KEY",
                    "base_url": "https://paper-api.alpaca.markets"
                },
                "interactive_brokers": {
                    "enabled": False,
                    "host": "127.0.0.1",
                    "port": 7497,
                    "client_id": 1
                },
                "crypto_exchange": {
                    "enabled": True,
                    "exchange": "binance",
                    "api_key": "YOUR_BINANCE_API_KEY",
                    "secret": "YOUR_BINANCE_SECRET",
                    "sandbox": True
                }
            },
            "risk_limits": {
                "max_daily_loss": 5000,
                "max_position_size": 10000,
                "max_order_value": 50000,
                "position_size_percent": 2.0
            },
            "execution_settings": {
                "default_order_type": "limit",
                "limit_offset_percent": 0.1,
                "max_slippage_percent": 0.5,
                "order_timeout_seconds": 300
            }
        }

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.logger.info(f"Loaded broker config from {config_file}")
        except FileNotFoundError:
            config = default_config
            # Save default config
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            self.logger.info(f"Created default broker config at {config_file}")

        return config

    async def connect_alpaca(self):
        """Connect to Alpaca broker"""
        try:
            if not ALPACA_AVAILABLE:
                self.logger.warning("Alpaca library not available")
                return False

            alpaca_config = self.config["brokers"]["alpaca"]
            if not alpaca_config["enabled"]:
                return False

            api_key = alpaca_config["api_key"]
            secret_key = alpaca_config["secret_key"]
            base_url = alpaca_config["base_url"]

            if api_key == "YOUR_ALPACA_API_KEY":
                self.logger.warning("Alpaca API keys not configured - using simulation mode")
                self.brokers_connected["alpaca"] = "simulation"
                return True

            # Initialize Alpaca API
            self.alpaca_api = tradeapi.REST(
                api_key,
                secret_key,
                base_url,
                api_version='v2'
            )

            # Test connection
            account = self.alpaca_api.get_account()
            self.logger.info(f"✅ Connected to Alpaca: {account.status}")
            self.brokers_connected["alpaca"] = "connected"

            return True

        except Exception as e:
            self.logger.error(f"Alpaca connection failed: {e}")
            self.brokers_connected["alpaca"] = "failed"
            return False

    async def connect_interactive_brokers(self):
        """Connect to Interactive Brokers"""
        try:
            if not IB_AVAILABLE:
                self.logger.warning("IB library not available")
                return False

            ib_config = self.config["brokers"]["interactive_brokers"]
            if not ib_config["enabled"]:
                return False

            self.ib_client = IB()
            await self.ib_client.connectAsync(
                ib_config["host"],
                ib_config["port"],
                clientId=ib_config["client_id"]
            )

            self.logger.info("✅ Connected to Interactive Brokers")
            self.brokers_connected["interactive_brokers"] = "connected"

            return True

        except Exception as e:
            self.logger.error(f"IB connection failed: {e}")
            self.brokers_connected["interactive_brokers"] = "failed"
            return False

    async def connect_crypto_exchange(self):
        """Connect to cryptocurrency exchange"""
        try:
            if not CRYPTO_AVAILABLE:
                self.logger.warning("CCXT library not available")
                return False

            crypto_config = self.config["brokers"]["crypto_exchange"]
            if not crypto_config["enabled"]:
                return False

            exchange_name = crypto_config["exchange"]
            api_key = crypto_config["api_key"]
            secret = crypto_config["secret"]
            sandbox = crypto_config["sandbox"]

            if api_key == "YOUR_BINANCE_API_KEY":
                self.logger.warning("Crypto API keys not configured - using simulation mode")
                self.brokers_connected["crypto"] = "simulation"
                return True

            # Initialize exchange
            exchange_class = getattr(ccxt, exchange_name)
            self.crypto_exchange = exchange_class({
                'apiKey': api_key,
                'secret': secret,
                'sandbox': sandbox,
                'enableRateLimit': True,
            })

            # Test connection
            balance = self.crypto_exchange.fetch_balance()
            self.logger.info(f"✅ Connected to {exchange_name} crypto exchange")
            self.brokers_connected["crypto"] = "connected"

            return True

        except Exception as e:
            self.logger.error(f"Crypto exchange connection failed: {e}")
            self.brokers_connected["crypto"] = "failed"
            return False

    async def initialize_connections(self):
        """Initialize all broker connections"""
        self.logger.info("🔗 Initializing broker connections...")

        connection_tasks = [
            self.connect_alpaca(),
            self.connect_interactive_brokers(),
            self.connect_crypto_exchange()
        ]

        results = await asyncio.gather(*connection_tasks, return_exceptions=True)

        connected_brokers = sum(1 for broker, status in self.brokers_connected.items()
                               if status in ["connected", "simulation"])

        self.logger.info(f"✅ {connected_brokers} brokers connected/available")
        return connected_brokers > 0

    def validate_order(self, order: TradingOrder) -> bool:
        """Validate order before execution"""
        try:
            # Check risk limits
            order_value = (order.price or 100) * order.quantity

            if order_value > self.max_position_size:
                self.logger.warning(f"Order exceeds max position size: ${order_value}")
                return False

            if self.daily_pnl < -self.max_daily_loss:
                self.logger.warning(f"Daily loss limit exceeded: ${self.daily_pnl}")
                return False

            # Check if market is open (simplified)
            now = datetime.now()
            if now.hour < 9 or now.hour > 16:  # Basic market hours check
                if "/USD" not in order.symbol:  # Allow crypto 24/7
                    self.logger.warning(f"Market closed for {order.symbol}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Order validation error: {e}")
            return False

    async def execute_order_alpaca(self, order: TradingOrder) -> bool:
        """Execute order through Alpaca"""
        try:
            if self.brokers_connected.get("alpaca") == "simulation":
                return await self.simulate_order_execution(order)

            if self.brokers_connected.get("alpaca") != "connected":
                return False

            # Prepare Alpaca order
            alpaca_order = {
                'symbol': order.symbol,
                'qty': order.quantity,
                'side': order.action.lower(),
                'type': order.order_type.value,
                'time_in_force': order.time_in_force
            }

            if order.order_type == OrderType.LIMIT:
                alpaca_order['limit_price'] = order.price

            # Submit order
            submitted_order = self.alpaca_api.submit_order(**alpaca_order)

            # Update order status
            order.status = OrderStatus.SUBMITTED
            order.order_id = submitted_order.id

            self.logger.info(f"📤 Alpaca order submitted: {order.symbol} {order.action} {order.quantity}")

            # Monitor order
            await self.monitor_alpaca_order(order, submitted_order.id)

            return True

        except Exception as e:
            self.logger.error(f"Alpaca execution error: {e}")
            order.status = OrderStatus.REJECTED
            return False

    async def simulate_order_execution(self, order: TradingOrder) -> bool:
        """Simulate order execution for testing"""
        try:
            # Simulate execution delay
            await asyncio.sleep(np.random.uniform(0.1, 2.0))

            # Simulate fill (90% success rate)
            if np.random.random() > 0.1:
                # Successful fill
                slippage = np.random.uniform(-0.01, 0.01)  # ±1% slippage
                fill_price = (order.price or 100) * (1 + slippage)

                order.status = OrderStatus.FILLED
                order.filled_price = fill_price
                order.filled_quantity = order.quantity
                order.commission = order.quantity * 0.005  # $0.005 per share

                # Update positions
                self.update_position(order)

                self.logger.info(f"✅ SIMULATED FILL: {order.symbol} {order.action} {order.quantity} @ ${fill_price:.2f}")

                return True
            else:
                # Rejected
                order.status = OrderStatus.REJECTED
                self.logger.warning(f"❌ SIMULATED REJECT: {order.symbol}")
                return False

        except Exception as e:
            self.logger.error(f"Simulation error: {e}")
            return False

    async def monitor_alpaca_order(self, order: TradingOrder, alpaca_order_id: str):
        """Monitor Alpaca order until filled or cancelled"""
        try:
            timeout = time.time() + 300  # 5 minute timeout

            while time.time() < timeout:
                alpaca_order = self.alpaca_api.get_order(alpaca_order_id)

                if alpaca_order.status == 'filled':
                    order.status = OrderStatus.FILLED
                    order.filled_price = float(alpaca_order.filled_avg_price)
                    order.filled_quantity = int(alpaca_order.filled_qty)

                    self.update_position(order)
                    self.logger.info(f"✅ FILLED: {order.symbol} @ ${order.filled_price:.2f}")
                    break

                elif alpaca_order.status in ['cancelled', 'rejected']:
                    order.status = OrderStatus.REJECTED
                    self.logger.warning(f"❌ ORDER REJECTED: {order.symbol}")
                    break

                await asyncio.sleep(1)

        except Exception as e:
            self.logger.error(f"Order monitoring error: {e}")

    def update_position(self, order: TradingOrder):
        """Update position after order fill"""
        symbol = order.symbol

        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                avg_price=0.0,
                market_value=0.0,
                unrealized_pnl=0.0
            )

        position = self.positions[symbol]

        if order.action == "BUY":
            # Calculate new average price
            total_cost = (position.quantity * position.avg_price) + (order.filled_quantity * order.filled_price)
            total_quantity = position.quantity + order.filled_quantity

            if total_quantity > 0:
                position.avg_price = total_cost / total_quantity

            position.quantity = total_quantity

        elif order.action == "SELL":
            # Calculate realized P&L
            if position.quantity > 0:
                pnl = (order.filled_price - position.avg_price) * order.filled_quantity
                position.realized_pnl += pnl
                self.daily_pnl += pnl

            position.quantity -= order.filled_quantity

        # Update tracking
        self.total_trades += 1
        self.total_commission += order.commission

        if order.status == OrderStatus.FILLED:
            self.successful_trades += 1

    async def process_trading_signal(self, signal: Dict):
        """Process trading signal from GPU systems"""
        try:
            # Create order from signal
            order = TradingOrder(
                order_id=f"GPU_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}",
                symbol=signal.get('symbol', 'SPY'),
                action=signal.get('action', 'BUY').upper(),
                quantity=signal.get('quantity', 100),
                order_type=OrderType(signal.get('order_type', 'limit').lower()),
                price=signal.get('price'),
                stop_price=signal.get('stop_price'),
                strategy_source=signal.get('strategy_source', 'GPU_GENERATED')
            )

            # Validate order
            if not self.validate_order(order):
                self.logger.warning(f"Order validation failed: {order.symbol}")
                return False

            # Add to pending orders
            self.pending_orders[order.order_id] = order

            # Execute order
            success = await self.execute_order_alpaca(order)

            if success:
                self.executed_orders[order.order_id] = order
                del self.pending_orders[order.order_id]

            return success

        except Exception as e:
            self.logger.error(f"Signal processing error: {e}")
            return False

    async def start_execution_engine(self):
        """Start the execution engine"""
        self.execution_active = True
        self.logger.info("🚀 Starting live execution engine...")

        # Initialize broker connections
        if not await self.initialize_connections():
            self.logger.error("No broker connections available")
            return

        # Process execution queue
        while self.execution_active:
            try:
                # Check for new signals (in real implementation, would receive from GPU systems)
                await asyncio.sleep(1)

                # Process any pending orders
                await self.process_pending_orders()

                # Update positions
                await self.update_positions_market_value()

            except Exception as e:
                self.logger.error(f"Execution engine error: {e}")
                await asyncio.sleep(5)

    async def process_pending_orders(self):
        """Process any pending orders"""
        for order_id, order in list(self.pending_orders.items()):
            if order.status == OrderStatus.PENDING:
                # Retry execution
                success = await self.execute_order_alpaca(order)
                if success:
                    self.executed_orders[order_id] = order
                    del self.pending_orders[order_id]

    async def update_positions_market_value(self):
        """Update market values of positions"""
        for symbol, position in self.positions.items():
            if position.quantity != 0:
                # In real implementation, get current market price
                current_price = 100 + np.random.uniform(-5, 5)  # Simulated
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity

    def get_execution_status(self) -> Dict:
        """Get execution engine status"""
        return {
            "execution_active": self.execution_active,
            "brokers_connected": self.brokers_connected,
            "pending_orders": len(self.pending_orders),
            "executed_orders": len(self.executed_orders),
            "total_trades": self.total_trades,
            "successful_trades": self.successful_trades,
            "success_rate": (self.successful_trades / max(self.total_trades, 1)) * 100,
            "daily_pnl": self.daily_pnl,
            "total_commission": self.total_commission,
            "positions": len([p for p in self.positions.values() if p.quantity != 0])
        }

    def stop_execution_engine(self):
        """Stop execution engine"""
        self.execution_active = False
        self.logger.info("🛑 Execution engine stopped")

    def get_system_status(self) -> Dict:
        """Get overall system status"""
        return {
            "broker_connections": {
                "alpaca": ALPACA_AVAILABLE and hasattr(self, 'alpaca_api'),
                "interactive_brokers": IB_AVAILABLE and hasattr(self, 'ib_client'),
                "crypto_exchanges": CRYPTO_AVAILABLE and hasattr(self, 'crypto_exchanges')
            },
            "execution_active": self.execution_active,
            "total_orders": len(self.pending_orders) + len(self.executed_orders),
            "successful_orders": self.successful_trades,
            "failed_orders": self.total_trades - self.successful_trades,
            "monitoring_active": self.execution_active
        }

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        total_cash = self.starting_cash
        total_equity = 0
        position_count = 0

        for position in self.positions.values():
            if position.quantity != 0:
                total_equity += position.market_value
                position_count += 1

        return {
            "total_cash": total_cash,
            "total_equity": total_equity,
            "total_value": total_cash + total_equity,
            "position_count": position_count,
            "daily_pnl": self.daily_pnl,
            "total_commission": self.total_commission
        }

    def validate_order(self, order_data: Dict) -> Dict:
        """Validate an order before execution"""
        warnings = []
        is_valid = True

        # Basic validation
        if not order_data.get('symbol'):
            warnings.append("Missing symbol")
            is_valid = False

        if not order_data.get('quantity') or order_data['quantity'] <= 0:
            warnings.append("Invalid quantity")
            is_valid = False

        if order_data.get('action') not in ['BUY', 'SELL']:
            warnings.append("Invalid action")
            is_valid = False

        # Risk checks
        if order_data.get('quantity', 0) > 1000:
            warnings.append("Large position size")

        return {
            "is_valid": is_valid,
            "risk_warnings": warnings
        }

    async def execute_order(self, order_data: Dict) -> Dict:
        """Execute an order and return result"""
        try:
            # Create order object
            order = TradingOrder(
                order_id=f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}",
                symbol=order_data.get('symbol', 'SPY'),
                action=order_data.get('action', 'BUY').upper(),
                quantity=order_data.get('quantity', 100),
                order_type=OrderType(order_data.get('order_type', 'market').lower()),
                price=order_data.get('limit_price'),
                strategy_source=order_data.get('strategy_id', 'TEST')
            )

            # For demo purposes, simulate execution
            success = await self.simulate_order_execution(order)

            return {
                "order_id": order.order_id,
                "status": order.status.value,
                "filled_price": getattr(order, 'filled_price', None),
                "commission": getattr(order, 'commission', 0)
            }

        except Exception as e:
            return {
                "order_id": "ERROR",
                "status": "failed",
                "error": str(e)
            }

async def demo_live_execution():
    """Demo the live execution engine"""
    print("="*80)
    print("LIVE BROKER EXECUTION ENGINE DEMO")
    print("Real-time order execution for autonomous trading")
    print("="*80)

    # Initialize engine
    engine = LiveBrokerExecutionEngine()

    print(f"\n🔗 Initializing broker connections...")
    connected = await engine.initialize_connections()

    if connected:
        print(f"✅ Broker connections established")

        # Demo trading signals
        demo_signals = [
            {"symbol": "AAPL", "action": "BUY", "quantity": 100, "price": 175.50, "strategy_source": "GPU_AI_AGENT"},
            {"symbol": "MSFT", "action": "BUY", "quantity": 50, "price": 350.25, "strategy_source": "GPU_PATTERNS"},
            {"symbol": "SPY", "action": "BUY", "quantity": 200, "price": 445.75, "strategy_source": "GPU_SCANNER"}
        ]

        print(f"\n📤 Processing {len(demo_signals)} demo trading signals...")

        for signal in demo_signals:
            success = await engine.process_trading_signal(signal)
            print(f"{'✅' if success else '❌'} {signal['symbol']} {signal['action']} {signal['quantity']}")

        # Show execution status
        status = engine.get_execution_status()
        print(f"\n📊 EXECUTION STATUS:")
        print(f"   Total trades: {status['total_trades']}")
        print(f"   Success rate: {status['success_rate']:.1f}%")
        print(f"   Daily P&L: ${status['daily_pnl']:.2f}")
        print(f"   Positions: {status['positions']}")

    else:
        print(f"⚠️ No broker connections - demo mode only")

    print(f"\n✅ Live execution engine ready for autonomous trading!")

if __name__ == "__main__":
    asyncio.run(demo_live_execution())