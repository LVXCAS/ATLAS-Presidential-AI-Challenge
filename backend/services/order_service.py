"""
Order Execution Service for Bloomberg Terminal
High-performance order management with Alpaca broker integration.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame, OrderSide, OrderType, TimeInForce

from core.config import get_settings
from core.database import DatabaseService
from core.redis_manager import get_redis_manager

logger = logging.getLogger(__name__)
settings = get_settings()


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING_NEW = "PENDING_NEW"
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class OrderRequest:
    """Order request structure."""
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    client_order_id: Optional[str] = None
    agent_name: Optional[str] = None
    strategy_name: Optional[str] = None


@dataclass
class Order:
    """Order tracking structure."""
    id: str
    client_order_id: str
    broker_order_id: Optional[str]
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: Optional[float]
    stop_price: Optional[float]
    time_in_force: str
    status: OrderStatus
    filled_quantity: int
    average_fill_price: Optional[float]
    commission: float
    created_at: datetime
    updated_at: datetime
    agent_name: Optional[str]
    strategy_name: Optional[str]
    rejection_reason: Optional[str] = None


@dataclass
class Trade:
    """Trade execution record."""
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    commission: float
    timestamp: datetime
    agent_name: Optional[str]
    strategy_name: Optional[str]


class OrderService:
    """High-performance order execution and management service."""
    
    def __init__(self):
        self.settings = settings
        self.redis_manager = get_redis_manager()
        self.orders: Dict[str, Order] = {}  # Active orders cache
        self.trades: List[Trade] = []
        self.order_callbacks: List[Callable] = []
        self.is_running = False
        self._tasks: List[asyncio.Task] = []
        
        # Performance metrics
        self.orders_placed = 0
        self.orders_filled = 0
        self.orders_rejected = 0
        self.total_commission = 0.0
        self.avg_execution_time = 0.0
        self._execution_times = []
        
        # Initialize Alpaca API
        if settings.alpaca.api_key and settings.alpaca.secret_key:
            self.alpaca_api = tradeapi.REST(
                settings.alpaca.api_key,
                settings.alpaca.secret_key,
                settings.alpaca.base_url
            )
            
            # Initialize trading stream for order updates
            self.trading_stream = tradeapi.Stream(
                settings.alpaca.api_key,
                settings.alpaca.secret_key,
                base_url=tradeapi.common.URL(settings.alpaca.base_url)
            )
        else:
            self.alpaca_api = None
            self.trading_stream = None
            logger.warning("Alpaca API credentials not configured")
    
    async def start_service(self) -> None:
        """Start the order execution service."""
        if self.is_running:
            return
        
        logger.info("Starting order execution service...")
        
        try:
            # Initialize Redis
            await self.redis_manager.initialize()
            
            # Load existing orders from database
            await self._load_existing_orders()
            
            # Start trading stream for order updates
            if self.trading_stream:
                await self._start_trading_stream()
            
            # Start background tasks
            self._tasks = [
                asyncio.create_task(self._monitor_orders()),
                asyncio.create_task(self._performance_monitor()),
                asyncio.create_task(self._risk_monitor())
            ]
            
            self.is_running = True
            logger.info("Order execution service started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start order service: {e}")
            await self.stop_service()
            raise
    
    async def stop_service(self) -> None:
        """Stop the order execution service."""
        logger.info("Stopping order execution service...")
        
        self.is_running = False
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Close trading stream
        if self.trading_stream:
            await self.trading_stream.stop_ws()
        
        logger.info("Order execution service stopped")
    
    async def place_order(self, order_request: OrderRequest) -> Optional[Order]:
        """Place a new order with risk checks."""
        try:
            start_time = time.time()
            
            # Generate order ID if not provided
            if not order_request.client_order_id:
                order_request.client_order_id = str(uuid.uuid4())
            
            # Pre-trade risk checks
            risk_check_result = await self._pre_trade_risk_check(order_request)
            if not risk_check_result["approved"]:
                logger.warning(f"Order rejected by risk check: {risk_check_result['reason']}")
                return await self._create_rejected_order(order_request, risk_check_result["reason"])
            
            # Create order object
            order = Order(
                id=str(uuid.uuid4()),
                client_order_id=order_request.client_order_id,
                broker_order_id=None,
                symbol=order_request.symbol,
                side=order_request.side,
                quantity=order_request.quantity,
                order_type=order_request.order_type,
                price=order_request.price,
                stop_price=order_request.stop_price,
                time_in_force=order_request.time_in_force,
                status=OrderStatus.PENDING_NEW,
                filled_quantity=0,
                average_fill_price=None,
                commission=0.0,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                agent_name=order_request.agent_name,
                strategy_name=order_request.strategy_name
            )
            
            # Store in cache and database
            self.orders[order.id] = order
            await self._store_order_in_db(order)
            
            # Submit to broker
            if self.alpaca_api:
                broker_order = await self._submit_to_alpaca(order_request)
                if broker_order:
                    order.broker_order_id = broker_order.id
                    order.status = OrderStatus.NEW
                else:
                    order.status = OrderStatus.REJECTED
                    order.rejection_reason = "Broker submission failed"
            else:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = "No broker connection available"
            
            # Update order
            order.updated_at = datetime.now(timezone.utc)
            await self._update_order_in_db(order)
            
            # Notify callbacks
            await self._notify_order_callbacks(order)
            
            # Update metrics
            execution_time = time.time() - start_time
            self._execution_times.append(execution_time)
            if len(self._execution_times) > 100:
                self._execution_times = self._execution_times[-100:]
            
            self.orders_placed += 1
            self.avg_execution_time = sum(self._execution_times) / len(self._execution_times)
            
            logger.info(f"Order placed: {order.symbol} {order.side.value} {order.quantity} @ {order.price or 'MKT'}")
            return order
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            self.orders_rejected += 1
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            if order_id not in self.orders:
                logger.warning(f"Cannot cancel unknown order: {order_id}")
                return False
            
            order = self.orders[order_id]
            
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                logger.warning(f"Cannot cancel order in status {order.status.value}")
                return False
            
            # Cancel with broker
            if order.broker_order_id and self.alpaca_api:
                try:
                    self.alpaca_api.cancel_order(order.broker_order_id)
                except Exception as e:
                    logger.error(f"Failed to cancel order with broker: {e}")
                    return False
            
            # Update order status
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now(timezone.utc)
            
            # Update database
            await self._update_order_in_db(order)
            
            # Notify callbacks
            await self._notify_order_callbacks(order)
            
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def cancel_all_orders(self) -> int:
        """Cancel all active orders (emergency function)."""
        logger.critical("CANCELLING ALL ORDERS - EMERGENCY STOP")
        
        cancelled_count = 0
        
        for order in list(self.orders.values()):
            if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING_NEW]:
                if await self.cancel_order(order.id):
                    cancelled_count += 1
        
        logger.critical(f"Emergency cancellation complete: {cancelled_count} orders cancelled")
        return cancelled_count
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    async def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol."""
        return [order for order in self.orders.values() if order.symbol == symbol]
    
    async def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        return [
            order for order in self.orders.values()
            if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING_NEW]
        ]
    
    async def get_order_history(self, limit: int = 100) -> List[Order]:
        """Get order history from database."""
        try:
            query = """
            SELECT * FROM orders 
            ORDER BY timestamp DESC 
            LIMIT :limit
            """
            rows = await DatabaseService.execute_query(query, {"limit": limit})
            
            orders = []
            for row in rows:
                # Convert row to Order object
                order = self._row_to_order(row)
                orders.append(order)
            
            return orders
            
        except Exception as e:
            logger.error(f"Error getting order history: {e}")
            return []
    
    # Private methods
    
    async def _pre_trade_risk_check(self, order_request: OrderRequest) -> Dict[str, Any]:
        """Perform pre-trade risk checks."""
        try:
            # Get current portfolio metrics
            redis_client = await self.redis_manager.get_client()
            portfolio_data = await redis_client.hgetall("portfolio:metrics")
            
            if not portfolio_data:
                return {"approved": False, "reason": "Portfolio data not available"}
            
            current_value = float(portfolio_data.get("total_value", 0))
            buying_power = float(portfolio_data.get("buying_power", 0))
            
            # Calculate order value
            if order_request.order_type == OrderType.MARKET:
                # Estimate using last price
                last_price = await self._get_last_price(order_request.symbol)
                if not last_price:
                    return {"approved": False, "reason": f"No price data for {order_request.symbol}"}
                order_value = last_price * order_request.quantity
            elif order_request.price:
                order_value = order_request.price * order_request.quantity
            else:
                return {"approved": False, "reason": "Cannot determine order value"}
            
            # Check buying power
            if order_request.side == OrderSide.BUY and order_value > buying_power:
                return {"approved": False, "reason": f"Insufficient buying power: ${order_value:.2f} > ${buying_power:.2f}"}
            
            # Check position size limits
            max_position_value = settings.trading.max_position_size
            if order_value > max_position_value:
                return {"approved": False, "reason": f"Order value exceeds position limit: ${order_value:.2f} > ${max_position_value:.2f}"}
            
            # Check portfolio concentration
            if current_value > 0:
                position_percentage = order_value / current_value
                if position_percentage > 0.20:  # Max 20% in single position
                    return {"approved": False, "reason": f"Position would exceed 20% of portfolio"}
            
            # All checks passed
            return {"approved": True, "reason": "Risk checks passed"}
            
        except Exception as e:
            logger.error(f"Error in pre-trade risk check: {e}")
            return {"approved": False, "reason": f"Risk check failed: {e}"}
    
    async def _get_last_price(self, symbol: str) -> Optional[float]:
        """Get last price for a symbol."""
        try:
            redis_client = await self.redis_manager.get_client()
            price_data = await redis_client.hget(f"price:{symbol}", "price")
            return float(price_data) if price_data else None
        except:
            return None
    
    async def _submit_to_alpaca(self, order_request: OrderRequest) -> Optional[Any]:
        """Submit order to Alpaca broker."""
        try:
            # Convert order types
            alpaca_side = "buy" if order_request.side == OrderSide.BUY else "sell"
            alpaca_type = order_request.order_type.value.lower()
            
            # Submit order
            order = self.alpaca_api.submit_order(
                symbol=order_request.symbol,
                qty=order_request.quantity,
                side=alpaca_side,
                type=alpaca_type,
                time_in_force=order_request.time_in_force,
                limit_price=order_request.price,
                stop_price=order_request.stop_price,
                client_order_id=order_request.client_order_id
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Error submitting order to Alpaca: {e}")
            return None
    
    async def _execute_order(self, order: Order, quantity: int, price: float) -> None:
        """Execute order (fill or partial fill)."""
        try:
            # Create trade record
            trade = Trade(
                id=str(uuid.uuid4()),
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                quantity=quantity,
                price=price,
                commission=0.0,  # Assuming no commission for now
                timestamp=datetime.now(timezone.utc),
                agent_name=order.agent_name,
                strategy_name=order.strategy_name
            )
            
            # Update order
            order.filled_quantity += quantity
            order.average_fill_price = price  # Simplified - should be weighted average
            order.commission += trade.commission
            
            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
                self.orders_filled += 1
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
            
            order.updated_at = datetime.now(timezone.utc)
            
            # Store trade in database
            await self._store_trade_in_db(trade)
            
            # Update order in database
            await self._update_order_in_db(order)
            
            # Add to trades list
            self.trades.append(trade)
            
            # Notify callbacks
            await self._notify_order_callbacks(order)
            await self._notify_trade_callbacks(trade)
            
            logger.info(f"Trade executed: {trade.symbol} {trade.side.value} {trade.quantity} @ ${trade.price:.2f}")
            
        except Exception as e:
            logger.error(f"Error executing order: {e}")
    
    async def _create_rejected_order(self, order_request: OrderRequest, reason: str) -> Order:
        """Create a rejected order record."""
        order = Order(
            id=str(uuid.uuid4()),
            client_order_id=order_request.client_order_id or str(uuid.uuid4()),
            broker_order_id=None,
            symbol=order_request.symbol,
            side=order_request.side,
            quantity=order_request.quantity,
            order_type=order_request.order_type,
            price=order_request.price,
            stop_price=order_request.stop_price,
            time_in_force=order_request.time_in_force,
            status=OrderStatus.REJECTED,
            filled_quantity=0,
            average_fill_price=None,
            commission=0.0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            agent_name=order_request.agent_name,
            strategy_name=order_request.strategy_name,
            rejection_reason=reason
        )
        
        self.orders[order.id] = order
        await self._store_order_in_db(order)
        self.orders_rejected += 1
        
        return order
    
    async def _load_existing_orders(self) -> None:
        """Load existing active orders from database."""
        try:
            query = """
            SELECT * FROM orders 
            WHERE status IN ('NEW', 'PARTIALLY_FILLED', 'PENDING_NEW')
            ORDER BY timestamp DESC
            """
            rows = await DatabaseService.execute_query(query)
            
            for row in rows:
                order = self._row_to_order(row)
                self.orders[order.id] = order
            
            logger.info(f"Loaded {len(self.orders)} existing orders from database")
            
        except Exception as e:
            logger.error(f"Error loading existing orders: {e}")
    
    def _row_to_order(self, row) -> Order:
        """Convert database row to Order object."""
        return Order(
            id=str(row[0]),  # Assuming first column is id
            client_order_id=row[1],
            broker_order_id=row[2],
            symbol=row[3],
            side=OrderSide(row[5]),
            quantity=row[6],
            order_type=OrderType(row[4]),
            price=float(row[7]) if row[7] else None,
            stop_price=float(row[8]) if row[8] else None,
            time_in_force=row[9],
            status=OrderStatus(row[10]),
            filled_quantity=row[11],
            average_fill_price=float(row[12]) if row[12] else None,
            commission=float(row[13]),
            created_at=row[14],
            updated_at=row[14],
            agent_name=row[15],
            strategy_name=row[16],
            rejection_reason=row[17]
        )
    
    async def _store_order_in_db(self, order: Order) -> None:
        """Store order in database."""
        try:
            query = """
            INSERT INTO orders (
                id, client_order_id, broker_order_id, symbol, order_type, side, quantity,
                price, stop_price, time_in_force, status, filled_quantity, average_fill_price,
                commission, timestamp, agent_name, strategy_name, rejection_reason
            ) VALUES (
                :id, :client_order_id, :broker_order_id, :symbol, :order_type, :side, :quantity,
                :price, :stop_price, :time_in_force, :status, :filled_quantity, :average_fill_price,
                :commission, :timestamp, :agent_name, :strategy_name, :rejection_reason
            )
            """
            
            await DatabaseService.execute_insert(query, {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "broker_order_id": order.broker_order_id,
                "symbol": order.symbol,
                "order_type": order.order_type.value,
                "side": order.side.value,
                "quantity": order.quantity,
                "price": order.price,
                "stop_price": order.stop_price,
                "time_in_force": order.time_in_force,
                "status": order.status.value,
                "filled_quantity": order.filled_quantity,
                "average_fill_price": order.average_fill_price,
                "commission": order.commission,
                "timestamp": order.created_at,
                "agent_name": order.agent_name,
                "strategy_name": order.strategy_name,
                "rejection_reason": order.rejection_reason
            })
            
        except Exception as e:
            logger.error(f"Error storing order in database: {e}")
    
    async def _update_order_in_db(self, order: Order) -> None:
        """Update order in database."""
        try:
            query = """
            UPDATE orders SET
                status = :status,
                filled_quantity = :filled_quantity,
                average_fill_price = :average_fill_price,
                commission = :commission,
                broker_order_id = :broker_order_id,
                rejection_reason = :rejection_reason
            WHERE id = :id
            """
            
            await DatabaseService.execute_update(query, {
                "id": order.id,
                "status": order.status.value,
                "filled_quantity": order.filled_quantity,
                "average_fill_price": order.average_fill_price,
                "commission": order.commission,
                "broker_order_id": order.broker_order_id,
                "rejection_reason": order.rejection_reason
            })
            
        except Exception as e:
            logger.error(f"Error updating order in database: {e}")
    
    async def _store_trade_in_db(self, trade: Trade) -> None:
        """Store trade in database."""
        try:
            query = """
            INSERT INTO trades (
                id, order_id, symbol, side, quantity, price, commission,
                timestamp, agent_name, strategy_name
            ) VALUES (
                :id, :order_id, :symbol, :side, :quantity, :price, :commission,
                :timestamp, :agent_name, :strategy_name
            )
            """
            
            await DatabaseService.execute_insert(query, {
                "id": trade.id,
                "order_id": trade.order_id,
                "symbol": trade.symbol,
                "side": trade.side.value,
                "quantity": trade.quantity,
                "price": trade.price,
                "commission": trade.commission,
                "timestamp": trade.timestamp,
                "agent_name": trade.agent_name,
                "strategy_name": trade.strategy_name
            })
            
        except Exception as e:
            logger.error(f"Error storing trade in database: {e}")
    
    async def _start_trading_stream(self) -> None:
        """Start Alpaca trading stream for order updates."""
        try:
            # Subscribe to trade updates
            self.trading_stream.subscribe_trade_updates(self._handle_trade_update)
            
            # Start the stream in background
            asyncio.create_task(self.trading_stream.run())
            
        except Exception as e:
            logger.error(f"Error starting trading stream: {e}")
    
    async def _handle_trade_update(self, trade_update) -> None:
        """Handle real-time trade updates from Alpaca."""
        try:
            client_order_id = trade_update.order.get("client_order_id")
            if not client_order_id:
                return
            
            # Find the order
            order = None
            for o in self.orders.values():
                if o.client_order_id == client_order_id:
                    order = o
                    break
            
            if not order:
                logger.warning(f"Received update for unknown order: {client_order_id}")
                return
            
            # Update order based on trade update
            order.broker_order_id = trade_update.order.get("id")
            order.status = OrderStatus(trade_update.order.get("status", "NEW"))
            order.filled_quantity = int(trade_update.order.get("filled_qty", 0))
            order.updated_at = datetime.now(timezone.utc)
            
            if trade_update.order.get("filled_avg_price"):
                order.average_fill_price = float(trade_update.order.get("filled_avg_price"))
            
            # Update in database
            await self._update_order_in_db(order)
            
            # Notify callbacks
            await self._notify_order_callbacks(order)
            
        except Exception as e:
            logger.error(f"Error handling trade update: {e}")
    
    async def _monitor_orders(self) -> None:
        """Monitor orders for timeouts and updates."""
        while self.is_running:
            try:
                current_time = datetime.now(timezone.utc)
                
                for order in list(self.orders.values()):
                    # Check for expired orders
                    if order.time_in_force == "DAY" and order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                        hours_old = (current_time - order.created_at).total_seconds() / 3600
                        if hours_old > 24:  # Orders expire after 24 hours
                            order.status = OrderStatus.EXPIRED
                            await self._update_order_in_db(order)
                            await self._notify_order_callbacks(order)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in order monitor: {e}")
    
    async def _performance_monitor(self) -> None:
        """Monitor order execution performance."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                fill_rate = (self.orders_filled / self.orders_placed * 100) if self.orders_placed > 0 else 0
                
                logger.info(
                    f"Order performance: {self.orders_placed} placed, "
                    f"{self.orders_filled} filled ({fill_rate:.1f}%), "
                    f"{self.orders_rejected} rejected, "
                    f"avg execution: {self.avg_execution_time*1000:.1f}ms"
                )
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
    
    async def _risk_monitor(self) -> None:
        """Monitor for risk events related to orders."""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check for excessive rejected orders
                if self.orders_placed > 0:
                    rejection_rate = self.orders_rejected / self.orders_placed
                    if rejection_rate > 0.1:  # More than 10% rejection rate
                        logger.warning(f"High order rejection rate: {rejection_rate:.1%}")
                
                # Check for orders stuck in pending state
                current_time = datetime.now(timezone.utc)
                for order in self.orders.values():
                    if order.status == OrderStatus.PENDING_NEW:
                        minutes_pending = (current_time - order.created_at).total_seconds() / 60
                        if minutes_pending > 5:  # Stuck for more than 5 minutes
                            logger.error(f"Order stuck in PENDING_NEW state: {order.id}")
                
            except Exception as e:
                logger.error(f"Error in risk monitor: {e}")
    
    async def _notify_order_callbacks(self, order: Order) -> None:
        """Notify all registered order callbacks."""
        for callback in self.order_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order)
                else:
                    callback(order)
            except Exception as e:
                logger.error(f"Error in order callback: {e}")
    
    async def _notify_trade_callbacks(self, trade: Trade) -> None:
        """Notify trade callbacks (placeholder for future expansion)."""
        # Placeholder for trade-specific callbacks
        pass
    
    # Public API methods
    
    def add_order_callback(self, callback: Callable) -> None:
        """Add callback for order updates."""
        self.order_callbacks.append(callback)
    
    def remove_order_callback(self, callback: Callable) -> None:
        """Remove order callback."""
        if callback in self.order_callbacks:
            self.order_callbacks.remove(callback)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get order execution performance metrics."""
        fill_rate = (self.orders_filled / self.orders_placed * 100) if self.orders_placed > 0 else 0
        rejection_rate = (self.orders_rejected / self.orders_placed * 100) if self.orders_placed > 0 else 0
        
        return {
            "orders_placed": self.orders_placed,
            "orders_filled": self.orders_filled,
            "orders_rejected": self.orders_rejected,
            "fill_rate_percent": fill_rate,
            "rejection_rate_percent": rejection_rate,
            "avg_execution_time_ms": self.avg_execution_time * 1000,
            "total_commission": self.total_commission,
            "active_orders": len([o for o in self.orders.values() if o.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]])
        }
    
    def is_healthy(self) -> bool:
        """Check if the order service is healthy."""
        return self.is_running