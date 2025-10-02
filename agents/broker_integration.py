"""
Broker Integration Agent - Alpaca API Integration for Order Execution

This module implements comprehensive broker integration with Alpaca API for:
- Order lifecycle management (submit, fill, cancel)
- Position reconciliation and trade reporting
- Error handling for API failures and rejections
- Real-time order status monitoring
"""

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
import json
import time
import random

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError, TimeFrame
from alpaca_trade_api.entity import Order as AlpacaOrder, Position as AlpacaPosition
import pandas as pd
from pydantic import BaseModel, Field, validator

from config.settings import get_settings
from config.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()


class OrderSide(str, Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(str, Enum):
    """Order status enumeration"""
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
    SUBMITTED = "submitted" # Added for mock orders


class TimeInForce(str, Enum):
    """Time in force enumeration"""
    DAY = "day"
    GTC = "gtc"  # Good Till Canceled
    OPG = "opg"  # At the Opening
    CLS = "cls"  # At the Close
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


@dataclass
class OrderRequest:
    """Order request data structure"""
    symbol: str
    qty: Union[int, float, Decimal]
    side: OrderSide
    type: OrderType
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: Optional[Union[float, Decimal]] = None
    stop_price: Optional[Union[float, Decimal]] = None
    trail_price: Optional[Union[float, Decimal]] = None
    trail_percent: Optional[Union[float, Decimal]] = None
    extended_hours: bool = False
    client_order_id: Optional[str] = None
    order_class: Optional[str] = None  # simple, bracket, oco, oto
    take_profit: Optional[Dict[str, Any]] = None
    stop_loss: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate order request after initialization"""
        if self.type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.limit_price is None:
            raise ValueError(f"limit_price required for {self.type} orders")
        
        if self.type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError(f"stop_price required for {self.type} orders")
        
        if self.type == OrderType.TRAILING_STOP and not (self.trail_price or self.trail_percent):
            raise ValueError("trail_price or trail_percent required for trailing_stop orders")


@dataclass
class OrderResponse:
    """Order response data structure"""
    id: str
    client_order_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    submitted_at: Optional[datetime]
    filled_at: Optional[datetime]
    expired_at: Optional[datetime]
    canceled_at: Optional[datetime]
    failed_at: Optional[datetime]
    replaced_at: Optional[datetime]
    symbol: str
    asset_id: str
    asset_class: str
    qty: Decimal
    filled_qty: Decimal
    type: OrderType
    side: OrderSide
    time_in_force: TimeInForce
    limit_price: Optional[Decimal]
    stop_price: Optional[Decimal]
    status: OrderStatus
    extended_hours: bool
    legs: Optional[List[Dict[str, Any]]] = None
    trail_price: Optional[Decimal] = None
    trail_percent: Optional[Decimal] = None
    hwm: Optional[Decimal] = None  # High Water Mark for trailing stops
    quantity: Optional[Decimal] = None # Added for mock orders
    price: Optional[Decimal] = None # Added for mock orders
    filled_price: Optional[Decimal] = None # Added for mock orders
    filled_quantity: Optional[Decimal] = None # Added for mock orders
    commission: Optional[Decimal] = None # Added for mock orders
    slippage: Optional[Decimal] = None # Added for mock orders
    metadata: Optional[Dict[str, Any]] = None # Added for mock orders

    @classmethod
    def from_alpaca_order(cls, alpaca_order: AlpacaOrder) -> 'OrderResponse':
        """Create OrderResponse from Alpaca Order object"""
        return cls(
            id=alpaca_order.id,
            client_order_id=alpaca_order.client_order_id,
            created_at=alpaca_order.created_at,
            updated_at=alpaca_order.updated_at,
            submitted_at=alpaca_order.submitted_at,
            filled_at=alpaca_order.filled_at,
            expired_at=alpaca_order.expired_at,
            canceled_at=alpaca_order.canceled_at,
            failed_at=alpaca_order.failed_at,
            replaced_at=alpaca_order.replaced_at,
            symbol=alpaca_order.symbol,
            asset_id=alpaca_order.asset_id,
            asset_class=alpaca_order.asset_class,
            qty=Decimal(str(alpaca_order.qty)),
            filled_qty=Decimal(str(alpaca_order.filled_qty)),
            type=OrderType(alpaca_order.order_type),
            side=OrderSide(alpaca_order.side),
            time_in_force=TimeInForce(alpaca_order.time_in_force),
            limit_price=Decimal(str(alpaca_order.limit_price)) if alpaca_order.limit_price else None,
            stop_price=Decimal(str(alpaca_order.stop_price)) if alpaca_order.stop_price else None,
            status=OrderStatus(alpaca_order.status),
            extended_hours=alpaca_order.extended_hours,
            legs=alpaca_order.legs,
            trail_price=Decimal(str(alpaca_order.trail_price)) if alpaca_order.trail_price else None,
            trail_percent=Decimal(str(alpaca_order.trail_percent)) if alpaca_order.trail_percent else None,
            hwm=Decimal(str(alpaca_order.hwm)) if alpaca_order.hwm else None,
            quantity=Decimal(str(alpaca_order.qty)),
            price=Decimal(str(alpaca_order.limit_price)) if alpaca_order.limit_price else None,
            filled_price=Decimal(str(alpaca_order.filled_avg_price)) if hasattr(alpaca_order, 'filled_avg_price') and alpaca_order.filled_avg_price else None,
            filled_quantity=Decimal(str(alpaca_order.filled_qty)) if hasattr(alpaca_order, 'filled_qty') and alpaca_order.filled_qty else None,
            commission=Decimal(str(alpaca_order.commission)) if hasattr(alpaca_order, 'commission') and alpaca_order.commission else None,
            slippage=None,  # Alpaca doesn't provide slippage directly
            metadata=getattr(alpaca_order, 'metadata', {})
        )


@dataclass
class PositionInfo:
    """Position information data structure"""
    asset_id: str
    symbol: str
    exchange: str
    asset_class: str
    avg_entry_price: Decimal
    qty: Decimal
    side: str  # long or short
    market_value: Decimal
    cost_basis: Decimal
    unrealized_pl: Decimal
    unrealized_plpc: Decimal
    unrealized_intraday_pl: Decimal
    unrealized_intraday_plpc: Decimal
    current_price: Optional[Decimal]
    lastday_price: Optional[Decimal]
    change_today: Optional[Decimal]
    
    @classmethod
    def from_alpaca_position(cls, alpaca_position: AlpacaPosition) -> 'PositionInfo':
        """Create PositionInfo from Alpaca Position object"""
        return cls(
            asset_id=alpaca_position.asset_id,
            symbol=alpaca_position.symbol,
            exchange=alpaca_position.exchange,
            asset_class=alpaca_position.asset_class,
            avg_entry_price=Decimal(str(alpaca_position.avg_entry_price)),
            qty=Decimal(str(alpaca_position.qty)),
            side=alpaca_position.side,
            market_value=Decimal(str(alpaca_position.market_value)),
            cost_basis=Decimal(str(alpaca_position.cost_basis)),
            unrealized_pl=Decimal(str(alpaca_position.unrealized_pl)),
            unrealized_plpc=Decimal(str(alpaca_position.unrealized_plpc)),
            unrealized_intraday_pl=Decimal(str(alpaca_position.unrealized_intraday_pl)),
            unrealized_intraday_plpc=Decimal(str(alpaca_position.unrealized_intraday_plpc)),
            current_price=Decimal(str(alpaca_position.current_price)) if alpaca_position.current_price else None,
            lastday_price=Decimal(str(alpaca_position.lastday_price)) if alpaca_position.lastday_price else None,
            change_today=Decimal(str(alpaca_position.change_today)) if alpaca_position.change_today else None
        )


@dataclass
class TradeReport:
    """Trade execution report"""
    order_id: str
    symbol: str
    side: OrderSide
    qty: Decimal
    price: Decimal
    timestamp: datetime
    commission: Decimal = Decimal('0')
    fees: Decimal = Decimal('0')
    execution_id: Optional[str] = None
    liquidity_flag: Optional[str] = None  # Added, Removed, etc.


@dataclass
class BrokerError:
    """Broker error information"""
    error_code: Optional[str]
    error_message: str
    timestamp: datetime
    order_id: Optional[str] = None
    symbol: Optional[str] = None
    retry_count: int = 0
    is_retryable: bool = False


class AlpacaBrokerIntegration:
    """
    Alpaca Broker Integration for order execution and portfolio management
    
    Provides comprehensive integration with Alpaca API including:
    - Order lifecycle management
    - Position reconciliation
    - Trade reporting
    - Error handling and retry logic
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 secret_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 paper_trading: bool = True):
        """
        Initialize Alpaca broker integration
        
        Args:
            api_key: Alpaca API key (defaults to settings)
            secret_key: Alpaca secret key (defaults to settings)
            base_url: Alpaca base URL (defaults to settings)
            paper_trading: Whether to use paper trading (default: True)
        """
        self.api_key = api_key or settings.ALPACA_API_KEY
        self.secret_key = secret_key or settings.ALPACA_SECRET_KEY
        self.base_url = base_url or (
            settings.ALPACA_PAPER_BASE_URL if paper_trading 
            else settings.ALPACA_LIVE_BASE_URL
        )
        self.paper_trading = paper_trading
        
        # Initialize Alpaca API client
        if paper_trading and (not self.api_key or not self.secret_key):
            # Create mock API client for paper trading without real credentials
            logger.info("Paper trading mode detected - using mock API client")
            self.api = None
        else:
            try:
                self.api = tradeapi.REST(
                    key_id=self.api_key,
                    secret_key=self.secret_key,
                    base_url=self.base_url,
                    api_version='v2'
                )
            except Exception as e:
                if paper_trading:
                    logger.warning(f"Failed to initialize Alpaca API for paper trading: {e}")
                    logger.info("Continuing with mock API client for paper trading")
                    self.api = None
                else:
                    raise e
        
        # Order tracking
        self.active_orders: Dict[str, OrderResponse] = {}
        self.completed_orders: Dict[str, OrderResponse] = {}
        self.error_log: List[BrokerError] = []
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        
        logger.info(f"Initialized Alpaca broker integration (paper_trading={paper_trading})")
    
    async def submit_order(self, order_request: OrderRequest) -> OrderResponse:
        """
        Submit an order to Alpaca
        
        Args:
            order_request: Order request details
            
        Returns:
            OrderResponse: Order response with execution details
            
        Raises:
            BrokerError: If order submission fails after retries
        """
        # Handle paper trading mode without real API
        if self.paper_trading and self.api is None:
            logger.info(f"Paper trading mode - simulating order submission: {order_request.symbol} {order_request.side} {order_request.qty}")
            
            # Create mock order response for paper trading
            mock_order_id = f"PAPER_{order_request.symbol}_{int(time.time())}_{random.randint(1000, 9999)}"
            current_time = datetime.now(timezone.utc)
            
            order_response = OrderResponse(
                id=mock_order_id,
                client_order_id=order_request.client_order_id,
                created_at=current_time,
                updated_at=current_time,
                submitted_at=current_time,
                filled_at=None,
                expired_at=None,
                canceled_at=None,
                failed_at=None,
                replaced_at=None,
                symbol=order_request.symbol,
                asset_id=f"mock_{order_request.symbol}",
                asset_class="equity",
                qty=Decimal(str(order_request.qty)),
                filled_qty=Decimal('0'),
                type=order_request.type,
                side=order_request.side,
                time_in_force=order_request.time_in_force,
                limit_price=Decimal(str(order_request.limit_price)) if order_request.limit_price else None,
                stop_price=Decimal(str(order_request.stop_price)) if order_request.stop_price else None,
                status=OrderStatus.SUBMITTED,
                extended_hours=order_request.extended_hours,
                legs=None,
                trail_price=Decimal(str(order_request.trail_price)) if order_request.trail_price else None,
                trail_percent=Decimal(str(order_request.trail_percent)) if order_request.trail_percent else None,
                hwm=None,
                quantity=Decimal(str(order_request.qty)),
                price=Decimal(str(order_request.limit_price)) if order_request.limit_price else None,
                filled_price=None,
                filled_quantity=Decimal('0'),
                commission=Decimal('0'),
                slippage=Decimal('0'),
                metadata={'paper_trading': True, 'mock_order': True}
            )
            
            # Track the order
            self.active_orders[mock_order_id] = order_response
            
            logger.info(f"Mock order submitted successfully for paper trading: {mock_order_id}")
            return order_response
        
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                logger.info(f"Submitting order: {order_request.symbol} {order_request.side} {order_request.qty}")
                
                # Prepare order parameters
                order_params = {
                    'symbol': order_request.symbol,
                    'qty': str(order_request.qty),
                    'side': order_request.side.value,
                    'type': order_request.type.value,
                    'time_in_force': order_request.time_in_force.value,
                    'extended_hours': order_request.extended_hours
                }
                
                # Add optional parameters
                if order_request.limit_price is not None:
                    order_params['limit_price'] = str(order_request.limit_price)
                
                if order_request.stop_price is not None:
                    order_params['stop_price'] = str(order_request.stop_price)
                
                if order_request.trail_price is not None:
                    order_params['trail_price'] = str(order_request.trail_price)
                
                if order_request.trail_percent is not None:
                    order_params['trail_percent'] = str(order_request.trail_percent)
                
                if order_request.client_order_id is not None:
                    order_params['client_order_id'] = order_request.client_order_id
                
                if order_request.order_class is not None:
                    order_params['order_class'] = order_request.order_class
                
                if order_request.take_profit is not None:
                    order_params['take_profit'] = order_request.take_profit
                
                if order_request.stop_loss is not None:
                    order_params['stop_loss'] = order_request.stop_loss
                
                # Submit order to Alpaca
                alpaca_order = self.api.submit_order(**order_params)
                
                # Convert to our order response format
                order_response = OrderResponse.from_alpaca_order(alpaca_order)
                
                # Track the order
                self.active_orders[order_response.id] = order_response
                
                logger.info(f"Order submitted successfully: {order_response.id}")
                return order_response
                
            except APIError as e:
                last_error = e
                retry_count += 1
                
                # Log the error
                broker_error = BrokerError(
                    error_code=str(e.code) if hasattr(e, 'code') else None,
                    error_message=str(e),
                    timestamp=datetime.now(timezone.utc),
                    symbol=order_request.symbol,
                    retry_count=retry_count,
                    is_retryable=self._is_retryable_error(e)
                )
                self.error_log.append(broker_error)
                
                logger.warning(f"Order submission failed (attempt {retry_count}): {e}")
                
                # Check if error is retryable
                if not self._is_retryable_error(e) or retry_count > self.max_retries:
                    break
                
                # Wait before retry
                await asyncio.sleep(self.retry_delay * retry_count)
            
            except Exception as e:
                last_error = e
                broker_error = BrokerError(
                    error_code=None,
                    error_message=str(e),
                    timestamp=datetime.now(timezone.utc),
                    symbol=order_request.symbol,
                    retry_count=retry_count,
                    is_retryable=False
                )
                self.error_log.append(broker_error)
                logger.error(f"Unexpected error submitting order: {e}")
                break
        
        # If we get here, all retries failed
        raise Exception(f"Order submission failed after {retry_count} attempts: {last_error}")
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            bool: True if cancellation successful, False otherwise
        """
        try:
            logger.info(f"Canceling order: {order_id}")
            
            # Cancel order via Alpaca API
            self.api.cancel_order(order_id)
            
            # Update order status in our tracking
            if order_id in self.active_orders:
                self.active_orders[order_id].status = OrderStatus.CANCELED
                self.active_orders[order_id].canceled_at = datetime.now(timezone.utc)
            
            logger.info(f"Order canceled successfully: {order_id}")
            return True
            
        except APIError as e:
            broker_error = BrokerError(
                error_code=str(e.code) if hasattr(e, 'code') else None,
                error_message=str(e),
                timestamp=datetime.now(timezone.utc),
                order_id=order_id,
                is_retryable=False
            )
            self.error_log.append(broker_error)
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
        
        except Exception as e:
            broker_error = BrokerError(
                error_code=None,
                error_message=str(e),
                timestamp=datetime.now(timezone.utc),
                order_id=order_id,
                is_retryable=False
            )
            self.error_log.append(broker_error)
            logger.error(f"Unexpected error canceling order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[OrderResponse]:
        """
        Get current order status
        
        Args:
            order_id: Order ID to check
            
        Returns:
            OrderResponse: Current order status or None if not found
        """
        try:
            # Get order from Alpaca API
            alpaca_order = self.api.get_order(order_id)
            order_response = OrderResponse.from_alpaca_order(alpaca_order)
            
            # Update our tracking
            if order_response.status in [OrderStatus.FILLED, OrderStatus.CANCELED, 
                                       OrderStatus.EXPIRED, OrderStatus.REJECTED]:
                # Move to completed orders
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
                self.completed_orders[order_id] = order_response
            else:
                # Update active orders
                self.active_orders[order_id] = order_response
            
            return order_response
            
        except APIError as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error getting order status for {order_id}: {e}")
            return None

    async def get_order(self, order_id: str) -> Optional[OrderResponse]:
        """
        Get order by ID (alias for get_order_status for compatibility)

        Args:
            order_id: Order ID to retrieve

        Returns:
            OrderResponse: Order details or None if not found
        """
        return await self.get_order_status(order_id)

    async def get_all_orders(self, 
                           status: Optional[str] = None,
                           limit: int = 100,
                           after: Optional[datetime] = None,
                           until: Optional[datetime] = None,
                           direction: str = 'desc',
                           nested: bool = True) -> List[OrderResponse]:
        """
        Get all orders with optional filtering
        
        Args:
            status: Filter by order status
            limit: Maximum number of orders to return
            after: Filter orders after this timestamp
            until: Filter orders until this timestamp
            direction: Sort direction ('asc' or 'desc')
            nested: Include nested multi-leg orders
            
        Returns:
            List[OrderResponse]: List of orders
        """
        try:
            # Get orders from Alpaca API
            alpaca_orders = self.api.list_orders(
                status=status,
                limit=limit,
                after=after,
                until=until,
                direction=direction,
                nested=nested
            )
            
            # Convert to our format
            orders = [OrderResponse.from_alpaca_order(order) for order in alpaca_orders]
            
            logger.info(f"Retrieved {len(orders)} orders")
            return orders
            
        except APIError as e:
            logger.error(f"Failed to get orders: {e}")
            return []
        
        except Exception as e:
            logger.error(f"Unexpected error getting orders: {e}")
            return []
    
    async def get_positions(self) -> List[PositionInfo]:
        """
        Get all current positions
        
        Returns:
            List[PositionInfo]: List of current positions
        """
        try:
            # Get positions from Alpaca API
            alpaca_positions = self.api.list_positions()
            
            # Convert to our format
            positions = [PositionInfo.from_alpaca_position(pos) for pos in alpaca_positions]
            
            logger.info(f"Retrieved {len(positions)} positions")
            return positions
            
        except APIError as e:
            logger.error(f"Failed to get positions: {e}")
            return []
        
        except Exception as e:
            logger.error(f"Unexpected error getting positions: {e}")
            return []
    
    async def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """
        Get position for a specific symbol
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            PositionInfo: Position information or None if no position
        """
        try:
            # Get position from Alpaca API
            alpaca_position = self.api.get_position(symbol)
            position = PositionInfo.from_alpaca_position(alpaca_position)
            
            logger.info(f"Retrieved position for {symbol}: {position.qty} shares")
            return position
            
        except APIError as e:
            if "position does not exist" in str(e).lower():
                logger.info(f"No position found for {symbol}")
                return None
            logger.error(f"Failed to get position for {symbol}: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error getting position for {symbol}: {e}")
            return None
    
    async def close_position(self, symbol: str, qty: Optional[Union[int, float]] = None) -> Optional[OrderResponse]:
        """
        Close a position (or partial position)
        
        Args:
            symbol: Symbol to close position for
            qty: Quantity to close (None for full position)
            
        Returns:
            OrderResponse: Order response for the closing order
        """
        try:
            logger.info(f"Closing position for {symbol}" + (f" (qty: {qty})" if qty else " (full position)"))
            
            # Close position via Alpaca API
            if qty is not None:
                alpaca_order = self.api.close_position(symbol, qty=str(qty))
            else:
                alpaca_order = self.api.close_position(symbol)
            
            # Convert to our format
            order_response = OrderResponse.from_alpaca_order(alpaca_order)
            
            # Track the order
            self.active_orders[order_response.id] = order_response
            
            logger.info(f"Position close order submitted: {order_response.id}")
            return order_response
            
        except APIError as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error closing position for {symbol}: {e}")
            return None
    
    async def close_all_positions(self) -> bool:
        """
        Close all positions
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Closing all positions")
            
            # Close all positions via Alpaca API
            self.api.close_all_positions()
            
            logger.info("All positions closed successfully")
            return True
            
        except APIError as e:
            logger.error(f"Failed to close all positions: {e}")
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error closing all positions: {e}")
            return False
    
    async def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Get account information
        
        Returns:
            Dict: Account information or None if failed
        """
        try:
            # Get account from Alpaca API
            account = self.api.get_account()
            
            account_info = {
                'id': account.id,
                'account_number': account.account_number,
                'status': account.status,
                'currency': account.currency,
                'buying_power': float(account.buying_power),
                'regt_buying_power': float(account.regt_buying_power),
                'daytrading_buying_power': float(account.daytrading_buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked,
                'created_at': account.created_at,
                'trade_suspended_by_user': account.trade_suspended_by_user,
                'multiplier': float(account.multiplier),
                'shorting_enabled': account.shorting_enabled,
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'long_market_value': float(account.long_market_value),
                'short_market_value': float(account.short_market_value),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'last_maintenance_margin': float(account.last_maintenance_margin),
                'sma': float(account.sma),
                'daytrade_count': account.daytrade_count
            }
            
            logger.info(f"Retrieved account info: {account_info['account_number']}")
            return account_info
            
        except APIError as e:
            logger.error(f"Failed to get account info: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error getting account info: {e}")
            return None
    
    async def reconcile_positions(self) -> Dict[str, Any]:
        """
        Reconcile positions between our tracking and broker
        
        Returns:
            Dict: Reconciliation report
        """
        try:
            logger.info("Starting position reconciliation")
            
            # Get current positions from broker
            broker_positions = await self.get_positions()
            
            # Create reconciliation report
            reconciliation_report = {
                'timestamp': datetime.now(timezone.utc),
                'broker_positions_count': len(broker_positions),
                'positions': [],
                'discrepancies': [],
                'total_market_value': Decimal('0'),
                'total_unrealized_pl': Decimal('0')
            }
            
            for position in broker_positions:
                position_data = {
                    'symbol': position.symbol,
                    'qty': position.qty,
                    'avg_entry_price': position.avg_entry_price,
                    'market_value': position.market_value,
                    'unrealized_pl': position.unrealized_pl,
                    'side': position.side
                }
                reconciliation_report['positions'].append(position_data)
                reconciliation_report['total_market_value'] += position.market_value
                reconciliation_report['total_unrealized_pl'] += position.unrealized_pl
            
            logger.info(f"Position reconciliation completed: {len(broker_positions)} positions")
            return reconciliation_report
            
        except Exception as e:
            logger.error(f"Error during position reconciliation: {e}")
            return {
                'timestamp': datetime.now(timezone.utc),
                'error': str(e),
                'positions': [],
                'discrepancies': []
            }
    
    async def generate_trade_report(self, 
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate comprehensive trade report
        
        Args:
            start_date: Start date for report (optional)
            end_date: End date for report (optional)
            
        Returns:
            Dict: Trade report with statistics
        """
        try:
            logger.info("Generating trade report")
            
            # Get orders for the specified period
            orders = await self.get_all_orders(
                status='filled',
                after=start_date,
                until=end_date,
                limit=1000
            )
            
            # Calculate trade statistics
            total_trades = len(orders)
            buy_orders = [o for o in orders if o.side == OrderSide.BUY]
            sell_orders = [o for o in orders if o.side == OrderSide.SELL]
            
            total_volume = sum(o.qty * (o.limit_price or Decimal('0')) for o in orders)
            
            # Group trades by symbol
            symbol_trades = {}
            for order in orders:
                if order.symbol not in symbol_trades:
                    symbol_trades[order.symbol] = []
                symbol_trades[order.symbol].append(order)
            
            trade_report = {
                'timestamp': datetime.now(timezone.utc),
                'period': {
                    'start_date': start_date,
                    'end_date': end_date
                },
                'summary': {
                    'total_trades': total_trades,
                    'buy_orders': len(buy_orders),
                    'sell_orders': len(sell_orders),
                    'total_volume': float(total_volume),
                    'unique_symbols': len(symbol_trades)
                },
                'by_symbol': {},
                'orders': [
                    {
                        'id': o.id,
                        'symbol': o.symbol,
                        'side': o.side.value,
                        'qty': float(o.qty),
                        'filled_qty': float(o.filled_qty),
                        'limit_price': float(o.limit_price) if o.limit_price else None,
                        'status': o.status.value,
                        'filled_at': o.filled_at,
                        'created_at': o.created_at
                    }
                    for o in orders
                ]
            }
            
            # Add per-symbol statistics
            for symbol, symbol_orders in symbol_trades.items():
                symbol_buy_orders = [o for o in symbol_orders if o.side == OrderSide.BUY]
                symbol_sell_orders = [o for o in symbol_orders if o.side == OrderSide.SELL]
                
                trade_report['by_symbol'][symbol] = {
                    'total_trades': len(symbol_orders),
                    'buy_orders': len(symbol_buy_orders),
                    'sell_orders': len(symbol_sell_orders),
                    'total_qty': float(sum(o.qty for o in symbol_orders))
                }
            
            logger.info(f"Trade report generated: {total_trades} trades")
            return trade_report
            
        except Exception as e:
            logger.error(f"Error generating trade report: {e}")
            return {
                'timestamp': datetime.now(timezone.utc),
                'error': str(e),
                'summary': {},
                'orders': []
            }
    
    def _is_retryable_error(self, error: APIError) -> bool:
        """
        Determine if an API error is retryable
        
        Args:
            error: API error to check
            
        Returns:
            bool: True if error is retryable
        """
        # Common retryable error codes/messages
        retryable_codes = [429, 500, 502, 503, 504]  # Rate limit, server errors
        retryable_messages = [
            'timeout',
            'connection',
            'network',
            'temporary',
            'rate limit'
        ]
        
        # Check error code
        if hasattr(error, 'code') and error.code in retryable_codes:
            return True
        
        # Check error message
        error_message = str(error).lower()
        for message in retryable_messages:
            if message in error_message:
                return True
        
        return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on broker connection
        
        Returns:
            Dict: Health check results
        """
        health_status = {
            'timestamp': datetime.now(timezone.utc),
            'broker': 'alpaca',
            'paper_trading': self.paper_trading,
            'connection_status': 'unknown',
            'account_accessible': False,
            'orders_accessible': False,
            'positions_accessible': False,
            'errors': []
        }
        
        try:
            # Test account access
            account_info = await self.get_account_info()
            if account_info:
                health_status['account_accessible'] = True
                health_status['account_status'] = account_info.get('status')
            
            # Test orders access
            orders = await self.get_all_orders(limit=1)
            health_status['orders_accessible'] = True
            
            # Test positions access
            positions = await self.get_positions()
            health_status['positions_accessible'] = True
            
            # If all tests pass, connection is healthy
            if all([
                health_status['account_accessible'],
                health_status['orders_accessible'],
                health_status['positions_accessible']
            ]):
                health_status['connection_status'] = 'healthy'
            else:
                health_status['connection_status'] = 'degraded'
            
        except Exception as e:
            health_status['connection_status'] = 'unhealthy'
            health_status['errors'].append(str(e))
            logger.error(f"Broker health check failed: {e}")
        
        return health_status


# Convenience functions for common operations
async def create_market_order(broker: AlpacaBrokerIntegration,
                            symbol: str,
                            qty: Union[int, float],
                            side: OrderSide) -> OrderResponse:
    """Create and submit a market order"""
    order_request = OrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY
    )
    return await broker.submit_order(order_request)


async def create_limit_order(broker: AlpacaBrokerIntegration,
                           symbol: str,
                           qty: Union[int, float],
                           side: OrderSide,
                           limit_price: Union[float, Decimal],
                           time_in_force: TimeInForce = TimeInForce.DAY) -> OrderResponse:
    """Create and submit a limit order"""
    order_request = OrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        type=OrderType.LIMIT,
        limit_price=limit_price,
        time_in_force=time_in_force
    )
    return await broker.submit_order(order_request)


async def create_stop_loss_order(broker: AlpacaBrokerIntegration,
                                symbol: str,
                                qty: Union[int, float],
                                side: OrderSide,
                                stop_price: Union[float, Decimal]) -> OrderResponse:
    """Create and submit a stop loss order"""
    order_request = OrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        type=OrderType.STOP,
        stop_price=stop_price,
        time_in_force=TimeInForce.GTC
    )
    return await broker.submit_order(order_request)