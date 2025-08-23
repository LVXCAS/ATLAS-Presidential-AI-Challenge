"""
Market simulator for realistic backtesting with slippage, latency, and market microstructure effects.
"""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class MarketData:
    """Market data point with OHLCV and microstructure data."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    spread: float
    volatility: float


@dataclass
class Order:
    """Order representation for backtesting."""
    id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    price: Optional[float]  # None for market orders
    order_type: str  # 'market', 'limit', 'stop'
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    fills: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.fills is None:
            self.fills = []


@dataclass
class Fill:
    """Order fill representation."""
    order_id: str
    timestamp: datetime
    price: float
    quantity: float
    commission: float
    slippage: float


class MarketSimulator:
    """
    Realistic market simulator that models:
    - Market microstructure effects
    - Order book dynamics
    - Slippage and latency
    - Partial fills
    - Market impact
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.current_time = None
        self.market_data: Dict[str, MarketData] = {}
        self.order_book: Dict[str, List[Order]] = {}
        self.filled_orders: List[Order] = []
        
        # Market microstructure parameters
        self.bid_ask_spread_bps = self.config.get('bid_ask_spread_bps', 5)  # 0.05%
        self.market_impact_factor = self.config.get('market_impact_factor', 0.001)
        self.latency_ms = self.config.get('latency_ms', 50)
        self.volatility_multiplier = self.config.get('volatility_multiplier', 1.0)
        
        # Slippage model parameters
        self.base_slippage_bps = self.config.get('base_slippage_bps', 2)  # 0.02%
        self.volume_slippage_factor = self.config.get('volume_slippage_factor', 0.1)
        self.volatility_slippage_factor = self.config.get('volatility_slippage_factor', 0.5)
        
        # Commission structure
        self.commission_per_share = self.config.get('commission_per_share', 0.005)
        self.min_commission = self.config.get('min_commission', 1.0)
        self.max_commission_rate = self.config.get('max_commission_rate', 0.005)  # 0.5%
        
        logger.info(f"MarketSimulator initialized with config: {self.config}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for market simulation."""
        return {
            'bid_ask_spread_bps': 5,
            'market_impact_factor': 0.001,
            'latency_ms': 50,
            'volatility_multiplier': 1.0,
            'base_slippage_bps': 2,
            'volume_slippage_factor': 0.1,
            'volatility_slippage_factor': 0.5,
            'commission_per_share': 0.005,
            'min_commission': 1.0,
            'max_commission_rate': 0.005
        }
    
    def update_market_data(self, symbol: str, ohlcv: Dict[str, float], timestamp: datetime):
        """Update market data for a symbol."""
        # Calculate bid-ask spread based on volatility and volume
        mid_price = ohlcv['close']
        volume = ohlcv['volume']
        high_low_range = ohlcv['high'] - ohlcv['low']
        volatility = high_low_range / mid_price if mid_price > 0 else 0.01
        
        # Dynamic spread based on volatility and volume
        base_spread = mid_price * (self.bid_ask_spread_bps / 10000)
        vol_adjustment = volatility * self.volatility_multiplier * mid_price * 0.01
        volume_adjustment = max(0, (1000000 - volume) / 1000000) * mid_price * 0.001
        
        spread = base_spread + vol_adjustment + volume_adjustment
        
        self.market_data[symbol] = MarketData(
            timestamp=timestamp,
            symbol=symbol,
            open=ohlcv['open'],
            high=ohlcv['high'],
            low=ohlcv['low'],
            close=ohlcv['close'],
            volume=int(volume),
            bid=mid_price - spread/2,
            ask=mid_price + spread/2,
            bid_size=random.randint(100, 1000),
            ask_size=random.randint(100, 1000),
            spread=spread,
            volatility=volatility
        )
        
        self.current_time = timestamp
    
    def submit_order(self, order: Order) -> str:
        """Submit an order to the market simulator."""
        if order.symbol not in self.order_book:
            self.order_book[order.symbol] = []
        
        # Add latency simulation
        order.timestamp = order.timestamp + timedelta(milliseconds=self.latency_ms)
        
        self.order_book[order.symbol].append(order)
        logger.debug(f"Order submitted: {order.id} - {order.side.value} {order.quantity} {order.symbol}")
        
        return order.id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        for symbol, orders in self.order_book.items():
            for order in orders:
                if order.id == order_id and order.status == OrderStatus.PENDING:
                    order.status = OrderStatus.CANCELLED
                    logger.debug(f"Order cancelled: {order_id}")
                    return True
        return False
    
    def process_orders(self) -> List[Fill]:
        """Process all pending orders against current market data."""
        fills = []
        
        for symbol, orders in self.order_book.items():
            if symbol not in self.market_data:
                continue
            
            market_data = self.market_data[symbol]
            
            # Process orders in timestamp order
            pending_orders = [o for o in orders if o.status == OrderStatus.PENDING]
            pending_orders.sort(key=lambda x: x.timestamp)
            
            for order in pending_orders:
                if order.timestamp <= self.current_time:
                    order_fills = self._execute_order(order, market_data)
                    fills.extend(order_fills)
        
        return fills
    
    def _execute_order(self, order: Order, market_data: MarketData) -> List[Fill]:
        """Execute a single order with realistic market conditions."""
        fills = []
        
        if order.order_type == 'market':
            fills = self._execute_market_order(order, market_data)
        elif order.order_type == 'limit':
            fills = self._execute_limit_order(order, market_data)
        elif order.order_type == 'stop':
            fills = self._execute_stop_order(order, market_data)
        
        # Update order status
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            order.status = OrderStatus.PARTIAL
        
        return fills
    
    def _execute_market_order(self, order: Order, market_data: MarketData) -> List[Fill]:
        """Execute market order with realistic slippage."""
        fills = []
        remaining_quantity = order.quantity - order.filled_quantity
        
        if remaining_quantity <= 0:
            return fills
        
        # Determine execution price based on order side
        if order.side == OrderSide.BUY:
            base_price = market_data.ask
            available_quantity = market_data.ask_size
        else:
            base_price = market_data.bid  
            available_quantity = market_data.bid_size
        
        # Calculate slippage
        slippage = self._calculate_slippage(
            order, market_data, remaining_quantity
        )
        
        # Apply market impact
        market_impact = self._calculate_market_impact(
            remaining_quantity, market_data.volume, base_price
        )
        
        # Final execution price
        if order.side == OrderSide.BUY:
            execution_price = base_price + slippage + market_impact
        else:
            execution_price = base_price - slippage - market_impact
        
        # Determine fill quantity (partial fills possible)
        fill_quantity = min(remaining_quantity, available_quantity)
        
        # Random partial fill simulation
        if random.random() < 0.1:  # 10% chance of partial fill
            fill_quantity = min(fill_quantity, remaining_quantity * random.uniform(0.3, 0.8))
        
        # Calculate commission
        commission = self._calculate_commission(fill_quantity, execution_price)
        
        # Create fill
        fill = Fill(
            order_id=order.id,
            timestamp=self.current_time,
            price=execution_price,
            quantity=fill_quantity,
            commission=commission,
            slippage=slippage
        )
        
        fills.append(fill)
        order.fills.append({
            'timestamp': self.current_time.isoformat(),
            'price': execution_price,
            'quantity': fill_quantity,
            'commission': commission,
            'slippage': slippage
        })
        
        # Update order
        order.filled_quantity += fill_quantity
        if order.filled_quantity > 0:
            total_value = sum(f.price * f.quantity for f in order.fills)
            total_quantity = sum(f.quantity for f in order.fills)
            order.avg_fill_price = total_value / total_quantity if total_quantity > 0 else 0
        
        logger.debug(f"Market order fill: {order.id} - {fill_quantity}@{execution_price:.4f}")
        
        return fills
    
    def _execute_limit_order(self, order: Order, market_data: MarketData) -> List[Fill]:
        """Execute limit order if price conditions are met."""
        fills = []
        remaining_quantity = order.quantity - order.filled_quantity
        
        if remaining_quantity <= 0:
            return fills
        
        # Check if limit order can be executed
        can_execute = False
        if order.side == OrderSide.BUY and market_data.ask <= order.price:
            can_execute = True
            execution_price = min(order.price, market_data.ask)
            available_quantity = market_data.ask_size
        elif order.side == OrderSide.SELL and market_data.bid >= order.price:
            can_execute = True
            execution_price = max(order.price, market_data.bid)
            available_quantity = market_data.bid_size
        
        if not can_execute:
            return fills
        
        # Determine fill quantity
        fill_quantity = min(remaining_quantity, available_quantity)
        
        # Random partial fill for limit orders
        if random.random() < 0.15:  # 15% chance of partial fill
            fill_quantity = min(fill_quantity, remaining_quantity * random.uniform(0.2, 0.9))
        
        # Calculate commission
        commission = self._calculate_commission(fill_quantity, execution_price)
        
        # Create fill
        fill = Fill(
            order_id=order.id,
            timestamp=self.current_time,
            price=execution_price,
            quantity=fill_quantity,
            commission=commission,
            slippage=0.0  # No slippage for limit orders
        )
        
        fills.append(fill)
        order.fills.append({
            'timestamp': self.current_time.isoformat(),
            'price': execution_price,
            'quantity': fill_quantity,
            'commission': commission,
            'slippage': 0.0
        })
        
        # Update order
        order.filled_quantity += fill_quantity
        if order.filled_quantity > 0:
            total_value = sum(f['price'] * f['quantity'] for f in order.fills)
            total_quantity = sum(f['quantity'] for f in order.fills)
            order.avg_fill_price = total_value / total_quantity if total_quantity > 0 else 0
        
        logger.debug(f"Limit order fill: {order.id} - {fill_quantity}@{execution_price:.4f}")
        
        return fills
    
    def _execute_stop_order(self, order: Order, market_data: MarketData) -> List[Fill]:
        """Execute stop order when stop price is reached."""
        # Stop orders convert to market orders when triggered
        # Implementation would check if stop price is hit and convert to market order
        return []
    
    def _calculate_slippage(self, order: Order, market_data: MarketData, quantity: float) -> float:
        """Calculate realistic slippage based on order size, volatility, and market conditions."""
        base_price = market_data.ask if order.side == OrderSide.BUY else market_data.bid
        
        # Base slippage
        base_slippage = base_price * (self.base_slippage_bps / 10000)
        
        # Volume-based slippage
        volume_ratio = quantity / max(market_data.volume, 1)
        volume_slippage = base_price * volume_ratio * self.volume_slippage_factor
        
        # Volatility-based slippage  
        volatility_slippage = base_price * market_data.volatility * self.volatility_slippage_factor
        
        # Random component
        random_slippage = base_slippage * random.uniform(-0.5, 1.5)
        
        total_slippage = base_slippage + volume_slippage + volatility_slippage + random_slippage
        
        return max(0, total_slippage)
    
    def _calculate_market_impact(self, quantity: float, daily_volume: int, price: float) -> float:
        """Calculate market impact based on order size relative to daily volume."""
        if daily_volume <= 0:
            return 0.0
        
        participation_rate = quantity / daily_volume
        impact = price * participation_rate * self.market_impact_factor
        
        # Non-linear impact for large orders
        if participation_rate > 0.01:  # > 1% of daily volume
            impact *= (1 + participation_rate * 10)
        
        return impact
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission based on quantity and price."""
        # Per-share commission
        commission = quantity * self.commission_per_share
        
        # Apply minimum commission
        commission = max(commission, self.min_commission)
        
        # Apply maximum commission rate
        notional = quantity * price
        max_commission = notional * self.max_commission_rate
        commission = min(commission, max_commission)
        
        return commission
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get status of a specific order."""
        for orders in self.order_book.values():
            for order in orders:
                if order.id == order_id:
                    return order
        return None
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders, optionally filtered by symbol."""
        open_orders = []
        
        for sym, orders in self.order_book.items():
            if symbol and sym != symbol:
                continue
            for order in orders:
                if order.status in [OrderStatus.PENDING, OrderStatus.PARTIAL]:
                    open_orders.append(order)
        
        return open_orders
    
    def get_filled_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all filled orders, optionally filtered by symbol."""
        filled_orders = []
        
        for sym, orders in self.order_book.items():
            if symbol and sym != symbol:
                continue
            for order in orders:
                if order.status == OrderStatus.FILLED:
                    filled_orders.append(order)
        
        return filled_orders
    
    def reset(self):
        """Reset simulator state."""
        self.current_time = None
        self.market_data.clear()
        self.order_book.clear()
        self.filled_orders.clear()
        logger.info("MarketSimulator reset")