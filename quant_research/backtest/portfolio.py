"""Portfolio management for backtesting."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
from dataclasses import dataclass, field
from enum import Enum

from structlog import get_logger

logger = get_logger(__name__)


class PositionType(Enum):
    """Position type enumeration."""
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """Individual position in a security."""
    
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    position_type: PositionType = PositionType.LONG
    
    # Current state
    current_price: float = 0.0
    last_update: Optional[datetime] = None
    
    # Tracking
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        if self.position_type == PositionType.LONG:
            return self.quantity * self.current_price
        else:  # SHORT
            return -self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L of position."""
        if self.position_type == PositionType.LONG:
            return self.quantity * (self.current_price - self.entry_price)
        else:  # SHORT
            return self.quantity * (self.entry_price - self.current_price)
    
    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def return_pct(self) -> float:
        """Return percentage."""
        if self.entry_price == 0:
            return 0.0
        
        if self.position_type == PositionType.LONG:
            return (self.current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            return (self.entry_price - self.current_price) / self.entry_price
    
    @property
    def cost_basis(self) -> float:
        """Cost basis of position."""
        return abs(self.quantity) * self.entry_price
    
    def update_price(self, new_price: float, timestamp: datetime):
        """Update current price of position.
        
        Args:
            new_price: New market price
            timestamp: Update timestamp
        """
        self.current_price = new_price
        self.last_update = timestamp
    
    def reduce_position(self, quantity: float, exit_price: float) -> float:
        """Reduce position size and realize P&L.
        
        Args:
            quantity: Quantity to reduce (positive number)
            exit_price: Exit price
            
        Returns:
            Realized P&L from the reduction
        """
        if quantity <= 0:
            return 0.0
        
        # Limit reduction to current position size
        actual_reduction = min(abs(quantity), abs(self.quantity))
        
        # Calculate realized P&L
        if self.position_type == PositionType.LONG:
            realized = actual_reduction * (exit_price - self.entry_price)
        else:  # SHORT
            realized = actual_reduction * (self.entry_price - exit_price)
        
        # Update position
        if self.quantity > 0:
            self.quantity = max(0, self.quantity - actual_reduction)
        else:
            self.quantity = min(0, self.quantity + actual_reduction)
        
        self.realized_pnl += realized
        
        return realized
    
    def close_position(self, exit_price: float) -> float:
        """Close entire position.
        
        Args:
            exit_price: Exit price
            
        Returns:
            Realized P&L from closing
        """
        return self.reduce_position(abs(self.quantity), exit_price)


class Portfolio:
    """Portfolio management class."""
    
    def __init__(
        self,
        initial_capital: float,
        margin_requirement: float = 0.5,
        commission_rate: float = 0.001
    ):
        """Initialize portfolio.
        
        Args:
            initial_capital: Starting capital
            margin_requirement: Margin requirement for short positions
            commission_rate: Commission rate per trade
        """
        self.initial_capital = initial_capital
        self.margin_requirement = margin_requirement
        self.commission_rate = commission_rate
        
        # Core portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        
        # Performance tracking
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.trades_count = 0
        
        # Historical tracking
        self.trade_history: List[Dict[str, Any]] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        
        logger.info(f"Portfolio initialized with ${initial_capital:,.2f}")
    
    @property
    def market_value(self) -> float:
        """Total market value of all positions."""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)."""
        return self.cash + self.market_value
    
    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def equity(self) -> float:
        """Portfolio equity."""
        return self.total_value
    
    @property
    def total_return(self) -> float:
        """Total return percentage."""
        if self.initial_capital == 0:
            return 0.0
        return (self.total_value - self.initial_capital) / self.initial_capital
    
    @property
    def buying_power(self) -> float:
        """Available buying power."""
        # Simplified calculation
        return self.cash
    
    @property
    def position_count(self) -> int:
        """Number of open positions."""
        return len([pos for pos in self.positions.values() if pos.quantity != 0])
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Position object or None
        """
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if portfolio has position in symbol.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            True if position exists and quantity != 0
        """
        pos = self.get_position(symbol)
        return pos is not None and pos.quantity != 0
    
    def get_position_quantity(self, symbol: str) -> float:
        """Get position quantity for symbol.
        
        Args:
            symbol: Symbol to get quantity for
            
        Returns:
            Position quantity (0 if no position)
        """
        pos = self.get_position(symbol)
        return pos.quantity if pos else 0.0
    
    def update_price(self, symbol: str, price: float, timestamp: datetime):
        """Update price for a symbol.
        
        Args:
            symbol: Symbol to update
            price: New price
            timestamp: Update timestamp
        """
        if symbol in self.positions:
            self.positions[symbol].update_price(price, timestamp)
    
    def open_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        position_type: PositionType = PositionType.LONG,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Open a new position or add to existing position.
        
        Args:
            symbol: Symbol to trade
            quantity: Quantity to trade (positive for long, negative for short)
            price: Entry price
            timestamp: Trade timestamp
            position_type: Position type
            metadata: Additional metadata
            
        Returns:
            True if trade was successful
        """
        try:
            # Calculate trade cost
            trade_value = abs(quantity) * price
            commission = trade_value * self.commission_rate
            
            # Check if we have enough capital
            required_capital = trade_value + commission
            
            if position_type == PositionType.SHORT:
                # For short positions, require margin
                required_capital = trade_value * self.margin_requirement + commission
            
            if required_capital > self.buying_power:
                logger.warning(f"Insufficient capital for trade: {symbol}")
                return False
            
            # Get or create position
            if symbol in self.positions:
                position = self.positions[symbol]
                
                # If changing direction (long to short or vice versa)
                if (position.quantity > 0 and quantity < 0) or (position.quantity < 0 and quantity > 0):
                    # Close existing position first
                    self.close_position(symbol, price, timestamp)
                    
                    # Open new position with remaining quantity
                    remaining_quantity = quantity + position.quantity
                    if abs(remaining_quantity) > 0:
                        return self.open_position(
                            symbol, remaining_quantity, price, timestamp, 
                            position_type, metadata
                        )
                    return True
                
                # Add to existing position (same direction)
                else:
                    # Calculate new average entry price
                    total_quantity = position.quantity + quantity
                    total_cost = (position.quantity * position.entry_price) + (quantity * price)
                    new_avg_price = total_cost / total_quantity if total_quantity != 0 else price
                    
                    position.quantity = total_quantity
                    position.entry_price = new_avg_price
                    position.current_price = price
                    position.last_update = timestamp
            else:
                # Create new position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    entry_time=timestamp,
                    position_type=position_type,
                    current_price=price,
                    last_update=timestamp,
                    metadata=metadata or {}
                )
            
            # Update cash
            self.cash -= (trade_value + commission)
            self.fees_paid += commission
            self.trades_count += 1
            
            # Record trade
            trade_record = {
                'timestamp': timestamp,
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'trade_value': trade_value,
                'commission': commission,
                'action': 'BUY' if quantity > 0 else 'SELL',
                'position_type': position_type.value
            }
            self.trade_history.append(trade_record)
            
            logger.info(f"Opened position: {symbol} {quantity} @ ${price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening position for {symbol}: {e}")
            return False
    
    def close_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        quantity: Optional[float] = None
    ) -> bool:
        """Close position (partially or completely).
        
        Args:
            symbol: Symbol to close
            price: Exit price
            timestamp: Trade timestamp
            quantity: Quantity to close (None for full position)
            
        Returns:
            True if trade was successful
        """
        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return False
        
        position = self.positions[symbol]
        
        if position.quantity == 0:
            logger.warning(f"Position for {symbol} already closed")
            return False
        
        try:
            # Determine quantity to close
            if quantity is None:
                close_quantity = abs(position.quantity)
            else:
                close_quantity = min(abs(quantity), abs(position.quantity))
            
            # Calculate trade details
            trade_value = close_quantity * price
            commission = trade_value * self.commission_rate
            
            # Realize P&L
            realized_pnl = position.reduce_position(close_quantity, price)
            
            # Update portfolio
            self.cash += trade_value - commission
            self.realized_pnl += realized_pnl
            self.fees_paid += commission
            self.trades_count += 1
            
            # Record trade
            trade_record = {
                'timestamp': timestamp,
                'symbol': symbol,
                'quantity': -close_quantity if position.position_type == PositionType.LONG else close_quantity,
                'price': price,
                'trade_value': trade_value,
                'commission': commission,
                'realized_pnl': realized_pnl,
                'action': 'SELL' if position.position_type == PositionType.LONG else 'COVER'
            }
            self.trade_history.append(trade_record)
            
            # Remove position if fully closed
            if position.quantity == 0:
                del self.positions[symbol]
                logger.info(f"Closed position: {symbol} @ ${price:.2f}, P&L: ${realized_pnl:.2f}")
            else:
                logger.info(f"Reduced position: {symbol} by {close_quantity} @ ${price:.2f}, P&L: ${realized_pnl:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return False
    
    def get_state(self) -> Dict[str, Any]:
        """Get current portfolio state.
        
        Returns:
            Dictionary containing portfolio state
        """
        return {
            'cash': self.cash,
            'market_value': self.market_value,
            'total_value': self.total_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'total_return': self.total_return,
            'position_count': self.position_count,
            'trades_count': self.trades_count,
            'fees_paid': self.fees_paid
        }
    
    def get_positions_summary(self) -> pd.DataFrame:
        """Get summary of all positions.
        
        Returns:
            DataFrame with position details
        """
        if not self.positions:
            return pd.DataFrame()
        
        positions_data = []
        
        for symbol, position in self.positions.items():
            if position.quantity != 0:
                positions_data.append({
                    'symbol': symbol,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'market_value': position.market_value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'return_pct': position.return_pct * 100,
                    'position_type': position.position_type.value,
                    'entry_time': position.entry_time
                })
        
        return pd.DataFrame(positions_data)
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame.
        
        Returns:
            DataFrame with trade history
        """
        if not self.trade_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_history)