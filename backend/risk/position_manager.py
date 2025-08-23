"""
Position Manager for Bloomberg Terminal
Advanced position tracking and management with real-time updates.
"""

import asyncio
import logging
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from events.event_bus import EventBus, Event, EventType, get_event_bus
from agents.base_agent import TradingSignal, SignalType

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    """Position status types."""
    OPEN = "open"
    CLOSED = "closed"
    CLOSING = "closing"
    SUSPENDED = "suspended"


class PositionType(Enum):
    """Position types."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class Trade:
    """Individual trade record."""
    id: str
    position_id: str
    symbol: str
    side: str  # buy/sell
    quantity: float
    price: float
    value: float
    timestamp: datetime
    order_id: Optional[str] = None
    commission: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Position data structure."""
    id: str
    symbol: str
    position_type: PositionType
    status: PositionStatus
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    cost_basis: float
    
    # Risk metrics
    daily_var: float
    beta: float
    volatility: float
    max_loss: float
    max_gain: float
    
    # Timestamps
    opened_at: datetime
    last_updated: datetime
    closed_at: Optional[datetime] = None
    
    # Trading info
    trades: List[Trade] = field(default_factory=list)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Metadata
    sector: str = "Unknown"
    strategy: Optional[str] = None
    notes: str = ""
    
    def calculate_metrics(self) -> None:
        """Calculate position metrics."""
        if self.quantity == 0:
            return
            
        # P&L calculations
        if self.position_type == PositionType.LONG:
            self.unrealized_pnl = (self.current_price - self.avg_entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.avg_entry_price - self.current_price) * abs(self.quantity)
            
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
        self.market_value = abs(self.quantity) * self.current_price
        
        # Risk metrics (simplified)
        price_change = abs(self.current_price - self.avg_entry_price) / self.avg_entry_price
        self.daily_var = self.market_value * self.volatility / np.sqrt(252)  # Daily VaR
        
        # Track max gain/loss during position lifetime
        current_pnl_pct = self.unrealized_pnl / self.cost_basis if self.cost_basis > 0 else 0
        self.max_gain = max(self.max_gain, current_pnl_pct)
        self.max_loss = min(self.max_loss, current_pnl_pct)


class PositionManager:
    """
    Comprehensive position management system providing:
    - Real-time position tracking and updates
    - P&L calculation and monitoring
    - Risk-adjusted position sizing
    - Stop-loss and take-profit management
    - Position lifecycle management
    - Performance analytics
    - Integration with risk engine
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            # Position management
            'auto_stop_loss': True,
            'default_stop_loss_pct': 0.05,  # 5% stop loss
            'default_take_profit_pct': 0.15,  # 15% take profit
            'trailing_stop_enabled': True,
            'trailing_stop_pct': 0.03,  # 3% trailing stop
            
            # Risk controls
            'max_position_size': 100000,  # $100k max position
            'max_daily_loss': 0.02,  # 2% max daily loss
            'position_timeout_hours': 48,  # Auto-close after 48h
            'min_hold_time_minutes': 30,  # Minimum hold time
            
            # Performance tracking
            'track_performance': True,
            'benchmark_symbol': 'SPY',
            'performance_lookback_days': 30,
            
            # Data retention
            'keep_closed_positions_days': 30,
            'max_trades_per_position': 100,
            
            # Sector mappings
            'sector_mapping': {
                'AAPL': 'Technology', 'GOOGL': 'Technology', 'MSFT': 'Technology',
                'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
                'JPM': 'Financials', 'BAC': 'Financials', 'GS': 'Financials',
                'SPY': 'Index', 'QQQ': 'Index', 'VTI': 'Index'
            }
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.event_bus: EventBus = get_event_bus()
        
        # Position tracking
        self.positions: Dict[str, Position] = {}  # position_id -> Position
        self.positions_by_symbol: Dict[str, List[str]] = {}  # symbol -> position_ids
        self.closed_positions: List[Position] = []
        
        # Performance tracking
        self.portfolio_value_history: List[Dict] = []
        self.daily_pnl_history: List[Dict] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Market data cache
        self.price_cache: Dict[str, Dict] = {}
        self.volatility_cache: Dict[str, float] = {}
        
        self.is_running = False
        
    async def initialize(self) -> None:
        """Initialize the position manager."""
        try:
            logger.info("Initializing Position Manager")
            
            # Setup event subscriptions
            await self._setup_event_subscriptions()
            
            # Initialize mock positions for demonstration
            await self._initialize_demo_positions()
            
            logger.info("Position Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Position Manager: {e}")
            raise
    
    async def start(self) -> None:
        """Start the position manager."""
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._position_monitoring_loop())
        asyncio.create_task(self._stop_loss_monitoring_loop())
        asyncio.create_task(self._performance_tracking_loop())
        asyncio.create_task(self._cleanup_loop())
        
        logger.info("Position Manager started")
    
    async def stop(self) -> None:
        """Stop the position manager."""
        self.is_running = False
        logger.info("Position Manager stopped")
    
    async def open_position(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        signal: Optional[TradingSignal] = None
    ) -> str:
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity (positive=long, negative=short)
            entry_price: Entry price
            signal: Optional trading signal that triggered the position
            
        Returns:
            Position ID
        """
        try:
            position_id = str(uuid.uuid4())
            
            # Determine position type
            position_type = PositionType.LONG if quantity > 0 else PositionType.SHORT
            
            # Create initial trade
            trade = Trade(
                id=str(uuid.uuid4()),
                position_id=position_id,
                symbol=symbol,
                side="buy" if quantity > 0 else "sell",
                quantity=abs(quantity),
                price=entry_price,
                value=abs(quantity) * entry_price,
                timestamp=datetime.now(timezone.utc),
                metadata={'source': 'position_open'}
            )
            
            # Create position
            position = Position(
                id=position_id,
                symbol=symbol,
                position_type=position_type,
                status=PositionStatus.OPEN,
                quantity=quantity,
                avg_entry_price=entry_price,
                current_price=entry_price,
                market_value=abs(quantity) * entry_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                total_pnl=0.0,
                cost_basis=abs(quantity) * entry_price,
                daily_var=0.0,
                beta=1.0,
                volatility=self.volatility_cache.get(symbol, 0.02),
                max_loss=0.0,
                max_gain=0.0,
                opened_at=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc),
                trades=[trade],
                sector=self.config['sector_mapping'].get(symbol, 'Unknown')
            )
            
            # Set stop-loss and take-profit if enabled
            if self.config['auto_stop_loss']:
                if position_type == PositionType.LONG:
                    position.stop_loss = entry_price * (1 - self.config['default_stop_loss_pct'])
                    position.take_profit = entry_price * (1 + self.config['default_take_profit_pct'])
                else:
                    position.stop_loss = entry_price * (1 + self.config['default_stop_loss_pct'])
                    position.take_profit = entry_price * (1 - self.config['default_take_profit_pct'])
            
            # Add signal context if available
            if signal:
                position.strategy = signal.agent_name
                position.notes = f"Opened from {signal.agent_name} signal (conf: {signal.confidence:.2f})"
                if signal.stop_loss:
                    position.stop_loss = signal.stop_loss
                if signal.target_price:
                    position.take_profit = signal.target_price
            
            # Store position
            self.positions[position_id] = position
            
            if symbol not in self.positions_by_symbol:
                self.positions_by_symbol[symbol] = []
            self.positions_by_symbol[symbol].append(position_id)
            
            # Publish position event
            await self.event_bus.publish(Event(
                id=str(uuid.uuid4()),
                event_type=EventType.POSITION_UPDATE,
                timestamp=datetime.now(timezone.utc),
                source='PositionManager',
                data={
                    'action': 'position_opened',
                    'position_id': position_id,
                    'symbol': symbol,
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'market_value': position.market_value
                }
            ))
            
            logger.info(f"Opened {position_type.value} position for {symbol}: {quantity} @ ${entry_price:.2f}")
            
            return position_id
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            raise
    
    async def close_position(self, position_id: str, close_price: Optional[float] = None) -> bool:
        """
        Close an existing position.
        
        Args:
            position_id: Position to close
            close_price: Optional close price (will use current market price if None)
            
        Returns:
            Success status
        """
        try:
            if position_id not in self.positions:
                logger.error(f"Position {position_id} not found")
                return False
            
            position = self.positions[position_id]
            
            if position.status != PositionStatus.OPEN:
                logger.warning(f"Position {position_id} is not open (status: {position.status.value})")
                return False
            
            # Use current market price if not specified
            if close_price is None:
                close_price = self.price_cache.get(position.symbol, {}).get('price', position.current_price)
            
            # Create closing trade
            close_side = "sell" if position.position_type == PositionType.LONG else "buy"
            close_trade = Trade(
                id=str(uuid.uuid4()),
                position_id=position_id,
                symbol=position.symbol,
                side=close_side,
                quantity=abs(position.quantity),
                price=close_price,
                value=abs(position.quantity) * close_price,
                timestamp=datetime.now(timezone.utc),
                metadata={'source': 'position_close'}
            )
            
            # Update position
            position.trades.append(close_trade)
            position.current_price = close_price
            position.status = PositionStatus.CLOSED
            position.closed_at = datetime.now(timezone.utc)
            
            # Calculate final P&L
            if position.position_type == PositionType.LONG:
                position.realized_pnl = (close_price - position.avg_entry_price) * position.quantity
            else:
                position.realized_pnl = (position.avg_entry_price - close_price) * abs(position.quantity)
            
            position.unrealized_pnl = 0.0
            position.total_pnl = position.realized_pnl
            position.last_updated = datetime.now(timezone.utc)
            
            # Move to closed positions
            self.closed_positions.append(position)
            del self.positions[position_id]
            
            # Remove from symbol mapping
            if position.symbol in self.positions_by_symbol:
                if position_id in self.positions_by_symbol[position.symbol]:
                    self.positions_by_symbol[position.symbol].remove(position_id)
                if not self.positions_by_symbol[position.symbol]:
                    del self.positions_by_symbol[position.symbol]
            
            # Publish position event
            await self.event_bus.publish(Event(
                id=str(uuid.uuid4()),
                event_type=EventType.POSITION_UPDATE,
                timestamp=datetime.now(timezone.utc),
                source='PositionManager',
                data={
                    'action': 'position_closed',
                    'position_id': position_id,
                    'symbol': position.symbol,
                    'realized_pnl': position.realized_pnl,
                    'close_price': close_price,
                    'holding_period': (position.closed_at - position.opened_at).total_seconds() / 3600  # hours
                }
            ))
            
            logger.info(f"Closed position {position.symbol}: P&L ${position.realized_pnl:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    async def update_position_prices(self, price_updates: Dict[str, float]) -> None:
        """Update positions with current market prices."""
        try:
            for symbol, price in price_updates.items():
                # Update price cache
                self.price_cache[symbol] = {
                    'price': price,
                    'timestamp': datetime.now(timezone.utc)
                }
                
                # Update positions for this symbol
                if symbol in self.positions_by_symbol:
                    for position_id in self.positions_by_symbol[symbol]:
                        if position_id in self.positions:
                            position = self.positions[position_id]
                            position.current_price = price
                            position.calculate_metrics()
                            position.last_updated = datetime.now(timezone.utc)
                            
        except Exception as e:
            logger.error(f"Error updating position prices: {e}")
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        try:
            if not self.positions:
                return {
                    'total_positions': 0,
                    'total_value': 0,
                    'total_pnl': 0,
                    'unrealized_pnl': 0,
                    'realized_pnl': 0,
                    'daily_pnl': 0,
                    'positions_by_status': {},
                    'positions_by_sector': {},
                    'top_positions': [],
                    'worst_positions': []
                }
            
            # Calculate totals
            total_value = sum(pos.market_value for pos in self.positions.values())
            total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_realized = sum(pos.realized_pnl for pos in self.closed_positions[-30:])  # Last 30 days
            total_pnl = total_unrealized + total_realized
            
            # Position breakdowns
            positions_by_status = {}
            positions_by_sector = {}
            
            for position in self.positions.values():
                # By status
                status = position.status.value
                if status not in positions_by_status:
                    positions_by_status[status] = {'count': 0, 'value': 0}
                positions_by_status[status]['count'] += 1
                positions_by_status[status]['value'] += position.market_value
                
                # By sector
                sector = position.sector
                if sector not in positions_by_sector:
                    positions_by_sector[sector] = {'count': 0, 'value': 0, 'pnl': 0}
                positions_by_sector[sector]['count'] += 1
                positions_by_sector[sector]['value'] += position.market_value
                positions_by_sector[sector]['pnl'] += position.unrealized_pnl
            
            # Top and worst positions
            sorted_positions = sorted(self.positions.values(), key=lambda p: p.unrealized_pnl, reverse=True)
            top_positions = [
                {
                    'symbol': pos.symbol,
                    'pnl': pos.unrealized_pnl,
                    'pnl_pct': (pos.unrealized_pnl / pos.cost_basis) * 100 if pos.cost_basis > 0 else 0,
                    'value': pos.market_value
                }
                for pos in sorted_positions[:5]
            ]
            
            worst_positions = [
                {
                    'symbol': pos.symbol,
                    'pnl': pos.unrealized_pnl,
                    'pnl_pct': (pos.unrealized_pnl / pos.cost_basis) * 100 if pos.cost_basis > 0 else 0,
                    'value': pos.market_value
                }
                for pos in sorted_positions[-5:]
            ]
            
            return {
                'total_positions': len(self.positions),
                'total_value': total_value,
                'total_pnl': total_pnl,
                'unrealized_pnl': total_unrealized,
                'realized_pnl': total_realized,
                'daily_pnl': self._calculate_daily_pnl(),
                'positions_by_status': positions_by_status,
                'positions_by_sector': positions_by_sector,
                'top_positions': top_positions,
                'worst_positions': worst_positions,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {'error': str(e)}
    
    async def get_position_details(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific position."""
        try:
            position = self.positions.get(position_id)
            if not position:
                return None
            
            return {
                'id': position.id,
                'symbol': position.symbol,
                'position_type': position.position_type.value,
                'status': position.status.value,
                'quantity': position.quantity,
                'avg_entry_price': position.avg_entry_price,
                'current_price': position.current_price,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl,
                'total_pnl': position.total_pnl,
                'pnl_pct': (position.unrealized_pnl / position.cost_basis) * 100 if position.cost_basis > 0 else 0,
                'cost_basis': position.cost_basis,
                'daily_var': position.daily_var,
                'beta': position.beta,
                'volatility': position.volatility,
                'max_loss': position.max_loss,
                'max_gain': position.max_gain,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'opened_at': position.opened_at.isoformat(),
                'last_updated': position.last_updated.isoformat(),
                'closed_at': position.closed_at.isoformat() if position.closed_at else None,
                'holding_period_hours': (datetime.now(timezone.utc) - position.opened_at).total_seconds() / 3600,
                'sector': position.sector,
                'strategy': position.strategy,
                'notes': position.notes,
                'trades_count': len(position.trades),
                'trades': [
                    {
                        'id': trade.id,
                        'side': trade.side,
                        'quantity': trade.quantity,
                        'price': trade.price,
                        'value': trade.value,
                        'timestamp': trade.timestamp.isoformat(),
                        'commission': trade.commission
                    }
                    for trade in position.trades
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting position details: {e}")
            return None
    
    async def get_positions_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Get all positions for a specific symbol."""
        try:
            positions = []
            
            if symbol in self.positions_by_symbol:
                for position_id in self.positions_by_symbol[symbol]:
                    if position_id in self.positions:
                        position_data = await self.get_position_details(position_id)
                        if position_data:
                            positions.append(position_data)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions by symbol: {e}")
            return []
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate daily P&L."""
        try:
            # This would calculate P&L since market open
            # For now, return sum of unrealized P&L changes
            daily_pnl = 0.0
            
            for position in self.positions.values():
                # Simplified - would use actual daily price changes
                if position.opened_at.date() == datetime.now().date():
                    daily_pnl += position.unrealized_pnl
                else:
                    # Use a fraction of unrealized P&L as daily change
                    daily_pnl += position.unrealized_pnl * 0.1
            
            return daily_pnl
            
        except Exception:
            return 0.0
    
    async def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions."""
        try:
            # Subscribe to market data updates
            await self.event_bus.subscribe(
                [EventType.MARKET_DATA_UPDATE],
                self._handle_market_data_event
            )
            
            # Subscribe to order events
            await self.event_bus.subscribe(
                [EventType.ORDER_FILLED],
                self._handle_order_filled_event
            )
            
        except Exception as e:
            logger.error(f"Failed to setup event subscriptions: {e}")
    
    async def _handle_market_data_event(self, event: Event) -> None:
        """Handle market data updates."""
        try:
            market_data = event.data
            symbol = market_data.get('symbol')
            price = market_data.get('price')
            
            if symbol and price:
                await self.update_position_prices({symbol: price})
                
        except Exception as e:
            logger.error(f"Error handling market data event: {e}")
    
    async def _handle_order_filled_event(self, event: Event) -> None:
        """Handle order filled events."""
        try:
            order_data = event.data
            symbol = order_data.get('symbol')
            quantity = order_data.get('quantity', 0)
            price = order_data.get('price', 0)
            
            if symbol and quantity and price:
                # Check if this creates a new position or modifies existing
                existing_positions = await self.get_positions_by_symbol(symbol)
                
                if not existing_positions:
                    # Open new position
                    await self.open_position(symbol, quantity, price)
                else:
                    # Add to existing position (simplified)
                    logger.debug(f"Would add to existing position: {symbol}")
                    
        except Exception as e:
            logger.error(f"Error handling order filled event: {e}")
    
    async def _initialize_demo_positions(self) -> None:
        """Initialize demo positions for testing."""
        try:
            demo_positions = [
                ('AAPL', 100, 150.0),
                ('GOOGL', -50, 2500.0),
                ('MSFT', 75, 300.0),
                ('TSLA', 25, 800.0),
            ]
            
            for symbol, quantity, price in demo_positions:
                await self.open_position(symbol, quantity, price)
                
                # Simulate some price movement
                new_price = price * np.random.uniform(0.95, 1.05)
                await self.update_position_prices({symbol: new_price})
                
        except Exception as e:
            logger.error(f"Error initializing demo positions: {e}")
    
    async def _position_monitoring_loop(self) -> None:
        """Background position monitoring loop."""
        while self.is_running:
            try:
                # Update all positions
                for position in self.positions.values():
                    position.calculate_metrics()
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in position monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _stop_loss_monitoring_loop(self) -> None:
        """Monitor stop-loss and take-profit levels."""
        while self.is_running:
            try:
                positions_to_close = []
                
                for position in self.positions.values():
                    if position.status != PositionStatus.OPEN:
                        continue
                    
                    current_price = position.current_price
                    
                    # Check stop-loss
                    if position.stop_loss:
                        if position.position_type == PositionType.LONG and current_price <= position.stop_loss:
                            positions_to_close.append((position.id, "stop_loss"))
                        elif position.position_type == PositionType.SHORT and current_price >= position.stop_loss:
                            positions_to_close.append((position.id, "stop_loss"))
                    
                    # Check take-profit
                    if position.take_profit:
                        if position.position_type == PositionType.LONG and current_price >= position.take_profit:
                            positions_to_close.append((position.id, "take_profit"))
                        elif position.position_type == PositionType.SHORT and current_price <= position.take_profit:
                            positions_to_close.append((position.id, "take_profit"))
                
                # Close positions that hit stops
                for position_id, reason in positions_to_close:
                    logger.info(f"Closing position {position_id} due to {reason}")
                    await self.close_position(position_id)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in stop-loss monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _performance_tracking_loop(self) -> None:
        """Track portfolio performance metrics."""
        while self.is_running:
            try:
                # Record current portfolio state
                portfolio_summary = await self.get_portfolio_summary()
                
                self.portfolio_value_history.append({
                    'timestamp': datetime.now(timezone.utc),
                    'total_value': portfolio_summary.get('total_value', 0),
                    'total_pnl': portfolio_summary.get('total_pnl', 0),
                    'unrealized_pnl': portfolio_summary.get('unrealized_pnl', 0),
                    'position_count': portfolio_summary.get('total_positions', 0)
                })
                
                # Keep limited history
                if len(self.portfolio_value_history) > 1440:  # 24 hours of minute data
                    self.portfolio_value_history = self.portfolio_value_history[-1440:]
                
                await asyncio.sleep(60)  # Record every minute
                
            except Exception as e:
                logger.error(f"Error in performance tracking loop: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self) -> None:
        """Cleanup old closed positions and data."""
        while self.is_running:
            try:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config['keep_closed_positions_days'])
                
                # Clean up old closed positions
                self.closed_positions = [
                    pos for pos in self.closed_positions
                    if pos.closed_at and pos.closed_at > cutoff_date
                ]
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(1800)


# Convenience function
def create_position_manager(**kwargs) -> PositionManager:
    """Create a position manager with configuration."""
    return PositionManager(kwargs)