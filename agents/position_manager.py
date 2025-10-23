"""
Enhanced Position Management System
Handles position tracking, exit strategies, and risk management
"""

import sys
import os
from pathlib import Path

# Add project root to Python path to ensure local config is imported
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from agents.broker_integration import AlpacaBrokerIntegration, OrderRequest, OrderSide, OrderType, TimeInForce
from agents.options_trading_agent import OptionsTrader, OptionsPosition, OptionsStrategy
from config.logging_config import get_logger

logger = get_logger(__name__)

class PositionType(str, Enum):
    """Types of positions"""
    STOCK_LONG = "stock_long"
    STOCK_SHORT = "stock_short"
    OPTIONS = "options"

class ExitReason(str, Enum):
    """Reasons for position exit"""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TIME_BASED = "time_based"
    RISK_MANAGEMENT = "risk_management"
    SIGNAL_REVERSAL = "signal_reversal"
    MANUAL = "manual"
    TRAILING_STOP = "trailing_stop"

@dataclass
class ExitRule:
    """Exit rule configuration"""
    rule_type: ExitReason
    trigger_value: float
    is_percentage: bool = True
    time_limit: Optional[timedelta] = None
    conditions: Optional[Dict[str, Any]] = None
    priority: int = 1  # Lower number = higher priority

@dataclass
class StockPosition:
    """Stock position tracking"""
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    position_type: PositionType
    exit_rules: List[ExitRule] = field(default_factory=list)
    
    # Current state
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_percent: Optional[float] = None
    
    # Exit tracking
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    realized_pnl: Optional[float] = None
    
    # Risk management
    highest_price: Optional[float] = None  # For trailing stops
    lowest_price: Optional[float] = None   # For trailing stops
    max_loss: Optional[float] = None
    max_gain: Optional[float] = None
    
    # Metadata
    strategy_source: str = "unknown"
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_current_price(self, price: float):
        """Update current price and derived metrics"""
        self.current_price = price
        
        # Update P&L
        if self.position_type == PositionType.STOCK_LONG:
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
        
        self.unrealized_pnl_percent = self.unrealized_pnl / (self.entry_price * abs(self.quantity))
        
        # Update price extremes for trailing stops
        if self.highest_price is None or price > self.highest_price:
            self.highest_price = price
        if self.lowest_price is None or price < self.lowest_price:
            self.lowest_price = price
        
        # Update max gain/loss
        if self.max_gain is None or self.unrealized_pnl > self.max_gain:
            self.max_gain = self.unrealized_pnl
        if self.max_loss is None or self.unrealized_pnl < self.max_loss:
            self.max_loss = self.unrealized_pnl
    
    def should_exit(self) -> Tuple[bool, Optional[ExitReason], str]:
        """Check if position should be exited based on rules"""
        
        if not self.current_price or not self.exit_rules:
            return False, None, ""
        
        # Sort rules by priority
        sorted_rules = sorted(self.exit_rules, key=lambda x: x.priority)
        
        for rule in sorted_rules:
            should_exit, reason = self._check_exit_rule(rule)
            if should_exit:
                return True, rule.rule_type, reason
        
        return False, None, ""
    
    def _check_exit_rule(self, rule: ExitRule) -> Tuple[bool, str]:
        """Check individual exit rule"""
        
        if rule.rule_type == ExitReason.TAKE_PROFIT:
            if rule.is_percentage:
                target = self.unrealized_pnl_percent >= rule.trigger_value / 100
            else:
                target = self.unrealized_pnl >= rule.trigger_value
            
            if target:
                return True, f"Take profit triggered: {self.unrealized_pnl_percent:.1%}"
        
        elif rule.rule_type == ExitReason.STOP_LOSS:
            if rule.is_percentage:
                target = self.unrealized_pnl_percent <= -rule.trigger_value / 100
            else:
                target = self.unrealized_pnl <= -rule.trigger_value
            
            if target:
                return True, f"Stop loss triggered: {self.unrealized_pnl_percent:.1%}"
        
        elif rule.rule_type == ExitReason.TIME_BASED:
            if rule.time_limit:
                time_held = datetime.now() - self.entry_time
                if time_held >= rule.time_limit:
                    return True, f"Time limit reached: {time_held.total_seconds() / 3600:.1f}h"
        
        elif rule.rule_type == ExitReason.TRAILING_STOP:
            if self.position_type == PositionType.STOCK_LONG and self.highest_price:
                # Trailing stop for long position
                trail_price = self.highest_price * (1 - rule.trigger_value / 100)
                if self.current_price <= trail_price:
                    return True, f"Trailing stop triggered: ${self.current_price:.2f} <= ${trail_price:.2f}"
            
            elif self.position_type == PositionType.STOCK_SHORT and self.lowest_price:
                # Trailing stop for short position
                trail_price = self.lowest_price * (1 + rule.trigger_value / 100)
                if self.current_price >= trail_price:
                    return True, f"Trailing stop triggered: ${self.current_price:.2f} >= ${trail_price:.2f}"
        
        return False, ""

class PositionManager:
    """Enhanced position management system"""
    
    def __init__(self, broker: Optional[AlpacaBrokerIntegration] = None):
        self.broker = broker
        self.stock_positions: Dict[str, StockPosition] = {}
        self.options_trader = OptionsTrader(broker)
        
        # Default exit rules
        self.default_exit_rules = [
            ExitRule(ExitReason.STOP_LOSS, 10.0, True, priority=1),      # 10% stop loss
            ExitRule(ExitReason.TAKE_PROFIT, 20.0, True, priority=2),    # 20% take profit
            ExitRule(ExitReason.TRAILING_STOP, 5.0, True, priority=3),   # 5% trailing stop
            ExitRule(ExitReason.TIME_BASED, 0, True, timedelta(days=7), priority=4)  # 7 day max hold
        ]
        
        # Risk management settings
        self.max_position_size = 0.05  # 5% of portfolio per position
        self.max_sector_allocation = 0.20  # 20% per sector
        self.max_total_risk = 0.15  # 15% total portfolio risk
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.trade_history: List[Dict] = []
    
    async def open_stock_position(self, symbol: str, quantity: int, entry_price: float, 
                                position_type: PositionType, strategy_source: str = "unknown",
                                confidence: float = 0.5, custom_exit_rules: Optional[List[ExitRule]] = None) -> str:
        """Open a new stock position with exit rules"""
        
        position_id = f"{symbol}_{position_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Use custom rules or defaults
        exit_rules = custom_exit_rules if custom_exit_rules else self.default_exit_rules.copy()
        
        # Adjust exit rules based on confidence
        if confidence > 0.7:  # High confidence - wider stops, higher targets
            for rule in exit_rules:
                if rule.rule_type == ExitReason.STOP_LOSS:
                    rule.trigger_value *= 1.5  # Wider stop loss
                elif rule.rule_type == ExitReason.TAKE_PROFIT:
                    rule.trigger_value *= 1.3  # Higher target
        elif confidence < 0.3:  # Low confidence - tighter stops
            for rule in exit_rules:
                if rule.rule_type == ExitReason.STOP_LOSS:
                    rule.trigger_value *= 0.7  # Tighter stop loss
                elif rule.rule_type == ExitReason.TAKE_PROFIT:
                    rule.trigger_value *= 0.8  # Lower target
        
        position = StockPosition(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=datetime.now(),
            position_type=position_type,
            exit_rules=exit_rules,
            strategy_source=strategy_source,
            confidence=confidence,
            current_price=entry_price
        )
        
        position.update_current_price(entry_price)
        self.stock_positions[position_id] = position
        
        logger.info(f"Opened {position_type} position: {symbol} x{quantity} @ ${entry_price:.2f}, confidence: {confidence:.1%}")
        return position_id
    
    async def update_position_prices(self, price_data: Dict[str, float]):
        """Update all position prices with latest market data"""
        
        for position_id, position in self.stock_positions.items():
            if position.symbol in price_data:
                position.update_current_price(price_data[position.symbol])
    
    async def monitor_positions(self) -> List[Dict]:
        """Monitor all positions and execute exits when needed"""
        
        actions_taken = []
        
        # Monitor stock positions
        for position_id, position in list(self.stock_positions.items()):
            try:
                should_exit, exit_reason, reason_detail = position.should_exit()
                
                if should_exit:
                    success = await self.close_stock_position(position_id, exit_reason, reason_detail)
                    if success:
                        actions_taken.append({
                            'action': 'CLOSE_STOCK',
                            'position_id': position_id,
                            'symbol': position.symbol,
                            'reason': exit_reason,
                            'detail': reason_detail,
                            'pnl': position.realized_pnl
                        })
                
            except Exception as e:
                logger.error(f"Error monitoring position {position_id}: {e}")
        
        # Monitor options positions
        try:
            options_actions = await self.options_trader.monitor_options_positions()
            actions_taken.extend(options_actions)
        except Exception as e:
            logger.error(f"Error monitoring options positions: {e}")
        
        return actions_taken
    
    async def close_stock_position(self, position_id: str, exit_reason: ExitReason, 
                                 reason_detail: str = "") -> bool:
        """Close a stock position"""
        
        if position_id not in self.stock_positions:
            return False
        
        position = self.stock_positions[position_id]
        
        try:
            # Execute closing order if broker available
            if self.broker:
                # Determine closing side
                close_side = OrderSide.SELL if position.position_type == PositionType.STOCK_LONG else OrderSide.BUY
                
                order_request = OrderRequest(
                    symbol=position.symbol,
                    qty=abs(position.quantity),
                    side=close_side,
                    type=OrderType.MARKET,
                    client_order_id=f"CLOSE_{position_id}"
                )
                
                order_response = await self.broker.submit_order(order_request)
                position.exit_price = position.current_price  # Would get actual fill price from order
            else:
                position.exit_price = position.current_price
            
            # Update position with exit info
            position.exit_time = datetime.now()
            position.exit_reason = exit_reason
            position.realized_pnl = position.unrealized_pnl
            
            # Update performance tracking
            self.total_trades += 1
            self.total_pnl += position.realized_pnl
            
            if position.realized_pnl > 0:
                self.winning_trades += 1
            
            # Add to trade history
            trade_record = {
                'symbol': position.symbol,
                'position_type': position.position_type,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'exit_price': position.exit_price,
                'entry_time': position.entry_time,
                'exit_time': position.exit_time,
                'exit_reason': exit_reason,
                'pnl': position.realized_pnl,
                'pnl_percent': position.unrealized_pnl_percent,
                'hold_time': position.exit_time - position.entry_time,
                'strategy_source': position.strategy_source,
                'confidence': position.confidence
            }
            self.trade_history.append(trade_record)
            
            logger.info(f"Closed position {position.symbol}: {exit_reason} - P&L: ${position.realized_pnl:.2f} ({position.unrealized_pnl_percent:.1%})")
            
            # Remove from active positions
            del self.stock_positions[position_id]
            return True
            
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
            return False
    
    async def execute_trade_with_management(self, symbol: str, signal: str, quantity: int, 
                                          price: float, confidence: float, 
                                          strategy_source: str = "unknown") -> Optional[str]:
        """Execute a trade with automatic position management"""
        
        try:
            # Determine position type
            if signal.upper() == "BUY":
                position_type = PositionType.STOCK_LONG
                order_side = OrderSide.BUY
            elif signal.upper() == "SELL":
                position_type = PositionType.STOCK_SHORT
                order_side = OrderSide.SELL
            else:
                return None
            
            # Execute order through broker if available
            if self.broker:
                order_request = OrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    type=OrderType.MARKET
                )
                
                order_response = await self.broker.submit_order(order_request)
                logger.info(f"Executed {signal} order: {symbol} x{quantity} @ ~${price:.2f}, Order ID: {order_response.id}")
            
            # Create managed position
            position_id = await self.open_stock_position(
                symbol=symbol,
                quantity=quantity if signal.upper() == "BUY" else -quantity,
                entry_price=price,
                position_type=position_type,
                strategy_source=strategy_source,
                confidence=confidence
            )
            
            return position_id
            
        except Exception as e:
            logger.error(f"Error executing managed trade for {symbol}: {e}")
            return None
    
    async def execute_options_trade(self, symbol: str, price: float, volatility: float, 
                                  rsi: float, price_change: float) -> Optional[str]:
        """Execute options trade with automatic management"""
        
        try:
            # Get options chain
            contracts = await self.options_trader.get_options_chain(symbol)
            if not contracts:
                return None
            
            # Find best strategy
            strategy_result = self.options_trader.find_best_options_strategy(
                symbol, price, volatility, rsi, price_change
            )
            
            if not strategy_result:
                return None
            
            strategy, selected_contracts = strategy_result
            
            # Execute strategy
            position = await self.options_trader.execute_options_strategy(
                strategy, selected_contracts, quantity=1
            )
            
            if position:
                logger.info(f"Executed options strategy {strategy} for {symbol}")
                return position.symbol
            
        except Exception as e:
            logger.error(f"Error executing options trade for {symbol}: {e}")
        
        return None
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        
        # Stock positions summary
        total_stock_value = 0
        total_unrealized_pnl = 0
        positions_by_symbol = {}
        
        for position_id, position in self.stock_positions.items():
            if position.current_price:
                position_value = abs(position.quantity) * position.current_price
                total_stock_value += position_value
                total_unrealized_pnl += position.unrealized_pnl or 0
                
                if position.symbol not in positions_by_symbol:
                    positions_by_symbol[position.symbol] = []
                positions_by_symbol[position.symbol].append({
                    'position_id': position_id,
                    'quantity': position.quantity,
                    'type': position.position_type,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'pnl': position.unrealized_pnl,
                    'pnl_percent': position.unrealized_pnl_percent
                })
        
        # Options summary
        options_summary = self.options_trader.get_positions_summary()
        
        # Performance metrics
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        avg_trade_pnl = self.total_pnl / self.total_trades if self.total_trades > 0 else 0
        
        return {
            'stock_positions': {
                'total_positions': len(self.stock_positions),
                'total_value': total_stock_value,
                'total_unrealized_pnl': total_unrealized_pnl,
                'positions_by_symbol': positions_by_symbol
            },
            'options_positions': options_summary,
            'performance': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': win_rate,
                'total_pnl': self.total_pnl,
                'avg_trade_pnl': avg_trade_pnl
            },
            'recent_trades': self.trade_history[-10:] if self.trade_history else []
        }
    
    def add_exit_rule_to_position(self, position_id: str, exit_rule: ExitRule) -> bool:
        """Add custom exit rule to existing position"""
        
        if position_id in self.stock_positions:
            self.stock_positions[position_id].exit_rules.append(exit_rule)
            logger.info(f"Added exit rule {exit_rule.rule_type} to position {position_id}")
            return True
        return False
    
    def get_position_details(self, position_id: str) -> Optional[Dict]:
        """Get detailed information about a specific position"""
        
        if position_id in self.stock_positions:
            position = self.stock_positions[position_id]
            return {
                'position_id': position_id,
                'symbol': position.symbol,
                'position_type': position.position_type,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'entry_time': position.entry_time,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_percent': position.unrealized_pnl_percent,
                'highest_price': position.highest_price,
                'lowest_price': position.lowest_price,
                'exit_rules': [
                    {
                        'type': rule.rule_type,
                        'trigger_value': rule.trigger_value,
                        'is_percentage': rule.is_percentage,
                        'time_limit': rule.time_limit,
                        'priority': rule.priority
                    } for rule in position.exit_rules
                ],
                'strategy_source': position.strategy_source,
                'confidence': position.confidence
            }
        return None