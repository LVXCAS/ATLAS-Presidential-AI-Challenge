"""
Event-Driven Backtesting Engine for LangGraph Trading System

This module provides a comprehensive backtesting framework that supports:
- Event-driven simulation with realistic market conditions
- Realistic slippage and commission modeling
- Performance metrics calculation (Sharpe, drawdown, etc.)
- Walk-forward analysis capability
- Multi-strategy backtesting with signal fusion
- Synthetic scenario testing

Requirements: Requirement 4 (Backtesting and Historical Validation)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod
import warnings
import json
from pathlib import Path

# Import existing components
from strategies.technical_indicators import TechnicalIndicator, IndicatorResult


class OrderType(Enum):
    """Order types supported by the backtesting engine"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status tracking"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class MarketData:
    """Market data structure for backtesting"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None


@dataclass
class Order:
    """Order structure for backtesting"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    strategy: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class Trade:
    """Trade execution record"""
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    slippage: float
    strategy: Optional[str] = None
    pnl: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class Position:
    """Position tracking"""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_price: float = 0.0
    strategy: Optional[str] = None


@dataclass
class Portfolio:
    """Portfolio state tracking"""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    total_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    drawdown: float = 0.0
    max_drawdown: float = 0.0
    peak_value: float = 0.0


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    information_ratio: Optional[float] = None
    var_95: float = 0.0
    cvar_95: float = 0.0


class SlippageModel(ABC):
    """Abstract base class for slippage models"""
    
    @abstractmethod
    def calculate_slippage(self, order: Order, market_data: MarketData) -> float:
        """Calculate slippage for an order"""
        pass


class LinearSlippageModel(SlippageModel):
    """Linear slippage model based on order size and spread"""
    
    def __init__(self, base_slippage: float = 0.0001, volume_impact: float = 0.00001):
        self.base_slippage = base_slippage
        self.volume_impact = volume_impact
    
    def calculate_slippage(self, order: Order, market_data: MarketData) -> float:
        """Calculate linear slippage"""
        # Base slippage
        slippage = self.base_slippage
        
        # Volume impact (simplified model)
        if market_data.volume > 0:
            volume_ratio = order.quantity / market_data.volume
            slippage += volume_ratio * self.volume_impact
        
        # Spread impact
        if market_data.spread:
            slippage += market_data.spread * 0.5
        
        return slippage


class CommissionModel(ABC):
    """Abstract base class for commission models"""
    
    @abstractmethod
    def calculate_commission(self, order: Order, fill_price: float) -> float:
        """Calculate commission for an order"""
        pass


class PerShareCommissionModel(CommissionModel):
    """Per-share commission model"""
    
    def __init__(self, per_share: float = 0.005, minimum: float = 1.0):
        self.per_share = per_share
        self.minimum = minimum
    
    def calculate_commission(self, order: Order, fill_price: float) -> float:
        """Calculate per-share commission"""
        commission = order.quantity * self.per_share
        return max(commission, self.minimum)


class PercentageCommissionModel(CommissionModel):
    """Percentage-based commission model"""
    
    def __init__(self, percentage: float = 0.001, minimum: float = 1.0):
        self.percentage = percentage
        self.minimum = minimum
    
    def calculate_commission(self, order: Order, fill_price: float) -> float:
        """Calculate percentage-based commission"""
        commission = order.quantity * fill_price * self.percentage
        return max(commission, self.minimum)


class BacktestingEngine:
    """
    Event-driven backtesting engine with realistic market simulation
    
    Features:
    - Event-driven architecture for realistic simulation
    - Multiple slippage and commission models
    - Comprehensive performance metrics
    - Walk-forward analysis capability
    - Multi-strategy support
    - Synthetic scenario testing
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        slippage_model: Optional[SlippageModel] = None,
        commission_model: Optional[CommissionModel] = None,
        benchmark_symbol: Optional[str] = "SPY",
        random_seed: Optional[int] = None
    ):
        self.initial_capital = initial_capital
        self.slippage_model = slippage_model or LinearSlippageModel()
        self.commission_model = commission_model or PerShareCommissionModel()
        self.benchmark_symbol = benchmark_symbol
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize state
        self.reset()
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def reset(self):
        """Reset backtesting engine to initial state"""
        self.portfolio = Portfolio(cash=self.initial_capital)
        self.portfolio.total_value = self.initial_capital
        self.portfolio.peak_value = self.initial_capital
        
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.portfolio_history: List[Dict] = []
        self.current_time: Optional[datetime] = None
        self.order_counter = 0
        self.trade_counter = 0
        self._price_history = {}
        self._volume_history = {}
        
        # Performance tracking
        self.daily_returns: List[float] = []
        self.benchmark_returns: List[float] = []
        self.equity_curve: List[float] = [self.initial_capital]
        self.drawdown_curve: List[float] = [0.0]
    
    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        strategy: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Submit an order to the backtesting engine"""
        self.order_counter += 1
        order_id = f"ORDER_{self.order_counter:06d}"
        
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=abs(quantity),  # Ensure positive quantity
            price=price,
            stop_price=stop_price,
            timestamp=self.current_time,
            strategy=strategy,
            metadata=metadata or {}
        )
        
        self.orders.append(order)
        self.logger.debug(f"Order submitted: {order_id} - {side.value} {quantity} {symbol}")
        
        return order_id
    
    def process_market_data(self, market_data: MarketData):
        """Process market data and execute pending orders"""
        self.current_time = market_data.timestamp
        
        # Update positions with current market prices
        self._update_positions(market_data)
        
        # Process pending orders
        self._process_orders(market_data)
        
        # Update portfolio metrics
        self._update_portfolio_metrics()
        
        # Record portfolio state
        self._record_portfolio_state()
    
    def _update_positions(self, market_data: MarketData):
        """Update position values with current market data"""
        if market_data.symbol in self.portfolio.positions:
            position = self.portfolio.positions[market_data.symbol]
            position.last_price = market_data.close
            position.market_value = position.quantity * market_data.close
            
            if position.quantity != 0:
                position.unrealized_pnl = (market_data.close - position.avg_price) * position.quantity
    
    def _process_orders(self, market_data: MarketData):
        """Process pending orders against market data"""
        orders_to_process = []
        
        for order in self.orders:
            if order.status != OrderStatus.PENDING or order.symbol != market_data.symbol:
                continue
            
            # Check if order should be filled
            fill_price = self._get_fill_price(order, market_data)
            
            if fill_price is not None:
                self._execute_order(order, fill_price, market_data)
                orders_to_process.append(order)
    
    def _get_fill_price(self, order: Order, market_data: MarketData) -> Optional[float]:
        """Determine if order should be filled and at what price"""
        if order.order_type == OrderType.MARKET:
            # Market orders fill at current price
            return market_data.close
        
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and market_data.low <= order.price:
                return order.price  # Fill at limit price
            elif order.side == OrderSide.SELL and market_data.high >= order.price:
                return order.price  # Fill at limit price
        
        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY and market_data.high >= order.stop_price:
                return market_data.close
            elif order.side == OrderSide.SELL and market_data.low <= order.stop_price:
                return market_data.close
        
        return None
    
    def _execute_order(self, order: Order, fill_price: float, market_data: MarketData):
        """Execute an order at the given fill price"""
        # Calculate slippage
        slippage = self.slippage_model.calculate_slippage(order, market_data)
        
        # Apply slippage to fill price
        if order.side == OrderSide.BUY:
            actual_fill_price = fill_price * (1 + slippage)
        else:
            actual_fill_price = fill_price * (1 - slippage)
        
        # Calculate commission
        commission = self.commission_model.calculate_commission(order, actual_fill_price)
        
        # Check if we have enough cash for buy orders
        total_cost = order.quantity * actual_fill_price + commission
        if order.side == OrderSide.BUY and total_cost > self.portfolio.cash:
            order.status = OrderStatus.REJECTED
            self.logger.warning(f"Order {order.id} rejected: insufficient cash")
            return
        
        # Execute the trade
        self.trade_counter += 1
        trade_id = f"TRADE_{self.trade_counter:06d}"
        
        trade = Trade(
            id=trade_id,
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=actual_fill_price,
            timestamp=self.current_time,
            commission=commission,
            slippage=slippage,
            strategy=order.strategy,
            metadata=order.metadata
        )
        
        self.trades.append(trade)
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = actual_fill_price
        order.commission = commission
        order.slippage = slippage
        
        # Update portfolio
        self._update_portfolio_from_trade(trade)
        
        self.logger.info(f"Trade executed: {trade_id} - {trade.side.value} {trade.quantity} {trade.symbol} @ {actual_fill_price:.4f}")
    
    def _update_portfolio_from_trade(self, trade: Trade):
        """Update portfolio state from executed trade"""
        symbol = trade.symbol
        
        # Initialize position if it doesn't exist
        if symbol not in self.portfolio.positions:
            self.portfolio.positions[symbol] = Position(symbol=symbol, strategy=trade.strategy)
        
        position = self.portfolio.positions[symbol]
        
        if trade.side == OrderSide.BUY:
            # Update average price for long positions
            if position.quantity >= 0:
                total_cost = position.quantity * position.avg_price + trade.quantity * trade.price
                position.quantity += trade.quantity
                position.avg_price = total_cost / position.quantity if position.quantity > 0 else 0
            else:
                # Covering short position
                if trade.quantity >= abs(position.quantity):
                    # Full cover plus new long
                    realized_pnl = (position.avg_price - trade.price) * abs(position.quantity)
                    position.realized_pnl += realized_pnl
                    self.portfolio.realized_pnl += realized_pnl
                    
                    remaining_quantity = trade.quantity - abs(position.quantity)
                    position.quantity = remaining_quantity
                    position.avg_price = trade.price if remaining_quantity > 0 else 0
                else:
                    # Partial cover
                    realized_pnl = (position.avg_price - trade.price) * trade.quantity
                    position.realized_pnl += realized_pnl
                    self.portfolio.realized_pnl += realized_pnl
                    position.quantity += trade.quantity
            
            # Update cash
            self.portfolio.cash -= trade.quantity * trade.price + trade.commission
        
        else:  # SELL
            # Update for short positions or selling long
            if position.quantity > 0:
                # Selling long position
                if trade.quantity >= position.quantity:
                    # Full sale plus new short
                    realized_pnl = (trade.price - position.avg_price) * position.quantity
                    position.realized_pnl += realized_pnl
                    self.portfolio.realized_pnl += realized_pnl
                    
                    remaining_quantity = trade.quantity - position.quantity
                    position.quantity = -remaining_quantity
                    position.avg_price = trade.price if remaining_quantity > 0 else 0
                else:
                    # Partial sale
                    realized_pnl = (trade.price - position.avg_price) * trade.quantity
                    position.realized_pnl += realized_pnl
                    self.portfolio.realized_pnl += realized_pnl
                    position.quantity -= trade.quantity
            else:
                # Adding to short position
                if position.quantity <= 0:
                    total_proceeds = abs(position.quantity) * position.avg_price + trade.quantity * trade.price
                    position.quantity -= trade.quantity
                    position.avg_price = total_proceeds / abs(position.quantity) if position.quantity < 0 else 0
            
            # Update cash
            self.portfolio.cash += trade.quantity * trade.price - trade.commission
    
    def _update_portfolio_metrics(self):
        """Update portfolio-level metrics"""
        # Calculate total portfolio value
        total_position_value = sum(pos.market_value for pos in self.portfolio.positions.values())
        self.portfolio.total_value = self.portfolio.cash + total_position_value
        
        # Calculate unrealized P&L
        self.portfolio.unrealized_pnl = sum(pos.unrealized_pnl for pos in self.portfolio.positions.values())
        
        # Calculate total P&L
        self.portfolio.total_pnl = self.portfolio.realized_pnl + self.portfolio.unrealized_pnl
        
        # Update peak value and drawdown
        if self.portfolio.total_value > self.portfolio.peak_value:
            self.portfolio.peak_value = self.portfolio.total_value
        
        self.portfolio.drawdown = (self.portfolio.peak_value - self.portfolio.total_value) / self.portfolio.peak_value
        self.portfolio.max_drawdown = max(self.portfolio.max_drawdown, self.portfolio.drawdown)
    
    def _record_portfolio_state(self):
        """Record current portfolio state for analysis"""
        state = {
            'timestamp': self.current_time,
            'total_value': self.portfolio.total_value,
            'cash': self.portfolio.cash,
            'realized_pnl': self.portfolio.realized_pnl,
            'unrealized_pnl': self.portfolio.unrealized_pnl,
            'total_pnl': self.portfolio.total_pnl,
            'drawdown': self.portfolio.drawdown,
            'positions': {symbol: {
                'quantity': pos.quantity,
                'avg_price': pos.avg_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl
            } for symbol, pos in self.portfolio.positions.items()}
        }
        
        self.portfolio_history.append(state)
        
        # Update equity curve and returns
        self.equity_curve.append(self.portfolio.total_value)
        self.drawdown_curve.append(self.portfolio.drawdown)
        
        if len(self.equity_curve) > 1:
            daily_return = (self.portfolio.total_value - self.equity_curve[-2]) / self.equity_curve[-2]
            self.daily_returns.append(daily_return)
    
    def run_backtest(
        self,
        market_data: List[MarketData],
        strategy_func: Callable,
        strategy_params: Optional[Dict] = None
    ) -> Dict:
        """
        Run a complete backtest with the given market data and strategy
        
        Args:
            market_data: List of MarketData objects in chronological order
            strategy_func: Strategy function that takes (engine, market_data, params)
            strategy_params: Parameters to pass to the strategy function
        
        Returns:
            Dictionary containing backtest results and performance metrics
        """
        self.reset()
        strategy_params = strategy_params or {}
        
        self.logger.info(f"Starting backtest with {len(market_data)} data points")
        
        # Process each market data point
        for i, data in enumerate(market_data):
            # Process market data (updates positions and executes orders)
            self.process_market_data(data)
            
            # Run strategy logic
            try:
                strategy_func(self, data, strategy_params)
            except Exception as e:
                self.logger.error(f"Strategy error at {data.timestamp}: {e}")
                continue
            
            # Progress logging
            if i % 1000 == 0:
                self.logger.info(f"Processed {i}/{len(market_data)} data points")
        
        # Calculate final performance metrics
        performance_metrics = self.calculate_performance_metrics()
        
        # Prepare results
        results = {
            'performance_metrics': performance_metrics,
            'portfolio_history': self.portfolio_history,
            'trades': [self._trade_to_dict(trade) for trade in self.trades],
            'final_portfolio': self._portfolio_to_dict(self.portfolio),
            'equity_curve': self.equity_curve,
            'drawdown_curve': self.drawdown_curve,
            'daily_returns': self.daily_returns
        }
        
        self.logger.info(f"Backtest completed. Final value: ${self.portfolio.total_value:,.2f}")
        
        return results
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if len(self.daily_returns) == 0:
            return PerformanceMetrics(
                total_return=0.0, annualized_return=0.0, volatility=0.0,
                sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
                max_drawdown=0.0, max_drawdown_duration=0, win_rate=0.0,
                profit_factor=0.0, avg_win=0.0, avg_loss=0.0, total_trades=0,
                winning_trades=0, losing_trades=0, largest_win=0.0,
                largest_loss=0.0, avg_trade_duration=0.0
            )
        
        returns = np.array(self.daily_returns)
        
        # Basic return metrics
        total_return = (self.portfolio.total_value - self.initial_capital) / self.initial_capital
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Risk-adjusted metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 1 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / self.portfolio.max_drawdown if self.portfolio.max_drawdown > 0 else 0
        
        # Drawdown duration
        max_drawdown_duration = self._calculate_max_drawdown_duration()
        
        # Trade-based metrics
        trade_pnls = []
        for trade in self.trades:
            if trade.pnl is not None:
                trade_pnls.append(trade.pnl)
        
        if not trade_pnls:
            # Calculate P&L from position changes
            trade_pnls = self._calculate_trade_pnls()
        
        winning_trades = len([pnl for pnl in trade_pnls if pnl > 0])
        losing_trades = len([pnl for pnl in trade_pnls if pnl < 0])
        total_trades = len(trade_pnls)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        winning_pnls = [pnl for pnl in trade_pnls if pnl > 0]
        losing_pnls = [pnl for pnl in trade_pnls if pnl < 0]
        
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        largest_win = max(winning_pnls) if winning_pnls else 0
        largest_loss = min(losing_pnls) if losing_pnls else 0
        
        gross_profit = sum(winning_pnls) if winning_pnls else 0
        gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns) > 0 and np.any(returns <= var_95) else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=self.portfolio.max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration=0.0,  # TODO: Calculate based on position holding periods
            var_95=var_95,
            cvar_95=cvar_95
        )
    
    def _calculate_max_drawdown_duration(self) -> int:
        """Calculate maximum drawdown duration in days"""
        if not self.drawdown_curve:
            return 0
        
        max_duration = 0
        current_duration = 0
        
        for drawdown in self.drawdown_curve:
            if drawdown > 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def _calculate_trade_pnls(self) -> List[float]:
        """Calculate P&L for each trade (simplified)"""
        # This is a simplified calculation
        # In practice, you'd want to track round-trip trades
        trade_pnls = []
        
        for trade in self.trades:
            # Simplified P&L calculation
            if trade.side == OrderSide.BUY:
                # For buy trades, we'll estimate P&L based on subsequent price movement
                # This is simplified - real implementation would track position lifecycle
                pnl = 0.0  # Placeholder
            else:
                pnl = 0.0  # Placeholder
            
            trade_pnls.append(pnl)
        
        return trade_pnls
    
    def _trade_to_dict(self, trade: Trade) -> Dict:
        """Convert trade to dictionary for serialization"""
        return {
            'id': trade.id,
            'order_id': trade.order_id,
            'symbol': trade.symbol,
            'side': trade.side.value,
            'quantity': trade.quantity,
            'price': trade.price,
            'timestamp': trade.timestamp.isoformat(),
            'commission': trade.commission,
            'slippage': trade.slippage,
            'strategy': trade.strategy,
            'pnl': trade.pnl if trade.pnl is not None else 0.0,
            'metadata': trade.metadata
        }
    
    def _portfolio_to_dict(self, portfolio: Portfolio) -> Dict:
        """Convert portfolio to dictionary for serialization"""
        return {
            'cash': portfolio.cash,
            'total_value': portfolio.total_value,
            'unrealized_pnl': portfolio.unrealized_pnl,
            'realized_pnl': portfolio.realized_pnl,
            'total_pnl': portfolio.total_pnl,
            'drawdown': portfolio.drawdown,
            'max_drawdown': portfolio.max_drawdown,
            'peak_value': portfolio.peak_value,
            'positions': {symbol: {
                'quantity': pos.quantity,
                'avg_price': pos.avg_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'realized_pnl': pos.realized_pnl,
                'last_price': pos.last_price,
                'strategy': pos.strategy
            } for symbol, pos in portfolio.positions.items()}
        }
    
    def walk_forward_analysis(
        self,
        market_data: List[MarketData],
        strategy_func: Callable,
        training_period: int = 252,  # 1 year
        testing_period: int = 63,   # 3 months
        step_size: int = 21,        # 1 month
        strategy_params: Optional[Dict] = None
    ) -> Dict:
        """
        Perform walk-forward analysis
        
        Args:
            market_data: Historical market data
            strategy_func: Strategy function to test
            training_period: Number of periods for training
            testing_period: Number of periods for testing
            step_size: Step size for rolling window
            strategy_params: Base strategy parameters
        
        Returns:
            Dictionary containing walk-forward analysis results
        """
        self.logger.info("Starting walk-forward analysis")
        
        results = []
        total_periods = len(market_data)
        
        for start_idx in range(0, total_periods - training_period - testing_period, step_size):
            train_end_idx = start_idx + training_period
            test_end_idx = train_end_idx + testing_period
            
            if test_end_idx > total_periods:
                break
            
            # Training data
            train_data = market_data[start_idx:train_end_idx]
            
            # Testing data
            test_data = market_data[train_end_idx:test_end_idx]
            
            # Run backtest on test period
            test_results = self.run_backtest(test_data, strategy_func, strategy_params)
            
            period_result = {
                'period': len(results) + 1,
                'train_start': train_data[0].timestamp,
                'train_end': train_data[-1].timestamp,
                'test_start': test_data[0].timestamp,
                'test_end': test_data[-1].timestamp,
                'performance': test_results['performance_metrics']
            }
            
            results.append(period_result)
            
            self.logger.info(f"Completed walk-forward period {len(results)}")
        
        # Aggregate results
        all_returns = []
        all_sharpe_ratios = []
        all_max_drawdowns = []
        
        for result in results:
            perf = result['performance']
            all_returns.append(perf.total_return)
            all_sharpe_ratios.append(perf.sharpe_ratio)
            all_max_drawdowns.append(perf.max_drawdown)
        
        aggregate_metrics = {
            'avg_return': np.mean(all_returns),
            'std_return': np.std(all_returns),
            'avg_sharpe': np.mean(all_sharpe_ratios),
            'avg_max_drawdown': np.mean(all_max_drawdowns),
            'consistency_ratio': len([r for r in all_returns if r > 0]) / len(all_returns) if all_returns else 0
        }
        
        return {
            'periods': results,
            'aggregate_metrics': aggregate_metrics,
            'total_periods': len(results)
        }
    
    def synthetic_scenario_testing(
        self,
        base_data: List[MarketData],
        strategy_func: Callable,
        scenarios: List[str] = None,
        strategy_params: Optional[Dict] = None
    ) -> Dict:
        """
        Test strategy against synthetic market scenarios
        
        Args:
            base_data: Base market data to modify
            strategy_func: Strategy function to test
            scenarios: List of scenario names to test
            strategy_params: Strategy parameters
        
        Returns:
            Dictionary containing scenario test results
        """
        if scenarios is None:
            scenarios = [
                'trending_up', 'trending_down', 'mean_reverting',
                'high_volatility', 'low_volatility', 'news_shock',
                'flash_crash', 'gradual_decline', 'sideways_market',
                'volatility_spike'
            ]
        
        self.logger.info(f"Running synthetic scenario testing for {len(scenarios)} scenarios")
        
        results = {}
        
        for scenario in scenarios:
            # Generate synthetic data for scenario
            synthetic_data = self._generate_synthetic_scenario(base_data, scenario)
            
            # Run backtest
            scenario_results = self.run_backtest(synthetic_data, strategy_func, strategy_params)
            
            results[scenario] = {
                'performance': scenario_results['performance_metrics'],
                'final_value': scenario_results['final_portfolio']['total_value'],
                'max_drawdown': scenario_results['performance_metrics'].max_drawdown,
                'total_trades': scenario_results['performance_metrics'].total_trades
            }
            
            self.logger.info(f"Completed scenario: {scenario}")
        
        return results
    
    def _generate_synthetic_scenario(self, base_data: List[MarketData], scenario: str) -> List[MarketData]:
        """Generate synthetic market data for testing scenarios"""
        # Use a fixed seed for scenario generation to ensure reproducibility
        scenario_seed = hash(scenario) % 2**32
        np.random.seed(scenario_seed)
        
        synthetic_data = []
        
        for i, data in enumerate(base_data):
            new_data = MarketData(
                timestamp=data.timestamp,
                symbol=data.symbol,
                open=data.open,
                high=data.high,
                low=data.low,
                close=data.close,
                volume=data.volume,
                vwap=data.vwap,
                bid=data.bid,
                ask=data.ask,
                spread=data.spread
            )
            
            # Apply scenario modifications
            if scenario == 'trending_up':
                trend_factor = 1 + (i / len(base_data)) * 0.5  # 50% uptrend
                new_data.close *= trend_factor
                new_data.high *= trend_factor
                new_data.low *= trend_factor
                new_data.open *= trend_factor
            
            elif scenario == 'trending_down':
                trend_factor = 1 - (i / len(base_data)) * 0.3  # 30% downtrend
                new_data.close *= trend_factor
                new_data.high *= trend_factor
                new_data.low *= trend_factor
                new_data.open *= trend_factor
            
            elif scenario == 'high_volatility':
                volatility_multiplier = 3.0
                price_change = (new_data.close - new_data.open) * volatility_multiplier
                new_data.close = new_data.open + price_change
                new_data.high = max(new_data.open, new_data.close) * 1.02
                new_data.low = min(new_data.open, new_data.close) * 0.98
            
            elif scenario == 'low_volatility':
                volatility_multiplier = 0.3
                price_change = (new_data.close - new_data.open) * volatility_multiplier
                new_data.close = new_data.open + price_change
                new_data.high = max(new_data.open, new_data.close) * 1.001
                new_data.low = min(new_data.open, new_data.close) * 0.999
            
            elif scenario == 'flash_crash' and i == len(base_data) // 2:
                # Flash crash in the middle
                new_data.close *= 0.9  # 10% drop
                new_data.low *= 0.85   # 15% intraday drop
            
            elif scenario == 'news_shock':
                # Random news shocks
                if np.random.random() < 0.05:  # 5% chance per period
                    shock_magnitude = np.random.normal(0, 0.03)  # 3% std dev
                    new_data.close *= (1 + shock_magnitude)
                    new_data.high *= (1 + max(0, shock_magnitude))
                    new_data.low *= (1 + min(0, shock_magnitude))
            
            # Add more scenarios as needed...
            
            synthetic_data.append(new_data)
        
        return synthetic_data
    
    def generate_report(self, results: Dict, output_path: Optional[str] = None) -> str:
        """Generate a comprehensive backtest report"""
        performance = results['performance_metrics']
        
        report = f"""
# Backtesting Report

## Summary
- **Total Return**: {performance.total_return:.2%}
- **Annualized Return**: {performance.annualized_return:.2%}
- **Volatility**: {performance.volatility:.2%}
- **Sharpe Ratio**: {performance.sharpe_ratio:.2f}
- **Maximum Drawdown**: {performance.max_drawdown:.2%}

## Risk Metrics
- **Sortino Ratio**: {performance.sortino_ratio:.2f}
- **Calmar Ratio**: {performance.calmar_ratio:.2f}
- **VaR (95%)**: {performance.var_95:.2%}
- **CVaR (95%)**: {performance.cvar_95:.2%}
- **Max Drawdown Duration**: {performance.max_drawdown_duration} days

## Trading Statistics
- **Total Trades**: {performance.total_trades}
- **Win Rate**: {performance.win_rate:.2%}
- **Profit Factor**: {performance.profit_factor:.2f}
- **Average Win**: ${performance.avg_win:.2f}
- **Average Loss**: ${performance.avg_loss:.2f}
- **Largest Win**: ${performance.largest_win:.2f}
- **Largest Loss**: ${performance.largest_loss:.2f}

## Final Portfolio
- **Total Value**: ${results['final_portfolio']['total_value']:,.2f}
- **Cash**: ${results['final_portfolio']['cash']:,.2f}
- **Realized P&L**: ${results['final_portfolio']['realized_pnl']:,.2f}
- **Unrealized P&L**: ${results['final_portfolio']['unrealized_pnl']:,.2f}
"""
        
        if output_path:
            Path(output_path).write_text(report)
            self.logger.info(f"Report saved to {output_path}")
        
        return report
    
    def _trade_to_dict(self, trade: Trade) -> Dict:
        """Convert Trade object to dictionary"""
        return {
            'id': trade.id,
            'order_id': trade.order_id,
            'symbol': trade.symbol,
            'side': trade.side.value,
            'quantity': trade.quantity,
            'price': trade.price,
            'timestamp': trade.timestamp.isoformat(),
            'commission': trade.commission,
            'slippage': trade.slippage,
            'strategy': trade.strategy,
            'pnl': trade.pnl,
            'metadata': trade.metadata
        }
    
    def _portfolio_to_dict(self, portfolio: Portfolio) -> Dict:
        """Convert Portfolio object to dictionary"""
        return {
            'cash': portfolio.cash,
            'total_value': portfolio.total_value,
            'unrealized_pnl': portfolio.unrealized_pnl,
            'realized_pnl': portfolio.realized_pnl,
            'total_pnl': portfolio.total_pnl,
            'drawdown': portfolio.drawdown,
            'max_drawdown': portfolio.max_drawdown,
            'peak_value': portfolio.peak_value,
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl,
                    'last_price': pos.last_price,
                    'strategy': pos.strategy
                } for symbol, pos in portfolio.positions.items()
            }
        }


# Example strategy functions for testing
def simple_momentum_strategy(engine: BacktestingEngine, market_data: MarketData, params: Dict):
    """Simple momentum strategy for testing"""
    # This is a placeholder - in practice, you'd use the actual strategy agents
    # For now, implement a simple moving average crossover
    
    # Get parameters
    short_window = params.get('short_window', 10)
    long_window = params.get('long_window', 30)
    
    # This would normally use historical data to calculate moving averages
    # For this example, we'll use a simplified approach
    
    # Initialize price history
    if not hasattr(engine, '_price_history'):
        engine._price_history = {}
    
    symbol = market_data.symbol
    
    if symbol not in engine._price_history:
        engine._price_history[symbol] = []
        
    # Update history
    engine._price_history[symbol].append(market_data.close)
    
    prices = engine._price_history[symbol]

    if len(prices) >= long_window:
        short_ma = np.mean(prices[-short_window:])
        long_ma = np.mean(prices[-long_window:])
        
        # Get current position
        current_position = engine.portfolio.positions.get(market_data.symbol)
        current_qty = current_position.quantity if current_position else 0
        
        # Simple crossover strategy
        if short_ma > long_ma and current_qty <= 0:
            # Buy signal
            order_qty = 100  # Fixed quantity for simplicity
            engine.submit_order(
                symbol=market_data.symbol,
                side=OrderSide.BUY,
                quantity=order_qty,
                order_type=OrderType.MARKET,
                strategy='momentum'
            )
        
        elif short_ma < long_ma and current_qty > 0:
            # Sell signal
            engine.submit_order(
                symbol=market_data.symbol,
                side=OrderSide.SELL,
                quantity=current_qty,
                order_type=OrderType.MARKET,
                strategy='momentum'
            )


def buy_and_hold_strategy(engine: BacktestingEngine, market_data: MarketData, params: Dict):
    """Simple buy and hold strategy for testing"""
    # Buy once at the beginning
    if not hasattr(engine, '_initial_buy_done'):
        engine._initial_buy_done = True
        
        # Calculate quantity based on available cash
        quantity = int(engine.portfolio.cash * 0.95 / market_data.close)  # Use 95% of cash
        
        if quantity > 0:
            engine.submit_order(
                symbol=market_data.symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
                strategy='buy_and_hold'
            )