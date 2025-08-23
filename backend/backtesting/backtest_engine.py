"""
Comprehensive backtesting engine with realistic market simulation, portfolio tracking, and performance analytics.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
import uuid
import numpy as np
import pandas as pd

from .market_simulator import MarketSimulator, Order, OrderSide, OrderStatus, Fill
from agents.base_agent import TradingSignal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Portfolio position tracking."""
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_timestamp: datetime = None
    last_updated: datetime = None
    trades: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def market_value(self) -> float:
        return abs(self.quantity) * self.current_price
    
    @property 
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl
    
    def update_price(self, price: float, timestamp: datetime):
        """Update position with current market price."""
        self.current_price = price
        self.last_updated = timestamp
        
        # Calculate unrealized P&L
        if self.quantity != 0:
            if self.quantity > 0:  # Long position
                self.unrealized_pnl = self.quantity * (price - self.avg_entry_price)
            else:  # Short position
                self.unrealized_pnl = abs(self.quantity) * (self.avg_entry_price - price)
        else:
            self.unrealized_pnl = 0.0


@dataclass
class Trade:
    """Individual trade record."""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    slippage: float
    signal_id: Optional[str] = None
    pnl: float = 0.0


@dataclass
class BacktestResults:
    """Complete backtest results with performance metrics."""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    total_commission: float
    total_slippage: float
    trades: List[Trade]
    daily_returns: List[float]
    equity_curve: List[Tuple[datetime, float]]
    positions_history: List[Dict[str, Position]]
    performance_metrics: Dict[str, Any]


class Portfolio:
    """Portfolio management for backtesting."""
    
    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []
        
        # Performance tracking
        self.peak_equity = initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.max_drawdown_duration = 0
        self.drawdown_start = None
        self.current_drawdown_duration = 0
        
        logger.info(f"Portfolio initialized with ${initial_capital:,.2f}")
    
    @property
    def total_equity(self) -> float:
        """Calculate total portfolio equity."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    @property
    def positions_value(self) -> float:
        """Total value of all positions."""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def realized_pnl(self) -> float:
        """Total realized P&L."""
        return sum(pos.realized_pnl for pos in self.positions.values())
    
    def update_prices(self, price_data: Dict[str, float], timestamp: datetime):
        """Update all position prices and calculate equity."""
        for symbol, position in self.positions.items():
            if symbol in price_data:
                position.update_price(price_data[symbol], timestamp)
        
        # Track equity curve
        current_equity = self.total_equity
        self.equity_curve.append((timestamp, current_equity))
        
        # Update drawdown tracking
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            if self.drawdown_start:
                # End of drawdown period
                self.drawdown_start = None
                self.current_drawdown_duration = 0
        else:
            # Calculate current drawdown
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            
            if not self.drawdown_start:
                self.drawdown_start = timestamp
                self.current_drawdown_duration = 0
            else:
                self.current_drawdown_duration = (timestamp - self.drawdown_start).days
            
            # Update max drawdown
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
            
            if self.current_drawdown_duration > self.max_drawdown_duration:
                self.max_drawdown_duration = self.current_drawdown_duration
    
    def execute_trade(self, fill: Fill, signal_id: Optional[str] = None) -> Trade:
        """Execute a trade from a fill."""
        # Create trade record
        trade = Trade(
            id=str(uuid.uuid4()),
            symbol=fill.order_id.split('_')[1] if '_' in fill.order_id else 'UNKNOWN',
            side='buy' if fill.quantity > 0 else 'sell',
            quantity=abs(fill.quantity),
            price=fill.price,
            timestamp=fill.timestamp,
            commission=fill.commission,
            slippage=fill.slippage,
            signal_id=signal_id
        )
        
        symbol = trade.symbol
        trade_value = trade.quantity * trade.price
        
        # Update cash
        if trade.side == 'buy':
            self.cash -= (trade_value + trade.commission)
        else:
            self.cash += (trade_value - trade.commission)
        
        # Update or create position
        if symbol not in self.positions:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=trade.quantity if trade.side == 'buy' else -trade.quantity,
                avg_entry_price=trade.price,
                entry_timestamp=trade.timestamp
            )
        else:
            # Update existing position
            pos = self.positions[symbol]
            
            if trade.side == 'buy':
                # Adding to position or reducing short
                if pos.quantity >= 0:  # Long or flat
                    # Adding to long position
                    total_cost = (pos.quantity * pos.avg_entry_price) + (trade.quantity * trade.price)
                    total_quantity = pos.quantity + trade.quantity
                    pos.avg_entry_price = total_cost / total_quantity if total_quantity > 0 else 0
                    pos.quantity = total_quantity
                else:  # Short position
                    # Reducing short position
                    if trade.quantity >= abs(pos.quantity):
                        # Closing short and potentially going long
                        closing_quantity = abs(pos.quantity)
                        realized_pnl = closing_quantity * (pos.avg_entry_price - trade.price)
                        pos.realized_pnl += realized_pnl
                        trade.pnl = realized_pnl
                        
                        remaining_quantity = trade.quantity - closing_quantity
                        if remaining_quantity > 0:
                            # Going long with remaining quantity
                            pos.quantity = remaining_quantity
                            pos.avg_entry_price = trade.price
                        else:
                            # Flat
                            pos.quantity = 0
                            pos.avg_entry_price = 0
                    else:
                        # Partially reducing short position
                        realized_pnl = trade.quantity * (pos.avg_entry_price - trade.price)
                        pos.realized_pnl += realized_pnl
                        pos.quantity += trade.quantity  # Still negative
                        trade.pnl = realized_pnl
            
            else:  # sell
                # Reducing position or adding to short
                if pos.quantity > 0:  # Long position
                    # Reducing long position
                    if trade.quantity >= pos.quantity:
                        # Closing long and potentially going short
                        closing_quantity = pos.quantity
                        realized_pnl = closing_quantity * (trade.price - pos.avg_entry_price)
                        pos.realized_pnl += realized_pnl
                        trade.pnl = realized_pnl
                        
                        remaining_quantity = trade.quantity - closing_quantity
                        if remaining_quantity > 0:
                            # Going short with remaining quantity
                            pos.quantity = -remaining_quantity
                            pos.avg_entry_price = trade.price
                        else:
                            # Flat
                            pos.quantity = 0
                            pos.avg_entry_price = 0
                    else:
                        # Partially reducing long position
                        realized_pnl = trade.quantity * (trade.price - pos.avg_entry_price)
                        pos.realized_pnl += realized_pnl
                        pos.quantity -= trade.quantity
                        trade.pnl = realized_pnl
                else:  # Short or flat position
                    # Adding to short position
                    if pos.quantity == 0:
                        # Opening short position
                        pos.quantity = -trade.quantity
                        pos.avg_entry_price = trade.price
                    else:
                        # Adding to existing short position
                        total_cost = (abs(pos.quantity) * pos.avg_entry_price) + (trade.quantity * trade.price)
                        total_quantity = abs(pos.quantity) + trade.quantity
                        pos.avg_entry_price = total_cost / total_quantity
                        pos.quantity = -total_quantity
        
        # Add trade to position history
        self.positions[symbol].trades.append({
            'id': trade.id,
            'timestamp': trade.timestamp.isoformat(),
            'side': trade.side,
            'quantity': trade.quantity,
            'price': trade.price,
            'commission': trade.commission,
            'slippage': trade.slippage,
            'pnl': trade.pnl
        })
        
        self.trades.append(trade)
        
        logger.debug(f"Trade executed: {trade.side} {trade.quantity} {symbol} @ ${trade.price:.4f}")
        
        return trade
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        return self.positions.get(symbol)
    
    def close_position(self, symbol: str, price: float, timestamp: datetime) -> Optional[Trade]:
        """Close entire position for a symbol."""
        if symbol not in self.positions or self.positions[symbol].quantity == 0:
            return None
        
        pos = self.positions[symbol]
        quantity = abs(pos.quantity)
        side = 'sell' if pos.quantity > 0 else 'buy'
        
        # Create synthetic fill for closing trade
        fill = Fill(
            order_id=f"close_{symbol}",
            timestamp=timestamp,
            price=price,
            quantity=quantity if side == 'sell' else -quantity,
            commission=quantity * 0.005,  # Default commission
            slippage=0.0
        )
        
        return self.execute_trade(fill)


class BacktestEngine:
    """Comprehensive backtesting engine with realistic market simulation."""
    
    def __init__(self, initial_capital: float = 1000000.0, config: Dict[str, Any] = None):
        self.initial_capital = initial_capital
        self.config = config or {}
        
        # Initialize components
        self.market_simulator = MarketSimulator(config.get('market_simulator', {}))
        self.portfolio = Portfolio(initial_capital)
        
        # Backtest state
        self.current_time = None
        self.start_time = None
        self.end_time = None
        self.data_feed: Optional[Callable] = None
        self.strategy_signals: List[TradingSignal] = []
        
        # Performance tracking
        self.benchmark_returns: List[float] = []
        self.benchmark_data: Dict[str, List[float]] = {}
        
        logger.info(f"BacktestEngine initialized with ${initial_capital:,.2f}")
    
    def add_data_feed(self, data_feed: Callable[[datetime], Dict[str, Dict[str, float]]]):
        """Add historical data feed function."""
        self.data_feed = data_feed
    
    def add_benchmark(self, symbol: str, data: List[Dict[str, float]]):
        """Add benchmark data for comparison."""
        self.benchmark_data[symbol] = data
    
    async def run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        strategy_function: Callable[[datetime, Dict[str, Any]], List[TradingSignal]],
        symbols: List[str],
        frequency: str = '1D'
    ) -> BacktestResults:
        """
        Run complete backtest simulation.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date  
            strategy_function: Function that generates trading signals
            symbols: List of symbols to trade
            frequency: Data frequency ('1D', '1H', '15m', etc.)
        """
        logger.info(f"Starting backtest: {start_date} to {end_date}")
        
        self.start_time = start_date
        self.end_time = end_date
        self.current_time = start_date
        
        # Generate time periods
        time_periods = self._generate_time_periods(start_date, end_date, frequency)
        
        previous_equity = self.initial_capital
        
        for i, timestamp in enumerate(time_periods):
            self.current_time = timestamp
            
            # Get market data for this timestamp
            if self.data_feed:
                market_data = self.data_feed(timestamp)
            else:
                # Use mock data if no feed provided
                market_data = self._generate_mock_data(symbols, timestamp)
            
            # Update market simulator with new data
            for symbol, ohlcv in market_data.items():
                if symbol in symbols:
                    self.market_simulator.update_market_data(symbol, ohlcv, timestamp)
            
            # Update portfolio positions with current prices
            current_prices = {symbol: data['close'] for symbol, data in market_data.items()}
            self.portfolio.update_prices(current_prices, timestamp)
            
            # Process any pending orders
            fills = self.market_simulator.process_orders()
            for fill in fills:
                self.portfolio.execute_trade(fill)
            
            # Generate strategy signals
            context = {
                'current_prices': current_prices,
                'portfolio': self.portfolio,
                'timestamp': timestamp,
                'market_data': market_data
            }
            
            signals = strategy_function(timestamp, context)
            
            # Process signals into orders
            for signal in signals:
                await self._process_signal(signal)
            
            # Calculate daily returns
            current_equity = self.portfolio.total_equity
            if i > 0:
                daily_return = (current_equity - previous_equity) / previous_equity
                self.portfolio.daily_returns.append(daily_return)
            
            previous_equity = current_equity
            
            # Log progress
            if i % max(1, len(time_periods) // 10) == 0:
                progress = (i / len(time_periods)) * 100
                logger.info(f"Backtest progress: {progress:.1f}% - Equity: ${current_equity:,.2f}")
        
        # Generate final results
        return self._generate_results()
    
    async def _process_signal(self, signal: TradingSignal):
        """Process a trading signal into market orders."""
        # Calculate position size based on signal strength and risk management
        position_size = self._calculate_position_size(signal)
        
        if position_size <= 0:
            return
        
        # Determine order side
        if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            side = OrderSide.BUY
        elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            side = OrderSide.SELL
        else:
            return  # HOLD signal
        
        # Create order
        order = Order(
            id=f"signal_{signal.id}_{uuid.uuid4().hex[:8]}",
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            side=side,
            quantity=position_size,
            price=None,  # Market order
            order_type='market'
        )
        
        # Submit to market simulator
        self.market_simulator.submit_order(order)
        
        logger.debug(f"Signal processed: {signal.signal_type.value} {signal.symbol} - Size: {position_size}")
    
    def _calculate_position_size(self, signal: TradingSignal) -> float:
        """Calculate position size based on signal and risk management."""
        # Get current market data
        if signal.symbol not in self.market_simulator.market_data:
            return 0.0
        
        market_data = self.market_simulator.market_data[signal.symbol]
        current_price = market_data.close
        
        # Base position size (percentage of equity)
        equity = self.portfolio.total_equity
        base_allocation = self.config.get('base_position_size', 0.02)  # 2% default
        
        # Adjust based on signal strength
        strength_multiplier = signal.confidence * signal.strength
        
        # Risk-adjusted position size
        position_value = equity * base_allocation * strength_multiplier
        position_size = position_value / current_price
        
        # Apply position limits
        max_position_value = equity * self.config.get('max_position_size', 0.10)  # 10% max
        max_position_size = max_position_value / current_price
        
        return min(position_size, max_position_size)
    
    def _generate_time_periods(self, start: datetime, end: datetime, frequency: str) -> List[datetime]:
        """Generate time periods for backtesting."""
        periods = []
        current = start
        
        if frequency == '1D':
            delta = timedelta(days=1)
        elif frequency == '1H':
            delta = timedelta(hours=1)
        elif frequency == '15m':
            delta = timedelta(minutes=15)
        else:
            delta = timedelta(days=1)  # Default
        
        while current <= end:
            periods.append(current)
            current += delta
        
        return periods
    
    def _generate_mock_data(self, symbols: List[str], timestamp: datetime) -> Dict[str, Dict[str, float]]:
        """Generate mock market data for testing."""
        data = {}
        
        for symbol in symbols:
            # Simple random walk with trend
            base_price = 100.0
            volatility = 0.02
            trend = 0.0005
            
            # Add some randomness based on timestamp
            np.random.seed(int(timestamp.timestamp()) + hash(symbol))
            price_change = np.random.normal(trend, volatility)
            price = base_price * (1 + price_change)
            
            data[symbol] = {
                'open': price * 0.999,
                'high': price * 1.01,
                'low': price * 0.995,
                'close': price,
                'volume': np.random.randint(100000, 1000000)
            }
        
        return data
    
    def _generate_results(self) -> BacktestResults:
        """Generate comprehensive backtest results."""
        final_equity = self.portfolio.total_equity
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Calculate annualized return
        days = (self.end_time - self.start_time).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Calculate volatility (annualized)
        if len(self.portfolio.daily_returns) > 1:
            volatility = np.std(self.portfolio.daily_returns) * np.sqrt(252)  # Annualized
        else:
            volatility = 0.0
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate win rate
        winning_trades = [t for t in self.portfolio.trades if t.pnl > 0]
        losing_trades = [t for t in self.portfolio.trades if t.pnl < 0]
        win_rate = len(winning_trades) / len(self.portfolio.trades) if self.portfolio.trades else 0
        
        # Calculate profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate average trade duration
        trade_durations = []
        for symbol, position in self.portfolio.positions.items():
            if len(position.trades) >= 2:
                for i in range(len(position.trades) - 1):
                    start = datetime.fromisoformat(position.trades[i]['timestamp'])
                    end = datetime.fromisoformat(position.trades[i + 1]['timestamp'])
                    duration = (end - start).total_seconds() / 3600  # Hours
                    trade_durations.append(duration)
        
        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
        
        # Calculate total costs
        total_commission = sum(t.commission for t in self.portfolio.trades)
        total_slippage = sum(t.slippage for t in self.portfolio.trades)
        
        return BacktestResults(
            start_date=self.start_time,
            end_date=self.end_time,
            initial_capital=self.initial_capital,
            final_capital=final_equity,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=self.portfolio.max_drawdown,
            max_drawdown_duration=self.portfolio.max_drawdown_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.portfolio.trades),
            avg_trade_duration=avg_trade_duration,
            total_commission=total_commission,
            total_slippage=total_slippage,
            trades=self.portfolio.trades,
            daily_returns=self.portfolio.daily_returns,
            equity_curve=self.portfolio.equity_curve,
            positions_history=[dict(self.portfolio.positions)],
            performance_metrics={
                'total_return_pct': total_return * 100,
                'annualized_return_pct': annualized_return * 100,
                'volatility_pct': volatility * 100,
                'max_drawdown_pct': self.portfolio.max_drawdown * 100,
                'win_rate_pct': win_rate * 100,
                'avg_winning_trade': np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
                'avg_losing_trade': np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
                'largest_winner': max([t.pnl for t in winning_trades]) if winning_trades else 0,
                'largest_loser': min([t.pnl for t in losing_trades]) if losing_trades else 0,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'net_profit': gross_profit + gross_loss,
                'total_cost': total_commission + total_slippage
            }
        )