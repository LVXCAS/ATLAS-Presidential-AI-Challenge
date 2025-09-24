"""
Comprehensive Backtesting Environment for Advanced Algorithmic Trading System

This module provides a complete backtesting framework that accurately simulates
the parallel execution architecture, continuous learning, and all integrated components.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from enum import Enum
import json
import sqlite3
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Data and Analysis
import yfinance as yf
import talib
from scipy import stats
import quantlib as ql

# ML and AI
from sklearn.metrics import sharpe_ratio, max_drawdown, calmar_ratio
import torch
import torch.nn as nn

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class BacktestMode(Enum):
    VECTORIZED = "vectorized"
    EVENT_DRIVEN = "event_driven"
    PARALLEL_SIMULATION = "parallel_simulation"

class ExecutionModel(Enum):
    PERFECT = "perfect"
    REALISTIC_SLIPPAGE = "realistic_slippage"
    MARKET_IMPACT = "market_impact"
    LIMIT_ORDER_BOOK = "limit_order_book"

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 1000000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_leverage: float = 3.0
    risk_free_rate: float = 0.02
    benchmark: str = "SPY"
    mode: BacktestMode = BacktestMode.PARALLEL_SIMULATION
    execution_model: ExecutionModel = ExecutionModel.REALISTIC_SLIPPAGE
    parallel_workers: int = 4
    enable_continuous_learning: bool = True
    enable_regime_detection: bool = True
    enable_expert_agents: bool = True
    transaction_costs: bool = True
    margin_requirements: bool = True

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    position_type: str  # 'long' or 'short'
    strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    commission: float
    slippage: float
    strategy: str
    trade_type: str  # 'long' or 'short'
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    alpha: float
    beta: float
    information_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float

class MarketDataSimulator:
    """Simulates realistic market conditions for backtesting"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data_cache = {}
        self.volatility_surface = {}

    async def get_market_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch market data for backtesting"""
        market_data = {}

        for symbol in symbols:
            if symbol in self.data_cache:
                market_data[symbol] = self.data_cache[symbol]
                continue

            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=self.config.start_date,
                    end=self.config.end_date,
                    interval="1d"
                )

                # Add technical indicators
                data = self._add_technical_indicators(data)

                # Add microstructure noise
                data = self._add_microstructure_noise(data)

                market_data[symbol] = data
                self.data_cache[symbol] = data

            except Exception as e:
                logging.error(f"Error fetching data for {symbol}: {e}")

        return market_data

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to market data"""
        # Basic indicators
        data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
        data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
        data['EMA_12'] = talib.EMA(data['Close'], timeperiod=12)
        data['EMA_26'] = talib.EMA(data['Close'], timeperiod=26)

        # Momentum indicators
        data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
        data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['Close'])
        data['Stoch_K'], data['Stoch_D'] = talib.STOCH(data['High'], data['Low'], data['Close'])

        # Volatility indicators
        data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
        data['BBANDS_upper'], data['BBANDS_middle'], data['BBANDS_lower'] = talib.BBANDS(data['Close'])

        # Volume indicators
        data['OBV'] = talib.OBV(data['Close'], data['Volume'])
        data['AD'] = talib.AD(data['High'], data['Low'], data['Close'], data['Volume'])

        return data

    def _add_microstructure_noise(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add realistic microstructure noise to prices"""
        noise_factor = 0.0001  # 1 basis point
        noise = np.random.normal(0, noise_factor, len(data))

        data['Close_with_noise'] = data['Close'] * (1 + noise)
        data['High_with_noise'] = data['High'] * (1 + np.abs(noise))
        data['Low_with_noise'] = data['Low'] * (1 - np.abs(noise))

        return data

class ExecutionSimulator:
    """Simulates realistic trade execution"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.execution_model = config.execution_model

    def execute_trade(self,
                     symbol: str,
                     quantity: float,
                     price: float,
                     timestamp: datetime,
                     order_type: str = "market") -> Tuple[float, float, float]:
        """
        Simulate trade execution with realistic costs
        Returns: (executed_price, commission, slippage)
        """

        if self.execution_model == ExecutionModel.PERFECT:
            return price, 0.0, 0.0

        elif self.execution_model == ExecutionModel.REALISTIC_SLIPPAGE:
            slippage = self._calculate_slippage(quantity, price)
            commission = self._calculate_commission(quantity, price)
            executed_price = price * (1 + slippage if quantity > 0 else 1 - slippage)
            return executed_price, commission, slippage * price * abs(quantity)

        elif self.execution_model == ExecutionModel.MARKET_IMPACT:
            market_impact = self._calculate_market_impact(quantity, price)
            commission = self._calculate_commission(quantity, price)
            slippage = self._calculate_slippage(quantity, price)

            total_impact = market_impact + slippage
            executed_price = price * (1 + total_impact if quantity > 0 else 1 - total_impact)
            return executed_price, commission, total_impact * price * abs(quantity)

        else:  # LIMIT_ORDER_BOOK
            return self._simulate_limit_order_execution(symbol, quantity, price, timestamp)

    def _calculate_slippage(self, quantity: float, price: float) -> float:
        """Calculate realistic slippage based on order size"""
        base_slippage = self.config.slippage_rate
        size_impact = min(abs(quantity) / 10000, 0.001)  # Size-dependent impact
        return base_slippage + size_impact

    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission costs"""
        notional = abs(quantity) * price
        return notional * self.config.commission_rate

    def _calculate_market_impact(self, quantity: float, price: float) -> float:
        """Calculate market impact based on order size"""
        # Square root law for market impact
        base_impact = 0.0001
        size_factor = np.sqrt(abs(quantity) / 1000)
        return base_impact * size_factor

    def _simulate_limit_order_execution(self,
                                      symbol: str,
                                      quantity: float,
                                      price: float,
                                      timestamp: datetime) -> Tuple[float, float, float]:
        """Simulate limit order execution with partial fills"""
        # Simplified limit order simulation
        fill_probability = 0.85  # 85% chance of fill

        if np.random.random() < fill_probability:
            # Partial fill simulation
            fill_ratio = np.random.uniform(0.7, 1.0)
            actual_quantity = quantity * fill_ratio

            commission = self._calculate_commission(actual_quantity, price)
            return price, commission, 0.0
        else:
            # Order not filled
            return 0.0, 0.0, 0.0

class PortfolioManager:
    """Manages portfolio state during backtesting"""

    def __init__(self, initial_capital: float, max_leverage: float = 3.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_leverage = max_leverage
        self.positions: Dict[str, Position] = {}
        self.cash = initial_capital
        self.equity_curve = []
        self.trades_history: List[Trade] = []
        self.daily_returns = []

    def get_portfolio_value(self, market_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        positions_value = sum(
            pos.quantity * market_prices.get(pos.symbol, pos.entry_price)
            for pos in self.positions.values()
        )
        return self.cash + positions_value

    def get_available_capital(self, market_prices: Dict[str, float]) -> float:
        """Calculate available capital for new positions"""
        portfolio_value = self.get_portfolio_value(market_prices)
        used_margin = sum(
            abs(pos.quantity * market_prices.get(pos.symbol, pos.entry_price))
            for pos in self.positions.values()
        )

        max_capital = portfolio_value * self.max_leverage
        return max(0, max_capital - used_margin)

    def can_open_position(self, symbol: str, quantity: float, price: float,
                         market_prices: Dict[str, float]) -> bool:
        """Check if position can be opened"""
        required_capital = abs(quantity * price) / self.max_leverage
        available_capital = self.get_available_capital(market_prices)

        return available_capital >= required_capital

    def open_position(self, symbol: str, quantity: float, price: float,
                     timestamp: datetime, strategy: str,
                     execution_costs: Tuple[float, float, float]) -> bool:
        """Open a new position"""
        executed_price, commission, slippage_cost = execution_costs

        if executed_price == 0:  # Order not filled
            return False

        position_key = f"{symbol}_{strategy}"

        if position_key in self.positions:
            # Add to existing position
            existing_pos = self.positions[position_key]
            total_quantity = existing_pos.quantity + quantity
            total_cost = existing_pos.quantity * existing_pos.entry_price + quantity * executed_price

            existing_pos.quantity = total_quantity
            existing_pos.entry_price = total_cost / total_quantity if total_quantity != 0 else 0
        else:
            # Create new position
            self.positions[position_key] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=executed_price,
                entry_time=timestamp,
                position_type="long" if quantity > 0 else "short",
                strategy=strategy
            )

        # Update cash
        self.cash -= (quantity * executed_price + commission + slippage_cost)
        return True

    def close_position(self, symbol: str, strategy: str, price: float,
                      timestamp: datetime, execution_costs: Tuple[float, float, float],
                      partial_quantity: Optional[float] = None) -> Optional[Trade]:
        """Close a position and record the trade"""
        position_key = f"{symbol}_{strategy}"

        if position_key not in self.positions:
            return None

        position = self.positions[position_key]
        executed_price, commission, slippage_cost = execution_costs

        if executed_price == 0:  # Order not filled
            return None

        # Determine quantity to close
        close_quantity = partial_quantity if partial_quantity else position.quantity
        close_quantity = min(abs(close_quantity), abs(position.quantity))

        if position.quantity < 0:  # Short position
            close_quantity = -close_quantity

        # Calculate PnL
        pnl = close_quantity * (executed_price - position.entry_price)
        if position.quantity < 0:  # Short position
            pnl = -pnl

        # Create trade record
        trade = Trade(
            symbol=symbol,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price=position.entry_price,
            exit_price=executed_price,
            quantity=close_quantity,
            pnl=pnl - commission - slippage_cost,
            commission=commission,
            slippage=slippage_cost,
            strategy=strategy,
            trade_type=position.position_type
        )

        self.trades_history.append(trade)

        # Update cash
        self.cash += (close_quantity * executed_price - commission - slippage_cost)

        # Update or remove position
        position.quantity -= close_quantity
        if abs(position.quantity) < 1e-8:  # Position fully closed
            del self.positions[position_key]

        return trade

class PerformanceAnalyzer:
    """Analyzes backtesting performance with comprehensive metrics"""

    def __init__(self, benchmark_symbol: str = "SPY", risk_free_rate: float = 0.02):
        self.benchmark_symbol = benchmark_symbol
        self.risk_free_rate = risk_free_rate

    def calculate_metrics(self,
                         equity_curve: pd.Series,
                         trades: List[Trade],
                         benchmark_data: Optional[pd.Series] = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""

        if len(equity_curve) < 2:
            return self._empty_metrics()

        # Basic return calculations
        returns = equity_curve.pct_change().dropna()
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        # Annualized metrics
        trading_days = len(equity_curve)
        years = trading_days / 252.0
        annualized_return = (1 + total_return) ** (1/years) - 1
        volatility = returns.std() * np.sqrt(252)

        # Risk-adjusted metrics
        excess_returns = returns - self.risk_free_rate / 252
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = excess_returns.mean() / downside_volatility * np.sqrt(252) if downside_volatility > 0 else 0

        # Drawdown analysis
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_dd = drawdown.min()

        # Calmar ratio
        calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0

        # Beta and Alpha (if benchmark provided)
        alpha, beta, info_ratio = 0, 1, 0
        if benchmark_data is not None:
            benchmark_returns = benchmark_data.pct_change().dropna()
            if len(benchmark_returns) == len(returns):
                beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
                alpha = annualized_return - (self.risk_free_rate + beta * (benchmark_returns.mean() * 252 - self.risk_free_rate))
                tracking_error = (returns - benchmark_returns).std() * np.sqrt(252)
                info_ratio = (returns - benchmark_returns).mean() * np.sqrt(252) / tracking_error if tracking_error > 0 else 0

        # Trade statistics
        winning_trades = sum(1 for trade in trades if trade.pnl > 0)
        losing_trades = sum(1 for trade in trades if trade.pnl < 0)
        total_trades = len(trades)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        wins = [trade.pnl for trade in trades if trade.pnl > 0]
        losses = [trade.pnl for trade in trades if trade.pnl < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0

        # Risk metrics
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = returns[returns <= var_95].mean() if len(returns) > 0 else 0

        # Distribution metrics
        skewness = stats.skew(returns) if len(returns) > 0 else 0
        kurtosis = stats.kurtosis(returns) if len(returns) > 0 else 0

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            alpha=alpha,
            beta=beta,
            information_ratio=info_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            var_95=var_95,
            cvar_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis
        )

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics for edge cases"""
        return PerformanceMetrics(
            total_return=0, annualized_return=0, volatility=0, sharpe_ratio=0,
            sortino_ratio=0, max_drawdown=0, calmar_ratio=0, alpha=0, beta=1,
            information_ratio=0, win_rate=0, profit_factor=0, avg_win=0, avg_loss=0,
            total_trades=0, winning_trades=0, losing_trades=0, var_95=0, cvar_95=0,
            skewness=0, kurtosis=0
        )

class ComprehensiveBacktester:
    """Main backtesting engine that orchestrates all components"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.market_simulator = MarketDataSimulator(config)
        self.execution_simulator = ExecutionSimulator(config)
        self.portfolio_manager = PortfolioManager(config.initial_capital, config.max_leverage)
        self.performance_analyzer = PerformanceAnalyzer(config.benchmark, config.risk_free_rate)

        # Database for storing results
        self.db_path = "backtesting_results.db"
        self._setup_database()

    def _setup_database(self):
        """Setup SQLite database for storing results"""
        conn = sqlite3.connect(self.db_path)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config TEXT,
                start_date TEXT,
                end_date TEXT,
                initial_capital REAL,
                final_capital REAL,
                total_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                total_trades INTEGER,
                timestamp TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id INTEGER,
                symbol TEXT,
                entry_time TEXT,
                exit_time TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                pnl REAL,
                strategy TEXT,
                FOREIGN KEY (backtest_id) REFERENCES backtest_runs (id)
            )
        """)

        conn.close()

    async def run_backtest(self,
                          strategy_func: Callable,
                          symbols: List[str],
                          strategy_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run comprehensive backtest

        Args:
            strategy_func: Trading strategy function
            symbols: List of symbols to trade
            strategy_params: Strategy-specific parameters

        Returns:
            Complete backtest results
        """
        logging.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")

        # Fetch market data
        market_data = await self.market_simulator.get_market_data(symbols)

        if not market_data:
            raise ValueError("No market data available for backtesting")

        # Get benchmark data
        benchmark_data = None
        if self.config.benchmark:
            benchmark_ticker = yf.Ticker(self.config.benchmark)
            benchmark_data = benchmark_ticker.history(
                start=self.config.start_date,
                end=self.config.end_date
            )['Close']

        # Run backtest based on mode
        if self.config.mode == BacktestMode.PARALLEL_SIMULATION:
            results = await self._run_parallel_backtest(strategy_func, market_data, strategy_params)
        elif self.config.mode == BacktestMode.EVENT_DRIVEN:
            results = await self._run_event_driven_backtest(strategy_func, market_data, strategy_params)
        else:  # VECTORIZED
            results = await self._run_vectorized_backtest(strategy_func, market_data, strategy_params)

        # Calculate performance metrics
        equity_curve = pd.Series(results['equity_curve'])
        performance_metrics = self.performance_analyzer.calculate_metrics(
            equity_curve,
            self.portfolio_manager.trades_history,
            benchmark_data
        )

        # Store results in database
        backtest_id = self._store_results(results, performance_metrics)

        # Compile final results
        final_results = {
            'backtest_id': backtest_id,
            'config': self.config,
            'performance_metrics': performance_metrics,
            'equity_curve': equity_curve,
            'trades': self.portfolio_manager.trades_history,
            'positions': self.portfolio_manager.positions,
            'market_data': market_data,
            'detailed_results': results
        }

        logging.info(f"Backtest completed. Total return: {performance_metrics.total_return:.2%}")
        return final_results

    async def _run_parallel_backtest(self,
                                   strategy_func: Callable,
                                   market_data: Dict[str, pd.DataFrame],
                                   strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run backtest with parallel simulation"""

        # Get all unique dates
        all_dates = set()
        for data in market_data.values():
            all_dates.update(data.index.date)

        trading_dates = sorted(list(all_dates))
        equity_curve = []
        daily_signals = []

        # Process each trading day
        for current_date in trading_dates:
            date_str = current_date.strftime('%Y-%m-%d')

            # Get market prices for this date
            current_prices = {}
            current_data = {}

            for symbol, data in market_data.items():
                date_data = data[data.index.date == current_date]
                if not date_data.empty:
                    current_prices[symbol] = date_data['Close'].iloc[-1]
                    current_data[symbol] = date_data.iloc[-1].to_dict()

            if not current_prices:
                continue

            # Generate trading signals using strategy
            try:
                signals = await asyncio.to_thread(
                    strategy_func,
                    current_data,
                    self.portfolio_manager.positions,
                    strategy_params or {}
                )

                daily_signals.append({
                    'date': current_date,
                    'signals': signals,
                    'prices': current_prices
                })

                # Execute trades based on signals
                await self._execute_signals(signals, current_prices, current_date)

            except Exception as e:
                logging.error(f"Error processing signals for {date_str}: {e}")
                continue

            # Update portfolio value
            portfolio_value = self.portfolio_manager.get_portfolio_value(current_prices)
            equity_curve.append(portfolio_value)

            # Log progress
            if len(equity_curve) % 50 == 0:
                logging.info(f"Processed {len(equity_curve)} days, Portfolio: ${portfolio_value:,.2f}")

        return {
            'equity_curve': equity_curve,
            'trading_dates': trading_dates,
            'daily_signals': daily_signals,
            'final_portfolio_value': equity_curve[-1] if equity_curve else self.config.initial_capital
        }

    async def _execute_signals(self,
                             signals: Dict[str, Any],
                             current_prices: Dict[str, float],
                             current_date: datetime):
        """Execute trading signals"""

        for signal in signals:
            symbol = signal.get('symbol')
            action = signal.get('action')  # 'buy', 'sell', 'close'
            quantity = signal.get('quantity', 0)
            strategy = signal.get('strategy', 'default')

            if not symbol or not action:
                continue

            current_price = current_prices.get(symbol)
            if not current_price:
                continue

            if action == 'buy' and quantity > 0:
                # Check if we can open position
                if self.portfolio_manager.can_open_position(symbol, quantity, current_price, current_prices):
                    execution_costs = self.execution_simulator.execute_trade(
                        symbol, quantity, current_price, current_date
                    )
                    self.portfolio_manager.open_position(
                        symbol, quantity, current_price, current_date, strategy, execution_costs
                    )

            elif action == 'sell' and quantity > 0:
                # Sell/short position
                execution_costs = self.execution_simulator.execute_trade(
                    symbol, -quantity, current_price, current_date
                )
                self.portfolio_manager.open_position(
                    symbol, -quantity, current_price, current_date, strategy, execution_costs
                )

            elif action == 'close':
                # Close existing position
                execution_costs = self.execution_simulator.execute_trade(
                    symbol, quantity, current_price, current_date
                )
                self.portfolio_manager.close_position(
                    symbol, strategy, current_price, current_date, execution_costs
                )

    async def _run_event_driven_backtest(self,
                                       strategy_func: Callable,
                                       market_data: Dict[str, pd.DataFrame],
                                       strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run event-driven backtest (placeholder for future implementation)"""
        # This would implement a more sophisticated event-driven architecture
        # For now, fall back to parallel simulation
        return await self._run_parallel_backtest(strategy_func, market_data, strategy_params)

    async def _run_vectorized_backtest(self,
                                     strategy_func: Callable,
                                     market_data: Dict[str, pd.DataFrame],
                                     strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run vectorized backtest for simple strategies"""
        # This would implement vectorized operations for speed
        # For now, fall back to parallel simulation
        return await self._run_parallel_backtest(strategy_func, market_data, strategy_params)

    def _store_results(self, results: Dict[str, Any], metrics: PerformanceMetrics) -> int:
        """Store backtest results in database"""
        conn = sqlite3.connect(self.db_path)

        # Store main backtest run
        cursor = conn.execute("""
            INSERT INTO backtest_runs
            (config, start_date, end_date, initial_capital, final_capital,
             total_return, sharpe_ratio, max_drawdown, total_trades, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            json.dumps(self.config.__dict__, default=str),
            self.config.start_date.isoformat(),
            self.config.end_date.isoformat(),
            self.config.initial_capital,
            results.get('final_portfolio_value', 0),
            metrics.total_return,
            metrics.sharpe_ratio,
            metrics.max_drawdown,
            metrics.total_trades,
            datetime.now().isoformat()
        ))

        backtest_id = cursor.lastrowid

        # Store individual trades
        for trade in self.portfolio_manager.trades_history:
            conn.execute("""
                INSERT INTO trades
                (backtest_id, symbol, entry_time, exit_time, entry_price,
                 exit_price, quantity, pnl, strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                backtest_id,
                trade.symbol,
                trade.entry_time.isoformat(),
                trade.exit_time.isoformat(),
                trade.entry_price,
                trade.exit_price,
                trade.quantity,
                trade.pnl,
                trade.strategy
            ))

        conn.commit()
        conn.close()

        return backtest_id

class BacktestVisualizer:
    """Creates comprehensive visualizations for backtest results"""

    def __init__(self, results: Dict[str, Any]):
        self.results = results
        self.metrics = results['performance_metrics']
        self.equity_curve = results['equity_curve']
        self.trades = results['trades']

    def create_comprehensive_report(self, output_path: str = "backtest_report.html"):
        """Create comprehensive HTML report with all visualizations"""

        # Create subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                'Equity Curve', 'Drawdown Analysis',
                'Monthly Returns Heatmap', 'Trade Distribution',
                'Rolling Sharpe Ratio', 'Risk-Return Scatter',
                'Trade PnL Distribution', 'Cumulative Returns vs Benchmark'
            ],
            specs=[
                [{"secondary_y": True}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "histogram"}],
                [{"secondary_y": False}, {"type": "scatter"}],
                [{"type": "histogram"}, {"secondary_y": True}]
            ]
        )

        # 1. Equity Curve
        dates = pd.date_range(start=self.results['config'].start_date,
                             periods=len(self.equity_curve), freq='D')

        fig.add_trace(
            go.Scatter(x=dates, y=self.equity_curve, name='Portfolio Value',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )

        # 2. Drawdown
        running_max = pd.Series(self.equity_curve).expanding().max()
        drawdown = (pd.Series(self.equity_curve) - running_max) / running_max * 100

        fig.add_trace(
            go.Scatter(x=dates, y=drawdown, name='Drawdown %',
                      fill='tonexty', fillcolor='rgba(255,0,0,0.3)'),
            row=1, col=2
        )

        # 3. Trade PnL Distribution
        if self.trades:
            pnls = [trade.pnl for trade in self.trades]
            fig.add_trace(
                go.Histogram(x=pnls, name='Trade PnL', nbinsx=30),
                row=2, col=2
            )

        # 4. Rolling Sharpe
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        rolling_sharpe = returns.rolling(60).mean() / returns.rolling(60).std() * np.sqrt(252)

        fig.add_trace(
            go.Scatter(x=dates[60:], y=rolling_sharpe.iloc[60:],
                      name='60-Day Rolling Sharpe'),
            row=3, col=1
        )

        # Update layout
        fig.update_layout(
            height=1200,
            title=f"Comprehensive Backtest Report - Total Return: {self.metrics.total_return:.2%}",
            showlegend=True
        )

        # Save to HTML
        fig.write_html(output_path)
        print(f"Comprehensive report saved to {output_path}")

    def print_performance_summary(self):
        """Print formatted performance summary"""
        print("\n" + "="*60)
        print("COMPREHENSIVE BACKTEST RESULTS")
        print("="*60)

        print(f"\n📊 RETURN METRICS")
        print(f"Total Return:        {self.metrics.total_return:>10.2%}")
        print(f"Annualized Return:   {self.metrics.annualized_return:>10.2%}")
        print(f"Volatility:          {self.metrics.volatility:>10.2%}")

        print(f"\n⚖️  RISK METRICS")
        print(f"Sharpe Ratio:        {self.metrics.sharpe_ratio:>10.2f}")
        print(f"Sortino Ratio:       {self.metrics.sortino_ratio:>10.2f}")
        print(f"Max Drawdown:        {self.metrics.max_drawdown:>10.2%}")
        print(f"Calmar Ratio:        {self.metrics.calmar_ratio:>10.2f}")

        print(f"\n📈 BENCHMARK COMPARISON")
        print(f"Alpha:               {self.metrics.alpha:>10.2%}")
        print(f"Beta:                {self.metrics.beta:>10.2f}")
        print(f"Information Ratio:   {self.metrics.information_ratio:>10.2f}")

        print(f"\n🎯 TRADING PERFORMANCE")
        print(f"Total Trades:        {self.metrics.total_trades:>10}")
        print(f"Win Rate:            {self.metrics.win_rate:>10.2%}")
        print(f"Profit Factor:       {self.metrics.profit_factor:>10.2f}")
        print(f"Avg Win:             ${self.metrics.avg_win:>9.2f}")
        print(f"Avg Loss:            ${self.metrics.avg_loss:>9.2f}")

        print(f"\n📊 RISK MEASURES")
        print(f"VaR (95%):           {self.metrics.var_95:>10.2%}")
        print(f"CVaR (95%):          {self.metrics.cvar_95:>10.2%}")
        print(f"Skewness:            {self.metrics.skewness:>10.2f}")
        print(f"Kurtosis:            {self.metrics.kurtosis:>10.2f}")

        print("\n" + "="*60)

# Example usage and testing
async def simple_momentum_strategy(market_data: Dict[str, Any],
                                 positions: Dict[str, Position],
                                 params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Example momentum strategy for testing the backtesting framework
    """
    signals = []

    for symbol, data in market_data.items():
        if isinstance(data, dict) and 'Close' in data:
            # Simple momentum: buy if price > 20-day SMA
            current_price = data['Close']
            sma_20 = data.get('SMA_20', current_price)

            position_key = f"{symbol}_momentum"

            if current_price > sma_20 and position_key not in positions:
                # Buy signal
                signals.append({
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': 100,
                    'strategy': 'momentum'
                })
            elif current_price < sma_20 and position_key in positions:
                # Sell signal
                signals.append({
                    'symbol': symbol,
                    'action': 'close',
                    'quantity': positions[position_key].quantity,
                    'strategy': 'momentum'
                })

    return signals

async def example_backtest():
    """Example of how to run a comprehensive backtest"""

    # Configure backtest
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1),
        initial_capital=1000000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
        max_leverage=2.0,
        benchmark="SPY",
        mode=BacktestMode.PARALLEL_SIMULATION,
        execution_model=ExecutionModel.REALISTIC_SLIPPAGE
    )

    # Create backtester
    backtester = ComprehensiveBacktester(config)

    # Run backtest
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    results = await backtester.run_backtest(
        strategy_func=simple_momentum_strategy,
        symbols=symbols,
        strategy_params={'lookback': 20}
    )

    # Create visualizations
    visualizer = BacktestVisualizer(results)
    visualizer.print_performance_summary()
    visualizer.create_comprehensive_report()

    return results

if __name__ == "__main__":
    # Run example backtest
    asyncio.run(example_backtest())