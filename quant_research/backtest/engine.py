"""Core backtesting engine."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, date
import asyncio
from dataclasses import dataclass, field
from enum import Enum

from ..config import BacktestConfig, StrategyBacktestConfig
from ..data.sources.manager import DataSourceManager
from ..data.sources.base import DataRequest
from .portfolio import Portfolio
from .execution import ExecutionEngine
from .risk import RiskManager
from .performance import PerformanceAnalyzer
from structlog import get_logger

logger = get_logger(__name__)


class BacktestStatus(Enum):
    """Backtest execution status."""
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BacktestResult:
    """Backtest execution results."""
    
    status: BacktestStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Performance metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Portfolio data
    portfolio_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    positions_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Benchmark comparison
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    correlation: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0  # Value at Risk
    cvar_95: float = 0.0  # Conditional Value at Risk
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """Backtest duration in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class BacktestEngine:
    """Core backtesting engine."""
    
    def __init__(
        self,
        config: BacktestConfig,
        data_manager: DataSourceManager,
        initial_capital: float = 100000.0
    ):
        """Initialize backtest engine.
        
        Args:
            config: Backtest configuration
            data_manager: Data source manager
            initial_capital: Starting capital
        """
        self.config = config
        self.data_manager = data_manager
        self.initial_capital = initial_capital
        
        # Initialize components
        self.portfolio = Portfolio(initial_capital)
        self.execution_engine = ExecutionEngine(config)
        self.risk_manager = RiskManager(config)
        self.performance_analyzer = PerformanceAnalyzer(config)
        
        # State tracking
        self.status = BacktestStatus.READY
        self.current_time: Optional[datetime] = None
        self.data: Dict[str, pd.DataFrame] = {}
        
        # Results
        self.results: Optional[BacktestResult] = None
        
        # Callbacks
        self.strategy_callback: Optional[Callable] = None
        self.pre_trade_callback: Optional[Callable] = None
        self.post_trade_callback: Optional[Callable] = None
        
        logger.info("Initialized backtest engine", capital=initial_capital)
    
    def set_strategy(self, strategy_callback: Callable):
        """Set the trading strategy callback.
        
        Args:
            strategy_callback: Function that takes current data and returns signals
        """
        self.strategy_callback = strategy_callback
        logger.info("Strategy callback set")
    
    def set_callbacks(
        self,
        pre_trade: Optional[Callable] = None,
        post_trade: Optional[Callable] = None
    ):
        """Set additional callbacks.
        
        Args:
            pre_trade: Called before each trade
            post_trade: Called after each trade
        """
        self.pre_trade_callback = pre_trade
        self.post_trade_callback = post_trade
        logger.info("Additional callbacks set")
    
    async def load_data(
        self,
        symbols: List[str],
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        frequency: str = "1D"
    ):
        """Load historical data for backtesting.
        
        Args:
            symbols: List of symbols to load
            start_date: Start date for data
            end_date: End date for data
            frequency: Data frequency
        """
        logger.info(f"Loading data for {len(symbols)} symbols", frequency=frequency)
        
        try:
            # Create data request
            request = DataRequest(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency
            )
            
            # Get data from data manager
            async with self.data_manager.connect_all():
                response = await self.data_manager.get_bars(request)
            
            if response.is_empty:
                raise ValueError("No data retrieved")
            
            # Store data by symbol
            df = response.to_pandas()
            
            for symbol in symbols:
                symbol_data = df[df['symbol'] == symbol].copy()
                
                if not symbol_data.empty:
                    # Ensure proper datetime index
                    if 'timestamp' in symbol_data.columns:
                        symbol_data = symbol_data.set_index('timestamp')
                    
                    symbol_data = symbol_data.sort_index()
                    self.data[symbol] = symbol_data
                    
                    logger.debug(f"Loaded {len(symbol_data)} records for {symbol}")
                else:
                    logger.warning(f"No data for symbol: {symbol}")
            
            logger.info(f"Data loading completed for {len(self.data)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    async def run(
        self,
        strategy: Optional[Callable] = None,
        rebalance_frequency: str = "daily"
    ) -> BacktestResult:
        """Run the backtest.
        
        Args:
            strategy: Trading strategy function (optional if already set)
            rebalance_frequency: How often to rebalance
            
        Returns:
            BacktestResult containing all results
        """
        if strategy:
            self.set_strategy(strategy)
        
        if not self.strategy_callback:
            raise ValueError("No strategy set")
        
        if not self.data:
            raise ValueError("No data loaded")
        
        logger.info("Starting backtest execution")
        
        try:
            # Initialize results
            self.results = BacktestResult(
                status=BacktestStatus.RUNNING,
                start_time=datetime.now()
            )
            self.status = BacktestStatus.RUNNING
            
            # Get combined time index from all symbols
            all_timestamps = set()
            for symbol_data in self.data.values():
                all_timestamps.update(symbol_data.index)
            
            timestamps = sorted(list(all_timestamps))
            
            if not timestamps:
                raise ValueError("No timestamps found in data")
            
            logger.info(f"Processing {len(timestamps)} time periods")
            
            # Main backtest loop
            portfolio_history = []
            trades_history = []
            
            for i, timestamp in enumerate(timestamps):
                self.current_time = timestamp
                
                # Get current data for all symbols
                current_data = {}
                for symbol, symbol_data in self.data.items():
                    if timestamp in symbol_data.index:
                        # Get data up to current timestamp for lookback
                        historical_data = symbol_data.loc[:timestamp]
                        current_data[symbol] = historical_data
                
                if not current_data:
                    continue
                
                # Update portfolio with current prices
                self._update_portfolio_prices(current_data, timestamp)
                
                # Generate trading signals
                try:
                    signals = self.strategy_callback(current_data, self.portfolio, timestamp)
                    
                    if signals and isinstance(signals, dict):
                        # Pre-trade callback
                        if self.pre_trade_callback:
                            signals = self.pre_trade_callback(signals, current_data, self.portfolio)
                        
                        # Risk management check
                        signals = self.risk_manager.filter_signals(
                            signals, self.portfolio, current_data
                        )
                        
                        # Execute trades
                        executed_trades = await self._execute_signals(
                            signals, current_data, timestamp
                        )
                        
                        if executed_trades:
                            trades_history.extend(executed_trades)
                            
                            # Post-trade callback
                            if self.post_trade_callback:
                                self.post_trade_callback(executed_trades, self.portfolio)
                
                except Exception as e:
                    logger.warning(f"Strategy error at {timestamp}: {e}")
                    continue
                
                # Record portfolio state
                portfolio_state = self.portfolio.get_state()
                portfolio_state['timestamp'] = timestamp
                portfolio_history.append(portfolio_state)
                
                # Progress logging
                if i % 1000 == 0:
                    progress = (i / len(timestamps)) * 100
                    logger.info(f"Backtest progress: {progress:.1f}%")
            
            # Finalize results
            self.results.status = BacktestStatus.COMPLETED
            self.results.end_time = datetime.now()
            
            # Store historical data
            if portfolio_history:
                self.results.portfolio_history = pd.DataFrame(portfolio_history)
                self.results.portfolio_history = self.results.portfolio_history.set_index('timestamp')
            
            if trades_history:
                self.results.trades_history = pd.DataFrame(trades_history)
            
            # Calculate performance metrics
            await self._calculate_performance_metrics()
            
            self.status = BacktestStatus.COMPLETED
            logger.info("Backtest completed successfully")
            
            return self.results
            
        except Exception as e:
            self.status = BacktestStatus.FAILED
            if self.results:
                self.results.status = BacktestStatus.FAILED
                self.results.end_time = datetime.now()
            
            logger.error(f"Backtest failed: {e}")
            raise
    
    def _update_portfolio_prices(
        self,
        current_data: Dict[str, pd.DataFrame],
        timestamp: datetime
    ):
        """Update portfolio with current market prices.
        
        Args:
            current_data: Current market data by symbol
            timestamp: Current timestamp
        """
        for symbol, data in current_data.items():
            if not data.empty and 'close' in data.columns:
                current_price = data['close'].iloc[-1]
                self.portfolio.update_price(symbol, current_price, timestamp)
    
    async def _execute_signals(
        self,
        signals: Dict[str, Any],
        current_data: Dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> List[Dict[str, Any]]:
        """Execute trading signals.
        
        Args:
            signals: Trading signals by symbol
            current_data: Current market data
            timestamp: Current timestamp
            
        Returns:
            List of executed trades
        """
        executed_trades = []
        
        for symbol, signal in signals.items():
            if symbol not in current_data:
                continue
            
            symbol_data = current_data[symbol]
            if symbol_data.empty or 'close' not in symbol_data.columns:
                continue
            
            current_price = symbol_data['close'].iloc[-1]
            
            # Execute trade based on signal
            trade = self.execution_engine.execute_signal(
                symbol=symbol,
                signal=signal,
                current_price=current_price,
                portfolio=self.portfolio,
                timestamp=timestamp
            )
            
            if trade:
                executed_trades.append(trade)
                logger.debug(f"Executed trade: {symbol} {signal} at {current_price}")
        
        return executed_trades
    
    async def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics."""
        if not self.results or self.results.portfolio_history.empty:
            return
        
        portfolio_df = self.results.portfolio_history
        
        # Calculate performance metrics using the analyzer
        metrics = await self.performance_analyzer.calculate_metrics(
            portfolio_df, self.results.trades_history
        )
        
        # Update results with calculated metrics
        for key, value in metrics.items():
            if hasattr(self.results, key):
                setattr(self.results, key, value)
        
        logger.info("Performance metrics calculated")
    
    async def run_walk_forward_analysis(
        self,
        strategy: Callable,
        walk_forward_periods: int = 12,
        retrain_frequency: int = 3
    ) -> List[BacktestResult]:
        """Run walk-forward analysis.
        
        Args:
            strategy: Trading strategy function
            walk_forward_periods: Number of periods for analysis
            retrain_frequency: How often to retrain (in periods)
            
        Returns:
            List of BacktestResult objects for each period
        """
        logger.info(f"Starting walk-forward analysis with {walk_forward_periods} periods")
        
        results = []
        
        # Implementation would split data into periods and run multiple backtests
        # This is a simplified version
        
        return results
    
    async def run_monte_carlo(
        self,
        strategy: Callable,
        n_simulations: int = 1000,
        randomize_returns: bool = True
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation.
        
        Args:
            strategy: Trading strategy function  
            n_simulations: Number of simulations to run
            randomize_returns: Whether to randomize returns
            
        Returns:
            Monte Carlo simulation results
        """
        logger.info(f"Starting Monte Carlo simulation with {n_simulations} runs")
        
        simulation_results = []
        
        for i in range(n_simulations):
            # Reset portfolio for each simulation
            self.portfolio = Portfolio(self.initial_capital)
            
            # Optionally randomize data
            if randomize_returns:
                # Implement return randomization logic
                pass
            
            # Run backtest
            result = await self.run(strategy)
            simulation_results.append(result)
            
            if i % 100 == 0:
                logger.info(f"Monte Carlo progress: {i}/{n_simulations}")
        
        # Aggregate results
        returns = [r.total_return for r in simulation_results]
        sharpe_ratios = [r.sharpe_ratio for r in simulation_results]
        max_drawdowns = [r.max_drawdown for r in simulation_results]
        
        monte_carlo_results = {
            "simulations": n_simulations,
            "returns": {
                "mean": np.mean(returns),
                "std": np.std(returns),
                "min": np.min(returns),
                "max": np.max(returns),
                "percentiles": {
                    "5th": np.percentile(returns, 5),
                    "25th": np.percentile(returns, 25),
                    "50th": np.percentile(returns, 50),
                    "75th": np.percentile(returns, 75),
                    "95th": np.percentile(returns, 95)
                }
            },
            "sharpe_ratios": {
                "mean": np.mean(sharpe_ratios),
                "std": np.std(sharpe_ratios)
            },
            "max_drawdowns": {
                "mean": np.mean(max_drawdowns),
                "worst": np.max(max_drawdowns)
            },
            "win_rate": len([r for r in returns if r > 0]) / len(returns)
        }
        
        logger.info("Monte Carlo simulation completed")
        
        return monte_carlo_results