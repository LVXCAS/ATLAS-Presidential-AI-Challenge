"""Backtesting configuration settings."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, date
import pandas as pd


class BacktestConfig(BaseModel):
    """Backtesting configuration."""
    
    # Time period
    start_date: date = Field(default_factory=lambda: pd.Timestamp.now().date() - pd.Timedelta(days=365))
    end_date: date = Field(default_factory=lambda: pd.Timestamp.now().date())
    
    # Capital and position sizing
    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # 10% of portfolio
    min_position_size: float = 0.01  # 1% of portfolio
    
    # Transaction costs
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005   # 0.05% slippage
    bid_ask_spread: float = 0.0002  # 0.02% spread
    
    # Risk management
    stop_loss: Optional[float] = 0.05  # 5% stop loss
    take_profit: Optional[float] = 0.15  # 15% take profit
    max_drawdown_limit: float = 0.2  # 20% max drawdown
    
    # Position management
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    allow_short_selling: bool = True
    margin_requirement: float = 0.5  # 50% margin requirement
    
    # Data settings
    benchmark: str = "SPY"
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    data_frequency: str = "1D"  # 1D, 1H, 15min, 5min, 1min
    
    # Performance metrics
    metrics_to_calculate: List[str] = [
        "total_return",
        "annual_return", 
        "volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "average_trade",
        "total_trades"
    ]
    
    # Optimization settings
    optimization_metric: str = "sharpe_ratio"
    optimization_method: str = "grid_search"  # grid_search, random_search, bayesian
    n_optimization_trials: int = 100
    
    # Walk-forward analysis
    enable_walk_forward: bool = False
    walk_forward_periods: int = 12  # months
    walk_forward_retrain_frequency: int = 3  # months
    
    # Monte Carlo simulation
    monte_carlo_runs: int = 1000
    monte_carlo_enabled: bool = False
    
    class Config:
        extra = "allow"


class StrategyBacktestConfig(BaseModel):
    """Strategy-specific backtest configuration."""
    
    strategy_name: str
    strategy_type: str  # momentum, mean_reversion, arbitrage, etc.
    
    # Strategy parameters
    parameters: Dict[str, Any] = {}
    
    # Entry/Exit rules
    entry_rules: Dict[str, Any] = {}
    exit_rules: Dict[str, Any] = {}
    
    # Position sizing
    position_sizing_method: str = "fixed_percent"  # fixed_percent, volatility_target, kelly
    position_sizing_params: Dict[str, Any] = {"allocation": 0.1}
    
    # Universe selection
    universe: List[str] = ["SPY"]  # Symbols to trade
    universe_selection_method: Optional[str] = None  # top_momentum, low_volatility, etc.
    universe_size: int = 10
    
    # Rebalancing
    rebalance_frequency: str = "monthly"
    rebalance_threshold: Optional[float] = 0.05  # 5% drift threshold
    
    # Risk management
    portfolio_risk_limit: float = 0.15  # 15% portfolio volatility target
    individual_risk_limit: float = 0.05  # 5% per position
    correlation_limit: float = 0.7  # Max correlation between positions


class BacktestEnvironmentConfig(BaseModel):
    """Backtesting environment configuration."""
    
    # Execution settings
    execution_model: str = "perfect"  # perfect, realistic, slippage_model
    market_impact_model: Optional[str] = None  # linear, sqrt, market_impact
    
    # Data handling
    data_adjustment: str = "splits_and_dividends"  # splits_only, splits_and_dividends, none
    survivorship_bias: bool = False  # Include delisted stocks
    point_in_time_data: bool = True  # Use point-in-time fundamental data
    
    # Market hours
    market_open: str = "09:30"
    market_close: str = "16:00"
    timezone: str = "US/Eastern"
    
    # Holidays and weekends
    skip_weekends: bool = True
    holiday_calendar: str = "NYSE"  # NYSE, NASDAQ, LSE, etc.
    
    # Performance settings
    parallel_processing: bool = True
    n_jobs: int = -1  # Use all available cores
    chunk_size: int = 1000  # For parallel processing
    
    # Memory management
    max_memory_usage: str = "8GB"
    cache_data: bool = True
    preload_data: bool = False


class ReportConfig(BaseModel):
    """Backtest report configuration."""
    
    # Report generation
    generate_report: bool = True
    report_format: str = "html"  # html, pdf, json
    save_trades: bool = True
    save_positions: bool = True
    
    # Visualizations
    generate_charts: bool = True
    chart_types: List[str] = [
        "equity_curve",
        "drawdown",
        "monthly_returns",
        "rolling_sharpe",
        "correlation_matrix",
        "position_heatmap"
    ]
    
    # Benchmark comparison
    compare_to_benchmark: bool = True
    benchmark_symbol: str = "SPY"
    
    # Attribution analysis
    performance_attribution: bool = True
    risk_attribution: bool = True
    
    # Output paths
    results_directory: str = "results/backtests"
    charts_directory: str = "results/charts" 
    reports_directory: str = "results/reports"


# Predefined backtest configurations
BACKTEST_CONFIGS = {
    "momentum_strategy": StrategyBacktestConfig(
        strategy_name="momentum_strategy",
        strategy_type="momentum",
        parameters={
            "lookback_period": 252,  # 1 year
            "holding_period": 21,    # 1 month
            "top_n": 10
        },
        universe=["SPY", "QQQ", "IWM", "DIA"],
        rebalance_frequency="monthly"
    ),
    
    "mean_reversion": StrategyBacktestConfig(
        strategy_name="mean_reversion",
        strategy_type="mean_reversion", 
        parameters={
            "z_score_threshold": 2.0,
            "lookback_window": 20,
            "holding_period": 5
        },
        universe=["AAPL", "MSFT", "GOOGL", "AMZN"],
        rebalance_frequency="daily"
    ),
    
    "pairs_trading": StrategyBacktestConfig(
        strategy_name="pairs_trading",
        strategy_type="arbitrage",
        parameters={
            "cointegration_lookback": 252,
            "z_score_entry": 2.0,
            "z_score_exit": 0.0,
            "stop_loss_z": 3.0
        },
        universe=["XLF", "XLE", "XLK", "XLV"],  # Sector ETFs
        rebalance_frequency="weekly"
    ),
    
    "options_momentum": StrategyBacktestConfig(
        strategy_name="options_momentum",
        strategy_type="options",
        parameters={
            "dte_min": 30,  # Days to expiration
            "dte_max": 60,
            "delta_target": 0.5,
            "iv_percentile_min": 20
        },
        universe=["SPY", "QQQ"],
        rebalance_frequency="weekly"
    )
}