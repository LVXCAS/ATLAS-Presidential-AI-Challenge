"""
Backtesting Package

A comprehensive backtesting engine with realistic market simulation,
strategy optimization, and performance analytics.
"""

from .market_simulator import MarketSimulator, Order, OrderSide, OrderStatus, Fill, MarketData
from .backtest_engine import BacktestEngine, BacktestResults, Portfolio, Position, Trade
from .data_loader import DataLoader, create_data_feed
from .strategy_tester import StrategyTester, ParameterRange, OptimizationResult, WalkForwardResult

__all__ = [
    # Market simulation
    'MarketSimulator',
    'Order',
    'OrderSide', 
    'OrderStatus',
    'Fill',
    'MarketData',
    
    # Backtesting engine
    'BacktestEngine',
    'BacktestResults',
    'Portfolio',
    'Position',
    'Trade',
    
    # Data management
    'DataLoader',
    'create_data_feed',
    
    # Strategy testing
    'StrategyTester',
    'ParameterRange',
    'OptimizationResult',
    'WalkForwardResult'
]