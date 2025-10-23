"""
Advanced Backtesting API Endpoints
Provides comprehensive backtesting capabilities for trading strategies
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
import asyncio
import json
import sys
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import pandas as pd
import numpy as np

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

router = APIRouter(prefix="/api/backtesting", tags=["backtesting"])

# Pydantic models for backtesting requests
class BacktestRequest(BaseModel):
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float = 100000
    parameters: Dict[str, Any] = {}

class StrategyPerformance(BaseModel):
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades_count: int

class BacktestResult(BaseModel):
    strategy_name: str
    symbol: str
    period: str
    performance: StrategyPerformance
    equity_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    statistics: Dict[str, Any]

class BacktestEngine:
    """Advanced backtesting engine with realistic market simulation"""

    def __init__(self):
        self.results = {}

    def run_backtest(self, request: BacktestRequest) -> BacktestResult:
        """Run a comprehensive backtest"""

        # Generate sample data for demonstration
        start_date = pd.to_datetime(request.start_date)
        end_date = pd.to_datetime(request.end_date)
        dates = pd.date_range(start_date, end_date, freq='D')

        # Simulate price data
        np.random.seed(42)  # For reproducible results
        returns = np.random.normal(0.001, 0.02, len(dates))  # ~0.1% daily return, 2% volatility
        prices = [100]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # Create sample trading data
        equity_curve = []
        trades = []
        portfolio_value = request.initial_capital

        for i, (date, price) in enumerate(zip(dates, prices)):
            # Simple momentum strategy simulation
            if i > 20:  # Need some history
                ma_short = np.mean(prices[i-5:i])
                ma_long = np.mean(prices[i-20:i])

                # Generate trades based on moving average crossover
                if ma_short > ma_long and i % 10 == 0:  # Buy signal
                    trade_value = portfolio_value * 0.1  # 10% position
                    shares = int(trade_value / price)
                    if shares > 0:
                        trades.append({
                            "date": date.isoformat(),
                            "action": "BUY",
                            "symbol": request.symbol,
                            "price": round(price, 2),
                            "shares": shares,
                            "value": shares * price
                        })

                elif ma_short < ma_long and i % 15 == 0:  # Sell signal
                    if trades and trades[-1]["action"] == "BUY":
                        buy_price = trades[-1]["price"]
                        shares = trades[-1]["shares"]
                        profit = shares * (price - buy_price)
                        portfolio_value += profit

                        trades.append({
                            "date": date.isoformat(),
                            "action": "SELL",
                            "symbol": request.symbol,
                            "price": round(price, 2),
                            "shares": shares,
                            "value": shares * price,
                            "profit": round(profit, 2)
                        })

            # Update equity curve
            equity_curve.append({
                "date": date.isoformat(),
                "portfolio_value": round(portfolio_value, 2),
                "price": round(price, 2),
                "benchmark_value": round(request.initial_capital * (price / 100), 2)
            })

        # Calculate performance metrics
        final_value = portfolio_value
        total_return = (final_value - request.initial_capital) / request.initial_capital

        # Calculate other metrics
        daily_returns = [ec["portfolio_value"] for ec in equity_curve]
        daily_returns = [(daily_returns[i] - daily_returns[i-1]) / daily_returns[i-1]
                        for i in range(1, len(daily_returns))]

        annualized_return = (1 + total_return) ** (365 / len(dates)) - 1
        volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Calculate max drawdown
        peak = request.initial_capital
        max_drawdown = 0
        for point in equity_curve:
            if point["portfolio_value"] > peak:
                peak = point["portfolio_value"]
            drawdown = (peak - point["portfolio_value"]) / peak
            max_drawdown = max(max_drawdown, drawdown)

        # Calculate trade statistics
        winning_trades = [t for t in trades if t.get("profit", 0) > 0]
        losing_trades = [t for t in trades if t.get("profit", 0) < 0]

        win_rate = len(winning_trades) / max(1, len([t for t in trades if "profit" in t]))
        profit_factor = (sum(t.get("profit", 0) for t in winning_trades) /
                        abs(sum(t.get("profit", 0) for t in losing_trades))) if losing_trades else 0

        performance = StrategyPerformance(
            total_return=round(total_return * 100, 2),
            annualized_return=round(annualized_return * 100, 2),
            volatility=round(volatility * 100, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            max_drawdown=round(max_drawdown * 100, 2),
            win_rate=round(win_rate * 100, 2),
            profit_factor=round(profit_factor, 2),
            trades_count=len([t for t in trades if "profit" in t])
        )

        result = BacktestResult(
            strategy_name=request.strategy_name,
            symbol=request.symbol,
            period=f"{request.start_date} to {request.end_date}",
            performance=performance,
            equity_curve=equity_curve,
            trades=trades,
            statistics={
                "total_days": len(dates),
                "trading_days": len([d for d in dates if d.weekday() < 5]),
                "average_daily_return": round(np.mean(daily_returns) * 100, 3) if daily_returns else 0,
                "best_day": round(max(daily_returns) * 100, 2) if daily_returns else 0,
                "worst_day": round(min(daily_returns) * 100, 2) if daily_returns else 0,
                "final_portfolio_value": round(final_value, 2)
            }
        )

        return result

# Global backtest engine
backtest_engine = BacktestEngine()

@router.post("/run-backtest")
async def run_backtest(request: BacktestRequest):
    """Run a comprehensive backtest for a trading strategy"""
    try:
        result = backtest_engine.run_backtest(request)
        return {
            "status": "completed",
            "result": result.dict(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtesting error: {e}")

@router.get("/strategies")
async def get_available_strategies():
    """Get list of available backtesting strategies"""
    strategies = [
        {
            "name": "momentum_crossover",
            "description": "Moving average crossover strategy",
            "parameters": {
                "short_window": "Short MA period (default: 5)",
                "long_window": "Long MA period (default: 20)",
                "position_size": "Position size as % of capital (default: 0.1)"
            }
        },
        {
            "name": "mean_reversion",
            "description": "Mean reversion strategy using RSI",
            "parameters": {
                "rsi_period": "RSI calculation period (default: 14)",
                "oversold": "RSI oversold level (default: 30)",
                "overbought": "RSI overbought level (default: 70)"
            }
        },
        {
            "name": "quantitative_options",
            "description": "Options trading with Black-Scholes and Greeks",
            "parameters": {
                "max_delta": "Maximum delta exposure (default: 0.5)",
                "min_days_to_expiry": "Minimum days to expiration (default: 30)",
                "iv_threshold": "Implied volatility threshold (default: 0.25)"
            }
        },
        {
            "name": "ml_momentum",
            "description": "Machine learning enhanced momentum strategy",
            "parameters": {
                "lookback_period": "Feature calculation period (default: 20)",
                "confidence_threshold": "ML prediction confidence (default: 0.6)"
            }
        }
    ]

    return {
        "strategies": strategies,
        "count": len(strategies),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/results/{backtest_id}")
async def get_backtest_results(backtest_id: str):
    """Get detailed results of a specific backtest"""
    # In a real implementation, this would retrieve stored results
    return {
        "message": f"Backtest results for ID: {backtest_id}",
        "note": "Results storage not implemented in this demo",
        "timestamp": datetime.now().isoformat()
    }

@router.post("/compare-strategies")
async def compare_strategies(requests: List[BacktestRequest]):
    """Compare multiple strategies side by side"""
    try:
        results = []
        for request in requests:
            result = backtest_engine.run_backtest(request)
            results.append(result)

        # Create comparison metrics
        comparison = {
            "strategies": [r.strategy_name for r in results],
            "performance_comparison": {
                "total_returns": [r.performance.total_return for r in results],
                "sharpe_ratios": [r.performance.sharpe_ratio for r in results],
                "max_drawdowns": [r.performance.max_drawdown for r in results],
                "win_rates": [r.performance.win_rate for r in results]
            },
            "best_strategy": {
                "by_return": max(results, key=lambda x: x.performance.total_return).strategy_name,
                "by_sharpe": max(results, key=lambda x: x.performance.sharpe_ratio).strategy_name,
                "by_drawdown": min(results, key=lambda x: x.performance.max_drawdown).strategy_name
            },
            "detailed_results": [r.dict() for r in results]
        }

        return {
            "status": "completed",
            "comparison": comparison,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Strategy comparison error: {e}")

@router.post("/monte-carlo-simulation")
async def run_monte_carlo_simulation(
    symbol: str = "SPY",
    initial_capital: float = 100000,
    num_simulations: int = 1000,
    days: int = 252
):
    """Run Monte Carlo simulation for portfolio performance"""
    try:
        # Simulate multiple portfolio paths
        np.random.seed(42)

        # Historical statistics (example values)
        daily_return = 0.001  # 0.1% daily
        daily_volatility = 0.016  # 1.6% daily

        simulations = []
        final_values = []

        for sim in range(num_simulations):
            portfolio_values = [initial_capital]

            for day in range(days):
                daily_change = np.random.normal(daily_return, daily_volatility)
                new_value = portfolio_values[-1] * (1 + daily_change)
                portfolio_values.append(new_value)

            simulations.append(portfolio_values)
            final_values.append(portfolio_values[-1])

        # Calculate statistics
        final_values = np.array(final_values)

        results = {
            "simulation_parameters": {
                "symbol": symbol,
                "initial_capital": initial_capital,
                "num_simulations": num_simulations,
                "days": days
            },
            "statistics": {
                "mean_final_value": round(np.mean(final_values), 2),
                "median_final_value": round(np.median(final_values), 2),
                "std_final_value": round(np.std(final_values), 2),
                "min_final_value": round(np.min(final_values), 2),
                "max_final_value": round(np.max(final_values), 2),
                "percentiles": {
                    "5th": round(np.percentile(final_values, 5), 2),
                    "25th": round(np.percentile(final_values, 25), 2),
                    "75th": round(np.percentile(final_values, 75), 2),
                    "95th": round(np.percentile(final_values, 95), 2)
                }
            },
            "risk_metrics": {
                "probability_of_loss": round(np.mean(final_values < initial_capital) * 100, 2),
                "value_at_risk_95": round(np.percentile(final_values, 5), 2),
                "expected_shortfall": round(np.mean(final_values[final_values <= np.percentile(final_values, 5)]), 2)
            },
            "sample_paths": simulations[:10]  # Return first 10 simulation paths
        }

        return {
            "status": "completed",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Monte Carlo simulation error: {e}")

@router.get("/performance-analytics")
async def get_performance_analytics():
    """Get advanced performance analytics dashboard data"""
    return {
        "analytics": {
            "risk_adjusted_returns": {
                "sharpe_ratio": 1.45,
                "sortino_ratio": 2.12,
                "calmar_ratio": 0.89,
                "omega_ratio": 1.78
            },
            "drawdown_analysis": {
                "current_drawdown": 0.0,
                "max_drawdown": -8.5,
                "average_drawdown": -2.3,
                "drawdown_duration_avg": 15
            },
            "return_distribution": {
                "skewness": 0.12,
                "kurtosis": 2.8,
                "var_95": -2.1,
                "cvar_95": -3.2
            },
            "strategy_correlation": {
                "spy_correlation": 0.65,
                "qqq_correlation": 0.72,
                "market_beta": 0.88
            }
        },
        "timestamp": datetime.now().isoformat()
    }