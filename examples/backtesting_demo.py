"""
Backtesting Engine Demo

This demo showcases the comprehensive backtesting capabilities including:
- Event-driven backtesting with realistic market simulation
- Multiple strategy testing
- Performance metrics calculation
- Walk-forward analysis
- Synthetic scenario testing
- Report generation

Requirements: Requirement 4 (Backtesting and Historical Validation)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

from strategies.backtesting_engine import (
    BacktestingEngine, MarketData, OrderSide, OrderType,
    LinearSlippageModel, PerShareCommissionModel,
    simple_momentum_strategy, buy_and_hold_strategy
)


def generate_realistic_market_data(
    symbol: str = "AAPL",
    start_date: datetime = datetime(2022, 1, 1),
    end_date: datetime = datetime(2023, 12, 31),
    initial_price: float = 150.0
) -> list:
    """Generate realistic market data for backtesting"""
    
    print(f"Generating market data for {symbol} from {start_date.date()} to {end_date.date()}")
    
    data = []
    current_date = start_date
    current_price = initial_price
    
    # Market parameters
    daily_volatility = 0.02  # 2% daily volatility
    trend_strength = 0.0001  # Slight upward trend
    
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        # Generate price movement
        random_return = np.random.normal(trend_strength, daily_volatility)
        
        # Calculate OHLC
        open_price = current_price
        close_price = open_price * (1 + random_return)
        
        # Generate realistic high/low
        intraday_range = abs(random_return) * 1.5
        high_price = max(open_price, close_price) * (1 + intraday_range * np.random.random())
        low_price = min(open_price, close_price) * (1 - intraday_range * np.random.random())
        
        # Generate volume (higher volume on larger price moves)
        base_volume = 1000000
        volume_multiplier = 1 + abs(random_return) * 5
        volume = int(base_volume * volume_multiplier * (0.5 + np.random.random()))
        
        # Create market data
        market_data = MarketData(
            timestamp=current_date.replace(hour=16, minute=0),  # Market close
            symbol=symbol,
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=volume,
            vwap=round((high_price + low_price + close_price) / 3, 2),
            spread=round(close_price * 0.0001, 4)  # 1 basis point spread
        )
        
        data.append(market_data)
        current_price = close_price
        current_date += timedelta(days=1)
    
    print(f"Generated {len(data)} trading days of data")
    return data


def advanced_momentum_strategy(engine, market_data, params):
    """
    Advanced momentum strategy with multiple indicators
    
    This strategy combines:
    - Moving average crossovers
    - RSI momentum
    - Volume confirmation
    - Position sizing based on volatility
    """
    
    # Parameters
    short_window = params.get('short_window', 20)
    long_window = params.get('long_window', 50)
    rsi_period = params.get('rsi_period', 14)
    volume_threshold = params.get('volume_threshold', 1.2)
    max_position_size = params.get('max_position_size', 0.1)  # 10% of portfolio
    
    # Initialize price and volume history
    if not hasattr(engine, '_price_history'):
        engine._price_history = {}
        engine._volume_history = {}
    
    symbol = market_data.symbol
    
    if symbol not in engine._price_history:
        engine._price_history[symbol] = []
        engine._volume_history[symbol] = []
    
    # Update history
    engine._price_history[symbol].append(market_data.close)
    engine._volume_history[symbol].append(market_data.volume)
    
    # Keep only required history
    max_history = max(long_window, rsi_period) + 10
    if len(engine._price_history[symbol]) > max_history:
        engine._price_history[symbol] = engine._price_history[symbol][-max_history:]
        engine._volume_history[symbol] = engine._volume_history[symbol][-max_history:]
    
    prices = engine._price_history[symbol]
    volumes = engine._volume_history[symbol]
    
    if len(prices) < long_window:
        return
    
    # Calculate indicators
    short_ma = np.mean(prices[-short_window:])
    long_ma = np.mean(prices[-long_window:])
    
    # Simple RSI calculation
    if len(prices) >= rsi_period + 1:
        price_changes = np.diff(prices[-rsi_period-1:])
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 100
    else:
        rsi = 50  # Neutral
    
    # Volume confirmation
    avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else volumes[-1]
    volume_ratio = market_data.volume / avg_volume if avg_volume > 0 else 1
    
    # Get current position
    current_position = engine.portfolio.positions.get(symbol)
    current_qty = current_position.quantity if current_position else 0
    
    # Calculate position size based on portfolio value
    portfolio_value = engine.portfolio.total_value
    max_position_value = portfolio_value * max_position_size
    max_shares = int(max_position_value / market_data.close)
    
    # Trading signals
    bullish_signal = (
        short_ma > long_ma and  # Trend is up
        rsi > 50 and  # Momentum is positive
        volume_ratio > volume_threshold  # Volume confirmation
    )
    
    bearish_signal = (
        short_ma < long_ma and  # Trend is down
        rsi < 50 and  # Momentum is negative
        volume_ratio > volume_threshold  # Volume confirmation
    )
    
    # Execute trades
    if bullish_signal and current_qty <= 0:
        # Buy signal
        if current_qty < 0:
            # Cover short first
            engine.submit_order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=abs(current_qty),
                order_type=OrderType.MARKET,
                strategy='momentum_cover'
            )
        
        # Go long
        long_quantity = min(max_shares, 1000)  # Cap at 1000 shares
        if long_quantity > 0:
            engine.submit_order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=long_quantity,
                order_type=OrderType.MARKET,
                strategy='momentum_long'
            )
    
    elif bearish_signal and current_qty >= 0:
        # Sell signal
        if current_qty > 0:
            # Sell long first
            engine.submit_order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=current_qty,
                order_type=OrderType.MARKET,
                strategy='momentum_sell'
            )
        
        # Go short (optional - can be disabled)
        if params.get('allow_short', False):
            short_quantity = min(max_shares, 500)  # Smaller short positions
            if short_quantity > 0:
                engine.submit_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=short_quantity,
                    order_type=OrderType.MARKET,
                    strategy='momentum_short'
                )


def mean_reversion_strategy(engine, market_data, params):
    """
    Mean reversion strategy using Bollinger Bands
    """
    
    # Parameters
    window = params.get('window', 20)
    num_std = params.get('num_std', 2.0)
    max_position_size = params.get('max_position_size', 0.05)  # 5% of portfolio
    
    # Initialize price history
    if not hasattr(engine, '_price_history'):
        engine._price_history = {}
    
    symbol = market_data.symbol
    
    if symbol not in engine._price_history:
        engine._price_history[symbol] = []
    
    # Update history
    engine._price_history[symbol].append(market_data.close)
    
    # Keep only required history
    if len(engine._price_history[symbol]) > window + 10:
        engine._price_history[symbol] = engine._price_history[symbol][-(window + 10):]
    
    prices = engine._price_history[symbol]
    
    if len(prices) < window:
        return
    
    # Calculate Bollinger Bands
    sma = np.mean(prices[-window:])
    std = np.std(prices[-window:])
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    
    current_price = market_data.close
    
    # Get current position
    current_position = engine.portfolio.positions.get(symbol)
    current_qty = current_position.quantity if current_position else 0
    
    # Calculate position size
    portfolio_value = engine.portfolio.total_value
    max_position_value = portfolio_value * max_position_size
    max_shares = int(max_position_value / current_price)
    
    # Mean reversion signals
    oversold = current_price < lower_band  # Price below lower band
    overbought = current_price > upper_band  # Price above upper band
    neutral = lower_band <= current_price <= upper_band
    
    # Execute trades
    if oversold and current_qty <= 0:
        # Buy on oversold condition
        buy_quantity = min(max_shares, 500)
        if buy_quantity > 0:
            engine.submit_order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=buy_quantity,
                order_type=OrderType.MARKET,
                strategy='mean_reversion_buy'
            )
    
    elif overbought and current_qty >= 0:
        # Sell on overbought condition
        if current_qty > 0:
            engine.submit_order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=current_qty,
                order_type=OrderType.MARKET,
                strategy='mean_reversion_sell'
            )
    
    elif neutral and current_qty != 0:
        # Close position when price returns to normal range
        if current_qty > 0:
            engine.submit_order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=current_qty,
                order_type=OrderType.MARKET,
                strategy='mean_reversion_close'
            )
        elif current_qty < 0:
            engine.submit_order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=abs(current_qty),
                order_type=OrderType.MARKET,
                strategy='mean_reversion_cover'
            )


def run_comprehensive_backtest_demo():
    """Run comprehensive backtesting demonstration"""
    
    print("=" * 80)
    print("LANGGRAPH TRADING SYSTEM - BACKTESTING ENGINE DEMO")
    print("=" * 80)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate market data
    print("\n1. GENERATING MARKET DATA")
    print("-" * 40)
    
    market_data = generate_realistic_market_data(
        symbol="AAPL",
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_price=150.0
    )
    
    # Initialize backtesting engine
    print("\n2. INITIALIZING BACKTESTING ENGINE")
    print("-" * 40)
    
    engine = BacktestingEngine(
        initial_capital=100000.0,
        slippage_model=LinearSlippageModel(base_slippage=0.0005, volume_impact=0.00001),
        commission_model=PerShareCommissionModel(per_share=0.005, minimum=1.0),
        random_seed=42
    )
    
    print(f"Initial Capital: ${engine.initial_capital:,.2f}")
    print(f"Slippage Model: Linear (base: 0.05%, volume impact: 0.001%)")
    print(f"Commission Model: Per-share ($0.005/share, min $1.00)")
    
    # Test different strategies
    strategies = [
        ("Buy and Hold", buy_and_hold_strategy, {}),
        ("Simple Momentum", simple_momentum_strategy, {'short_window': 10, 'long_window': 30}),
        ("Advanced Momentum", advanced_momentum_strategy, {
            'short_window': 20, 'long_window': 50, 'rsi_period': 14,
            'volume_threshold': 1.2, 'max_position_size': 0.1
        }),
        ("Mean Reversion", mean_reversion_strategy, {
            'window': 20, 'num_std': 2.0, 'max_position_size': 0.05
        })
    ]
    
    strategy_results = {}
    
    print("\n3. RUNNING STRATEGY BACKTESTS")
    print("-" * 40)
    
    for strategy_name, strategy_func, params in strategies:
        print(f"\nTesting {strategy_name} Strategy...")
        
        # Reset engine for each strategy
        engine.reset()
        
        # Run backtest
        results = engine.run_backtest(market_data, strategy_func, params)
        strategy_results[strategy_name] = results
        
        # Print summary
        perf = results['performance_metrics']
        print(f"  Final Value: ${engine.portfolio.total_value:,.2f}")
        print(f"  Total Return: {perf.total_return:.2%}")
        print(f"  Annualized Return: {perf.annualized_return:.2%}")
        print(f"  Sharpe Ratio: {perf.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {perf.max_drawdown:.2%}")
        print(f"  Total Trades: {perf.total_trades}")
        print(f"  Win Rate: {perf.win_rate:.2%}")
    
    # Performance comparison
    print("\n4. STRATEGY PERFORMANCE COMPARISON")
    print("-" * 40)
    
    print(f"{'Strategy':<20} {'Return':<10} {'Sharpe':<8} {'Drawdown':<10} {'Trades':<8}")
    print("-" * 60)
    
    for strategy_name, results in strategy_results.items():
        perf = results['performance_metrics']
        print(f"{strategy_name:<20} {perf.total_return:>8.2%} {perf.sharpe_ratio:>7.2f} "
              f"{perf.max_drawdown:>8.2%} {perf.total_trades:>7d}")
    
    # Walk-forward analysis
    print("\n5. WALK-FORWARD ANALYSIS")
    print("-" * 40)
    
    print("Running walk-forward analysis on Advanced Momentum strategy...")
    
    wf_results = engine.walk_forward_analysis(
        market_data,
        advanced_momentum_strategy,
        training_period=126,  # 6 months
        testing_period=63,    # 3 months
        step_size=21,         # 1 month
        strategy_params={
            'short_window': 20, 'long_window': 50, 'rsi_period': 14,
            'volume_threshold': 1.2, 'max_position_size': 0.1
        }
    )
    
    agg_metrics = wf_results['aggregate_metrics']
    print(f"Walk-Forward Periods: {wf_results['total_periods']}")
    print(f"Average Return: {agg_metrics['avg_return']:.2%}")
    print(f"Return Std Dev: {agg_metrics['std_return']:.2%}")
    print(f"Average Sharpe: {agg_metrics['avg_sharpe']:.2f}")
    print(f"Consistency Ratio: {agg_metrics['consistency_ratio']:.2%}")
    
    # Synthetic scenario testing
    print("\n6. SYNTHETIC SCENARIO TESTING")
    print("-" * 40)
    
    print("Testing Advanced Momentum strategy against synthetic scenarios...")
    
    scenarios = ['trending_up', 'trending_down', 'high_volatility', 'mean_reverting']
    scenario_results = engine.synthetic_scenario_testing(
        market_data[:252],  # Use 1 year of data
        advanced_momentum_strategy,
        scenarios,
        {
            'short_window': 20, 'long_window': 50, 'rsi_period': 14,
            'volume_threshold': 1.2, 'max_position_size': 0.1
        }
    )
    
    print(f"{'Scenario':<15} {'Final Value':<12} {'Max DD':<10} {'Trades':<8}")
    print("-" * 50)
    
    for scenario, result in scenario_results.items():
        print(f"{scenario:<15} ${result['final_value']:>10,.0f} "
              f"{result['max_drawdown']:>8.2%} {result['total_trades']:>7d}")
    
    # Generate detailed report
    print("\n7. GENERATING DETAILED REPORT")
    print("-" * 40)
    
    best_strategy = max(strategy_results.items(), 
                       key=lambda x: x[1]['performance_metrics'].sharpe_ratio)
    best_name, best_results = best_strategy
    
    print(f"Best performing strategy: {best_name}")
    
    report = engine.generate_report(best_results)
    
    # Save report to file
    report_path = Path("backtest_report.md")
    report_path.write_text(report)
    print(f"Detailed report saved to: {report_path.absolute()}")
    
    # Risk analysis
    print("\n8. RISK ANALYSIS")
    print("-" * 40)
    
    perf = best_results['performance_metrics']
    print(f"Value at Risk (95%): {perf.var_95:.2%}")
    print(f"Conditional VaR (95%): {perf.cvar_95:.2%}")
    print(f"Sortino Ratio: {perf.sortino_ratio:.2f}")
    print(f"Calmar Ratio: {perf.calmar_ratio:.2f}")
    print(f"Max Drawdown Duration: {perf.max_drawdown_duration} days")
    
    # Reproducibility test
    print("\n9. REPRODUCIBILITY TEST")
    print("-" * 40)
    
    print("Testing reproducibility with same random seed...")
    
    engine1 = BacktestingEngine(initial_capital=100000.0, random_seed=42)
    engine2 = BacktestingEngine(initial_capital=100000.0, random_seed=42)
    
    results1 = engine1.run_backtest(market_data[:100], simple_momentum_strategy, {})
    results2 = engine2.run_backtest(market_data[:100], simple_momentum_strategy, {})
    
    return1 = results1['performance_metrics'].total_return
    return2 = results2['performance_metrics'].total_return
    
    print(f"Run 1 Return: {return1:.6%}")
    print(f"Run 2 Return: {return2:.6%}")
    print(f"Difference: {abs(return1 - return2):.10f}")
    print(f"Reproducible: {'✓' if abs(return1 - return2) < 1e-10 else '✗'}")
    
    print("\n" + "=" * 80)
    print("BACKTESTING DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return strategy_results


if __name__ == "__main__":
    try:
        results = run_comprehensive_backtest_demo()
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"\nError running demo: {e}")
        import traceback
        traceback.print_exc()