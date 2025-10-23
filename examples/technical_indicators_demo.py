"""
Technical Indicators Library Demo

This script demonstrates the usage of all technical indicators
and parameter optimization capabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.technical_indicators import (
    IndicatorLibrary, calculate_ema, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_z_score
)
from strategies.parameter_optimization import (
    ParameterOptimizer, ParameterSpace, optimize_rsi, optimize_macd
)


def generate_sample_data(n_points: int = 252, trend: float = 0.1, volatility: float = 0.02) -> np.ndarray:
    """Generate sample price data for demonstration"""
    np.random.seed(42)
    
    # Generate returns with trend and volatility
    returns = np.random.normal(trend/252, volatility, n_points)
    
    # Convert to price series starting at 100
    prices = 100 * np.exp(np.cumsum(returns))
    
    return prices


def demo_individual_indicators():
    """Demonstrate individual indicator calculations"""
    print("=" * 60)
    print("INDIVIDUAL INDICATORS DEMONSTRATION")
    print("=" * 60)
    
    # Generate sample data
    prices = generate_sample_data(100)
    print(f"Generated {len(prices)} price points")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    # EMA Demonstration
    print("\n1. Exponential Moving Average (EMA)")
    print("-" * 40)
    ema_result = calculate_ema(prices, period=20)
    print(f"EMA Parameters: {ema_result.parameters}")
    print(f"Latest EMA value: ${ema_result.values[-1]:.2f}")
    print(f"Latest price: ${prices[-1]:.2f}")
    
    # RSI Demonstration
    print("\n2. Relative Strength Index (RSI)")
    print("-" * 40)
    rsi_result = calculate_rsi(prices, period=14)
    print(f"RSI Parameters: {rsi_result.parameters}")
    print(f"Latest RSI value: {rsi_result.values[-1]:.2f}")
    
    rsi_signal = "Overbought" if rsi_result.values[-1] > 70 else "Oversold" if rsi_result.values[-1] < 30 else "Neutral"
    print(f"RSI Signal: {rsi_signal}")
    
    # MACD Demonstration
    print("\n3. MACD (Moving Average Convergence Divergence)")
    print("-" * 40)
    macd_result = calculate_macd(prices)
    print(f"MACD Parameters: {macd_result.parameters}")
    
    macd_line = macd_result.values[-1, 0]
    signal_line = macd_result.values[-1, 1]
    histogram = macd_result.values[-1, 2]
    
    print(f"MACD Line: {macd_line:.4f}")
    print(f"Signal Line: {signal_line:.4f}")
    print(f"Histogram: {histogram:.4f}")
    
    macd_signal = "Bullish" if macd_line > signal_line else "Bearish"
    print(f"MACD Signal: {macd_signal}")
    
    # Bollinger Bands Demonstration
    print("\n4. Bollinger Bands")
    print("-" * 40)
    bb_result = calculate_bollinger_bands(prices, period=20, std_dev=2.0)
    print(f"Bollinger Bands Parameters: {bb_result.parameters}")
    
    upper_band = bb_result.values[-1, 0]
    middle_band = bb_result.values[-1, 1]
    lower_band = bb_result.values[-1, 2]
    current_price = prices[-1]
    
    print(f"Upper Band: ${upper_band:.2f}")
    print(f"Middle Band: ${middle_band:.2f}")
    print(f"Lower Band: ${lower_band:.2f}")
    print(f"Current Price: ${current_price:.2f}")
    
    if current_price > upper_band:
        bb_signal = "Overbought (above upper band)"
    elif current_price < lower_band:
        bb_signal = "Oversold (below lower band)"
    else:
        bb_signal = "Within bands"
    print(f"Bollinger Bands Signal: {bb_signal}")
    
    # Z-Score Demonstration
    print("\n5. Z-Score")
    print("-" * 40)
    zscore_result = calculate_z_score(prices, period=20)
    print(f"Z-Score Parameters: {zscore_result.parameters}")
    print(f"Latest Z-Score: {zscore_result.values[-1]:.2f}")
    
    zscore_signal = "Mean reversion opportunity" if abs(zscore_result.values[-1]) > 2 else "Normal range"
    print(f"Z-Score Signal: {zscore_signal}")


def demo_indicator_library():
    """Demonstrate the IndicatorLibrary class"""
    print("\n" + "=" * 60)
    print("INDICATOR LIBRARY DEMONSTRATION")
    print("=" * 60)
    
    # Generate sample data
    prices = generate_sample_data(150)
    library = IndicatorLibrary()
    
    print(f"Available indicators: {library.get_available_indicators()}")
    
    # Calculate multiple indicators at once
    indicators_config = {
        'ema': {'period': 20},
        'rsi': {'period': 14},
        'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        'bollinger_bands': {'period': 20, 'std_dev': 2.0},
        'z_score': {'period': 20}
    }
    
    print("\nCalculating multiple indicators...")
    results = library.calculate_multiple(prices, indicators_config)
    
    print("\nResults Summary:")
    for indicator_name, result in results.items():
        if result is not None:
            print(f"  {indicator_name}: [OK] Calculated successfully")
            if hasattr(result.values, 'shape'):
                if len(result.values.shape) == 1:
                    print(f"    Latest value: {result.values[-1]:.4f}")
                else:
                    print(f"    Shape: {result.values.shape}")
            else:
                print(f"    Latest value: {result.values[-1]:.4f}")
        else:
            print(f"  {indicator_name}: [X] Failed to calculate")


def demo_parameter_optimization():
    """Demonstrate parameter optimization"""
    print("\n" + "=" * 60)
    print("PARAMETER OPTIMIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Generate sample data with some trend
    prices = generate_sample_data(200, trend=0.15, volatility=0.02)
    optimizer = ParameterOptimizer()
    
    print("Optimizing RSI parameters...")
    print("-" * 40)
    
    # Optimize RSI with custom parameter space
    rsi_space = [ParameterSpace('period', 10, 25, 5, param_type='int')]
    
    rsi_optimization = optimizer.grid_search(
        'rsi', prices, rsi_space, 
        objective='sharpe_ratio', n_jobs=1
    )
    
    print(f"Best RSI parameters: {rsi_optimization.best_params}")
    print(f"Best Sharpe ratio: {rsi_optimization.best_score:.4f}")
    print(f"Total evaluations: {rsi_optimization.total_evaluations}")
    print(f"Optimization time: {rsi_optimization.optimization_time:.2f} seconds")
    
    print("\nOptimizing MACD parameters...")
    print("-" * 40)
    
    # Optimize MACD with limited search space for demo
    macd_space = [
        ParameterSpace('fast_period', 10, 15, 5, param_type='int'),
        ParameterSpace('slow_period', 20, 30, 5, param_type='int'),
        ParameterSpace('signal_period', 8, 12, 2, param_type='int')
    ]
    
    macd_optimization = optimizer.grid_search(
        'macd', prices, macd_space,
        objective='profit_factor', n_jobs=1
    )
    
    print(f"Best MACD parameters: {macd_optimization.best_params}")
    print(f"Best profit factor: {macd_optimization.best_score:.4f}")
    print(f"Total evaluations: {macd_optimization.total_evaluations}")
    print(f"Optimization time: {macd_optimization.optimization_time:.2f} seconds")
    
    print("\nRandom Search Optimization...")
    print("-" * 40)
    
    # Demonstrate random search
    random_optimization = optimizer.random_search(
        'rsi', prices, rsi_space,
        n_iterations=10, objective='sharpe_ratio', random_seed=42
    )
    
    print(f"Random search best parameters: {random_optimization.best_params}")
    print(f"Random search best score: {random_optimization.best_score:.4f}")
    print(f"Method: {random_optimization.method}")


def demo_trading_signals():
    """Demonstrate how to generate trading signals"""
    print("\n" + "=" * 60)
    print("TRADING SIGNALS DEMONSTRATION")
    print("=" * 60)
    
    # Generate sample data
    prices = generate_sample_data(100)
    library = IndicatorLibrary()
    
    # Calculate indicators
    ema_20 = library.calculate_indicator('ema', prices, period=20)
    ema_50 = library.calculate_indicator('ema', prices, period=50)
    rsi = library.calculate_indicator('rsi', prices, period=14)
    macd = library.calculate_indicator('macd', prices)
    bb = library.calculate_indicator('bollinger_bands', prices, period=20)
    
    # Generate signals
    signals = []
    
    # EMA Crossover Signal
    if ema_20.values[-1] > ema_50.values[-1] and ema_20.values[-2] <= ema_50.values[-2]:
        signals.append("[INFO] EMA Bullish Crossover: EMA(20) crossed above EMA(50)")
    elif ema_20.values[-1] < ema_50.values[-1] and ema_20.values[-2] >= ema_50.values[-2]:
        signals.append("[RED] EMA Bearish Crossover: EMA(20) crossed below EMA(50)")
    
    # RSI Signals
    if rsi.values[-1] < 30:
        signals.append(f"[GREEN] RSI Oversold: RSI = {rsi.values[-1]:.1f} (Buy signal)")
    elif rsi.values[-1] > 70:
        signals.append(f"[RED] RSI Overbought: RSI = {rsi.values[-1]:.1f} (Sell signal)")
    
    # MACD Signals
    macd_line = macd.values[-1, 0]
    signal_line = macd.values[-1, 1]
    prev_macd = macd.values[-2, 0]
    prev_signal = macd.values[-2, 1]
    
    if macd_line > signal_line and prev_macd <= prev_signal:
        signals.append("[INFO] MACD Bullish Crossover: MACD crossed above signal line")
    elif macd_line < signal_line and prev_macd >= prev_signal:
        signals.append("[RED] MACD Bearish Crossover: MACD crossed below signal line")
    
    # Bollinger Bands Signals
    current_price = prices[-1]
    upper_band = bb.values[-1, 0]
    lower_band = bb.values[-1, 2]
    
    if current_price <= lower_band:
        signals.append(f"[GREEN] Bollinger Bands: Price touched lower band (${current_price:.2f} <= ${lower_band:.2f})")
    elif current_price >= upper_band:
        signals.append(f"[RED] Bollinger Bands: Price touched upper band (${current_price:.2f} >= ${upper_band:.2f})")
    
    # Display signals
    print(f"Current Price: ${current_price:.2f}")
    print(f"Generated {len(signals)} trading signals:")
    
    if signals:
        for signal in signals:
            print(f"  {signal}")
    else:
        print("  No significant signals detected")
    
    # Display current indicator values
    print(f"\nCurrent Indicator Values:")
    print(f"  EMA(20): ${ema_20.values[-1]:.2f}")
    print(f"  EMA(50): ${ema_50.values[-1]:.2f}")
    print(f"  RSI(14): {rsi.values[-1]:.1f}")
    print(f"  MACD: {macd_line:.4f}")
    print(f"  Signal: {signal_line:.4f}")
    print(f"  BB Upper: ${upper_band:.2f}")
    print(f"  BB Lower: ${lower_band:.2f}")


def demo_performance_comparison():
    """Demonstrate performance characteristics"""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    import time
    
    # Test with different data sizes
    sizes = [100, 500, 1000, 5000]
    library = IndicatorLibrary()
    
    print("Performance test with different data sizes:")
    print("Size\tEMA\tRSI\tMACD\tBB\tZ-Score\tTotal")
    print("-" * 60)
    
    for size in sizes:
        prices = generate_sample_data(size)
        times = {}
        
        # Test each indicator
        indicators = ['ema', 'rsi', 'macd', 'bollinger_bands', 'z_score']
        params = [
            {'period': 20},
            {'period': 14},
            {},
            {'period': 20},
            {'period': 20}
        ]
        
        total_time = 0
        for indicator, param in zip(indicators, params):
            start_time = time.time()
            library.calculate_indicator(indicator, prices, **param)
            elapsed = time.time() - start_time
            times[indicator] = elapsed
            total_time += elapsed
        
        print(f"{size}\t{times['ema']:.3f}\t{times['rsi']:.3f}\t{times['macd']:.3f}\t{times['bollinger_bands']:.3f}\t{times['z_score']:.3f}\t{total_time:.3f}")
    
    print("\nAll times in seconds. Performance scales well with data size!")


def main():
    """Main demonstration function"""
    print("Technical Indicators Library - Comprehensive Demo")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        demo_individual_indicators()
        demo_indicator_library()
        demo_parameter_optimization()
        demo_trading_signals()
        demo_performance_comparison()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("[OK] All 5 technical indicators (EMA, RSI, MACD, Bollinger Bands, Z-Score)")
        print("[OK] Vectorized implementations for performance")
        print("[OK] Parameter optimization with grid search and random search")
        print("[OK] Trading signal generation")
        print("[OK] Performance characteristics")
        print("[OK] Comprehensive error handling and validation")
        
        print("\nNext Steps:")
        print("• Integrate indicators into trading strategies")
        print("• Use parameter optimization for strategy tuning")
        print("• Combine multiple indicators for signal confirmation")
        print("• Implement backtesting with optimized parameters")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()