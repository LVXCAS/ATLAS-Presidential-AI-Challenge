"""
Fibonacci Analysis Demo for LangGraph Trading System

This demo showcases the comprehensive Fibonacci analysis capabilities including:
- Swing point detection in real market data
- Fibonacci retracement and extension calculations
- Support/resistance level identification
- Confluence zone detection
- Integration potential with technical indicators

Run this script to see Fibonacci analysis in action with sample market data.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.fibonacci_analysis import (
    FibonacciAnalyzer, analyze_fibonacci_levels,
    calculate_fibonacci_retracements, detect_confluence_zones_simple
)


def generate_realistic_market_data(days=100, start_price=100.0, trend=0.1, volatility=0.02):
    """
    Generate realistic market data with trends, pullbacks, and volatility
    
    Args:
        days: Number of trading days
        start_price: Starting price
        trend: Daily trend (0.1 = 10% uptrend over period)
        volatility: Daily volatility
    
    Returns:
        DataFrame with OHLC data
    """
    np.random.seed(42)  # For reproducible results
    
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Generate base price movement with trend
    daily_returns = np.random.normal(trend/days, volatility, days)
    prices = [start_price]
    
    for i in range(1, days):
        # Add some mean reversion and momentum effects
        momentum = 0.1 * daily_returns[i-1]  # Momentum effect
        mean_reversion = -0.05 * (prices[-1] - start_price) / start_price  # Mean reversion
        
        new_return = daily_returns[i] + momentum + mean_reversion
        new_price = prices[-1] * (1 + new_return)
        prices.append(new_price)
    
    # Generate OHLC from close prices
    closes = np.array(prices)
    
    # Generate realistic high/low spreads
    daily_ranges = np.random.lognormal(mean=np.log(0.02), sigma=0.5, size=days)
    daily_ranges = np.clip(daily_ranges, 0.005, 0.1)  # Limit range
    
    highs = closes * (1 + daily_ranges * np.random.uniform(0.3, 1.0, days))
    lows = closes * (1 - daily_ranges * np.random.uniform(0.3, 1.0, days))
    
    # Ensure OHLC consistency
    opens = np.roll(closes, 1)
    opens[0] = start_price
    
    # Adjust for OHLC logic
    for i in range(days):
        high_candidates = [opens[i], closes[i], highs[i]]
        low_candidates = [opens[i], closes[i], lows[i]]
        
        highs[i] = max(high_candidates)
        lows[i] = min(low_candidates)
    
    return pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': np.random.randint(100000, 1000000, days)
    })


def demo_basic_fibonacci_analysis():
    """Demonstrate basic Fibonacci analysis functionality"""
    print("=" * 60)
    print("FIBONACCI ANALYSIS DEMO - Basic Functionality")
    print("=" * 60)
    
    # Generate sample market data
    print("\n1. Generating sample market data...")
    df = generate_realistic_market_data(days=60, start_price=100.0, trend=0.2, volatility=0.025)
    
    print(f"Generated {len(df)} days of market data")
    print(f"Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
    print(f"Final close: ${df['Close'].iloc[-1]:.2f}")
    
    # Perform Fibonacci analysis
    print("\n2. Performing comprehensive Fibonacci analysis...")
    
    result = analyze_fibonacci_levels(
        high_data=df['High'].values,
        low_data=df['Low'].values,
        close_data=df['Close'].values,
        timestamps=df['Date'].values,
        lookback_periods=3,
        min_touches=2,
        tolerance_pct=1.0
    )
    
    # Display results
    print(f"\nAnalysis Results:")
    print(f"- Swing highs detected: {len(result['swing_highs'])}")
    print(f"- Swing lows detected: {len(result['swing_lows'])}")
    print(f"- Fibonacci level sets: {len(result['fibonacci_levels'])}")
    print(f"- Fibonacci extensions: {len(result['fibonacci_extensions'])}")
    print(f"- Support/Resistance levels: {len(result['support_resistance'])}")
    print(f"- Confluence zones: {len(result['confluence_zones'])}")
    
    # Show some specific Fibonacci levels
    if result['fibonacci_levels']:
        print(f"\n3. Sample Fibonacci Retracement Levels:")
        fib_set = result['fibonacci_levels'][0]
        print(f"   Swing: ${fib_set['swing_low']['price']:.2f} -> ${fib_set['swing_high']['price']:.2f}")
        print(f"   Direction: {fib_set['direction']}")
        print(f"   Key levels:")
        
        for level_name, price in fib_set['level_prices'].items():
            percentage = fib_set['levels'][level_name] * 100
            print(f"     {percentage:5.1f}%: ${price:7.2f}")
    
    # Show confluence zones
    if result['confluence_zones']:
        print(f"\n4. Top Confluence Zones:")
        for i, zone in enumerate(result['confluence_zones'][:3]):
            print(f"   Zone {i+1}: ${zone['price']:.2f} (Strength: {zone['strength']:.2f})")
            print(f"           Components: {', '.join(zone['components'])}")
    
    return result, df


def demo_swing_point_detection():
    """Demonstrate swing point detection with visualization"""
    print("\n" + "=" * 60)
    print("SWING POINT DETECTION DEMO")
    print("=" * 60)
    
    # Generate data with clear swings
    df = generate_realistic_market_data(days=40, start_price=100.0, trend=0.0, volatility=0.03)
    
    analyzer = FibonacciAnalyzer(lookback_periods=3)
    
    # Detect swing points
    swing_highs, swing_lows = analyzer.swing_detector.detect_swing_points(
        df['High'].values, df['Low'].values, df['Date'].values
    )
    
    print(f"\nDetected {len(swing_highs)} swing highs and {len(swing_lows)} swing lows")
    
    # Show swing points
    print(f"\nSwing Highs:")
    for i, swing in enumerate(swing_highs[:5]):  # Show first 5
        date_str = pd.to_datetime(swing.timestamp).strftime('%Y-%m-%d') if swing.timestamp else f"Day {swing.index}"
        print(f"  {i+1}. ${swing.price:.2f} on {date_str}")
    
    print(f"\nSwing Lows:")
    for i, swing in enumerate(swing_lows[:5]):  # Show first 5
        date_str = pd.to_datetime(swing.timestamp).strftime('%Y-%m-%d') if swing.timestamp else f"Day {swing.index}"
        print(f"  {i+1}. ${swing.price:.2f} on {date_str}")
    
    return swing_highs, swing_lows, df


def demo_confluence_detection():
    """Demonstrate confluence zone detection"""
    print("\n" + "=" * 60)
    print("CONFLUENCE ZONE DETECTION DEMO")
    print("=" * 60)
    
    # Create price levels that should form confluence zones
    price_levels = [
        100.0, 100.2,  # Confluence around 100
        95.5, 95.8,    # Confluence around 95.6
        105.0,          # Isolated level
        90.0, 89.8, 90.3  # Confluence around 90
    ]
    
    print(f"Input price levels: {price_levels}")
    
    # Detect confluence zones
    confluence_zones = detect_confluence_zones_simple(price_levels, tolerance_pct=1.0)
    
    print(f"\nDetected {len(confluence_zones)} confluence zones:")
    
    for i, zone in enumerate(confluence_zones):
        print(f"\nZone {i+1}:")
        print(f"  Price: ${zone['price']:.2f}")
        print(f"  Strength: {zone['strength']:.2f}")
        print(f"  Level count: {zone['level_count']}")
        print(f"  Components: {zone['components']}")
    
    return confluence_zones


def demo_fibonacci_retracements():
    """Demonstrate Fibonacci retracement calculations"""
    print("\n" + "=" * 60)
    print("FIBONACCI RETRACEMENT CALCULATIONS DEMO")
    print("=" * 60)
    
    # Example swing: Stock moves from $80 to $120, then retraces
    swing_low_price = 80.0
    swing_high_price = 120.0
    
    print(f"Example swing: ${swing_low_price:.2f} -> ${swing_high_price:.2f}")
    print(f"Swing range: ${swing_high_price - swing_low_price:.2f}")
    
    # Calculate standard retracements
    fib_levels = calculate_fibonacci_retracements(
        swing_high_price=swing_high_price,
        swing_low_price=swing_low_price,
        swing_high_index=20,
        swing_low_index=10
    )
    
    print(f"\nFibonacci Retracement Levels ({fib_levels['direction']} trend):")
    print(f"{'Level':<10} {'Price':<10} {'Retracement':<12}")
    print("-" * 35)
    
    for level_name, price in fib_levels['level_prices'].items():
        percentage = fib_levels['levels'][level_name] * 100
        retracement = swing_high_price - price
        print(f"{percentage:5.1f}%    ${price:7.2f}   ${retracement:7.2f}")
    
    # Show potential entry zones
    print(f"\nPotential Entry Zones for Continuation:")
    key_levels = ['fib_382', 'fib_500', 'fib_618']
    for level in key_levels:
        if level in fib_levels['level_prices']:
            price = fib_levels['level_prices'][level]
            percentage = fib_levels['levels'][level] * 100
            print(f"  {percentage:5.1f}% level: ${price:.2f} (Strong support expected)")
    
    return fib_levels


def demo_support_resistance_detection():
    """Demonstrate support and resistance level detection"""
    print("\n" + "=" * 60)
    print("SUPPORT/RESISTANCE DETECTION DEMO")
    print("=" * 60)
    
    # Generate data with clear support/resistance levels
    df = generate_realistic_market_data(days=50, start_price=100.0, trend=0.05, volatility=0.02)
    
    analyzer = FibonacciAnalyzer(min_touches=2, tolerance_pct=1.0)
    
    # Detect support/resistance levels
    sr_levels = analyzer.sr_detector.detect_levels(
        df['High'].values, df['Low'].values, df['Close'].values
    )
    
    print(f"Detected {len(sr_levels)} support/resistance levels:")
    
    # Sort by strength
    sr_levels.sort(key=lambda x: x.strength, reverse=True)
    
    print(f"\n{'Type':<12} {'Price':<10} {'Strength':<10} {'Touches':<8}")
    print("-" * 45)
    
    for level in sr_levels[:8]:  # Show top 8 levels
        print(f"{level.level_type:<12} ${level.price:<9.2f} {level.strength:<9.2f} {level.touch_count:<8}")
    
    # Show strongest levels by type
    support_levels = [l for l in sr_levels if l.level_type == 'support']
    resistance_levels = [l for l in sr_levels if l.level_type == 'resistance']
    
    if support_levels:
        strongest_support = support_levels[0]
        print(f"\nStrongest Support: ${strongest_support.price:.2f} (Strength: {strongest_support.strength:.2f})")
    
    if resistance_levels:
        strongest_resistance = resistance_levels[0]
        print(f"Strongest Resistance: ${strongest_resistance.price:.2f} (Strength: {strongest_resistance.strength:.2f})")
    
    return sr_levels


def demo_integration_potential():
    """Demonstrate integration potential with technical indicators"""
    print("\n" + "=" * 60)
    print("INTEGRATION WITH TECHNICAL INDICATORS - CONCEPT DEMO")
    print("=" * 60)
    
    # Generate sample data
    df = generate_realistic_market_data(days=50, start_price=100.0, trend=0.15, volatility=0.025)
    
    # Perform Fibonacci analysis
    fib_result = analyze_fibonacci_levels(
        df['High'].values, df['Low'].values, df['Close'].values
    )
    
    # Simulate technical indicator levels (in real implementation, these would come from technical_indicators.py)
    current_price = df['Close'].iloc[-1]
    
    # Mock technical indicator levels
    ema_20 = current_price * 0.98  # 20-period EMA slightly below current price
    ema_50 = current_price * 0.95  # 50-period EMA further below
    bollinger_upper = current_price * 1.02
    bollinger_lower = current_price * 0.98
    
    print(f"Current Price: ${current_price:.2f}")
    print(f"\nTechnical Indicator Levels:")
    print(f"  EMA 20: ${ema_20:.2f}")
    print(f"  EMA 50: ${ema_50:.2f}")
    print(f"  Bollinger Upper: ${bollinger_upper:.2f}")
    print(f"  Bollinger Lower: ${bollinger_lower:.2f}")
    
    # Find Fibonacci levels near technical indicator levels
    print(f"\nFibonacci-Technical Indicator Confluence:")
    
    technical_levels = {
        'EMA_20': ema_20,
        'EMA_50': ema_50,
        'BB_Upper': bollinger_upper,
        'BB_Lower': bollinger_lower
    }
    
    confluence_found = False
    tolerance = 0.01  # 1% tolerance
    
    for fib_set in fib_result['fibonacci_levels']:
        for fib_name, fib_price in fib_set['level_prices'].items():
            for tech_name, tech_price in technical_levels.items():
                if abs(fib_price - tech_price) / tech_price <= tolerance:
                    confluence_found = True
                    percentage = fib_set['levels'][fib_name] * 100
                    print(f"  CONFLUENCE: {percentage:.1f}% Fib (${fib_price:.2f}) ≈ {tech_name} (${tech_price:.2f})")
    
    if not confluence_found:
        print("  No close confluence detected in this sample (try different parameters)")
    
    # Show how this could enhance trading signals
    print(f"\nSignal Enhancement Concept:")
    print(f"  - Fibonacci retracement + EMA support = Strong buy signal")
    print(f"  - Fibonacci extension + Bollinger upper = Take profit target")
    print(f"  - Multiple confluence zones = High-probability reversal areas")
    print(f"  - Support/resistance + Fibonacci = Enhanced entry/exit points")


def create_visualization_data(result, df):
    """Prepare data for visualization (conceptual - would need matplotlib/plotly)"""
    print("\n" + "=" * 60)
    print("VISUALIZATION DATA PREPARATION")
    print("=" * 60)
    
    print("Preparing data for potential chart visualization...")
    
    # Price data
    print(f"Price data points: {len(df)}")
    print(f"Date range: {df['Date'].iloc[0].strftime('%Y-%m-%d')} to {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    
    # Swing points for plotting
    swing_points = []
    for swing in result['swing_highs']:
        swing_points.append({
            'index': swing['index'],
            'price': swing['price'],
            'type': 'high'
        })
    
    for swing in result['swing_lows']:
        swing_points.append({
            'index': swing['index'],
            'price': swing['price'],
            'type': 'low'
        })
    
    print(f"Swing points for plotting: {len(swing_points)}")
    
    # Fibonacci levels for horizontal lines
    fib_lines = []
    for fib_set in result['fibonacci_levels']:
        for level_name, price in fib_set['level_prices'].items():
            percentage = fib_set['levels'][level_name] * 100
            fib_lines.append({
                'price': price,
                'label': f"{percentage:.1f}%",
                'type': 'fibonacci'
            })
    
    print(f"Fibonacci levels for horizontal lines: {len(fib_lines)}")
    
    # Confluence zones for highlighting
    confluence_zones = result['confluence_zones']
    print(f"Confluence zones for highlighting: {len(confluence_zones)}")
    
    print("\nVisualization would include:")
    print("  - Candlestick/line chart of price data")
    print("  - Swing high/low markers")
    print("  - Horizontal Fibonacci level lines")
    print("  - Highlighted confluence zones")
    print("  - Support/resistance level lines")
    print("  - Legend with level percentages and strengths")
    
    return {
        'price_data': df,
        'swing_points': swing_points,
        'fibonacci_lines': fib_lines,
        'confluence_zones': confluence_zones
    }


def main():
    """Run all Fibonacci analysis demos"""
    print("FIBONACCI ANALYSIS LIBRARY DEMONSTRATION")
    print("LangGraph Trading System")
    print("=" * 60)
    
    try:
        # Run all demos
        result, df = demo_basic_fibonacci_analysis()
        demo_swing_point_detection()
        demo_confluence_detection()
        demo_fibonacci_retracements()
        demo_support_resistance_detection()
        demo_integration_potential()
        create_visualization_data(result, df)
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nThe Fibonacci Analysis Library provides:")
        print("[OK] Comprehensive swing point detection")
        print("[OK] Accurate Fibonacci retracement calculations")
        print("[OK] Fibonacci extension projections")
        print("[OK] Support/resistance level identification")
        print("[OK] Intelligent confluence zone detection")
        print("[OK] Integration framework for technical indicators")
        print("[OK] Robust error handling and validation")
        print("[OK] Performance-optimized calculations")
        
        print(f"\nReady for integration with:")
        print("  - Momentum trading strategies")
        print("  - Mean reversion strategies")
        print("  - Technical indicator confluence")
        print("  - Risk management systems")
        print("  - Signal enhancement algorithms")
        
    except Exception as e:
        print(f"\nDemo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)