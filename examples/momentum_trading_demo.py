"""
Momentum Trading Agent Demo

This demo showcases the momentum trading agent's capabilities including:
- Technical indicator analysis (EMA, RSI, MACD)
- Fibonacci retracement integration
- Sentiment confirmation
- Volatility-adjusted position sizing
- Explainable AI with top-3 reasons
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict

# Add project root to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the momentum trading agent
from agents.momentum_trading_agent import (
    generate_momentum_signal,
    MomentumTradingAgent,
    MarketData,
    SentimentData
)


def create_sample_market_data(symbol: str, days: int = 60, scenario: str = "uptrend") -> List[Dict]:
    """Create sample market data for different scenarios"""
    data = []
    base_price = 100.0
    
    if scenario == "uptrend":
        # Create uptrending data with some volatility
        trend = np.linspace(0, 20, days)  # 20% gain over period
        noise = np.random.normal(0, 1, days)
        prices = base_price + trend + noise
    elif scenario == "downtrend":
        # Create downtrending data
        trend = np.linspace(0, -15, days)  # 15% loss over period
        noise = np.random.normal(0, 1, days)
        prices = base_price + trend + noise
    elif scenario == "sideways":
        # Create sideways market with oscillations
        prices = np.array([base_price + 3 * np.sin(i * 0.2) + np.random.normal(0, 0.5) for i in range(days)])
    elif scenario == "volatile":
        # Create high volatility market
        prices = np.array([base_price + np.random.normal(0, 3) for _ in range(days)])
    else:
        # Default mild uptrend
        trend = np.linspace(0, 10, days)
        noise = np.random.normal(0, 0.8, days)
        prices = base_price + trend + noise
    
    for i, price in enumerate(prices):
        # Ensure positive prices
        price = max(price, 10.0)
        
        data.append({
            'symbol': symbol,
            'timestamp': (datetime.utcnow() - timedelta(days=days-i)).isoformat(),
            'open': price - np.random.uniform(0.2, 0.8),
            'high': price + np.random.uniform(0.5, 2.0),
            'low': price - np.random.uniform(0.5, 2.0),
            'close': price,
            'volume': int(1000000 + np.random.randint(-200000, 200000)),
            'vwap': price + np.random.uniform(-0.3, 0.3)
        })
    
    return data


def create_sample_sentiment_data(symbol: str, sentiment_type: str = "positive") -> Dict:
    """Create sample sentiment data"""
    if sentiment_type == "positive":
        sentiment_score = np.random.uniform(0.2, 0.8)
    elif sentiment_type == "negative":
        sentiment_score = np.random.uniform(-0.8, -0.2)
    elif sentiment_type == "neutral":
        sentiment_score = np.random.uniform(-0.1, 0.1)
    else:
        sentiment_score = np.random.uniform(-0.5, 0.5)
    
    return {
        'symbol': symbol,
        'overall_sentiment': sentiment_score,
        'confidence': np.random.uniform(0.6, 0.9),
        'news_count': np.random.randint(5, 25),
        'social_sentiment': sentiment_score + np.random.uniform(-0.2, 0.2),
        'timestamp': datetime.utcnow().isoformat()
    }


async def demo_basic_signal_generation():
    """Demo basic momentum signal generation"""
    print("=" * 60)
    print("MOMENTUM TRADING AGENT - BASIC SIGNAL GENERATION DEMO")
    print("=" * 60)
    
    # Create sample data for AAPL with uptrend
    market_data = create_sample_market_data('AAPL', days=50, scenario='uptrend')
    sentiment_data = create_sample_sentiment_data('AAPL', sentiment_type='positive')
    
    print(f"Analyzing {len(market_data)} days of market data for AAPL...")
    print(f"Price range: ${market_data[0]['close']:.2f} -> ${market_data[-1]['close']:.2f}")
    print(f"Sentiment: {sentiment_data['overall_sentiment']:.2f} (confidence: {sentiment_data['confidence']:.2f})")
    print()
    
    # Generate momentum signal
    signal = await generate_momentum_signal('AAPL', market_data, sentiment_data)
    
    if signal:
        print("GENERATED MOMENTUM SIGNAL:")
        print("-" * 40)
        print(f"Symbol: {signal['symbol']}")
        print(f"Signal Type: {signal['signal_type']}")
        print(f"Signal Value: {signal['value']:.3f}")
        print(f"Confidence: {signal['confidence']:.3f}")
        print(f"Position Size: {signal['position_size_pct']:.2%}")
        print(f"Stop Loss: {signal['stop_loss_pct']:.2%}")
        print(f"Take Profit: {signal['take_profit_pct']:.2%}")
        print(f"Market Regime: {signal['market_regime']}")
        print()
        
        print("TOP 3 REASONS:")
        print("-" * 40)
        for reason in signal['top_3_reasons']:
            print(f"{reason['rank']}. {reason['factor']} (contribution: {reason['contribution']:.3f})")
            print(f"   {reason['explanation']}")
            print(f"   Confidence: {reason['confidence']:.2f}")
            print()
        
        print("TECHNICAL SIGNALS BREAKDOWN:")
        print("-" * 40)
        
        if signal['ema_signals']:
            print("EMA Signals:")
            for ema_signal in signal['ema_signals']:
                print(f"  - {ema_signal['indicator']}: {ema_signal['signal_type']} "
                      f"(strength: {ema_signal['strength']:.2f})")
                print(f"    {ema_signal['explanation']}")
        
        if signal['rsi_signals']:
            print("RSI Signals:")
            for rsi_signal in signal['rsi_signals']:
                print(f"  - {rsi_signal['indicator']}: {rsi_signal['signal_type']} "
                      f"(strength: {rsi_signal['strength']:.2f})")
                print(f"    {rsi_signal['explanation']}")
        
        if signal['macd_signals']:
            print("MACD Signals:")
            for macd_signal in signal['macd_signals']:
                print(f"  - {macd_signal['indicator']}: {macd_signal['signal_type']} "
                      f"(strength: {macd_signal['strength']:.2f})")
                print(f"    {macd_signal['explanation']}")
        
        if signal['fibonacci_signals']:
            print("Fibonacci Signals:")
            for fib_signal in signal['fibonacci_signals']:
                print(f"  - {fib_signal['level_type']}: {fib_signal['level_name']} "
                      f"at ${fib_signal['level_price']:.2f}")
                print(f"    Distance: {fib_signal['distance_pct']:.2f}% from current price")
                print(f"    {fib_signal['explanation']}")
    else:
        print("Failed to generate signal - insufficient data or error occurred")


async def demo_multiple_scenarios():
    """Demo signal generation across multiple market scenarios"""
    print("\n" + "=" * 60)
    print("MOMENTUM TRADING AGENT - MULTIPLE SCENARIOS DEMO")
    print("=" * 60)
    
    scenarios = [
        ('uptrend', 'positive'),
        ('downtrend', 'negative'),
        ('sideways', 'neutral'),
        ('volatile', 'mixed')
    ]
    
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT']
    
    for i, (market_scenario, sentiment_type) in enumerate(scenarios):
        symbol = symbols[i]
        
        print(f"\nScenario {i+1}: {market_scenario.upper()} market with {sentiment_type.upper()} sentiment")
        print("-" * 50)
        
        # Create scenario data
        market_data = create_sample_market_data(symbol, days=40, scenario=market_scenario)
        sentiment_data = create_sample_sentiment_data(symbol, sentiment_type=sentiment_type)
        
        # Generate signal
        signal = await generate_momentum_signal(symbol, market_data, sentiment_data)
        
        if signal:
            print(f"Symbol: {signal['symbol']}")
            print(f"Signal: {signal['signal_type']} (value: {signal['value']:.3f})")
            print(f"Confidence: {signal['confidence']:.3f}")
            print(f"Position Size: {signal['position_size_pct']:.2%}")
            print(f"Market Regime: {signal['market_regime']}")
            
            # Show top reason
            if signal['top_3_reasons']:
                top_reason = signal['top_3_reasons'][0]
                print(f"Top Reason: {top_reason['factor']} - {top_reason['explanation']}")
        else:
            print(f"No signal generated for {symbol}")


async def demo_fibonacci_integration():
    """Demo Fibonacci integration specifically"""
    print("\n" + "=" * 60)
    print("MOMENTUM TRADING AGENT - FIBONACCI INTEGRATION DEMO")
    print("=" * 60)
    
    # Create data with clear swing patterns for Fibonacci analysis
    symbol = 'FIBONACCI_TEST'
    
    # Create a pattern with clear swings
    base_prices = []
    # Upswing
    base_prices.extend(np.linspace(100, 120, 20))
    # Retracement
    base_prices.extend(np.linspace(120, 108, 15))
    # Extension
    base_prices.extend(np.linspace(108, 125, 15))
    
    # Add some noise
    prices = np.array(base_prices) + np.random.normal(0, 0.5, len(base_prices))
    
    market_data = []
    for i, price in enumerate(prices):
        market_data.append({
            'symbol': symbol,
            'timestamp': (datetime.utcnow() - timedelta(days=len(prices)-i)).isoformat(),
            'open': price - 0.3,
            'high': price + 0.8,
            'low': price - 0.8,
            'close': price,
            'volume': 1000000,
            'vwap': price
        })
    
    sentiment_data = create_sample_sentiment_data(symbol, sentiment_type='positive')
    
    print(f"Analyzing Fibonacci patterns in {len(market_data)} days of data...")
    print(f"Price pattern: ${market_data[0]['close']:.2f} -> ${market_data[19]['close']:.2f} -> "
          f"${market_data[34]['close']:.2f} -> ${market_data[-1]['close']:.2f}")
    print()
    
    signal = await generate_momentum_signal(symbol, market_data, sentiment_data)
    
    if signal and signal['fibonacci_signals']:
        print("FIBONACCI ANALYSIS RESULTS:")
        print("-" * 40)
        
        for fib_signal in signal['fibonacci_signals']:
            print(f"Level Type: {fib_signal['level_type']}")
            print(f"Level Name: {fib_signal['level_name']}")
            print(f"Level Price: ${fib_signal['level_price']:.2f}")
            print(f"Current Price: ${fib_signal['current_price']:.2f}")
            print(f"Distance: {fib_signal['distance_pct']:.2f}%")
            print(f"Confluence Strength: {fib_signal['confluence_strength']:.2f}")
            print(f"Explanation: {fib_signal['explanation']}")
            print()
        
        # Check if Fibonacci influenced the final signal
        fib_reasons = [r for r in signal['top_3_reasons'] if 'Fibonacci' in r['factor']]
        if fib_reasons:
            print("FIBONACCI INFLUENCE ON FINAL SIGNAL:")
            print("-" * 40)
            for reason in fib_reasons:
                print(f"Rank: {reason['rank']}")
                print(f"Contribution: {reason['contribution']:.3f}")
                print(f"Explanation: {reason['explanation']}")
    else:
        print("No Fibonacci signals detected in this data")


async def demo_sentiment_impact():
    """Demo the impact of sentiment on signals"""
    print("\n" + "=" * 60)
    print("MOMENTUM TRADING AGENT - SENTIMENT IMPACT DEMO")
    print("=" * 60)
    
    # Create the same market data but with different sentiment scenarios
    symbol = 'SENTIMENT_TEST'
    market_data = create_sample_market_data(symbol, days=30, scenario='uptrend')
    
    sentiment_scenarios = [
        ('Very Positive', 0.8),
        ('Positive', 0.3),
        ('Neutral', 0.0),
        ('Negative', -0.3),
        ('Very Negative', -0.8)
    ]
    
    print("Testing same market data with different sentiment scenarios:")
    print(f"Market: Uptrending from ${market_data[0]['close']:.2f} to ${market_data[-1]['close']:.2f}")
    print()
    
    for sentiment_name, sentiment_score in sentiment_scenarios:
        sentiment_data = {
            'symbol': symbol,
            'overall_sentiment': sentiment_score,
            'confidence': 0.8,
            'news_count': 15,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        signal = await generate_momentum_signal(symbol, market_data, sentiment_data)
        
        if signal:
            print(f"{sentiment_name} Sentiment ({sentiment_score:+.1f}):")
            print(f"  Signal: {signal['signal_type']} (value: {signal['value']:+.3f})")
            print(f"  Confidence: {signal['confidence']:.3f}")
            print(f"  Position Size: {signal['position_size_pct']:.2%}")
            
            # Check if sentiment is in top reasons
            sentiment_reasons = [r for r in signal['top_3_reasons'] if 'Sentiment' in r['factor']]
            if sentiment_reasons:
                reason = sentiment_reasons[0]
                print(f"  Sentiment Impact: Rank {reason['rank']} - {reason['explanation']}")
            print()


async def demo_volatility_adjustment():
    """Demo volatility adjustment impact"""
    print("\n" + "=" * 60)
    print("MOMENTUM TRADING AGENT - VOLATILITY ADJUSTMENT DEMO")
    print("=" * 60)
    
    volatility_scenarios = [
        ('Low Volatility', 'uptrend'),
        ('High Volatility', 'volatile'),
        ('Sideways Market', 'sideways')
    ]
    
    symbol = 'VOLATILITY_TEST'
    
    for scenario_name, market_scenario in volatility_scenarios:
        print(f"{scenario_name} Scenario:")
        print("-" * 30)
        
        market_data = create_sample_market_data(symbol, days=40, scenario=market_scenario)
        sentiment_data = create_sample_sentiment_data(symbol, sentiment_type='positive')
        
        signal = await generate_momentum_signal(symbol, market_data, sentiment_data)
        
        if signal:
            print(f"Market Regime: {signal['market_regime']}")
            print(f"Volatility Adjustment: {signal['volatility_adjustment']:.2f}")
            print(f"Position Size: {signal['position_size_pct']:.2%}")
            print(f"Signal Confidence: {signal['confidence']:.3f}")
            
            # Check if volatility is in top reasons
            vol_reasons = [r for r in signal['top_3_reasons'] if 'Volatility' in r['factor']]
            if vol_reasons:
                reason = vol_reasons[0]
                print(f"Volatility Impact: {reason['explanation']}")
            print()


async def main():
    """Run all demos"""
    print("MOMENTUM TRADING AGENT COMPREHENSIVE DEMO")
    print("=" * 60)
    print("This demo showcases the momentum trading agent's capabilities")
    print("including technical analysis, Fibonacci integration, sentiment")
    print("confirmation, and volatility-adjusted position sizing.")
    print()
    
    try:
        # Run all demo scenarios
        await demo_basic_signal_generation()
        await demo_multiple_scenarios()
        await demo_fibonacci_integration()
        await demo_sentiment_impact()
        await demo_volatility_adjustment()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("The momentum trading agent demonstrated:")
        print("[OK] Technical indicator analysis (EMA, RSI, MACD)")
        print("[OK] Fibonacci retracement integration")
        print("[OK] Sentiment confirmation and alignment")
        print("[OK] Volatility-adjusted position sizing")
        print("[OK] Explainable AI with top-3 reasons")
        print("[OK] Multiple market regime handling")
        print("[OK] Risk management (stop loss, take profit)")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run the demo
    asyncio.run(main())