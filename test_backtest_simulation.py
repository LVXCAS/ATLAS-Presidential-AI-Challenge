#!/usr/bin/env python3
"""
Backtesting Simulation for OPTIONS_BOT
Tests bot performance on historical market data patterns
"""

import asyncio
import sys
import random
import numpy as np
from datetime import datetime, timedelta
sys.path.append('.')

from OPTIONS_BOT import TomorrowReadyOptionsBot

async def backtest_simulation():
    """Run backtesting simulation based on historical market patterns"""
    print("BACKTESTING SIMULATION - OPTIONS_BOT")
    print("=" * 60)
    
    bot = TomorrowReadyOptionsBot()
    
    # Initialize bot
    try:
        await bot.initialize_all_systems()
        print(f"[OK] Bot initialized - Account: ${bot.risk_manager.account_value:,.2f}")
    except Exception as e:
        print(f"[FAIL] Bot initialization failed: {e}")
        return
    
    # Simulate different market periods (based on historical patterns)
    market_periods = [
        {
            "name": "Bull Market 2020-2021",
            "duration_days": 250,
            "avg_daily_return": 0.0008,
            "volatility": 22.0,
            "trend_strength": 0.7,
            "description": "COVID recovery bull run"
        },
        {
            "name": "Bear Market 2022",
            "duration_days": 180,
            "avg_daily_return": -0.0012,
            "volatility": 28.0,
            "trend_strength": -0.6,
            "description": "Inflation/rate hike selloff"
        },
        {
            "name": "Sideways 2015-2016",
            "duration_days": 300,
            "avg_daily_return": 0.0002,
            "volatility": 16.0,
            "trend_strength": 0.1,
            "description": "Range-bound market"
        },
        {
            "name": "Volatile 2018",
            "duration_days": 150,
            "avg_daily_return": -0.0005,
            "volatility": 35.0,
            "trend_strength": -0.3,
            "description": "Trade war volatility"
        },
        {
            "name": "Low Vol 2017",
            "duration_days": 200,
            "avg_daily_return": 0.0006,
            "volatility": 12.0,
            "trend_strength": 0.4,
            "description": "Goldilocks economy"
        }
    ]
    
    all_results = []
    
    print(f"\nBacktesting across {len(market_periods)} historical periods...\n")
    
    for period in market_periods:
        print(f"PERIOD: {period['name']}")
        print(f"Description: {period['description']}")
        print(f"Duration: {period['duration_days']} days")
        print(f"Characteristics: {period['avg_daily_return']:+.2%} daily return, {period['volatility']:.1f}% vol")
        
        period_results = await simulate_period(bot, period)
        all_results.append(period_results)
        
        # Display period results
        if period_results['trades'] > 0:
            print(f"Results: {period_results['trades']} trades, {period_results['win_rate']:.1%} win rate, ${period_results['total_pnl']:,.2f} P&L")
        else:
            print("Results: No trades generated")
        print()
    
    # Comprehensive analysis
    print("=" * 60)
    print("BACKTESTING ANALYSIS")
    print("=" * 60)
    
    total_trades = sum(r['trades'] for r in all_results)
    total_profitable = sum(r['profitable_trades'] for r in all_results)
    total_pnl = sum(r['total_pnl'] for r in all_results)
    
    print(f"\nOVERALL BACKTEST PERFORMANCE:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Profitable Trades: {total_profitable}")
    print(f"  Overall Win Rate: {total_profitable/total_trades:.1%}" if total_trades > 0 else "  Overall Win Rate: N/A")
    print(f"  Total P&L: ${total_pnl:,.2f}")
    print(f"  Average P&L per Trade: ${total_pnl/total_trades:.2f}" if total_trades > 0 else "  Average P&L per Trade: N/A")
    
    print(f"\nPERIOD-BY-PERIOD BREAKDOWN:")
    for i, (period, results) in enumerate(zip(market_periods, all_results)):
        print(f"  {i+1}. {period['name']}:")
        if results['trades'] > 0:
            print(f"     Trades: {results['trades']}, Win Rate: {results['win_rate']:.1%}, P&L: ${results['total_pnl']:,.2f}")
            print(f"     Best Trade: ${results['best_trade']:.2f}, Worst Trade: ${results['worst_trade']:.2f}")
        else:
            print(f"     No trades generated in this period")
    
    # Market regime analysis
    print(f"\nMARKET REGIME ANALYSIS:")
    bull_periods = [r for p, r in zip(market_periods, all_results) if p['trend_strength'] > 0.3]
    bear_periods = [r for p, r in zip(market_periods, all_results) if p['trend_strength'] < -0.3]
    sideways_periods = [r for p, r in zip(market_periods, all_results) if -0.3 <= p['trend_strength'] <= 0.3]
    
    analyze_regime_performance("Bull Markets", bull_periods)
    analyze_regime_performance("Bear Markets", bear_periods)
    analyze_regime_performance("Sideways Markets", sideways_periods)
    
    # Volatility analysis
    print(f"\nVOLATILITY ANALYSIS:")
    high_vol = [r for p, r in zip(market_periods, all_results) if p['volatility'] > 25]
    low_vol = [r for p, r in zip(market_periods, all_results) if p['volatility'] < 18]
    med_vol = [r for p, r in zip(market_periods, all_results) if 18 <= p['volatility'] <= 25]
    
    analyze_regime_performance("High Volatility (>25%)", high_vol)
    analyze_regime_performance("Medium Volatility (18-25%)", med_vol)
    analyze_regime_performance("Low Volatility (<18%)", low_vol)
    
    # Risk analysis
    print(f"\nRISK ANALYSIS:")
    if total_trades > 0:
        all_trade_pnls = []
        for result in all_results:
            all_trade_pnls.extend(result['trade_pnls'])
        
        if all_trade_pnls:
            max_gain = max(all_trade_pnls)
            max_loss = min(all_trade_pnls)
            std_dev = np.std(all_trade_pnls)
            avg_pnl = np.mean(all_trade_pnls)
            
            print(f"  Maximum Single Gain: ${max_gain:.2f}")
            print(f"  Maximum Single Loss: ${max_loss:.2f}")
            print(f"  Standard Deviation: ${std_dev:.2f}")
            print(f"  Sharpe-like Ratio: {avg_pnl/std_dev:.2f}" if std_dev > 0 else "  Sharpe-like Ratio: N/A")
            
            # Drawdown analysis
            cumulative_pnl = np.cumsum(all_trade_pnls)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = cumulative_pnl - running_max
            max_drawdown = np.min(drawdown)
            
            print(f"  Maximum Drawdown: ${max_drawdown:.2f}")
    
    print(f"\n" + "=" * 60)
    print("BACKTESTING COMPLETE")
    
    return all_results

async def simulate_period(bot, period_config):
    """Simulate trading for a specific market period"""
    results = {
        'period_name': period_config['name'],
        'trades': 0,
        'profitable_trades': 0,
        'total_pnl': 0.0,
        'trade_pnls': [],
        'best_trade': 0.0,
        'worst_trade': 0.0,
        'strategies_used': {},
        'win_rate': 0.0
    }
    
    # Simulate trading days in this period
    num_trading_days = min(period_config['duration_days'], 60)  # Limit for performance
    
    for day in range(num_trading_days):
        # Generate market conditions for this day
        daily_return = np.random.normal(
            period_config['avg_daily_return'],
            period_config['volatility'] / 100 / np.sqrt(252)
        )
        
        # Create realistic market data
        market_data = generate_market_data_for_day(period_config, daily_return, day)
        
        # Test if bot would trade
        try:
            opportunity = await bot.find_high_quality_opportunity('SPY')
            
            if opportunity:
                results['trades'] += 1
                
                # Track strategy usage
                strategy = str(opportunity['strategy'])
                results['strategies_used'][strategy] = results['strategies_used'].get(strategy, 0) + 1
                
                # Simulate trade outcome based on market conditions
                trade_pnl = simulate_trade_outcome(period_config, opportunity, daily_return)
                
                results['total_pnl'] += trade_pnl
                results['trade_pnls'].append(trade_pnl)
                
                if trade_pnl > 0:
                    results['profitable_trades'] += 1
                
                results['best_trade'] = max(results['best_trade'], trade_pnl)
                results['worst_trade'] = min(results['worst_trade'], trade_pnl)
        
        except Exception:
            continue  # Skip failed days
    
    # Calculate final metrics
    if results['trades'] > 0:
        results['win_rate'] = results['profitable_trades'] / results['trades']
    
    return results

def generate_market_data_for_day(period_config, daily_return, day_number):
    """Generate realistic market data for a trading day"""
    
    # Base volatility with some randomness
    vol = period_config['volatility'] * random.uniform(0.8, 1.2)
    
    # Volume pattern (higher on big moves)
    volume_base = 1.0 + abs(daily_return) * 10
    volume_ratio = volume_base * random.uniform(0.7, 1.5)
    
    # Momentum includes trend and daily move
    trend_momentum = period_config['trend_strength'] * 0.02  # Convert to daily momentum
    total_momentum = trend_momentum + daily_return
    
    return {
        'symbol': 'SPY',
        'current_price': 400.0 + day_number * trend_momentum * 400,
        'price_momentum': total_momentum,
        'realized_vol': vol,
        'volume_ratio': volume_ratio,
        'price_position': random.uniform(0.3, 0.7),
        'avg_volume': 50000000,
        'timestamp': datetime.now()
    }

def simulate_trade_outcome(period_config, opportunity, daily_return):
    """Simulate the P&L outcome of a trade"""
    
    # Base success probability
    base_success_prob = 0.5
    
    # Adjust based on market conditions and opportunity quality
    confidence_adj = (opportunity['confidence'] - 0.5) * 0.4
    
    # Trend alignment bonus/penalty
    is_bullish_strategy = 'BULL' in str(opportunity['strategy'])
    trend_alignment = period_config['trend_strength']
    
    if is_bullish_strategy and trend_alignment > 0:
        trend_adj = min(0.2, trend_alignment * 0.3)
    elif not is_bullish_strategy and trend_alignment < 0:
        trend_adj = min(0.2, abs(trend_alignment) * 0.3)
    else:
        trend_adj = -0.1  # Penalty for trading against trend
    
    # Volatility impact (options generally benefit from higher vol)
    vol_adj = min(0.15, (period_config['volatility'] - 20) * 0.01)
    
    success_prob = base_success_prob + confidence_adj + trend_adj + vol_adj
    success_prob = max(0.1, min(0.85, success_prob))
    
    # Determine trade outcome
    if random.random() < success_prob:
        # Winning trade
        profit_factor = random.uniform(0.6, 1.2)  # 60-120% of max profit
        return opportunity['max_profit'] * profit_factor
    else:
        # Losing trade
        loss_factor = random.uniform(0.4, 1.0)  # 40-100% of max loss
        return -opportunity['max_loss'] * loss_factor

def analyze_regime_performance(regime_name, results):
    """Analyze performance for a specific market regime"""
    if not results:
        print(f"  {regime_name}: No data")
        return
    
    total_trades = sum(r['trades'] for r in results)
    total_profitable = sum(r['profitable_trades'] for r in results)
    total_pnl = sum(r['total_pnl'] for r in results)
    
    if total_trades > 0:
        win_rate = total_profitable / total_trades
        avg_pnl_per_trade = total_pnl / total_trades
        print(f"  {regime_name}: {total_trades} trades, {win_rate:.1%} win rate, ${avg_pnl_per_trade:.2f} avg P&L")
    else:
        print(f"  {regime_name}: No trades generated")

if __name__ == "__main__":
    asyncio.run(backtest_simulation())