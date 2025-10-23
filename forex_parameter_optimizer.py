#!/usr/bin/env python3
"""
FOREX PARAMETER OPTIMIZER
Systematic grid search to find optimal parameters for 60%+ win rate

APPROACH:
1. Test multiple parameter combinations
2. Find settings that generate 100+ trades
3. Maximize win rate while maintaining sample size
4. Validate with walk-forward testing
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import json

from data.oanda_data_fetcher import OandaDataFetcher
from forex_v4_optimized import ForexV4OptimizedStrategy
from forex_v4_backtest import ForexV4Backtester


def optimize_parameters(pair='EUR_USD', candles=5000):
    """
    Grid search to find optimal parameters

    Tests combinations of:
    - EMA periods
    - RSI bounds
    - ADX threshold
    - Score threshold
    - Risk/reward ratio
    """

    print(f"\n{'='*80}")
    print(f"PARAMETER OPTIMIZATION: {pair}")
    print(f"{'='*80}")

    # Fetch data
    print(f"\n[1/3] Fetching {candles} candles...")
    fetcher = OandaDataFetcher()
    df = fetcher.get_bars(pair, timeframe='H1', limit=candles)

    if df is None or df.empty:
        print(f"[ERROR] Could not fetch data")
        return None

    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    print(f"[OK] Fetched {len(df)} candles")

    # Parameter grid (RELAXED for more signals)
    param_grid = {
        'ema_fast': [8, 10, 12],
        'ema_slow': [21],
        'rsi_long_lower': [45, 48, 50],
        'rsi_long_upper': [75, 80],
        'rsi_short_lower': [20, 25],
        'rsi_short_upper': [50, 52, 55],
        'adx_threshold': [0, 20, 25],  # 0 = disabled
        'score_threshold': [6.0, 7.0, 8.0],
        'risk_reward_ratio': [1.5, 2.0]
    }

    print(f"\n[2/3] Testing parameter combinations...")
    print(f"Parameter ranges:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    # Generate combinations
    from itertools import product

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    print(f"\nTotal combinations to test: {len(combinations)}")
    print(f"This may take a few minutes...")

    backtester = ForexV4Backtester(spread_pips=1.5, slippage_pips=0.5)
    results = []

    for i, combo in enumerate(combinations, 1):
        params = dict(zip(param_names, combo))

        # Create strategy
        strategy = ForexV4OptimizedStrategy(
            ema_fast=params['ema_fast'],
            ema_slow=params['ema_slow']
        )
        strategy.set_data_fetcher(fetcher)

        # Set parameters
        strategy.rsi_long_lower = params['rsi_long_lower']
        strategy.rsi_long_upper = params['rsi_long_upper']
        strategy.rsi_short_lower = params['rsi_short_lower']
        strategy.rsi_short_upper = params['rsi_short_upper']
        strategy.adx_threshold = params['adx_threshold']
        strategy.score_threshold = params['score_threshold']
        strategy.risk_reward_ratio = params['risk_reward_ratio']

        # Disable time filter for more signals
        from datetime import time
        strategy.trading_hours = {
            'start': time(0, 0),
            'end': time(23, 59)
        }

        # Disable volatility filter for more signals
        strategy.atr_percentile_min = 0
        strategy.atr_percentile_max = 100

        # Run backtest
        result = backtester.run_backtest(df, strategy, pair)

        if result and result['total_trades'] >= 10:  # At least 10 trades
            results.append({
                'params': params,
                'total_trades': result['total_trades'],
                'wins': result['wins'],
                'losses': result['losses'],
                'win_rate': result['win_rate'],
                'total_pips': result['total_pips'],
                'profit_factor': result['profit_factor'],
                'sharpe_ratio': result['sharpe_ratio'],
                'avg_win': result['avg_win'],
                'avg_loss': result['avg_loss']
            })

        # Progress update
        if i % 50 == 0:
            print(f"  Tested {i}/{len(combinations)} combinations... (found {len(results)} valid)")

    print(f"\n[OK] Found {len(results)} valid parameter sets")

    if len(results) == 0:
        print(f"[WARNING] No parameter sets generated enough trades. Filters too strict.")
        return None

    # Sort by win rate (then by total trades as tiebreaker)
    results.sort(key=lambda x: (x['win_rate'], x['total_trades']), reverse=True)

    print(f"\n[3/3] Top 10 Parameter Sets:")
    print(f"{'-'*120}")
    print(f"{'Rank':<6} {'WR%':<8} {'Trades':<8} {'Pips':<10} {'PF':<8} {'Sharpe':<8} {'Parameters':<60}")
    print(f"{'-'*120}")

    for i, r in enumerate(results[:10], 1):
        params_str = f"EMA:{r['params']['ema_fast']}/{r['params']['ema_slow']}, "
        params_str += f"RSI:L{r['params']['rsi_long_lower']}-{r['params']['rsi_long_upper']}/S{r['params']['rsi_short_lower']}-{r['params']['rsi_short_upper']}, "
        params_str += f"ADX:{r['params']['adx_threshold']}, Score:{r['params']['score_threshold']}, RR:{r['params']['risk_reward_ratio']}"

        print(f"{i:<6} {r['win_rate']:<8.1f} {r['total_trades']:<8} {r['total_pips']:<+10.1f} "
              f"{r['profit_factor']:<8.2f} {r['sharpe_ratio']:<8.2f} {params_str}")

    # Save results
    output_file = f'optimization_results_{pair}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[OK] Full results saved to: {output_file}")

    return results


def validate_top_params(pair='EUR_USD', params=None, candles=5000):
    """
    Validate top parameters with walk-forward testing

    Args:
        pair: Forex pair
        params: Parameter dict from optimization
        candles: Number of candles
    """

    print(f"\n{'='*80}")
    print(f"VALIDATING PARAMETERS: {pair}")
    print(f"{'='*80}")

    # Fetch data
    fetcher = OandaDataFetcher()
    df = fetcher.get_bars(pair, timeframe='H1', limit=candles)

    if df is None or df.empty:
        return None

    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    # Create strategy with optimized params
    strategy = ForexV4OptimizedStrategy(
        ema_fast=params['ema_fast'],
        ema_slow=params['ema_slow']
    )
    strategy.set_data_fetcher(fetcher)

    # Apply parameters
    strategy.rsi_long_lower = params['rsi_long_lower']
    strategy.rsi_long_upper = params['rsi_long_upper']
    strategy.rsi_short_lower = params['rsi_short_lower']
    strategy.rsi_short_upper = params['rsi_short_upper']
    strategy.adx_threshold = params['adx_threshold']
    strategy.score_threshold = params['score_threshold']
    strategy.risk_reward_ratio = params['risk_reward_ratio']

    # Disable time/volatility filters
    from datetime import time
    strategy.trading_hours = {'start': time(0, 0), 'end': time(23, 59)}
    strategy.atr_percentile_min = 0
    strategy.atr_percentile_max = 100

    print(f"\nOptimized Parameters:")
    print(f"  EMA: {params['ema_fast']}/{params['ema_slow']}/200")
    print(f"  RSI Long: {params['rsi_long_lower']}-{params['rsi_long_upper']}")
    print(f"  RSI Short: {params['rsi_short_lower']}-{params['rsi_short_upper']}")
    print(f"  ADX Threshold: {params['adx_threshold']}")
    print(f"  Score Threshold: {params['score_threshold']}")
    print(f"  R:R Ratio: {params['risk_reward_ratio']}:1")

    # Walk-forward validation
    backtester = ForexV4Backtester(spread_pips=1.5, slippage_pips=0.5)
    wf_results = backtester.walk_forward_test(df, strategy, pair, train_pct=0.7)

    # Display results
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD VALIDATION RESULTS")
    print(f"{'='*80}")

    if wf_results['in_sample']:
        in_s = wf_results['in_sample']
        print(f"\nIn-Sample (Training):")
        print(f"  Trades: {in_s['total_trades']}")
        print(f"  Win Rate: {in_s['win_rate']:.1f}%")
        print(f"  Total Pips: {in_s['total_pips']:+.1f}")
        print(f"  Profit Factor: {in_s['profit_factor']:.2f}x")

    if wf_results['out_sample']:
        out_s = wf_results['out_sample']
        print(f"\nOut-of-Sample (Test) *** KEY METRIC ***:")
        print(f"  Trades: {out_s['total_trades']}")
        print(f"  Win Rate: {out_s['win_rate']:.1f}%")
        print(f"  Total Pips: {out_s['total_pips']:+.1f}")
        print(f"  Profit Factor: {out_s['profit_factor']:.2f}x")
        print(f"  Sharpe Ratio: {out_s['sharpe_ratio']:.2f}")

        # Check if achieved target
        if out_s['win_rate'] >= 60 and out_s['total_trades'] >= 30:
            print(f"\n  STATUS: TARGET ACHIEVED on {pair}! [SUCCESS]")
        elif out_s['win_rate'] >= 60:
            print(f"\n  STATUS: Win rate achieved (need more trades)")
        else:
            print(f"\n  STATUS: Below 60% target")

    return wf_results


def main():
    """
    Main optimization workflow

    1. Grid search on each pair
    2. Find best parameters
    3. Validate with walk-forward
    4. Report results
    """

    print(f"\n{'='*80}")
    print(f"FOREX PARAMETER OPTIMIZATION SYSTEM")
    print(f"{'='*80}")
    print(f"\nObjective: Find parameters that achieve 60%+ win rate")
    print(f"Method: Grid search + walk-forward validation")
    print(f"\nThis will test hundreds of parameter combinations.")
    print(f"Estimated time: 5-10 minutes per pair")
    print(f"{'='*80}")

    pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']
    all_optimized = {}

    for pair in pairs:
        print(f"\n\nOPTIMIZING: {pair}")
        print(f"{'='*80}")

        # Run optimization
        results = optimize_parameters(pair, candles=5000)

        if results and len(results) > 0:
            # Get top params
            top_params = results[0]['params']
            all_optimized[pair] = {
                'params': top_params,
                'backtest': results[0]
            }

            # Validate with walk-forward
            print(f"\n\nValidating top parameters on {pair}...")
            wf = validate_top_params(pair, top_params, candles=5000)

            if wf:
                all_optimized[pair]['walk_forward'] = wf

    # Final summary
    print(f"\n\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE - FINAL RESULTS")
    print(f"{'='*80}")

    for pair, data in all_optimized.items():
        print(f"\n{pair}:")
        print(f"  Best Parameters: EMA {data['params']['ema_fast']}/{data['params']['ema_slow']}, "
              f"RSI L{data['params']['rsi_long_lower']}-{data['params']['rsi_long_upper']}, "
              f"ADX {data['params']['adx_threshold']}")

        if 'walk_forward' in data and data['walk_forward']['out_sample']:
            out_s = data['walk_forward']['out_sample']
            status = "[TARGET MET]" if out_s['win_rate'] >= 60 and out_s['total_trades'] >= 30 else "[NEEDS WORK]"
            print(f"  Out-of-Sample: {out_s['win_rate']:.1f}% WR ({out_s['total_trades']} trades, "
                  f"{out_s['total_pips']:+.1f} pips) {status}")

    print(f"\n{'='*80}")
    print(f"Optimization complete. Results saved to optimization_results_*.json")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
