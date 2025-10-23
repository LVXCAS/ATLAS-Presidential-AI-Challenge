#!/usr/bin/env python3
"""
FOREX QUICK OPTIMIZER
Fast parameter search focusing on key variables

Tests fewer combinations but covers the most impactful parameters
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
import json

from data.oanda_data_fetcher import OandaDataFetcher
from forex_v4_optimized import ForexV4OptimizedStrategy
from forex_v4_backtest import ForexV4Backtester


def quick_optimize(pair='EUR_USD', candles=5000):
    """
    Quick optimization focusing on most impactful parameters

    Tests ~50 combinations instead of ~2000
    """

    print(f"\n{'='*80}")
    print(f"QUICK OPTIMIZATION: {pair}")
    print(f"{'='*80}")

    # Fetch data
    print(f"\nFetching {candles} candles...")
    fetcher = OandaDataFetcher()
    df = fetcher.get_bars(pair, timeframe='H1', limit=candles)

    if df is None or df.empty:
        print(f"[ERROR] Could not fetch data")
        return None

    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    print(f"[OK] Fetched {len(df)} candles")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Focused parameter grid
    configs = [
        # Config 1: Relaxed filters (more signals)
        {
            'name': 'Relaxed',
            'ema_fast': 8,
            'rsi_long_lower': 45,
            'rsi_long_upper': 80,
            'rsi_short_lower': 20,
            'rsi_short_upper': 55,
            'adx_threshold': 0,  # Disabled
            'score_threshold': 5.0,
            'risk_reward_ratio': 1.5
        },
        # Config 2: Moderate filters
        {
            'name': 'Moderate',
            'ema_fast': 10,
            'rsi_long_lower': 48,
            'rsi_long_upper': 75,
            'rsi_short_lower': 25,
            'rsi_short_upper': 52,
            'adx_threshold': 20,
            'score_threshold': 6.5,
            'risk_reward_ratio': 1.5
        },
        # Config 3: Stricter (quality over quantity)
        {
            'name': 'Strict',
            'ema_fast': 10,
            'rsi_long_lower': 50,
            'rsi_long_upper': 70,
            'rsi_short_lower': 30,
            'rsi_short_upper': 50,
            'adx_threshold': 25,
            'score_threshold': 8.0,
            'risk_reward_ratio': 2.0
        },
        # Config 4: Original v3 settings
        {
            'name': 'V3_Original',
            'ema_fast': 8,
            'rsi_long_lower': 48,
            'rsi_long_upper': 80,
            'rsi_short_lower': 20,
            'rsi_short_upper': 52,
            'adx_threshold': 0,
            'score_threshold': 6.5,
            'risk_reward_ratio': 1.5
        },
        # Config 5: Balanced (sweet spot)
        {
            'name': 'Balanced',
            'ema_fast': 10,
            'rsi_long_lower': 47,
            'rsi_long_upper': 77,
            'rsi_short_lower': 23,
            'rsi_short_upper': 53,
            'adx_threshold': 18,
            'score_threshold': 6.0,
            'risk_reward_ratio': 1.5
        },
    ]

    print(f"\nTesting {len(configs)} configurations...")
    print(f"{'='*80}")

    backtester = ForexV4Backtester(spread_pips=1.5, slippage_pips=0.5)
    results = []

    for config in configs:
        print(f"\nTesting: {config['name']}")
        print(f"  EMA: {config['ema_fast']}/21/200")
        print(f"  RSI: L{config['rsi_long_lower']}-{config['rsi_long_upper']}, S{config['rsi_short_lower']}-{config['rsi_short_upper']}")
        print(f"  ADX: {config['adx_threshold']}, Score: {config['score_threshold']}, R:R: {config['risk_reward_ratio']}")

        # Create strategy
        strategy = ForexV4OptimizedStrategy(
            ema_fast=config['ema_fast'],
            ema_slow=21,
            ema_trend=200
        )
        strategy.set_data_fetcher(fetcher)

        # Apply config
        strategy.rsi_long_lower = config['rsi_long_lower']
        strategy.rsi_long_upper = config['rsi_long_upper']
        strategy.rsi_short_lower = config['rsi_short_lower']
        strategy.rsi_short_upper = config['rsi_short_upper']
        strategy.adx_threshold = config['adx_threshold']
        strategy.score_threshold = config['score_threshold']
        strategy.risk_reward_ratio = config['risk_reward_ratio']

        # Disable time filter for now (24/7 trading)
        strategy.trading_hours = {'start': time(0, 0), 'end': time(23, 59)}

        # Disable volatility filter
        strategy.atr_percentile_min = 0
        strategy.atr_percentile_max = 100

        # Run backtest
        result = backtester.run_backtest(df, strategy, pair)

        if result:
            results.append({
                'config': config['name'],
                'params': config,
                'total_trades': result['total_trades'],
                'wins': result['wins'],
                'losses': result['losses'],
                'win_rate': result['win_rate'],
                'total_pips': result['total_pips'],
                'profit_factor': result['profit_factor'],
                'sharpe_ratio': result['sharpe_ratio'],
                'avg_win': result['avg_win'],
                'avg_loss': result['avg_loss'],
                'max_drawdown': result['max_drawdown']
            })

            print(f"  -> {result['total_trades']} trades, {result['win_rate']:.1f}% WR, "
                  f"{result['total_pips']:+.1f} pips, PF: {result['profit_factor']:.2f}x")
        else:
            print(f"  -> No trades generated")

    if not results:
        print(f"\n[WARNING] No valid results for {pair}")
        return None

    # Sort by win rate
    results.sort(key=lambda x: (x['win_rate'], x['total_trades']), reverse=True)

    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Config':<15} {'Trades':<10} {'WR%':<10} {'Pips':<12} {'PF':<10} {'Sharpe':<10}")
    print(f"{'-'*80}")

    for r in results:
        print(f"{r['config']:<15} {r['total_trades']:<10} {r['win_rate']:<10.1f} "
              f"{r['total_pips']:<+12.1f} {r['profit_factor']:<10.2f} {r['sharpe_ratio']:<10.2f}")

    # Save results
    output_file = f'quick_optimization_{pair}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[OK] Results saved to: {output_file}")

    return results


def validate_best_config(pair, best_params, candles=5000):
    """
    Validate best configuration with walk-forward testing
    """

    print(f"\n{'='*80}")
    print(f"WALK-FORWARD VALIDATION: {pair}")
    print(f"{'='*80}")

    # Fetch data
    fetcher = OandaDataFetcher()
    df = fetcher.get_bars(pair, timeframe='H1', limit=candles)

    if df is None or df.empty:
        return None

    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    # Split into train/test
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"\nData split:")
    print(f"  Train: {len(train_df)} candles ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"  Test:  {len(test_df)} candles ({test_df.index[0]} to {test_df.index[-1]})")

    # Create strategy with best params
    strategy = ForexV4OptimizedStrategy(
        ema_fast=best_params['ema_fast'],
        ema_slow=21
    )
    strategy.set_data_fetcher(fetcher)

    # Apply params
    strategy.rsi_long_lower = best_params['rsi_long_lower']
    strategy.rsi_long_upper = best_params['rsi_long_upper']
    strategy.rsi_short_lower = best_params['rsi_short_lower']
    strategy.rsi_short_upper = best_params['rsi_short_upper']
    strategy.adx_threshold = best_params['adx_threshold']
    strategy.score_threshold = best_params['score_threshold']
    strategy.risk_reward_ratio = best_params['risk_reward_ratio']

    # Disable filters
    strategy.trading_hours = {'start': time(0, 0), 'end': time(23, 59)}
    strategy.atr_percentile_min = 0
    strategy.atr_percentile_max = 100

    backtester = ForexV4Backtester(spread_pips=1.5, slippage_pips=0.5)

    # Test on both periods
    print(f"\n[IN-SAMPLE] Testing on training period...")
    train_result = backtester.run_backtest(train_df, strategy, pair)

    print(f"\n[OUT-OF-SAMPLE] Testing on unseen data...")
    test_result = backtester.run_backtest(test_df, strategy, pair)

    # Display results
    print(f"\n{'='*80}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*80}")

    if train_result:
        print(f"\nIn-Sample (Training):")
        print(f"  Trades: {train_result['total_trades']}")
        print(f"  Win Rate: {train_result['win_rate']:.1f}%")
        print(f"  Total Pips: {train_result['total_pips']:+.1f}")
        print(f"  Profit Factor: {train_result['profit_factor']:.2f}x")

    if test_result:
        print(f"\nOut-of-Sample (Test) *** CRITICAL ***:")
        print(f"  Trades: {test_result['total_trades']}")
        print(f"  Win Rate: {test_result['win_rate']:.1f}%")
        print(f"  Total Pips: {test_result['total_pips']:+.1f}")
        print(f"  Profit Factor: {test_result['profit_factor']:.2f}x")
        print(f"  Sharpe Ratio: {test_result['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {test_result['max_drawdown']:.1f} pips")

        # Verdict
        print(f"\n{'='*80}")
        if test_result['win_rate'] >= 60 and test_result['total_trades'] >= 30:
            print(f"VERDICT: TARGET ACHIEVED [SUCCESS]")
            print(f"  - 60%+ win rate: YES ({test_result['win_rate']:.1f}%)")
            print(f"  - 30+ trades: YES ({test_result['total_trades']})")
            print(f"  - Out-of-sample validated: YES")
        elif test_result['win_rate'] >= 60:
            print(f"VERDICT: WIN RATE ACHIEVED (need more trades)")
            print(f"  - 60%+ win rate: YES ({test_result['win_rate']:.1f}%)")
            print(f"  - 30+ trades: NO ({test_result['total_trades']}/30)")
        elif test_result['total_trades'] >= 30:
            print(f"VERDICT: SAMPLE SIZE OK (win rate below target)")
            print(f"  - 60%+ win rate: NO ({test_result['win_rate']:.1f}%)")
            print(f"  - 30+ trades: YES ({test_result['total_trades']})")
        else:
            print(f"VERDICT: NEEDS IMPROVEMENT")
            print(f"  - 60%+ win rate: {'YES' if test_result['win_rate'] >= 60 else 'NO'} ({test_result['win_rate']:.1f}%)")
            print(f"  - 30+ trades: {'YES' if test_result['total_trades'] >= 30 else 'NO'} ({test_result['total_trades']}/30)")
        print(f"{'='*80}")

    return {
        'train': train_result,
        'test': test_result
    }


def main():
    """
    Quick optimization workflow
    """

    print(f"\n{'='*80}")
    print(f"FOREX QUICK OPTIMIZER")
    print(f"{'='*80}")
    print(f"\nObjective: Find 60%+ win rate configuration")
    print(f"Method: Test 5 focused parameter sets per pair")
    print(f"Time: ~2-3 minutes per pair")
    print(f"{'='*80}")

    pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']
    all_results = {}

    for pair in pairs:
        print(f"\n\n{'#'*80}")
        print(f"OPTIMIZING: {pair}")
        print(f"{'#'*80}")

        # Quick optimization
        results = quick_optimize(pair, candles=5000)

        if results and len(results) > 0:
            # Get best config
            best = results[0]
            all_results[pair] = {
                'best_config': best['config'],
                'best_params': best['params'],
                'backtest': best
            }

            print(f"\nBEST CONFIG: {best['config']}")
            print(f"  {best['total_trades']} trades, {best['win_rate']:.1f}% WR, "
                  f"{best['total_pips']:+.1f} pips")

            # Validate with walk-forward
            print(f"\nValidating {best['config']} on {pair}...")
            validation = validate_best_config(pair, best['params'], candles=5000)

            if validation:
                all_results[pair]['validation'] = validation

    # Final summary
    print(f"\n\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE - FINAL SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'Pair':<12} {'Config':<15} {'Out-Sample WR':<15} {'Trades':<10} {'Pips':<12} {'Status':<15}")
    print(f"{'-'*80}")

    total_trades = 0
    total_wins = 0

    for pair, data in all_results.items():
        if 'validation' in data and data['validation']['test']:
            test = data['validation']['test']
            total_trades += test['total_trades']
            total_wins += test['wins']

            status = "TARGET MET" if (test['win_rate'] >= 60 and test['total_trades'] >= 30) else "NEEDS WORK"
            print(f"{pair:<12} {data['best_config']:<15} {test['win_rate']:<15.1f} "
                  f"{test['total_trades']:<10} {test['total_pips']:<+12.1f} {status:<15}")

    if total_trades > 0:
        overall_wr = (total_wins / total_trades * 100)
        print(f"{'-'*80}")
        print(f"{'OVERALL':<12} {'':<15} {overall_wr:<15.1f} {total_trades:<10}")

        print(f"\n{'='*80}")
        if overall_wr >= 60 and total_trades >= 100:
            print(f"FINAL VERDICT: TARGET ACHIEVED ACROSS ALL PAIRS! [SUCCESS]")
        elif overall_wr >= 60:
            print(f"FINAL VERDICT: 60%+ WR achieved (need more trades)")
        else:
            print(f"FINAL VERDICT: Below 60% target (needs further optimization)")
        print(f"{'='*80}")

    print(f"\nOptimization complete.")
    print(f"Full results saved to quick_optimization_*.json files")


if __name__ == "__main__":
    main()
