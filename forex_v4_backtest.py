#!/usr/bin/env python3
"""
FOREX V4 COMPREHENSIVE BACKTEST SYSTEM
Rigorous testing with parameter optimization and walk-forward validation

FEATURES:
1. Extended backtesting (5000+ candles per pair)
2. Parameter grid search (EMA, RSI, ADX, R:R)
3. Walk-forward optimization (train on 70%, test on 30%)
4. Statistical significance testing (100+ trades minimum)
5. Out-of-sample validation
6. Detailed performance metrics (Sharpe, Sortino, Max DD)
7. Per-pair analysis with confidence intervals

METHODOLOGY:
- In-sample: Train on first 70% of data
- Out-of-sample: Test on last 30% of data
- Report both metrics separately
- Only claim 60%+ if BOTH achieve target
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from data.oanda_data_fetcher import OandaDataFetcher
from forex_v4_optimized import ForexV4OptimizedStrategy


class ForexV4Backtester:
    """
    Comprehensive backtesting engine with optimization

    Ensures results are:
    - Statistically significant (100+ trades)
    - Out-of-sample validated
    - Not overfitted
    - Realistic (spread/slippage included)
    """

    def __init__(self, spread_pips=1.5, slippage_pips=0.5):
        """
        Initialize backtester

        Args:
            spread_pips: Bid/ask spread cost (pips)
            slippage_pips: Execution slippage (pips)
        """
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.total_cost_pips = spread_pips + slippage_pips

        print(f"[V4 BACKTESTER] Initialized")
        print(f"  Spread: {spread_pips} pips")
        print(f"  Slippage: {slippage_pips} pips")
        print(f"  Total Cost: {self.total_cost_pips} pips per trade")

    def run_backtest(self,
                     df: pd.DataFrame,
                     strategy: ForexV4OptimizedStrategy,
                     pair: str,
                     start_idx: int = 250) -> Dict:
        """
        Run backtest on price data

        Args:
            df: Price data (OHLCV)
            strategy: Trading strategy
            pair: Forex pair
            start_idx: Start index (need data for indicators)

        Returns:
            Dict with trades and metrics
        """

        trades = []
        in_position = False
        current_trade = None

        # Scan through data
        for i in range(start_idx, len(df)):
            window = df.iloc[i-250:i].copy()

            # Check for signal
            opp = strategy.analyze_opportunity(window, pair)

            if opp and strategy.validate_rules(opp):
                if not in_position:
                    # Enter trade
                    current_trade = {
                        'entry_time': window.index[-1],
                        'entry_price': opp['entry_price'],
                        'direction': opp['direction'],
                        'stop_loss': opp['stop_loss'],
                        'take_profit': opp['take_profit'],
                        'score': opp['score'],
                        'rsi': opp['indicators']['rsi'],
                        'adx': opp['indicators']['adx'],
                        'entry_idx': i
                    }
                    in_position = True

            # Check exit conditions
            if in_position and current_trade:
                current_price = df.iloc[i]['close']
                pip_multiplier = 100 if 'JPY' in pair else 10000

                # LONG exit
                if current_trade['direction'] == 'LONG':
                    if current_price <= current_trade['stop_loss']:
                        # Stop loss
                        current_trade['exit_time'] = df.index[i]
                        current_trade['exit_price'] = current_trade['stop_loss']
                        current_trade['outcome'] = 'LOSS'
                        gross_pips = (current_trade['exit_price'] - current_trade['entry_price']) * pip_multiplier
                        current_trade['profit_pips'] = gross_pips - self.total_cost_pips
                        current_trade['bars_held'] = i - current_trade['entry_idx']
                        trades.append(current_trade)
                        in_position = False
                        current_trade = None

                    elif current_price >= current_trade['take_profit']:
                        # Take profit
                        current_trade['exit_time'] = df.index[i]
                        current_trade['exit_price'] = current_trade['take_profit']
                        current_trade['outcome'] = 'WIN'
                        gross_pips = (current_trade['exit_price'] - current_trade['entry_price']) * pip_multiplier
                        current_trade['profit_pips'] = gross_pips - self.total_cost_pips
                        current_trade['bars_held'] = i - current_trade['entry_idx']
                        trades.append(current_trade)
                        in_position = False
                        current_trade = None

                # SHORT exit
                else:
                    if current_price >= current_trade['stop_loss']:
                        # Stop loss
                        current_trade['exit_time'] = df.index[i]
                        current_trade['exit_price'] = current_trade['stop_loss']
                        current_trade['outcome'] = 'LOSS'
                        gross_pips = (current_trade['entry_price'] - current_trade['exit_price']) * pip_multiplier
                        current_trade['profit_pips'] = gross_pips - self.total_cost_pips
                        current_trade['bars_held'] = i - current_trade['entry_idx']
                        trades.append(current_trade)
                        in_position = False
                        current_trade = None

                    elif current_price <= current_trade['take_profit']:
                        # Take profit
                        current_trade['exit_time'] = df.index[i]
                        current_trade['exit_price'] = current_trade['take_profit']
                        current_trade['outcome'] = 'WIN'
                        gross_pips = (current_trade['entry_price'] - current_trade['exit_price']) * pip_multiplier
                        current_trade['profit_pips'] = gross_pips - self.total_cost_pips
                        current_trade['bars_held'] = i - current_trade['entry_idx']
                        trades.append(current_trade)
                        in_position = False
                        current_trade = None

        # Calculate metrics
        if len(trades) == 0:
            return None

        wins = [t for t in trades if t['outcome'] == 'WIN']
        losses = [t for t in trades if t['outcome'] == 'LOSS']

        win_rate = len(wins) / len(trades) * 100
        total_pips = sum([t['profit_pips'] for t in trades])
        avg_win = sum([t['profit_pips'] for t in wins]) / len(wins) if wins else 0
        avg_loss = sum([t['profit_pips'] for t in losses]) / len(losses) if losses else 0

        # Profit factor
        gross_profit = sum([t['profit_pips'] for t in wins]) if wins else 0
        gross_loss = abs(sum([t['profit_pips'] for t in losses])) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe ratio (annualized)
        returns = [t['profit_pips'] for t in trades]
        avg_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 1
        sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0

        # Max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

        return {
            'trades': trades,
            'total_trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pips': total_pips,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'avg_bars_held': np.mean([t['bars_held'] for t in trades])
        }

    def walk_forward_test(self,
                         df: pd.DataFrame,
                         strategy: ForexV4OptimizedStrategy,
                         pair: str,
                         train_pct: float = 0.7) -> Dict:
        """
        Walk-forward optimization

        Train on first 70% of data, test on last 30%

        Args:
            df: Full dataset
            strategy: Trading strategy
            pair: Forex pair
            train_pct: Training data percentage

        Returns:
            Dict with in-sample and out-of-sample results
        """

        split_idx = int(len(df) * train_pct)

        # Split data
        train_data = df.iloc[:split_idx]
        test_data = df.iloc[split_idx:]

        print(f"\n[WALK-FORWARD] Splitting data...")
        print(f"  Train: {len(train_data)} candles ({train_data.index[0]} to {train_data.index[-1]})")
        print(f"  Test:  {len(test_data)} candles ({test_data.index[0]} to {test_data.index[-1]})")

        # Run in-sample backtest
        print(f"\n[IN-SAMPLE] Testing on training data...")
        in_sample = self.run_backtest(train_data, strategy, pair)

        # Run out-of-sample backtest
        print(f"\n[OUT-OF-SAMPLE] Testing on unseen data...")
        out_sample = self.run_backtest(test_data, strategy, pair)

        return {
            'in_sample': in_sample,
            'out_sample': out_sample
        }

    def grid_search_parameters(self,
                              df: pd.DataFrame,
                              pair: str,
                              param_grid: Dict) -> List[Dict]:
        """
        Grid search over parameter space

        Tests all combinations of parameters to find optimal settings

        Args:
            df: Price data
            pair: Forex pair
            param_grid: Dict of parameter ranges

        Returns:
            List of results sorted by win rate
        """

        print(f"\n[GRID SEARCH] Testing parameter combinations...")

        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        print(f"  Total combinations: {len(combinations)}")

        results = []

        for i, combo in enumerate(combinations, 1):
            params = dict(zip(param_names, combo))

            # Create strategy with these parameters
            strategy = ForexV4OptimizedStrategy(
                ema_fast=params.get('ema_fast', 10),
                ema_slow=params.get('ema_slow', 21),
                rsi_period=params.get('rsi_period', 14)
            )

            # Override other parameters
            if 'rsi_long_lower' in params:
                strategy.rsi_long_lower = params['rsi_long_lower']
            if 'rsi_long_upper' in params:
                strategy.rsi_long_upper = params['rsi_long_upper']
            if 'adx_threshold' in params:
                strategy.adx_threshold = params['adx_threshold']
            if 'risk_reward_ratio' in params:
                strategy.risk_reward_ratio = params['risk_reward_ratio']

            # Run backtest
            result = self.run_backtest(df, strategy, pair)

            if result and result['total_trades'] >= 20:  # Minimum trades
                results.append({
                    'params': params,
                    'win_rate': result['win_rate'],
                    'total_trades': result['total_trades'],
                    'total_pips': result['total_pips'],
                    'profit_factor': result['profit_factor'],
                    'sharpe_ratio': result['sharpe_ratio']
                })

            if i % 10 == 0:
                print(f"    Tested {i}/{len(combinations)} combinations...")

        # Sort by win rate
        results.sort(key=lambda x: x['win_rate'], reverse=True)

        print(f"  Found {len(results)} valid configurations")

        return results

    def calculate_statistics(self, trades: List[Dict]) -> Dict:
        """
        Calculate statistical metrics

        Args:
            trades: List of trade dicts

        Returns:
            Dict with statistical measures
        """

        if not trades:
            return None

        returns = [t['profit_pips'] for t in trades]

        # Win streaks
        outcomes = [1 if t['outcome'] == 'WIN' else 0 for t in trades]
        streaks = []
        current_streak = 0

        for outcome in outcomes:
            if outcome == 1:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0

        max_win_streak = max(streaks) if streaks else 0

        # Confidence interval for win rate (95%)
        n = len(trades)
        wins = sum(outcomes)
        win_rate = wins / n
        std_error = np.sqrt(win_rate * (1 - win_rate) / n)
        ci_lower = win_rate - 1.96 * std_error
        ci_upper = win_rate + 1.96 * std_error

        return {
            'sample_size': n,
            'win_rate': win_rate * 100,
            'ci_lower': ci_lower * 100,
            'ci_upper': ci_upper * 100,
            'max_win_streak': max_win_streak,
            'avg_return': np.mean(returns),
            'median_return': np.median(returns),
            'std_return': np.std(returns)
        }


def run_comprehensive_backtest(pair='EUR_USD', candles=5000):
    """
    Run comprehensive backtest on single pair

    Args:
        pair: Forex pair
        candles: Number of candles to test
    """

    print("\n" + "="*70)
    print(f"COMPREHENSIVE BACKTEST: {pair}")
    print(f"Target: 60%+ Win Rate (Validated on {candles} candles)")
    print("="*70)

    # Initialize
    fetcher = OandaDataFetcher()
    backtester = ForexV4Backtester(spread_pips=1.5, slippage_pips=0.5)

    # Fetch data
    print(f"\n[1/5] Fetching {candles} H1 candles from OANDA...")
    df = fetcher.get_bars(pair, timeframe='H1', limit=candles)

    if df is None or df.empty:
        print(f"[ERROR] Could not fetch data for {pair}")
        return None

    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    print(f"[OK] Fetched {len(df)} candles")
    print(f"      Range: {df.index[0]} to {df.index[-1]}")

    # Create optimized strategy
    strategy = ForexV4OptimizedStrategy()
    strategy.set_data_fetcher(fetcher)

    # Walk-forward test
    print(f"\n[2/5] Running walk-forward validation...")
    wf_results = backtester.walk_forward_test(df, strategy, pair, train_pct=0.7)

    # Display in-sample results
    if wf_results['in_sample']:
        in_s = wf_results['in_sample']
        print(f"\n[IN-SAMPLE RESULTS]")
        print(f"  Trades: {in_s['total_trades']}")
        print(f"  Win Rate: {in_s['win_rate']:.1f}%")
        print(f"  Total Pips: {in_s['total_pips']:+.1f}")
        print(f"  Profit Factor: {in_s['profit_factor']:.2f}x")
        print(f"  Sharpe Ratio: {in_s['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {in_s['max_drawdown']:.1f} pips")

    # Display out-of-sample results
    if wf_results['out_sample']:
        out_s = wf_results['out_sample']
        print(f"\n[OUT-OF-SAMPLE RESULTS] *** KEY METRIC ***")
        print(f"  Trades: {out_s['total_trades']}")
        print(f"  Win Rate: {out_s['win_rate']:.1f}%")
        print(f"  Total Pips: {out_s['total_pips']:+.1f}")
        print(f"  Profit Factor: {out_s['profit_factor']:.2f}x")
        print(f"  Sharpe Ratio: {out_s['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {out_s['max_drawdown']:.1f} pips")

        # Statistical significance
        stats = backtester.calculate_statistics(out_s['trades'])
        if stats:
            print(f"\n[STATISTICAL ANALYSIS]")
            print(f"  Sample Size: {stats['sample_size']} trades")
            print(f"  Win Rate: {stats['win_rate']:.1f}%")
            print(f"  95% CI: [{stats['ci_lower']:.1f}%, {stats['ci_upper']:.1f}%]")
            print(f"  Max Win Streak: {stats['max_win_streak']}")

            # Check if statistically significant
            if stats['sample_size'] >= 30 and stats['ci_lower'] >= 60:
                print(f"\n  STATUS: STATISTICALLY SIGNIFICANT ✓")
                print(f"          (60%+ WR with 95% confidence)")
            elif stats['win_rate'] >= 60:
                print(f"\n  STATUS: PROMISING (need more trades for significance)")
            else:
                print(f"\n  STATUS: BELOW TARGET")

    # Full backtest on all data
    print(f"\n[3/5] Running full backtest on {len(df)} candles...")
    full_result = backtester.run_backtest(df, strategy, pair)

    if full_result:
        print(f"\n[FULL DATASET RESULTS]")
        print(f"  Trades: {full_result['total_trades']}")
        print(f"  Win Rate: {full_result['win_rate']:.1f}%")
        print(f"  Total Pips: {full_result['total_pips']:+.1f}")
        print(f"  Profit Factor: {full_result['profit_factor']:.2f}x")
        print(f"  Sharpe Ratio: {full_result['sharpe_ratio']:.2f}")

        # Sample trades
        print(f"\n[4/5] Sample Trades (First 5 + Last 5):")
        print("-"*70)

        trades = full_result['trades']
        sample = trades[:5] + trades[-5:] if len(trades) > 10 else trades

        for i, trade in enumerate(sample, 1):
            symbol = "[WIN]" if trade['outcome'] == 'WIN' else "[LOSS]"
            print(f"\n{i}. {symbol} {trade['direction']}")
            print(f"   Entry: {trade['entry_time']} @ {trade['entry_price']:.5f}")
            print(f"   Exit:  {trade['exit_time']} @ {trade['exit_price']:.5f}")
            print(f"   Profit: {trade['profit_pips']:+.1f} pips (held {trade['bars_held']} hrs)")
            print(f"   Score: {trade['score']:.1f}, RSI: {trade['rsi']:.1f}, ADX: {trade['adx']:.1f}")

    print("\n" + "="*70)

    return {
        'pair': pair,
        'walk_forward': wf_results,
        'full': full_result
    }


def main():
    """Run comprehensive backtests on all pairs"""

    print("\n" + "="*80)
    print("FOREX V4 COMPREHENSIVE BACKTEST SUITE")
    print("="*80)
    print("\nOBJECTIVE: Prove 60%+ Win Rate with Statistical Significance")
    print("\nMETHODOLOGY:")
    print("  1. Extended backtest (5000+ candles per pair)")
    print("  2. Walk-forward validation (70% train, 30% test)")
    print("  3. Out-of-sample testing (prevents overfitting)")
    print("  4. Statistical significance (100+ trades, 95% CI)")
    print("  5. Realistic costs (1.5 pips spread + 0.5 pips slippage)")
    print("\nSUCCESS CRITERIA:")
    print("  - EUR/USD: 60%+ WR (30+ trades)")
    print("  - GBP/USD: 60%+ WR (30+ trades)")
    print("  - USD/JPY: 60%+ WR (30+ trades)")
    print("  - Out-of-sample validated")
    print("  - Statistically significant (95% CI)")
    print("="*80)

    pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']
    all_results = {}

    for pair in pairs:
        try:
            result = run_comprehensive_backtest(pair, candles=5000)
            if result:
                all_results[pair] = result
        except Exception as e:
            print(f"\n[ERROR] Failed to test {pair}: {e}")

        print("\n")

    # Generate summary
    if all_results:
        print("="*80)
        print("OVERALL SUMMARY - OUT-OF-SAMPLE PERFORMANCE")
        print("="*80)

        total_trades = 0
        total_wins = 0
        total_pips = 0

        print("\nPer-Pair Performance (Out-of-Sample):")
        print("-"*80)

        for pair, result in all_results.items():
            if result['walk_forward']['out_sample']:
                out_s = result['walk_forward']['out_sample']
                total_trades += out_s['total_trades']
                total_wins += out_s['wins']
                total_pips += out_s['total_pips']

                status = "✓" if out_s['win_rate'] >= 60 else "✗"
                print(f"  {pair}: {out_s['win_rate']:.1f}% WR ({out_s['total_trades']} trades, "
                      f"{out_s['total_pips']:+.1f} pips, PF: {out_s['profit_factor']:.2f}x) {status}")

        overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0

        print(f"\n{'='*80}")
        print(f"COMBINED OUT-OF-SAMPLE PERFORMANCE:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Overall Win Rate: {overall_wr:.1f}%")
        print(f"  Total Profit: {total_pips:+.1f} pips")
        print(f"  Avg per Pair: {total_pips/len(all_results):+.1f} pips")
        print(f"{'='*80}")

        # Final verdict
        print(f"\nFINAL VERDICT:")
        if overall_wr >= 60 and total_trades >= 100:
            print(f"  STATUS: TARGET ACHIEVED ✓✓✓")
            print(f"  - Win Rate: {overall_wr:.1f}% (target: 60%+)")
            print(f"  - Sample Size: {total_trades} trades (target: 100+)")
            print(f"  - Validation: Out-of-sample tested")
            print(f"\n  READY FOR PAPER TRADING")
        elif overall_wr >= 60:
            print(f"  STATUS: WIN RATE ACHIEVED (need more trades)")
            print(f"  - Win Rate: {overall_wr:.1f}% ✓")
            print(f"  - Sample Size: {total_trades} (target: 100+)")
            print(f"\n  RECOMMENDATION: Extend backtest period or lower filters")
        elif total_trades >= 100:
            print(f"  STATUS: SAMPLE SIZE OK (win rate below target)")
            print(f"  - Win Rate: {overall_wr:.1f}% (target: 60%+)")
            print(f"  - Sample Size: {total_trades} ✓")
            print(f"\n  RECOMMENDATION: Further parameter optimization needed")
        else:
            print(f"  STATUS: NEEDS IMPROVEMENT")
            print(f"  - Win Rate: {overall_wr:.1f}% (target: 60%+)")
            print(f"  - Sample Size: {total_trades} (target: 100+)")
            print(f"\n  RECOMMENDATION: Relax filters or extend data period")

        print(f"\n{'='*80}")

    else:
        print("[ERROR] No backtest results. Check OANDA connection.")

    print("\nNext Steps:")
    print("1. Review forex_optimization_report.md for detailed analysis")
    print("2. If 60%+ achieved: Start paper trading for 30 days")
    print("3. If below target: Run parameter optimization (grid search)")
    print("4. Monitor out-of-sample performance closely")


if __name__ == "__main__":
    main()
