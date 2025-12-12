"""
Analyze Threshold Performance from Real Trade Data

After 24-48 hours of live trading, this script analyzes actual trade outcomes
to determine the optimal ATLAS score threshold.
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def load_all_trades():
    """Load all trades from session logs"""
    trades_dir = Path(__file__).parent / 'logs' / 'trades'
    all_trades = []

    for log_file in sorted(trades_dir.glob('session_*.json')):
        try:
            with open(log_file, 'r') as f:
                content = f.read().strip()
                if content.startswith('['):
                    # JSON array format
                    trades = json.loads(content)
                else:
                    # Line-delimited JSON
                    trades = []
                    for line in f:
                        if line.strip():
                            try:
                                trades.append(json.loads(line))
                            except:
                                pass

                all_trades.extend(trades)
        except Exception as e:
            print(f"Error loading {log_file.name}: {e}")

    return all_trades

def analyze_by_threshold(trades, threshold):
    """
    Analyze performance of trades that would have executed at given threshold.

    Args:
        trades: List of trade records
        threshold: Minimum score to include

    Returns:
        Dictionary with performance metrics
    """
    # Filter trades that meet threshold
    qualifying_trades = [t for t in trades if t.get('atlas_score', 0) >= threshold]

    if not qualifying_trades:
        return None

    # Separate wins/losses
    with_pnl = [t for t in qualifying_trades if t.get('pnl') is not None]

    if not with_pnl:
        return {
            'threshold': threshold,
            'total_trades': len(qualifying_trades),
            'closed_trades': 0,
            'note': 'No closed trades with P/L data yet'
        }

    wins = [t for t in with_pnl if t['pnl'] > 0]
    losses = [t for t in with_pnl if t['pnl'] < 0]

    total_pnl = sum(t['pnl'] for t in with_pnl)
    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0

    win_rate = len(wins) / len(with_pnl) if with_pnl else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)

    avg_win = gross_profit / len(wins) if wins else 0
    avg_loss = gross_loss / len(losses) if losses else 0
    expectancy = total_pnl / len(with_pnl)

    return {
        'threshold': threshold,
        'total_trades': len(qualifying_trades),
        'closed_trades': len(with_pnl),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
        'avg_score': sum(t.get('atlas_score', 0) for t in qualifying_trades) / len(qualifying_trades)
    }

def main():
    """Run threshold performance analysis"""
    print("="*80)
    print("ATLAS THRESHOLD PERFORMANCE ANALYSIS")
    print("="*80)
    print()

    # Load all trades
    print("Loading trade history...")
    all_trades = load_all_trades()
    print(f"Total trades loaded: {len(all_trades)}")

    # Filter to trades with ATLAS scores
    scored_trades = [t for t in all_trades if t.get('atlas_score') and t.get('atlas_score') > 0]
    print(f"Trades with ATLAS scores: {len(scored_trades)}")

    # Check how many have P/L data
    trades_with_pnl = [t for t in scored_trades if t.get('pnl') is not None]
    print(f"Closed trades with P/L: {len(trades_with_pnl)}")
    print()

    if len(trades_with_pnl) < 10:
        print("WARNING: Need at least 10 closed trades for meaningful analysis")
        print(f"Current: {len(trades_with_pnl)} trades")
        print()
        print("Recommendation: Wait for more trades to close, then re-run this script")
        return

    # Analyze different thresholds
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    print("="*80)
    print("PERFORMANCE BY THRESHOLD")
    print("="*80)
    print()

    results = []
    for threshold in thresholds:
        result = analyze_by_threshold(scored_trades, threshold)
        if result and result.get('closed_trades', 0) >= 5:  # Need at least 5 trades
            results.append(result)

    if not results:
        print("No thresholds have enough trades for analysis")
        return

    # Print results table
    print(f"{'Thresh':<8} {'Trades':<8} {'Wins':<6} {'Losses':<8} {'WR%':<8} {'P/L':<12} {'PF':<8} {'Expect':<10} {'AvgScore':<10}")
    print("-"*80)

    for r in results:
        print(f"{r['threshold']:<8.1f} "
              f"{r['closed_trades']:<8} "
              f"{r['wins']:<6} "
              f"{r['losses']:<8} "
              f"{r['win_rate']*100:<8.1f} "
              f"${r['total_pnl']:<11.2f} "
              f"{r['profit_factor']:<8.2f} "
              f"${r['expectancy']:<9.2f} "
              f"{r['avg_score']:<10.2f}")

    print()

    # Find optimal threshold
    # Optimize for highest expectancy (average P/L per trade)
    best_by_expectancy = max(results, key=lambda x: x['expectancy'])

    # Also consider profit factor
    best_by_pf = max(results, key=lambda x: x['profit_factor'])

    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()

    print(f"Best Threshold by EXPECTANCY: {best_by_expectancy['threshold']}")
    print(f"  Expectancy: ${best_by_expectancy['expectancy']:.2f} per trade")
    print(f"  Win Rate: {best_by_expectancy['win_rate']*100:.1f}%")
    print(f"  Profit Factor: {best_by_expectancy['profit_factor']:.2f}")
    print(f"  Total P/L: ${best_by_expectancy['total_pnl']:.2f}")
    print()

    print(f"Best Threshold by PROFIT FACTOR: {best_by_pf['threshold']}")
    print(f"  Profit Factor: {best_by_pf['profit_factor']:.2f}")
    print(f"  Win Rate: {best_by_pf['win_rate']*100:.1f}%")
    print(f"  Expectancy: ${best_by_pf['expectancy']:.2f} per trade")
    print(f"  Total P/L: ${best_by_pf['total_pnl']:.2f}")
    print()

    print("="*80)
    print()

    # Current threshold
    current_threshold = 1.5
    current_result = analyze_by_threshold(scored_trades, current_threshold)

    if current_result and current_result.get('closed_trades', 0) >= 5:
        print(f"CURRENT THRESHOLD ({current_threshold}) PERFORMANCE:")
        print(f"  Trades: {current_result['closed_trades']}")
        print(f"  Win Rate: {current_result['win_rate']*100:.1f}%")
        print(f"  Total P/L: ${current_result['total_pnl']:.2f}")
        print(f"  Expectancy: ${current_result['expectancy']:.2f}")
        print(f"  Profit Factor: {current_result['profit_factor']:.2f}")
        print()

        if best_by_expectancy['threshold'] != current_threshold:
            improvement = best_by_expectancy['expectancy'] - current_result['expectancy']
            print(f"Switching to threshold {best_by_expectancy['threshold']} would improve expectancy by ${improvement:.2f} per trade")
        else:
            print("Current threshold is already optimal!")

if __name__ == "__main__":
    main()
