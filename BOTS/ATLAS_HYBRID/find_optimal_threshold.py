"""
Analyze ATLAS trade logs to find optimal score threshold
Uses actual trade outcomes from session logs
"""
import json
import os
from typing import List, Dict, Tuple
from collections import defaultdict

def load_session_trades() -> List[Dict]:
    """Load all trades from session log files"""
    trades = []
    log_dir = "logs/trades"

    if not os.path.exists(log_dir):
        print(f"Error: {log_dir} not found")
        return []

    for filename in sorted(os.listdir(log_dir)):
        if not filename.startswith("session_"):
            continue

        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    trades.extend(data)
                    print(f"Loaded {len(data)} trades from {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    return trades

def calculate_trade_outcome(trade: Dict) -> Tuple[str, float]:
    """
    Determine if trade was win/loss and P/L amount
    Returns: (status, pnl) where status is 'win', 'loss', 'open', or 'unknown'
    """
    status = trade.get('status', 'unknown')
    pnl = trade.get('pnl')

    if status == 'closed' and pnl is not None:
        return ('win' if pnl > 0 else 'loss', pnl)
    elif status == 'open':
        return ('open', 0)
    elif status == 'failed':
        return ('failed', 0)
    else:
        return ('unknown', 0)

def analyze_by_threshold(trades: List[Dict], threshold: float) -> Dict:
    """Simulate performance with specific threshold"""

    # Filter trades that would qualify
    qualified = [t for t in trades if t.get('atlas_score', 0) >= threshold]

    # Separate by outcome
    wins = []
    losses = []
    open_trades = []
    failed = []

    for trade in qualified:
        outcome, pnl = calculate_trade_outcome(trade)
        if outcome == 'win':
            wins.append((trade, pnl))
        elif outcome == 'loss':
            losses.append((trade, pnl))
        elif outcome == 'open':
            open_trades.append(trade)
        elif outcome == 'failed':
            failed.append(trade)

    # Calculate statistics
    total_profit = sum(pnl for _, pnl in wins)
    total_loss = sum(pnl for _, pnl in losses)
    net_pnl = total_profit + total_loss

    closed = len(wins) + len(losses)
    win_rate = (len(wins) / closed * 100) if closed > 0 else 0

    avg_win = total_profit / len(wins) if wins else 0
    avg_loss = total_loss / len(losses) if losses else 0

    profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')

    return {
        'threshold': threshold,
        'qualified_trades': len(qualified),
        'wins': len(wins),
        'losses': len(losses),
        'open': len(open_trades),
        'failed': len(failed),
        'win_rate': win_rate,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'net_pnl': net_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'expectancy': (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
    }

def analyze_score_distribution(trades: List[Dict]):
    """Show distribution of scores"""
    scores = [t.get('atlas_score', 0) for t in trades if t.get('atlas_score')]

    if not scores:
        print("No scores found in trades")
        return

    print(f"\n=== SCORE DISTRIBUTION ===")
    print(f"Min Score: {min(scores):.2f}")
    print(f"Max Score: {max(scores):.2f}")
    print(f"Average Score: {sum(scores)/len(scores):.2f}")
    print(f"Median Score: {sorted(scores)[len(scores)//2]:.2f}")

    # Show distribution by range
    ranges = [(0, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 4), (4, 5), (5, 10)]
    print(f"\nScore Range Distribution:")
    for low, high in ranges:
        count = len([s for s in scores if low <= s < high])
        pct = count / len(scores) * 100
        print(f"  {low:.1f} - {high:.1f}: {count:4d} trades ({pct:5.1f}%)")

def main():
    print("=" * 80)
    print("ATLAS OPTIMAL THRESHOLD ANALYSIS")
    print("=" * 80)

    # Load all trades
    trades = load_session_trades()
    print(f"\nTotal trades loaded: {len(trades)}")

    if not trades:
        print("No trade data available!")
        return

    # Show score distribution
    analyze_score_distribution(trades)

    # Analyze different thresholds
    print("\n" + "=" * 80)
    print("THRESHOLD PERFORMANCE ANALYSIS")
    print("=" * 80)

    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    results = []

    print(f"\n{'Thresh':<7} {'Trades':<7} {'Wins':<6} {'Loss':<6} {'WR%':<7} {'Net P/L':<12} {'Avg Win':<10} {'Avg Loss':<10} {'PF':<6} {'Expect':<8}")
    print("-" * 95)

    for threshold in thresholds:
        result = analyze_by_threshold(trades, threshold)
        results.append(result)

        print(f"{result['threshold']:<7.1f} "
              f"{result['qualified_trades']:<7} "
              f"{result['wins']:<6} "
              f"{result['losses']:<6} "
              f"{result['win_rate']:<7.1f} "
              f"${result['net_pnl']:<11,.0f} "
              f"${result['avg_win']:<9,.0f} "
              f"${result['avg_loss']:<9,.0f} "
              f"{result['profit_factor']:<6.2f} "
              f"${result['expectancy']:<7,.0f}")

    # Find best thresholds
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Best by net P/L
    best_pnl = max(results, key=lambda x: x['net_pnl'])
    print(f"\n[BEST NET P/L] Threshold {best_pnl['threshold']}")
    print(f"   - Net P/L: ${best_pnl['net_pnl']:,.2f}")
    print(f"   - Win Rate: {best_pnl['win_rate']:.1f}%")
    print(f"   - Trades: {best_pnl['qualified_trades']}")

    # Best by win rate (min 20 trades)
    valid_results = [r for r in results if r['qualified_trades'] >= 20]
    if valid_results:
        best_wr = max(valid_results, key=lambda x: x['win_rate'])
        print(f"\n[BEST WIN RATE] (20+ trades): Threshold {best_wr['threshold']}")
        print(f"   - Win Rate: {best_wr['win_rate']:.1f}%")
        print(f"   - Net P/L: ${best_wr['net_pnl']:,.2f}")
        print(f"   - Trades: {best_wr['qualified_trades']}")

    # Best by profit factor
    pf_results = [r for r in results if r['profit_factor'] != float('inf') and r['qualified_trades'] >= 20]
    if pf_results:
        best_pf = max(pf_results, key=lambda x: x['profit_factor'])
        print(f"\n[BEST PROFIT FACTOR] (20+ trades): Threshold {best_pf['threshold']}")
        print(f"   - Profit Factor: {best_pf['profit_factor']:.2f}")
        print(f"   - Net P/L: ${best_pf['net_pnl']:,.2f}")
        print(f"   - Win Rate: {best_pf['win_rate']:.1f}%")

    # Current threshold (1.5) analysis
    current = [r for r in results if r['threshold'] == 1.5][0]
    print(f"\n[CURRENT THRESHOLD 1.5] PERFORMANCE:")
    print(f"   - Qualified Trades: {current['qualified_trades']}")
    print(f"   - Win Rate: {current['win_rate']:.1f}%")
    print(f"   - Net P/L: ${current['net_pnl']:,.2f}")
    print(f"   - Expectancy: ${current['expectancy']:,.2f} per trade")

    if current['net_pnl'] < 0:
        print(f"   WARNING: Negative expectancy - losing money!")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
