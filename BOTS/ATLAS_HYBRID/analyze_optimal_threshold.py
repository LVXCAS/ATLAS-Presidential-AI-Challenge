"""
Analyze historical trades to find optimal score threshold
"""
import json
import os
from typing import List, Dict
import statistics

def load_all_trades() -> List[Dict]:
    """Load all trades from session logs"""
    trades = []
    log_dir = "logs/trades"

    if not os.path.exists(log_dir):
        return []

    for filename in os.listdir(log_dir):
        if not filename.startswith("session_"):
            continue

        filepath = os.path.join(log_dir, filename)
        try:
            # Handle both JSON array and newline-delimited JSON
            with open(filepath, 'r') as f:
                content = f.read().strip()

            # Try parsing as JSON array
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    trades.extend(data)
                elif isinstance(data, dict):
                    trades.append(data)
            except json.JSONDecodeError:
                # Try newline-delimited JSON
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        try:
                            trades.append(json.loads(line))
                        except:
                            pass
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return trades

def analyze_by_threshold(trades: List[Dict], threshold: float):
    """Simulate what would happen with a specific threshold"""
    qualified = [t for t in trades if t.get('atlas_score', 0) >= threshold]

    wins = [t for t in qualified if t.get('status') == 'closed' and t.get('profit', 0) > 0]
    losses = [t for t in qualified if t.get('status') == 'closed' and t.get('profit', 0) < 0]

    total_profit = sum(t.get('profit', 0) for t in wins)
    total_loss = sum(t.get('profit', 0) for t in losses)
    net_profit = total_profit + total_loss

    win_rate = len(wins) / (len(wins) + len(losses)) * 100 if (wins or losses) else 0

    return {
        'threshold': threshold,
        'qualified_trades': len(qualified),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'net_profit': net_profit,
        'avg_win': total_profit / len(wins) if wins else 0,
        'avg_loss': total_loss / len(losses) if losses else 0,
    }

def main():
    print("=" * 80)
    print("ATLAS OPTIMAL THRESHOLD ANALYSIS")
    print("=" * 80)

    trades = load_all_trades()
    print(f"\nLoaded {len(trades)} total trades")

    # Filter to closed trades only
    closed_trades = [t for t in trades if t.get('status') == 'closed']
    print(f"Closed trades: {len(closed_trades)}")

    if not closed_trades:
        print("\nNo closed trades found - need more data!")
        return

    # Get score distribution
    scores = [t.get('atlas_score', 0) for t in closed_trades]
    print(f"\nScore range: {min(scores):.2f} - {max(scores):.2f}")
    print(f"Average score: {statistics.mean(scores):.2f}")
    print(f"Median score: {statistics.median(scores):.2f}")

    # Test different thresholds
    print("\n" + "=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)
    print(f"{'Threshold':<10} {'Trades':<8} {'Wins':<6} {'Losses':<8} {'WinRate':<10} {'Net P/L':<12} {'Avg Win':<10} {'Avg Loss':<10}")
    print("-" * 80)

    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    results = []

    for threshold in thresholds:
        result = analyze_by_threshold(closed_trades, threshold)
        results.append(result)

        print(f"{result['threshold']:<10.1f} "
              f"{result['qualified_trades']:<8} "
              f"{result['wins']:<6} "
              f"{result['losses']:<8} "
              f"{result['win_rate']:<10.1f}% "
              f"${result['net_profit']:<11,.2f} "
              f"${result['avg_win']:<9,.2f} "
              f"${result['avg_loss']:<9,.2f}")

    # Find optimal threshold (max net profit)
    if results:
        best = max(results, key=lambda x: x['net_profit'])
        print("\n" + "=" * 80)
        print(f"RECOMMENDED THRESHOLD: {best['threshold']}")
        print(f"  - Net Profit: ${best['net_profit']:,.2f}")
        print(f"  - Win Rate: {best['win_rate']:.1f}%")
        print(f"  - Trades: {best['qualified_trades']}")
        print("=" * 80)

if __name__ == "__main__":
    main()
