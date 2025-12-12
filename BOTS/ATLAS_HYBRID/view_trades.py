"""
Trade Performance Dashboard

View and analyze ATLAS trading performance from trade logs.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from core.trade_logger import TradeLogger


def print_separator(char="=", length=80):
    """Print separator line"""
    print(char * length)


def format_duration(minutes):
    """Format duration in minutes to human readable"""
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f}h"
    days = hours / 24
    return f"{days:.1f}d"


def view_today_trades():
    """View today's trades"""
    logger = TradeLogger()

    if not logger.daily_log_file.exists():
        print("\n[INFO] No trades logged today yet\n")
        return

    with open(logger.daily_log_file, 'r') as f:
        trades = json.load(f)

    print_separator()
    print(f"ATLAS TRADE LOG - {datetime.now().strftime('%Y-%m-%d')}")
    print_separator()
    print(f"\nTotal Decisions: {len(trades)}")

    # Separate by status
    pending = [t for t in trades if t.get('status') == 'pending']
    open_trades = [t for t in trades if t.get('status') == 'open']
    closed = [t for t in trades if t.get('status') == 'closed']
    failed = [t for t in trades if t.get('status') == 'failed']

    print(f"Pending: {len(pending)} | Open: {len(open_trades)} | Closed: {len(closed)} | Failed: {len(failed)}\n")

    # Show open trades
    if open_trades:
        print_separator("-")
        print("OPEN POSITIONS")
        print_separator("-")
        for trade in open_trades:
            print(f"\n{trade['trade_id']} | {trade['pair']} {trade['direction']}")
            print(f"  Entry: {trade['entry_price']:.5f} @ {trade['timestamp_entry'][:19]}")
            print(f"  Size: {trade['lots']:.1f} lots ({trade['units']:,} units)")
            print(f"  SL: {trade['stop_loss']:.5f} | TP: {trade['take_profit']:.5f}")
            print(f"  ATLAS Score: {trade['atlas_score']:.2f} (threshold: {trade['atlas_threshold']:.2f})")

    # Show closed trades
    if closed:
        print(f"\n")
        print_separator("-")
        print("CLOSED TRADES")
        print_separator("-")

        total_pnl = 0
        wins = 0
        losses = 0

        for trade in closed:
            pnl = trade.get('pnl', 0)
            total_pnl += pnl
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1

            duration = format_duration(trade.get('duration_minutes', 0))
            pips = trade.get('pips', 0)
            r_mult = trade.get('r_multiple', 0)

            status_icon = "✓" if pnl > 0 else "✗"
            print(f"\n{status_icon} {trade['trade_id']} | {trade['pair']} {trade['direction']}")
            print(f"  Entry: {trade['entry_price']:.5f} → Exit: {trade['exit_price']:.5f}")
            print(f"  P/L: ${pnl:+,.2f} ({pips:+.1f} pips, {r_mult:+.2f}R)")
            print(f"  Duration: {duration} | Exit: {trade['exit_reason']}")
            print(f"  Size: {trade['lots']:.1f} lots | Score: {trade['atlas_score']:.2f}")

        # Summary stats
        print(f"\n")
        print_separator("-")
        print("SUMMARY")
        print_separator("-")
        print(f"Total P/L: ${total_pnl:+,.2f}")
        print(f"Wins: {wins} | Losses: {losses}")
        if wins + losses > 0:
            win_rate = wins / (wins + losses) * 100
            print(f"Win Rate: {win_rate:.1f}%")
            print(f"Avg per Trade: ${total_pnl / (wins + losses):+,.2f}")

    # Show failed trades
    if failed:
        print(f"\n")
        print_separator("-")
        print("FAILED EXECUTIONS")
        print_separator("-")
        for trade in failed:
            print(f"\n✗ {trade['trade_id']} | {trade['pair']} {trade['direction']}")
            print(f"  Score: {trade['atlas_score']:.2f}")
            print(f"  Error: {trade['notes']}")

    print(f"\n")
    print_separator()


def view_performance_summary():
    """View overall performance summary"""
    logger = TradeLogger()
    summary = logger.get_performance_summary()

    print_separator()
    print("PERFORMANCE SUMMARY")
    print_separator()

    if 'error' in summary or 'message' in summary:
        print(f"\n{summary.get('error') or summary.get('message')}\n")
        print_separator()
        return

    print(f"\nTrades: {summary['total_trades']}")
    print(f"Wins: {summary['wins']} | Losses: {summary['losses']}")
    print(f"Win Rate: {summary['win_rate']:.1f}%")
    print(f"\nTotal P/L: ${summary['total_pnl']:+,.2f}")
    print(f"Avg Win: ${summary['avg_win']:+,.2f}")
    print(f"Avg Loss: ${summary['avg_loss']:+,.2f}")
    print(f"Expectancy: ${summary['expectancy']:+,.2f} per trade")
    print(f"\nProfit Factor: {summary['profit_factor']:.2f}")
    print(f"Largest Win: ${summary['largest_win']:+,.2f}")
    print(f"Largest Loss: ${summary['largest_loss']:+,.2f}")
    print(f"Avg Duration: {format_duration(summary['avg_duration_minutes'])}")

    print(f"\n")
    print_separator()


def view_agent_performance():
    """View which agents perform best"""
    logger = TradeLogger()

    if not logger.daily_log_file.exists():
        print("\n[INFO] No trades logged yet\n")
        return

    with open(logger.daily_log_file, 'r') as f:
        trades = json.load(f)

    closed = [t for t in trades if t.get('status') == 'closed']

    if not closed:
        print("\n[INFO] No closed trades to analyze\n")
        return

    # Track agent voting patterns
    agent_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0, 'voted_for': 0})

    for trade in closed:
        pnl = trade.get('pnl', 0)
        direction = trade['direction']
        agent_votes = trade.get('agent_votes', {})

        for agent_name, vote_data in agent_votes.items():
            vote = vote_data.get('vote')

            # Check if agent voted for this trade
            if (direction == 'BUY' and vote == 'BUY') or (direction == 'SELL' and vote == 'SELL'):
                agent_stats[agent_name]['voted_for'] += 1

                if pnl > 0:
                    agent_stats[agent_name]['wins'] += 1
                elif pnl < 0:
                    agent_stats[agent_name]['losses'] += 1

                agent_stats[agent_name]['total_pnl'] += pnl

    print_separator()
    print("AGENT PERFORMANCE ANALYSIS")
    print_separator()
    print(f"\nBased on {len(closed)} closed trades\n")

    # Sort by win rate
    sorted_agents = sorted(agent_stats.items(),
                          key=lambda x: x[1]['wins'] / max(x[1]['voted_for'], 1),
                          reverse=True)

    for agent_name, stats in sorted_agents:
        if stats['voted_for'] == 0:
            continue

        wins = stats['wins']
        losses = stats['losses']
        total = wins + losses
        win_rate = wins / total * 100 if total > 0 else 0
        pnl = stats['total_pnl']

        print(f"{agent_name:25s} | Votes: {stats['voted_for']:2d} | W/L: {wins:2d}/{losses:2d} | "
              f"WR: {win_rate:5.1f}% | P/L: ${pnl:+8.2f}")

    print(f"\n")
    print_separator()


def main():
    """Main menu"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == 'summary':
            view_performance_summary()
        elif command == 'agents':
            view_agent_performance()
        else:
            view_today_trades()
    else:
        # Default: show today's trades
        view_today_trades()


if __name__ == "__main__":
    main()
