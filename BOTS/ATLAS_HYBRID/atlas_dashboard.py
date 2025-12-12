"""
ATLAS Live Trading Dashboard
Real-time monitoring of ATLAS performance, agent votes, and trades

Usage:
    python atlas_dashboard.py         # Live auto-refresh view
    python atlas_dashboard.py --once  # Single snapshot
"""
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# Add parent to path
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv('../../.env')
os.environ['OANDA_API_KEY'] = os.getenv('OANDA_API_KEY')
os.environ['OANDA_ACCOUNT_ID'] = os.getenv('OANDA_ACCOUNT_ID')

from adapters.oanda_adapter import OandaAdapter


def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def load_coordinator_state():
    """Load coordinator state"""
    state_file = Path(__file__).parent / "learning" / "state" / "coordinator_state.json"
    if state_file.exists():
        with open(state_file, 'r') as f:
            return json.load(f)
    return {}


def load_recent_trades(hours=24):
    """Load recent trades from trade logger"""
    trades_dir = Path(__file__).parent / "logs" / "trades"
    if not trades_dir.exists():
        return []

    trades = []
    cutoff = datetime.now() - timedelta(hours=hours)

    # Check daily and session logs
    for log_file in sorted(trades_dir.glob("*.json"), reverse=True):
        if log_file.stem.startswith("summary"):
            continue
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                if isinstance(log_data, list):
                    for trade in log_data:
                        trade_time = datetime.fromisoformat(trade.get('timestamp_decision', '2000-01-01'))
                        if trade_time > cutoff:
                            trades.append(trade)
        except:
            pass

    return sorted(trades, key=lambda x: x.get('timestamp_decision', ''), reverse=True)


def format_score(score):
    """Format score with color indicators"""
    if score >= 2.0:
        return f"{score:+.2f} ████"
    elif score >= 1.5:
        return f"{score:+.2f} ███"
    elif score >= 1.0:
        return f"{score:+.2f} ██"
    elif score >= 0.5:
        return f"{score:+.2f} █"
    elif score <= -2.0:
        return f"{score:+.2f} ████"
    elif score <= -1.5:
        return f"{score:+.2f} ███"
    elif score <= -1.0:
        return f"{score:+.2f} ██"
    elif score <= -0.5:
        return f"{score:+.2f} █"
    else:
        return f"{score:+.2f}"


def display_dashboard(auto_refresh=True):
    """Display live dashboard"""

    try:
        while True:
            if auto_refresh:
                clear_screen()

            # Header
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("=" * 100)
            print(f"{'ATLAS LIVE TRADING DASHBOARD':^100}")
            print(f"{now:^100}")
            print("=" * 100)

            # Load data
            oanda = OandaAdapter()
            balance_data = oanda.get_account_balance()
            balance = balance_data.get('balance', 0) if isinstance(balance_data, dict) else balance_data
            positions = oanda.get_open_positions()
            state = load_coordinator_state()
            recent_trades = load_recent_trades(hours=24)

            # Account Summary
            print(f"\n[ACCOUNT STATUS]")
            print(f"  Balance:        ${balance:,.2f}")
            print(f"  Open Positions: {len(positions)}")

            unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in positions)
            print(f"  Unrealized P/L: ${unrealized_pnl:+,.2f}")

            # Open Positions
            if positions:
                print(f"\n[OPEN POSITIONS]")
                for pos in positions:
                    pair = pos.get('instrument', pos.get('symbol', 'UNKNOWN'))
                    units = pos['units']
                    lots = abs(units) / 100000
                    pnl = pos.get('unrealized_pnl', 0)
                    direction = 'LONG' if units > 0 else 'SHORT'
                    avg_price = pos.get('avg_price', 0)

                    pnl_str = f"${pnl:+,.2f}" if pnl != 0 else "$0.00"
                    print(f"  {pair:10} {direction:5} {lots:5.1f} lots @ {avg_price:.5f}  P/L: {pnl_str}")

            # Trading Statistics
            print(f"\n[TRADING STATISTICS]")
            total_decisions = state.get('total_decisions', 0)
            trades_executed = state.get('trades_executed', 0)
            exec_rate = (trades_executed / total_decisions * 100) if total_decisions > 0 else 0

            print(f"  Total Decisions:  {total_decisions:,}")
            print(f"  Trades Executed:  {trades_executed}")
            print(f"  Execution Rate:   {exec_rate:.1f}%")
            print(f"  Score Threshold:  {state.get('config', {}).get('score_threshold', 'N/A')}")
            print(f"  Training Phase:   {state.get('config', {}).get('training_phase', 'N/A').upper()}")

            # Recent Trades (Last 24h)
            if recent_trades:
                print(f"\n[RECENT TRADES - LAST 24 HOURS]")
                print(f"  {'Time':<10} {'Pair':<10} {'Dir':<5} {'Lots':<6} {'Score':<12} {'P/L':<12} {'Status':<10}")
                print(f"  {'-'*80}")

                total_pnl_24h = 0
                wins = 0
                losses = 0

                for trade in recent_trades[:10]:  # Show last 10
                    timestamp = trade.get('timestamp_decision', '')
                    if timestamp:
                        time_str = datetime.fromisoformat(timestamp).strftime("%H:%M")
                    else:
                        time_str = "N/A"

                    pair = trade.get('pair', 'N/A')
                    direction = trade.get('direction', 'N/A')
                    lots = trade.get('lots', 0)
                    score = trade.get('atlas_score', 0)
                    pnl = trade.get('pnl', 0)
                    status = trade.get('status', 'unknown')

                    if pnl:
                        total_pnl_24h += pnl
                        if pnl > 0:
                            wins += 1
                        else:
                            losses += 1

                    score_str = format_score(score)
                    pnl_str = f"${pnl:+,.2f}" if pnl else "Open"

                    print(f"  {time_str:<10} {pair:<10} {direction:<5} {lots:<6.1f} {score_str:<12} {pnl_str:<12} {status:<10}")

                # 24h Summary
                total_trades = wins + losses
                win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

                print(f"  {'-'*80}")
                print(f"  24h P/L: ${total_pnl_24h:+,.2f}  |  Win Rate: {win_rate:.1f}% ({wins}W-{losses}L)")
            else:
                print(f"\n[RECENT TRADES]")
                print(f"  No trades logged yet")

            # Agent Performance (if available)
            agent_state_dir = Path(__file__).parent / "learning" / "state"
            if agent_state_dir.exists():
                agent_stats = []
                for agent_file in agent_state_dir.glob("*agent_state.json"):
                    try:
                        with open(agent_file, 'r') as f:
                            agent_data = json.load(f)
                            if 'wins' in agent_data or 'losses' in agent_data:
                                agent_stats.append({
                                    'name': agent_file.stem.replace('agent_state', '').replace('_', ''),
                                    'wins': agent_data.get('wins', 0),
                                    'losses': agent_data.get('losses', 0),
                                    'weight': agent_data.get('weight', 1.0)
                                })
                    except:
                        pass

                if agent_stats:
                    print(f"\n[AGENT PERFORMANCE]")
                    print(f"  {'Agent':<25} {'W-L':<10} {'Win%':<10} {'Weight':<10}")
                    print(f"  {'-'*60}")

                    for agent in sorted(agent_stats, key=lambda x: x['wins'] + x['losses'], reverse=True)[:10]:
                        total = agent['wins'] + agent['losses']
                        win_pct = (agent['wins'] / total * 100) if total > 0 else 0
                        print(f"  {agent['name']:<25} {agent['wins']}-{agent['losses']:<8} {win_pct:5.1f}%     {agent['weight']:.2f}")

            print(f"\n{'='*100}")

            if auto_refresh:
                print(f"Auto-refreshing every 10 seconds... (Ctrl+C to exit)")
                time.sleep(10)
            else:
                break

    except KeyboardInterrupt:
        print(f"\n\nDashboard stopped.")
        return
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ATLAS Live Trading Dashboard")
    parser.add_argument('--once', action='store_true', help="Display once without auto-refresh")
    args = parser.parse_args()

    display_dashboard(auto_refresh=not args.once)
