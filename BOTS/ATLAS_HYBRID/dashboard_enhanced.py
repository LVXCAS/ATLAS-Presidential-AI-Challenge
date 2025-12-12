"""
ATLAS Enhanced Dashboard - Real-Time Monitoring
Shows balance, positions, recent trades, win rate, and profit tracking

Usage:
    python dashboard_enhanced.py         # Auto-refresh every 10 seconds
    python dashboard_enhanced.py --once  # Single snapshot
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


def calculate_performance_metrics(trades):
    """Calculate win rate, average profit, etc."""
    if not trades:
        return {
            'total': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0
        }

    wins = [t for t in trades if t.get('result') == 'win']
    losses = [t for t in trades if t.get('result') == 'loss']

    total_pnl = sum(t.get('pnl') or 0 for t in trades)
    avg_win = sum(t.get('pnl') or 0 for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.get('pnl') or 0 for t in losses) / len(losses) if losses else 0

    total_wins_pnl = sum(t.get('pnl') or 0 for t in wins)
    total_losses_pnl = abs(sum(t.get('pnl') or 0 for t in losses))
    profit_factor = total_wins_pnl / total_losses_pnl if total_losses_pnl > 0 else 0

    return {
        'total': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(trades) * 100 if trades else 0,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor
    }


def display_dashboard(auto_refresh=True):
    """Display enhanced dashboard"""

    STARTING_BALANCE = 173117.54  # Track against original balance

    try:
        while True:
            if auto_refresh:
                clear_screen()

            # Header
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("=" * 100)
            print(f"{'ATLAS ENHANCED DASHBOARD':^100}")
            print(f"{now:^100}")
            print("=" * 100)

            # Load data
            oanda = OandaAdapter()
            balance_data = oanda.get_account_balance()
            balance = balance_data.get('balance', 0) if isinstance(balance_data, dict) else balance_data
            positions = oanda.get_open_positions()
            state = load_coordinator_state()
            recent_trades_24h = load_recent_trades(hours=24)
            recent_trades_1h = load_recent_trades(hours=1)

            # Account Summary
            print(f"\n[ACCOUNT STATUS]")
            print(f"  Current Balance:    ${balance:,.2f}")

            session_pnl = balance - STARTING_BALANCE
            session_pnl_pct = (session_pnl / STARTING_BALANCE * 100)
            print(f"  Session P/L:        ${session_pnl:+,.2f} ({session_pnl_pct:+.2f}%)")

            print(f"  Open Positions:     {len(positions)}")

            unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in positions)
            print(f"  Unrealized P/L:     ${unrealized_pnl:+,.2f}")

            total_value = balance + unrealized_pnl
            print(f"  Total Equity:       ${total_value:,.2f}")

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
            print(f"\n[SYSTEM CONFIGURATION]")
            total_decisions = state.get('total_decisions', 0)
            trades_executed = state.get('trades_executed', 0)
            exec_rate = (trades_executed / total_decisions * 100) if total_decisions > 0 else 0

            print(f"  Score Threshold:    {state.get('config', {}).get('score_threshold', 'N/A')}")
            print(f"  Training Phase:     {state.get('config', {}).get('training_phase', 'N/A').upper()}")
            print(f"  Total Decisions:    {total_decisions:,}")
            print(f"  Trades Executed:    {trades_executed}")
            print(f"  Execution Rate:     {exec_rate:.1f}%")

            # Performance Metrics - Last 24 Hours
            metrics_24h = calculate_performance_metrics(recent_trades_24h)
            print(f"\n[PERFORMANCE - LAST 24 HOURS]")
            print(f"  Trades:             {metrics_24h['total']} ({metrics_24h['wins']}W / {metrics_24h['losses']}L)")
            print(f"  Win Rate:           {metrics_24h['win_rate']:.1f}%")
            print(f"  Total P/L:          ${metrics_24h['total_pnl']:+,.2f}")
            if metrics_24h['total'] > 0:
                print(f"  Avg Win:            ${metrics_24h['avg_win']:+,.2f}")
                print(f"  Avg Loss:           ${metrics_24h['avg_loss']:+,.2f}")
                print(f"  Profit Factor:      {metrics_24h['profit_factor']:.2f}")

            # Performance Metrics - Last 1 Hour
            metrics_1h = calculate_performance_metrics(recent_trades_1h)
            print(f"\n[PERFORMANCE - LAST 1 HOUR]")
            print(f"  Trades:             {metrics_1h['total']} ({metrics_1h['wins']}W / {metrics_1h['losses']}L)")
            if metrics_1h['total'] > 0:
                print(f"  Win Rate:           {metrics_1h['win_rate']:.1f}%")
                print(f"  P/L:                ${metrics_1h['total_pnl']:+,.2f}")

            # Recent Trades (Last 10)
            if recent_trades_24h:
                print(f"\n[RECENT TRADES - LAST 10]")
                print(f"  {'Time':<8} {'Pair':<10} {'Dir':<5} {'Lots':<6} {'Score':<7} {'P/L':<12} {'Result':<8}")
                print(f"  {'-'*80}")

                for trade in recent_trades_24h[:10]:
                    time_str = trade.get('timestamp_decision', '')[-8:-3] if trade.get('timestamp_decision') else 'N/A'
                    pair = trade.get('pair', 'UNKNOWN')
                    direction = trade.get('direction', 'N/A')[:3].upper()
                    score = trade.get('score', 0)
                    score_str = f"{score:+.2f}"

                    # Get lot size from units
                    units = trade.get('units', 0)
                    lots = abs(units) / 100000 if units else 0

                    pnl = trade.get('pnl') or 0
                    pnl_str = f"${pnl:+,.2f}" if pnl != 0 else "$0.00"

                    result = trade.get('result', trade.get('status', 'PENDING'))

                    print(f"  {time_str:<8} {pair:<10} {direction:<5} {lots:<6.1f} {score_str:<7} {pnl_str:<12} {result:<8}")

            # Status Footer
            print(f"\n{'='*100}")
            if auto_refresh:
                print(f"  Auto-refreshing every 10 seconds... (Ctrl+C to stop)")
                print(f"{'='*100}")
                time.sleep(10)
            else:
                break

    except KeyboardInterrupt:
        print(f"\n\n  Dashboard stopped by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    auto_refresh = "--once" not in sys.argv
    display_dashboard(auto_refresh=auto_refresh)
