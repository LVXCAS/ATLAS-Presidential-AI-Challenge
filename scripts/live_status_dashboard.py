#!/usr/bin/env python3
"""
LIVE STATUS DASHBOARD - Simple Real-Time Monitor
Shows what's happening right now across all systems
"""

import os
import time
from datetime import datetime
import pytz
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_system_status():
    """Get current status of all systems"""
    api = tradeapi.REST(
        key_id=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        base_url=os.getenv('ALPACA_BASE_URL'),
        api_version='v2'
    )

    account = api.get_account()
    positions = api.list_positions()
    orders_today = api.list_orders(status='all', limit=50)

    return {
        'account': account,
        'positions': positions,
        'orders_today': orders_today
    }

def get_scanner_status():
    """Check scanner log for recent activity"""
    try:
        with open('scanner_output.log', 'r') as f:
            lines = f.readlines()
            # Get last 50 lines
            recent = lines[-50:] if len(lines) > 50 else lines

            # Check if scanner is actively scanning
            for line in reversed(recent):
                if 'Progress:' in line and 'tickers scanned' in line:
                    return {'status': 'SCANNING', 'last_activity': line.strip()}
                elif 'WAITING' in line and '5 minutes' in line:
                    return {'status': 'WAITING', 'last_activity': 'Next scan in 5 minutes'}
                elif 'SCAN COMPLETE' in line:
                    return {'status': 'SCAN_COMPLETE', 'last_activity': 'Scan just completed'}

            return {'status': 'UNKNOWN', 'last_activity': 'Check scanner_output.log'}
    except:
        return {'status': 'ERROR', 'last_activity': 'Cannot read log'}

def get_stop_loss_status():
    """Check stop loss monitor activity"""
    try:
        with open('stop_loss_output.log', 'r') as f:
            lines = f.readlines()
            recent = lines[-20:] if len(lines) > 20 else lines

            triggers = sum(1 for line in recent if 'STOP LOSS TRIGGERED' in line)
            closed = sum(1 for line in recent if 'STOP LOSS EXECUTED' in line)

            return {
                'status': 'ACTIVE',
                'recent_triggers': triggers,
                'recent_closed': closed
            }
    except:
        return {'status': 'ERROR', 'recent_triggers': 0, 'recent_closed': 0}

def display_dashboard():
    """Display live dashboard"""

    while True:
        try:
            clear_screen()

            pdt = pytz.timezone('America/Los_Angeles')
            now = datetime.now(pdt)
            market_close = now.replace(hour=13, minute=0, second=0, microsecond=0)
            time_left = (market_close - now).total_seconds() / 3600

            print("=" * 80)
            print(f"LIVE TRADING DASHBOARD - {now.strftime('%I:%M:%S %p PDT')}")
            print("=" * 80)
            print(f"Market closes in: {time_left:.1f} hours")
            print()

            # Get data
            status = get_system_status()
            scanner_status = get_scanner_status()
            stop_loss_status = get_stop_loss_status()

            account = status['account']
            positions = status['positions']
            orders = status['orders_today']

            # Account Summary
            print("[ACCOUNT STATUS]")
            print("-" * 80)
            equity = float(account.equity)
            cash = float(account.cash)
            print(f"  Equity:           ${equity:,.2f}")
            print(f"  Cash:             ${cash:,.2f}")
            print(f"  Buying Power:     ${float(account.buying_power):,.2f}")
            print(f"  Options Power:    ${float(account.options_buying_power):,.2f}")
            print()

            # Positions Summary
            print("[POSITIONS - {} OPEN]".format(len(positions)))
            print("-" * 80)
            if len(positions) > 0:
                total_pl = sum(float(p.unrealized_pl) for p in positions)
                winners = sum(1 for p in positions if float(p.unrealized_pl) > 0)
                losers = sum(1 for p in positions if float(p.unrealized_pl) < 0)
                win_rate = (winners / (winners + losers) * 100) if (winners + losers) > 0 else 0

                print(f"  Total Unrealized P&L: ${total_pl:,.2f}")
                print(f"  Winners: {winners} | Losers: {losers} | Win Rate: {win_rate:.1f}%")
                print()

                # Show top 5 winners and top 5 losers
                sorted_positions = sorted(positions, key=lambda p: float(p.unrealized_pl), reverse=True)

                print("  Top 5 Winners:")
                for pos in sorted_positions[:5]:
                    if float(pos.unrealized_pl) > 0:
                        pl = float(pos.unrealized_pl)
                        pl_pct = float(pos.unrealized_plpc) * 100
                        print(f"    {pos.symbol:30s} +${pl:>8.2f} ({pl_pct:>+6.1f}%)")

                print()
                print("  Top 5 Losers:")
                for pos in sorted_positions[-5:]:
                    if float(pos.unrealized_pl) < 0:
                        pl = float(pos.unrealized_pl)
                        pl_pct = float(pos.unrealized_plpc) * 100
                        print(f"    {pos.symbol:30s} ${pl:>8.2f} ({pl_pct:>+6.1f}%)")
            else:
                print("  No open positions")
            print()

            # Scanner Status
            print("[SCANNER STATUS]")
            print("-" * 80)
            print(f"  Status: {scanner_status['status']}")
            print(f"  Activity: {scanner_status['last_activity']}")
            print()

            # Stop Loss Monitor
            print("[STOP LOSS MONITOR]")
            print("-" * 80)
            print(f"  Status: {stop_loss_status['status']}")
            print(f"  Recent triggers: {stop_loss_status['recent_triggers']}")
            print(f"  Recently closed: {stop_loss_status['recent_closed']}")
            print()

            # Orders Today
            print("[ORDERS TODAY - {} TOTAL]".format(len(orders)))
            print("-" * 80)
            filled = sum(1 for o in orders if o.status == 'filled')
            pending = sum(1 for o in orders if o.status in ['pending_new', 'accepted', 'new'])
            rejected = sum(1 for o in orders if o.status in ['rejected', 'canceled'])

            print(f"  Filled: {filled} | Pending: {pending} | Rejected: {rejected}")
            print()

            print("=" * 80)
            print("Press Ctrl+C to exit | Updates every 15 seconds")
            print("=" * 80)

            # Wait 15 seconds
            time.sleep(15)

        except KeyboardInterrupt:
            print("\n\nDashboard stopped.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(15)

if __name__ == "__main__":
    display_dashboard()
