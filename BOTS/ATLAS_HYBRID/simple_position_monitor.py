"""
Simple Position Monitor - Just tracks OANDA positions and reports when they close

Lightweight alternative that doesn't depend on TradeLogger internals.
"""

import os
import time
from datetime import datetime
from adapters.oanda_adapter import OandaAdapter

def monitor_positions(check_interval_seconds=30):
    """Monitor OANDA positions and report when they close."""

    print("="*80)
    print("SIMPLE POSITION MONITOR")
    print("="*80)
    print(f"Check interval: Every {check_interval_seconds} seconds")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()

    # Initialize OANDA adapter
    adapter = OandaAdapter(
        os.getenv('OANDA_API_KEY'),
        os.getenv('OANDA_ACCOUNT_ID'),
        practice=True
    )

    # Track positions we've seen
    last_positions = {}  # symbol -> position data
    check_count = 0
    last_balance = None

    while True:
        try:
            check_count += 1

            # Get current positions from OANDA
            current_positions = adapter.get_open_positions()
            balance_data = adapter.get_account_balance()

            # Build current position map
            current_map = {}
            if current_positions:
                for pos in current_positions:
                    symbol = pos.get('symbol', pos.get('instrument'))
                    current_map[symbol] = pos

            # Check for closed positions
            for symbol, old_pos in last_positions.items():
                if symbol not in current_map:
                    # Position closed!
                    print(f"\n{'='*80}")
                    print(f"POSITION CLOSED: {symbol}")
                    print(f"{'='*80}")
                    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Type: {old_pos.get('type', 'UNKNOWN')}")
                    print(f"Units: {old_pos.get('units', 0)}")

                    if last_balance and balance_data:
                        balance_change = balance_data['balance'] - last_balance
                        print(f"Balance change: ${balance_change:+.2f}")

                    if balance_data:
                        print(f"New balance: ${balance_data['balance']:.2f}")
                    print(f"{'='*80}")
                    print()

            # Check for new positions
            for symbol, new_pos in current_map.items():
                if symbol not in last_positions:
                    # New position opened
                    print(f"\n{'='*80}")
                    print(f"NEW POSITION OPENED: {symbol}")
                    print(f"{'='*80}")
                    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Type: {new_pos.get('type', 'UNKNOWN')}")
                    print(f"Units: {new_pos.get('units', 0)}")
                    print(f"Unrealized P/L: ${new_pos.get('unrealized_pnl', 0):.2f}")
                    print(f"{'='*80}")
                    print()

            # Update tracking
            last_positions = current_map.copy()
            if balance_data:
                last_balance = balance_data['balance']

            # Status update every 10 checks (~5 minutes)
            if check_count % 10 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ", end='')
                print(f"Tracking {len(current_map)} open position(s) | ", end='')
                if balance_data:
                    print(f"Balance: ${balance_data['balance']:.2f} | ", end='')
                    print(f"Unrealized P/L: ${balance_data['unrealized_pnl']:+.2f}")
                else:
                    print()

            time.sleep(check_interval_seconds)

        except KeyboardInterrupt:
            print("\n\nMonitor stopped by user")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(check_interval_seconds)

if __name__ == "__main__":
    import sys

    # Allow custom interval
    interval = 30
    if len(sys.argv) > 1:
        try:
            interval = int(sys.argv[1])
        except:
            pass

    monitor_positions(check_interval_seconds=interval)
