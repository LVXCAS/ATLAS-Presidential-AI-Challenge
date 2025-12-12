"""
Position Monitor - Tracks open positions and logs exits

Runs in background alongside live trading to detect when positions close
and log the final P/L for threshold optimization analysis.
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path
from adapters.oanda_adapter import OandaAdapter
from core.trade_logger import TradeLogger

def load_open_trades():
    """Load currently open trades from trade logger"""
    trade_logger = TradeLogger()
    return trade_logger.open_trades.copy()

def monitor_positions(check_interval_seconds=30):
    """
    Monitor open positions and log exits when they close.

    Args:
        check_interval_seconds: How often to check positions (default 30s)
    """
    print("="*80)
    print("ATLAS POSITION MONITOR")
    print("="*80)
    print(f"Check interval: Every {check_interval_seconds} seconds")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()

    # Initialize
    adapter = OandaAdapter(
        os.getenv('OANDA_API_KEY'),
        os.getenv('OANDA_ACCOUNT_ID'),
        practice=True
    )
    trade_logger = TradeLogger()

    # Track positions we're monitoring
    tracked_positions = {}  # trade_id -> {entry_price, pair, timestamp, ...}

    check_count = 0

    while True:
        try:
            check_count += 1

            # Get current open positions from OANDA
            oanda_positions = adapter.get_open_positions()
            oanda_trade_ids = set()

            if oanda_positions:
                for pos in oanda_positions:
                    if 'trade_id' in pos:
                        oanda_trade_ids.add(pos['trade_id'])

            # Get open trades from logger
            open_trades = trade_logger.open_trades.copy()

            # Check for newly closed positions
            for trade_id, trade_data in list(tracked_positions.items()):
                # If trade was open but is no longer in OANDA positions, it closed
                if trade_id not in oanda_trade_ids:
                    # Position closed - need to get final P/L
                    print(f"\n[CLOSED] {trade_id} - Fetching exit data...")

                    # Get transaction history to find close details
                    # Note: This is simplified - real implementation would query OANDA transactions API
                    balance_data = adapter.get_account_balance()

                    if balance_data:
                        # Estimate P/L from balance change (simplified)
                        # In production, would query OANDA transaction history

                        # For now, mark as closed with unknown P/L
                        # Real implementation: query /v3/accounts/{accountID}/transactions
                        # to get actual close price and P/L

                        print(f"  Trade {trade_id} closed")
                        print(f"  Entry: {trade_data.get('entry_price')}")
                        print(f"  Pair: {trade_data.get('pair')}")
                        print(f"  NOTE: Exit P/L tracking requires OANDA transaction API")
                        print(f"  Current balance: ${balance_data['balance']:.2f}")

                    # Remove from tracking
                    del tracked_positions[trade_id]

            # Add any new positions to tracking
            for trade_id, trade_obj in open_trades.items():
                if trade_id not in tracked_positions:
                    tracked_positions[trade_id] = {
                        'trade_id': trade_id,
                        'pair': trade_obj.pair,
                        'entry_price': trade_obj.entry_price,
                        'entry_time': trade_obj.timestamp_entry,
                        'stop_loss': trade_obj.stop_loss,
                        'take_profit': trade_obj.take_profit
                    }
                    print(f"\n[TRACKING] {trade_id}")
                    print(f"  Pair: {trade_obj.pair}")
                    print(f"  Entry: {trade_obj.entry_price}")

            # Status update every 10 checks
            if check_count % 10 == 0:
                print(f"\n[CHECK #{check_count}] {datetime.now().strftime('%H:%M:%S')}")
                print(f"  Tracking: {len(tracked_positions)} positions")
                if balance_data:
                    print(f"  Balance: ${balance_data['balance']:.2f}")

            time.sleep(check_interval_seconds)

        except KeyboardInterrupt:
            print("\n\nPosition monitor stopped by user")
            break
        except Exception as e:
            print(f"\nError in monitor loop: {e}")
            time.sleep(check_interval_seconds)

if __name__ == "__main__":
    monitor_positions(check_interval_seconds=30)
