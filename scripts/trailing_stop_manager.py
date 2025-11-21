"""
Trailing Stop Manager - Adjusts stop losses as positions become profitable
Runs in background alongside the main trading bot
"""
import os
import time
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.trades as trades

oanda_token = os.getenv('OANDA_API_KEY')
oanda_account_id = os.getenv('OANDA_ACCOUNT_ID', '101-001-37330890-001')

client = API(access_token=oanda_token, environment='practice')

# TRAILING STOP LEVELS (in pips)
BREAKEVEN_THRESHOLD = 50  # Move to breakeven after 50 pips profit
LOCK_HALF_THRESHOLD = 100  # Lock 50 pips profit after 100 pips
LOCK_MOST_THRESHOLD = 150  # Lock 100 pips profit after 150 pips

def get_pip_value(instrument):
    """Get pip value for an instrument (1 pip = 0.0001 for most, 0.01 for JPY pairs)"""
    if 'JPY' in instrument:
        return 0.01  # JPY pairs use 2 decimal places
    else:
        return 0.0001  # Most pairs use 4 decimal places

def get_open_trades():
    """Get all open trades"""
    try:
        r = trades.OpenTrades(accountID=oanda_account_id)
        response = client.request(r)
        return response.get('trades', [])
    except Exception as e:
        print(f"[ERROR] Getting trades: {e}")
        return []

def update_stop_loss(trade_id, new_stop_price):
    """Update stop loss for a trade"""
    try:
        data = {
            "stopLoss": {
                "price": str(new_stop_price)
            }
        }

        r = trades.TradeCRCDO(
            accountID=oanda_account_id,
            tradeID=trade_id,
            data=data
        )

        response = client.request(r)
        return True

    except Exception as e:
        print(f"[ERROR] Updating stop loss: {e}")
        return False

def calculate_profit_pips(trade):
    """Calculate profit in pips for a trade"""
    try:
        instrument = trade['instrument']
        units = float(trade['currentUnits'])
        entry_price = float(trade['price'])
        current_price = float(trade['unrealizedPL']) / units + entry_price  # Approximate

        pip_value = get_pip_value(instrument)

        if units > 0:  # LONG position
            profit_pips = (current_price - entry_price) / pip_value
        else:  # SHORT position
            profit_pips = (entry_price - current_price) / pip_value

        return profit_pips

    except Exception as e:
        print(f"[ERROR] Calculating pips: {e}")
        return 0

def manage_trailing_stops():
    """Main loop - check and adjust stops"""
    print("=" * 70)
    print("TRAILING STOP MANAGER - ACTIVE")
    print("=" * 70)
    print(f"Breakeven threshold: {BREAKEVEN_THRESHOLD} pips")
    print(f"Lock half threshold: {LOCK_HALF_THRESHOLD} pips")
    print(f"Lock most threshold: {LOCK_MOST_THRESHOLD} pips")
    print("=" * 70)

    last_check = {}

    while True:
        try:
            open_trades = get_open_trades()

            if not open_trades:
                # No open trades, wait and check again
                time.sleep(60)
                continue

            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking {len(open_trades)} open trades...")

            for trade in open_trades:
                trade_id = trade['id']
                instrument = trade['instrument']
                units = float(trade['currentUnits'])
                entry_price = float(trade['price'])
                current_stop = float(trade.get('stopLossOrder', {}).get('price', 0))
                unrealized_pl = float(trade['unrealizedPL'])

                direction = "LONG" if units > 0 else "SHORT"

                # Calculate profit in pips
                profit_pips = calculate_profit_pips(trade)

                print(f"  {instrument} {direction}: Entry {entry_price:.5f} | P/L: ${unrealized_pl:.2f} ({profit_pips:.1f} pips)")

                # Determine new stop level
                pip_value = get_pip_value(instrument)
                new_stop = None

                if profit_pips >= LOCK_MOST_THRESHOLD:
                    # Lock 100 pips profit
                    if direction == "LONG":
                        target_stop = entry_price + (100 * pip_value)
                        if current_stop < target_stop:
                            new_stop = target_stop
                            print(f"    -> Moving stop to +100 pips (locking ${100 * abs(units) / 10000:.0f} profit)")
                    else:  # SHORT
                        target_stop = entry_price - (100 * pip_value)
                        if current_stop > target_stop:
                            new_stop = target_stop
                            print(f"    -> Moving stop to +100 pips (locking ${100 * abs(units) / 10000:.0f} profit)")

                elif profit_pips >= LOCK_HALF_THRESHOLD:
                    # Lock 50 pips profit
                    if direction == "LONG":
                        target_stop = entry_price + (50 * pip_value)
                        if current_stop < target_stop:
                            new_stop = target_stop
                            print(f"    -> Moving stop to +50 pips (locking ${50 * abs(units) / 10000:.0f} profit)")
                    else:  # SHORT
                        target_stop = entry_price - (50 * pip_value)
                        if current_stop > target_stop:
                            new_stop = target_stop
                            print(f"    -> Moving stop to +50 pips (locking ${50 * abs(units) / 10000:.0f} profit)")

                elif profit_pips >= BREAKEVEN_THRESHOLD:
                    # Move to breakeven
                    if direction == "LONG":
                        if current_stop < entry_price:
                            new_stop = entry_price
                            print(f"    -> Moving stop to BREAKEVEN (no loss possible)")
                    else:  # SHORT
                        if current_stop > entry_price:
                            new_stop = entry_price
                            print(f"    -> Moving stop to BREAKEVEN (no loss possible)")

                # Update stop if needed
                if new_stop:
                    precision = 3 if 'JPY' in instrument else 5
                    new_stop_rounded = round(new_stop, precision)

                    if update_stop_loss(trade_id, new_stop_rounded):
                        print(f"    [SUCCESS] Stop updated to {new_stop_rounded:.5f}")
                        last_check[trade_id] = profit_pips
                    else:
                        print(f"    [FAILED] Could not update stop")

            # Check every 60 seconds
            time.sleep(60)

        except KeyboardInterrupt:
            print("\n[STOPPED] Trailing stop manager shutting down...")
            break
        except Exception as e:
            print(f"[ERROR] Main loop error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    manage_trailing_stops()
