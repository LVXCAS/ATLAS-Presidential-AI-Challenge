"""
Trailing Stop Manager V2 - DOLLAR-BASED THRESHOLDS
Fixes the Kelly Criterion position sizing incompatibility
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
import oandapyV20.endpoints.pricing as pricing

oanda_token = os.getenv('OANDA_API_KEY')
oanda_account_id = os.getenv('OANDA_ACCOUNT_ID', '101-001-37330890-001')

client = API(access_token=oanda_token, environment='practice')

# DOLLAR-BASED TRAILING STOP LEVELS (works with any position size!)
BREAKEVEN_PROFIT = 1000   # Move to breakeven after $1,000 profit
LOCK_HALF_PROFIT = 2000   # Lock half gains after $2,000 profit
LOCK_MOST_PROFIT = 3000   # Lock most gains after $3,000 profit

def get_pip_value(instrument):
    """Get pip value for an instrument"""
    if 'JPY' in instrument:
        return 0.01
    else:
        return 0.0001

def get_open_trades():
    """Get all open trades"""
    try:
        r = trades.OpenTrades(accountID=oanda_account_id)
        response = client.request(r)
        return response.get('trades', [])
    except Exception as e:
        print(f"[ERROR] Getting trades: {e}")
        return []

def get_current_price(instrument):
    """Get current market price"""
    try:
        params = {"instruments": instrument}
        r = pricing.PricingInfo(accountID=oanda_account_id, params=params)
        response = client.request(r)
        if response.get('prices'):
            return float(response['prices'][0]['closeoutBid'])
        return None
    except Exception as e:
        print(f"[ERROR] Getting price for {instrument}: {e}")
        return None

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

def calculate_stop_price_for_profit(entry_price, current_price, target_profit_pct, direction, pip_value):
    """
    Calculate what stop price would lock in a target percentage of current profit

    Args:
        entry_price: Entry price
        current_price: Current market price
        target_profit_pct: Percentage of profit to lock (0.5 = 50%, 0.75 = 75%)
        direction: 'LONG' or 'SHORT'
        pip_value: Pip size for the instrument

    Returns:
        Stop price that locks in target profit percentage
    """
    if direction == "LONG":
        # LONG: current is above entry (profit)
        total_move = current_price - entry_price
        target_move = total_move * target_profit_pct
        stop_price = entry_price + target_move
    else:  # SHORT
        # SHORT: current is below entry (profit)
        total_move = entry_price - current_price
        target_move = total_move * target_profit_pct
        stop_price = entry_price - target_move

    return stop_price

def manage_trailing_stops():
    """Main loop - check and adjust stops using DOLLAR thresholds"""
    print("=" * 70)
    print("TRAILING STOP MANAGER V2 - DOLLAR-BASED THRESHOLDS")
    print("=" * 70)
    print(f"Breakeven at: ${BREAKEVEN_PROFIT:,} profit")
    print(f"Lock 50% at: ${LOCK_HALF_PROFIT:,} profit")
    print(f"Lock 75% at: ${LOCK_MOST_PROFIT:,} profit")
    print("=" * 70)

    last_check = {}

    while True:
        try:
            open_trades = get_open_trades()

            if not open_trades:
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
                pip_value = get_pip_value(instrument)

                # Get current market price
                current_price = get_current_price(instrument)
                if not current_price:
                    print(f"  [WARN] Could not get price for {instrument}")
                    continue

                # Calculate pips for display
                if direction == "LONG":
                    profit_pips = (current_price - entry_price) / pip_value
                else:
                    profit_pips = (entry_price - current_price) / pip_value

                print(f"  {instrument} {direction}: Entry {entry_price:.5f} | Current {current_price:.5f} | P/L: ${unrealized_pl:.2f} ({profit_pips:.1f} pips)")

                # Determine new stop level based on DOLLAR profit
                new_stop = None
                reason = ""

                if unrealized_pl >= LOCK_MOST_PROFIT:
                    # Lock 75% of current profit
                    target_stop = calculate_stop_price_for_profit(
                        entry_price, current_price, 0.75, direction, pip_value
                    )

                    # Only update if this moves stop in favorable direction
                    should_update = False
                    if direction == "LONG" and target_stop > current_stop:
                        should_update = True
                    elif direction == "SHORT" and target_stop < current_stop:
                        should_update = True

                    if should_update or current_stop == 0:
                        new_stop = target_stop
                        locked_profit = unrealized_pl * 0.75
                        reason = f"Locking 75% of profit (${locked_profit:.2f})"

                elif unrealized_pl >= LOCK_HALF_PROFIT:
                    # Lock 50% of current profit
                    target_stop = calculate_stop_price_for_profit(
                        entry_price, current_price, 0.50, direction, pip_value
                    )

                    should_update = False
                    if direction == "LONG" and target_stop > current_stop:
                        should_update = True
                    elif direction == "SHORT" and target_stop < current_stop:
                        should_update = True

                    if should_update or current_stop == 0:
                        new_stop = target_stop
                        locked_profit = unrealized_pl * 0.50
                        reason = f"Locking 50% of profit (${locked_profit:.2f})"

                elif unrealized_pl >= BREAKEVEN_PROFIT:
                    # Move to breakeven (no loss possible)
                    should_update = False
                    if direction == "LONG" and entry_price > current_stop:
                        should_update = True
                        new_stop = entry_price
                    elif direction == "SHORT" and entry_price < current_stop:
                        should_update = True
                        new_stop = entry_price

                    if should_update or current_stop == 0:
                        new_stop = entry_price
                        reason = "Moving to BREAKEVEN (no loss possible)"

                # Update stop if needed
                if new_stop:
                    precision = 3 if 'JPY' in instrument else 5
                    new_stop_rounded = round(new_stop, precision)

                    print(f"    -> {reason}")
                    print(f"    -> New stop: {new_stop_rounded:.5f} (was {current_stop:.5f})")

                    if update_stop_loss(trade_id, new_stop_rounded):
                        print(f"    [SUCCESS] Stop updated!")
                        last_check[trade_id] = unrealized_pl
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
