"""
Monitor for new bot trades (10x leverage + news filter)
Run this to see when the upgraded bot places its first trade
"""
import os
import time
from datetime import datetime
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.positions as positions

# Load OANDA credentials
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

oanda_token = os.getenv('OANDA_API_KEY')
oanda_account_id = os.getenv('OANDA_ACCOUNT_ID', '101-001-37330890-001')

client = API(access_token=oanda_token, environment='practice')

def get_positions():
    """Get current OANDA positions"""
    try:
        r = positions.OpenPositions(accountID=oanda_account_id)
        response = client.request(r)
        return response.get('positions', [])
    except Exception as e:
        print(f"[ERROR] {e}")
        return []

print("=" * 70)
print("MONITORING FOR NEW BOT TRADES")
print("=" * 70)
print("Watching for:")
print("  - New positions with 1,000,000 units (10x leverage)")
print("  - Old positions have only 1,000 units")
print("=" * 70)

last_count = None

while True:
    current_positions = get_positions()

    if len(current_positions) != last_count:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Position count changed: {last_count} -> {len(current_positions)}")
        last_count = len(current_positions)

        if current_positions:
            print("\nCurrent Positions:")
            for pos in current_positions:
                instrument = pos['instrument']
                long_units = int(pos['long']['units']) if float(pos['long']['units']) > 0 else 0
                short_units = abs(int(pos['short']['units'])) if float(pos['short']['units']) < 0 else 0
                units = long_units if long_units > 0 else short_units
                direction = "LONG" if long_units > 0 else "SHORT"

                unrealized_pl = float(pos['long']['unrealizedPL']) if long_units > 0 else float(pos['short']['unrealizedPL'])

                # Flag if this is a NEW BOT trade (1M units)
                if units >= 100000:
                    print(f"  >>> NEW BOT: {instrument} {direction} {units:,} units | P/L: ${unrealized_pl:.2f}")
                else:
                    print(f"      Old bot: {instrument} {direction} {units:,} units | P/L: ${unrealized_pl:.2f}")

    time.sleep(30)  # Check every 30 seconds
