"""
Close all OANDA positions to give new bot clean slate
"""
import os
from dotenv import load_dotenv
load_dotenv()

import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.positions as positions

oanda_token = os.getenv('OANDA_API_KEY')
oanda_account_id = os.getenv('OANDA_ACCOUNT_ID', '101-001-37330890-001')

client = API(access_token=oanda_token, environment='practice')

# Get all open positions
r = positions.OpenPositions(accountID=oanda_account_id)
response = client.request(r)

open_positions = response.get('positions', [])

print("=" * 70)
print("CLOSING ALL POSITIONS")
print("=" * 70)

if not open_positions:
    print("No open positions to close")
else:
    for pos in open_positions:
        instrument = pos['instrument']
        long_units = float(pos['long']['units'])
        short_units = float(pos['short']['units'])

        if long_units > 0:
            print(f"\nClosing {instrument} LONG position ({int(long_units)} units)...")
            data = {"longUnits": "ALL"}
            r = positions.PositionClose(
                accountID=oanda_account_id,
                instrument=instrument,
                data=data
            )
            response = client.request(r)
            pnl = float(response['longOrderFillTransaction']['pl'])
            print(f"  Closed at P/L: ${pnl:.2f}")

        if short_units < 0:
            print(f"\nClosing {instrument} SHORT position ({int(abs(short_units))} units)...")
            data = {"shortUnits": "ALL"}
            r = positions.PositionClose(
                accountID=oanda_account_id,
                instrument=instrument,
                data=data
            )
            response = client.request(r)
            pnl = float(response['shortOrderFillTransaction']['pl'])
            print(f"  Closed at P/L: ${pnl:.2f}")

print("\n" + "=" * 70)
print("ALL POSITIONS CLOSED - Fresh slate for new bot")
print("=" * 70)
