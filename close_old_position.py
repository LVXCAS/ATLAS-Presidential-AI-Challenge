"""
Close the old USD_JPY position from last week
Start fresh for this week's trading
"""
import os
from dotenv import load_dotenv
load_dotenv()

import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.positions as positions

oanda_token = os.getenv('OANDA_API_KEY')
oanda_account = os.getenv('OANDA_ACCOUNT_ID')

client = API(access_token=oanda_token, environment='practice')

print("=" * 70)
print("CLOSING OLD USD_JPY POSITION FROM LAST WEEK")
print("=" * 70)
print()

# Close USD_JPY position
try:
    data = {
        "longUnits": "ALL"  # Close all long units
    }

    r = positions.PositionClose(
        accountID=oanda_account,
        instrument="USD_JPY",
        data=data
    )

    response = client.request(r)

    print("USD_JPY Position Closed:")
    print(f"  Long Units Closed: {response.get('longOrderFillTransaction', {}).get('units', 'N/A')}")
    print(f"  Fill Price: {response.get('longOrderFillTransaction', {}).get('price', 'N/A')}")
    print(f"  P/L: ${float(response.get('longOrderFillTransaction', {}).get('pl', 0)):.2f}")
    print()
    print("[SUCCESS] Position closed. Ready to start fresh!")
    print()

except Exception as e:
    print(f"[ERROR] Failed to close position: {e}")
    print()

print("=" * 70)
