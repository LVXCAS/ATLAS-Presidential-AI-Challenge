"""
Close USD_JPY Position
Closes the losing USD/JPY LONG position that's down -$112
Reason: Fundamentals say SHORT (Fed dovish, BOJ hawkish), not LONG
"""
import os
from dotenv import load_dotenv
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.positions as positions

load_dotenv()

oanda_token = os.getenv('OANDA_API_KEY')
oanda_account_id = os.getenv('OANDA_ACCOUNT_ID', '101-001-37330890-001')

client = API(access_token=oanda_token, environment='practice')

print("=" * 70)
print("CLOSING USD_JPY POSITION")
print("=" * 70)
print(f"Account: {oanda_account_id}")
print("Reason: Fundamentals misaligned (Fed dovish + BOJ hawkish = SHORT signal)")
print("Current P/L: -$112")
print("=" * 70)

try:
    # Close USD_JPY position
    # OANDA uses "long" vs "short" in the endpoint, we have a LONG position
    data = {"longUnits": "ALL"}  # Close all long units

    r = positions.PositionClose(
        accountID=oanda_account_id,
        instrument="USD_JPY",
        data=data
    )

    response = client.request(r)

    print("\n[SUCCESS] USD_JPY position closed")
    print(f"Closed units: {response.get('longOrderFillTransaction', {}).get('units', 'N/A')}")
    print(f"Price: {response.get('longOrderFillTransaction', {}).get('price', 'N/A')}")
    print(f"P/L: ${float(response.get('longOrderFillTransaction', {}).get('pl', 0)):.2f}")

except Exception as e:
    print(f"\n[ERROR] Failed to close position: {e}")
    print("Note: Position may already be closed or not exist")

print("\n" + "=" * 70)
print("Position closure complete")
print("=" * 70)
