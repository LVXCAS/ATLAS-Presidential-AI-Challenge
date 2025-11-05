#!/usr/bin/env python3
"""
EMERGENCY CLOSE - Close all positions immediately
"""
import os
from dotenv import load_dotenv
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.trades as trades

load_dotenv()

oanda_token = os.getenv('OANDA_API_KEY')
oanda_account_id = '101-001-37330890-001'

client = API(access_token=oanda_token, environment='practice')

# Get all open trades
r = trades.TradesList(accountID=oanda_account_id)
response = client.request(r)
trades_list = response.get('trades', [])

print("="*70)
print("EMERGENCY POSITION CLOSE")
print("="*70)

if not trades_list:
    print("No open positions to close")
else:
    for trade in trades_list:
        trade_id = trade['id']
        instrument = trade['instrument']
        units = int(trade['currentUnits'])
        entry = float(trade['price'])
        unrealized = float(trade['unrealizedPL'])

        direction = "LONG" if units > 0 else "SHORT"

        print(f"\nClosing {instrument} {direction}:")
        print(f"  Trade ID: {trade_id}")
        print(f"  Units: {abs(units):,}")
        print(f"  Entry: {entry:.5f}")
        print(f"  P/L: ${unrealized:,.2f}")

        # Close the trade
        r = trades.TradeClose(accountID=oanda_account_id, tradeID=trade_id)
        try:
            response = client.request(r)
            print(f"  [CLOSED] Trade closed successfully")
            print(f"  Final P/L: ${unrealized:,.2f}")
        except Exception as e:
            print(f"  [ERROR] Could not close trade: {e}")

print("\n" + "="*70)
print("All positions closed")
print("="*70)
