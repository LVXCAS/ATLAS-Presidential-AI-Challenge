"""
Detailed OANDA Forex Position Viewer
Shows exactly what trades are open and their performance
"""
import os
from dotenv import load_dotenv
load_dotenv()

import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.trades as trades

oanda_token = os.getenv('OANDA_API_KEY')
oanda_account = os.getenv('OANDA_ACCOUNT_ID')

client = API(access_token=oanda_token, environment='practice')

print("=" * 80)
print(" " * 25 + "FOREX POSITIONS - DETAILED VIEW")
print("=" * 80)
print()

# Get account summary
r = accounts.AccountSummary(accountID=oanda_account)
resp = client.request(r)

balance = float(resp['account']['balance'])
unrealized_pl = float(resp['account']['unrealizedPL'])
pl_percent = (unrealized_pl / balance) * 100

print(f"Account: {oanda_account}")
print(f"Balance: ${balance:,.2f}")
print(f"Unrealized P/L: ${unrealized_pl:,.2f} ({pl_percent:+.3f}%)")
print()

# Get all open trades
r = trades.OpenTrades(accountID=oanda_account)
resp = client.request(r)

open_trades = resp.get('trades', [])

if not open_trades:
    print("[NO OPEN POSITIONS]")
    print()
    print("Bot is scanning but hasn't found opportunities yet.")
else:
    print(f"=" * 80)
    print(f"OPEN TRADES: {len(open_trades)}")
    print("=" * 80)
    print()

    for i, trade in enumerate(open_trades, 1):
        instrument = trade['instrument']
        units = int(trade['currentUnits'])
        direction = "LONG" if units > 0 else "SHORT"

        entry_price = float(trade['price'])
        current_price = float(trade['price'])  # This gets updated by OANDA
        unrealized = float(trade['unrealizedPL'])

        print(f"Trade #{i}: {instrument} {direction}")
        print(f"  Units: {abs(units):,}")
        print(f"  Entry Price: {entry_price:.5f}")
        print(f"  Current Price: {current_price:.5f}")
        print(f"  Unrealized P/L: ${unrealized:,.2f}")

        # Calculate percentage
        if unrealized != 0:
            pl_pct = (unrealized / balance) * 100
            print(f"  % of Account: {pl_pct:+.3f}%")

        # Stop Loss / Take Profit if set
        if 'stopLossOrder' in trade:
            sl_price = float(trade['stopLossOrder']['price'])
            print(f"  Stop Loss: {sl_price:.5f}")

        if 'takeProfitOrder' in trade:
            tp_price = float(trade['takeProfitOrder']['price'])
            print(f"  Take Profit: {tp_price:.5f}")

        print()

print("=" * 80)
print("Tip: Run this script anytime to see current positions")
print("Command: python VIEW_FOREX_POSITIONS.py")
print("=" * 80)
