"""
Quick Position Summary - See current status at a glance
"""

import os
from dotenv import load_dotenv
load_dotenv()

import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.trades as trades
from datetime import datetime

oanda_token = os.getenv('OANDA_API_KEY')
oanda_account = os.getenv('OANDA_ACCOUNT_ID')
client = API(access_token=oanda_token, environment='practice')

# Get account summary
r = accounts.AccountSummary(accountID=oanda_account)
resp = client.request(r)

balance = float(resp['account']['balance'])
unrealized_pl = float(resp['account']['unrealizedPL'])
margin_used = float(resp['account']['marginUsed'])
margin_available = float(resp['account']['marginAvailable'])

print(f"\n{'='*60}")
print(f"FOREX ACCOUNT SUMMARY - {datetime.now().strftime('%I:%M %p, %b %d')}")
print(f"{'='*60}\n")

print(f"Balance: ${balance:,.2f}")
print(f"Unrealized P/L: ${unrealized_pl:,.2f} ({unrealized_pl/balance*100:+.3f}%)")
print(f"Margin Used: ${margin_used:,.2f}")
print(f"Margin Available: ${margin_available:,.2f}")

# Get open trades
r = trades.TradesList(accountID=oanda_account)
resp = client.request(r)
open_trades = resp.get('trades', [])

print(f"\nOpen Positions: {len(open_trades)}")

if open_trades:
    print(f"\n{'='*60}")
    for trade in open_trades:
        pair = trade['instrument']
        trade_id = trade['id']
        units = int(trade['currentUnits'])
        direction = "LONG" if units > 0 else "SHORT"
        entry = float(trade['price'])
        unrealized = float(trade['unrealizedPL'])
        pct = (unrealized / balance) * 100

        print(f"\n{pair} {direction} (Trade #{trade_id})")
        print(f"  Entry: {entry}")
        print(f"  Units: {abs(units):,}")
        print(f"  P/L: ${unrealized:,.2f} ({pct:+.3f}%)")

        if 'stopLossOrder' in trade:
            stop = float(trade['stopLossOrder']['price'])
            print(f"  Stop Loss: {stop}")

        if 'takeProfitOrder' in trade:
            target = float(trade['takeProfitOrder']['price'])
            print(f"  Take Profit: {target}")

print(f"\n{'='*60}\n")
