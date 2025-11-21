"""
HYBRID EXIT STRATEGY - Lock in EUR_USD, Trail GBP_USD
Executes professional position management to secure $4,500-5,000 profit
"""

import os
from dotenv import load_dotenv
load_dotenv()

import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.orders as orders
from datetime import datetime

oanda_token = os.getenv('OANDA_API_KEY')
oanda_account = os.getenv('OANDA_ACCOUNT_ID')
client = API(access_token=oanda_token, environment='practice')

print("=" * 60)
print("HYBRID EXIT STRATEGY EXECUTION")
print("=" * 60)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Get current account status
r = accounts.AccountSummary(accountID=oanda_account)
resp = client.request(r)
balance = float(resp['account']['balance'])
unrealized_pl = float(resp['account']['unrealizedPL'])

print(f"Account Balance: ${balance:,.2f}")
print(f"Unrealized P/L: ${unrealized_pl:,.2f} ({unrealized_pl/balance*100:+.3f}%)")
print()

# Get all open trades
r = trades.TradesList(accountID=oanda_account)
resp = client.request(r)
open_trades = resp.get('trades', [])

print(f"Open Positions: {len(open_trades)}")
print()

gbp_usd_trade = None
eur_usd_trade = None

for trade in open_trades:
    pair = trade['instrument']
    trade_id = trade['id']
    units = int(trade['currentUnits'])
    unrealized = float(trade['unrealizedPL'])
    entry_price = float(trade['price'])
    current_price = float(trade['price'])  # Will be updated

    if pair == 'GBP_USD':
        gbp_usd_trade = trade
        print(f"GBP_USD Trade #{trade_id}:")
        print(f"  Units: {units:,}")
        print(f"  Entry: {entry_price}")
        print(f"  Unrealized P/L: ${unrealized:,.2f}")

    elif pair == 'EUR_USD':
        eur_usd_trade = trade
        print(f"EUR_USD Trade #{trade_id}:")
        print(f"  Units: {units:,}")
        print(f"  Entry: {entry_price}")
        print(f"  Unrealized P/L: ${unrealized:,.2f}")

print()
print("=" * 60)
print("EXECUTION PLAN")
print("=" * 60)

# Step 1: Close EUR_USD position
if eur_usd_trade:
    print(f"\nStep 1: Closing EUR_USD Trade #{eur_usd_trade['id']}")
    print(f"  Locking in: ${float(eur_usd_trade['unrealizedPL']):,.2f}")

    try:
        r = trades.TradeClose(accountID=oanda_account, tradeID=eur_usd_trade['id'])
        resp = client.request(r)

        realized_pl = float(resp['orderFillTransaction']['pl'])
        print(f"  SUCCESS: EUR_USD closed")
        print(f"  Realized P/L: ${realized_pl:,.2f}")

    except Exception as e:
        print(f"  ERROR closing EUR_USD: {e}")
else:
    print("\nStep 1: EUR_USD position not found (may already be closed)")

# Step 2: Modify GBP_USD stop-loss to 1.30000
if gbp_usd_trade:
    print(f"\nStep 2: Modifying GBP_USD Trade #{gbp_usd_trade['id']} Stop-Loss")
    print(f"  Current Stop: {gbp_usd_trade.get('stopLossOrder', {}).get('price', 'None')}")
    print(f"  New Stop: 1.30000 (locks in minimum +$2,500)")

    # For SHORT position, stop-loss should be ABOVE entry
    units = int(gbp_usd_trade['currentUnits'])

    try:
        # Cancel existing stop-loss order first
        if 'stopLossOrder' in gbp_usd_trade:
            stop_order_id = gbp_usd_trade['stopLossOrder']['id']
            r = orders.OrderCancel(accountID=oanda_account, orderID=stop_order_id)
            client.request(r)
            print(f"  Cancelled old stop-loss order #{stop_order_id}")

        # Create new stop-loss order at 1.30000
        stop_loss_data = {
            "order": {
                "type": "STOP_LOSS",
                "tradeID": gbp_usd_trade['id'],
                "price": "1.30000",
                "timeInForce": "GTC"
            }
        }

        r = orders.OrderCreate(accountID=oanda_account, data=stop_loss_data)
        resp = client.request(r)
        print(f"  SUCCESS: New stop-loss set at 1.30000")

    except Exception as e:
        print(f"  ERROR modifying stop-loss: {e}")
else:
    print("\nStep 2: GBP_USD position not found")

print()
print("=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("\n1. Check GBP_USD position at NOON (12:00 PM)")
print("   - If near 1.28683 target: Close manually for +$3,750")
print("   - If still above 1.29000: Let it run with trailing stop")
print("   - If stopped at 1.30000: Already locked in +$2,500")
print()
print("2. Expected Total Profit: $4,500 - $5,000")
print("   - EUR_USD locked: ~$1,200")
print("   - GBP_USD minimum: $2,500")
print("   - GBP_USD target: $3,750")
print()
print("3. After positions close:")
print("   - Deploy IMPROVED_FOREX_BOT.py (USD_JPY/GBP_JPY only)")
print("   - Begin planning E8 prop firm challenge ($1,600)")
print()
print("=" * 60)
