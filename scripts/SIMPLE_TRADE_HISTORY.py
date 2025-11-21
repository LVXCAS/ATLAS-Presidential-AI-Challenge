"""
SIMPLE TRADE HISTORY - Get all closed positions from OANDA
"""
import os
import json
import oandapyV20
from oandapyV20 import API
from oandapyV20.endpoints import accounts
from datetime import datetime

# Load credentials
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

token = os.getenv('OANDA_API_KEY')
account_id = os.getenv('OANDA_ACCOUNT_ID', '101-004-29159709-001')

client = API(access_token=token, environment='practice')

print('='*80)
print('FOREX TRADING HISTORY - OANDA PRACTICE ACCOUNT')
print('='*80)

# Get account details
r = accounts.AccountDetails(accountID=account_id)
response = client.request(r)
account = response['account']

print(f"\nAccount ID: {account['id']}")
print(f"Currency: {account['currency']}")
print(f"Balance: ${float(account['balance']):,.2f}")
print(f"Total P/L (All Time): ${float(account['pl']):,.2f}")
print(f"Open Trades: {account['openTradeCount']}")
print(f"Last Transaction ID: {account['lastTransactionID']}")

# Get individual transaction details
print(f"\n{'='*80}")
print("Fetching recent transactions...")
print('='*80)

# Use the account changes endpoint
from oandapyV20.endpoints.accounts import AccountChanges

# Get changes from transaction ID 1 (beginning)
params = {'sinceTransactionID': '1'}
r = AccountChanges(accountID=account_id, params=params)

try:
    response = client.request(r)
    changes = response['changes']

    print(f"\nFound transaction changes:")
    print(f"  Trades closed: {len(changes.get('trades', []))}")
    print(f"  Positions: {len(changes.get('positions', []))}")
    print(f"  Orders created: {len(changes.get('orders', []))}")

    # Save full response for analysis
    with open('account_changes.json', 'w') as f:
        json.dump(response, f, indent=2)
    print(f"\nFull account changes saved to: account_changes.json")

except Exception as e:
    print(f"Error fetching changes: {e}")

# Try to get transaction stream
print(f"\n{'='*80}")
print("Attempting to get transaction stream...")
print('='*80)

from oandapyV20.endpoints.transactions import TransactionsSinceID

params = {'id': '1'}  # Start from transaction 1
r = TransactionsSinceID(accountID=account_id, params=params)

try:
    response = client.request(r)

    transactions = response.get('transactions', [])
    print(f"\nFound {len(transactions)} transactions")

    # Filter by type
    order_fills = [t for t in transactions if t.get('type') == 'ORDER_FILL']
    market_orders = [t for t in transactions if t.get('type') == 'MARKET_ORDER']

    print(f"  Market orders: {len(market_orders)}")
    print(f"  Order fills: {len(order_fills)}")

    # Show recent order fills
    if order_fills:
        print(f"\n{'='*80}")
        print("RECENT ORDER FILLS (Last 10)")
        print('='*80)

        for fill in order_fills[-10:]:
            time_str = datetime.strptime(fill['time'][:19], '%Y-%m-%dT%H:%M:%S').strftime('%b %d %I:%M%p')
            instrument = fill.get('instrument', 'UNKNOWN')
            units = float(fill.get('units', 0))
            direction = 'LONG' if units > 0 else 'SHORT'
            price = float(fill.get('price', 0))
            pl = float(fill.get('pl', 0))

            print(f"\n{fill['id']}: {instrument} {direction}")
            print(f"  Time: {time_str}")
            print(f"  Price: {price:.5f}")
            print(f"  Units: {abs(units):,.0f}")
            if pl != 0:
                print(f"  P/L: ${pl:,.2f}")

    # Save full transaction list
    with open('all_transactions.json', 'w') as f:
        json.dump(response, f, indent=2)
    print(f"\n\nFull transaction list saved to: all_transactions.json")

except Exception as e:
    print(f"Error: {e}")

print('\n' + '='*80)
print("Trade history extraction complete")
print('='*80)
