"""Check what order flow data is available from OANDA"""

import os
from dotenv import load_dotenv
from adapters.oanda_adapter import OandaAdapter
import requests

# Load environment
load_dotenv('../../.env')

# Initialize OANDA
oanda = OandaAdapter(
    api_key=os.getenv('OANDA_API_KEY'),
    account_id=os.getenv('OANDA_ACCOUNT_ID')
)

print("=" * 80)
print("OANDA ORDER FLOW & DATA ACCESS CHECK")
print("=" * 80)

# 1. Check current open positions
print("\n1. OPEN POSITIONS:")
print("-" * 80)
positions = oanda.get_open_positions()
if positions:
    for pos in positions:
        print(f"  Instrument: {pos.get('instrument')}")
        print(f"  Type: {pos.get('type')}")
        print(f"  Units: {pos.get('units')}")
        print(f"  Unrealized P/L: ${pos.get('unrealized_pnl', 0):,.2f}")
        print(f"  Avg Price: {pos.get('avg_price')}")
        print()
else:
    print("  No open positions")

# 2. Check recent orders
print("\n2. RECENT ORDERS:")
print("-" * 80)
url = f'https://api-fxpractice.oanda.com/v3/accounts/{oanda.account_id}/orders'
headers = {'Authorization': f'Bearer {oanda.api_key}'}

response = requests.get(url, headers=headers)
data = response.json()

if 'orders' in data:
    orders = data['orders']
    if orders:
        for order in orders[:5]:  # Last 5 orders
            print(f"  Order ID: {order.get('id')}")
            print(f"  Type: {order.get('type')}")
            print(f"  Instrument: {order.get('instrument')}")
            print(f"  Units: {order.get('units')}")
            print(f"  State: {order.get('state')}")
            print()
    else:
        print("  No pending orders")
else:
    print(f"  Error: {data}")

# 3. Check recent trades (last 10)
print("\n3. RECENT TRADES (Last 10):")
print("-" * 80)
url = f'https://api-fxpractice.oanda.com/v3/accounts/{oanda.account_id}/trades'
headers = {'Authorization': f'Bearer {oanda.api_key}'}
params = {'count': 10}

response = requests.get(url, headers=headers, params=params)
data = response.json()

if 'trades' in data:
    trades = data['trades']
    if trades:
        for trade in trades:
            print(f"  Trade ID: {trade.get('id')}")
            print(f"  Instrument: {trade.get('instrument')}")
            print(f"  Current Units: {trade.get('currentUnits')} ({'LONG' if float(trade.get('currentUnits', 0)) > 0 else 'SHORT'})")
            print(f"  Initial Units: {trade.get('initialUnits')}")
            print(f"  Price: {trade.get('price')}")
            print(f"  Unrealized P/L: ${float(trade.get('unrealizedPL', 0)):,.2f}")
            print(f"  State: {trade.get('state')}")
            print()
    else:
        print("  No open trades")
else:
    print(f"  No open trades")

# 4. Check recent transactions (actual fills)
print("\n4. RECENT ORDER FILLS (Last 5):")
print("-" * 80)
url = f'https://api-fxpractice.oanda.com/v3/accounts/{oanda.account_id}/transactions/sinceid'
headers = {'Authorization': f'Bearer {oanda.api_key}'}
params = {'id': 900}  # Get recent transactions

response = requests.get(url, headers=headers, params=params)
data = response.json()

if 'transactions' in data:
    fills = [t for t in data['transactions'] if t.get('type') == 'ORDER_FILL']
    if fills:
        print(f"  Found {len(fills)} order fills\n")
        for fill in reversed(fills[-5:]):  # Last 5
            units = float(fill.get('units', 0))
            direction = 'LONG' if units > 0 else 'SHORT'

            print(f"  Time: {fill.get('time', 'N/A')[:19]}")
            print(f"  Instrument: {fill.get('instrument')}")
            print(f"  Direction: {direction}")
            print(f"  Units: {units:+,.0f}")
            print(f"  Price: {fill.get('price')}")
            print(f"  Reason: {fill.get('reason')}")
            print(f"  Transaction ID: {fill.get('id')}")
            print()
    else:
        print("  No recent order fills")

# 5. Check order book (if available)
print("\n5. ORDER BOOK DATA:")
print("-" * 80)
print("  OANDA practice accounts have access to:")
print("  - Trade history (filled orders)")
print("  - Current positions")
print("  - Pending orders")
print("  - Transaction log")
print("\n  NOT available on practice accounts:")
print("  - Level 2 order book (market depth)")
print("  - Other traders' orders")
print("  - Institutional order flow")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nWe can track:")
print("  [OK] Our own order history (direction, units, fills)")
print("  [OK] Current open positions (long/short, units)")
print("  [OK] Transaction log (what was executed)")
print("  [OK] Price data (bid/ask/spread)")
print("\nThis is enough to verify if BUY orders execute as SHORT!")
print("=" * 80)
