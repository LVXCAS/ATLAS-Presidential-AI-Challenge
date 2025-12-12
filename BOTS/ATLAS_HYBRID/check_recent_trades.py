"""Check recent OANDA transactions to verify order direction"""

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

# Fetch recent transactions (use sinceID to get latest)
url = f'https://api-fxpractice.oanda.com/v3/accounts/{oanda.account_id}/transactions/sinceid'
headers = {'Authorization': f'Bearer {oanda.api_key}'}
params = {'id': 900}  # Get transactions after ID 900 (last ~68 transactions)

response = requests.get(url, headers=headers, params=params)
data = response.json()

print("=" * 80)
print("RECENT OANDA TRANSACTIONS")
print("=" * 80)

if 'transactions' in data:
    transactions = data['transactions']

    # Filter for ORDER_FILL transactions (actual trades)
    trades = [t for t in transactions if t.get('type') == 'ORDER_FILL']

    if trades:
        print(f"\nFound {len(trades)} recent trades:\n")
        for t in reversed(trades[-10:]):  # Last 10 trades
            time = t.get('time', 'N/A')[:19]
            instrument = t.get('instrument', 'N/A')
            units = float(t.get('units', 0))
            price = t.get('price', 'N/A')
            reason = t.get('reason', 'N/A')

            # Determine direction
            direction = 'LONG' if units > 0 else 'SHORT'

            print(f"{time} | {instrument:10} | {direction:6} | Units: {units:>8.0f} | Price: {price} | {reason}")
    else:
        print("\nNo ORDER_FILL transactions found in last 50 transactions")

    # Also show MARKET_ORDER transactions
    market_orders = [t for t in transactions if t.get('type') == 'MARKET_ORDER']
    if market_orders:
        print(f"\n\nMarket Orders ({len(market_orders)} found):\n")
        for t in reversed(market_orders[-10:]):
            time = t.get('time', 'N/A')[:19]
            instrument = t.get('instrument', 'N/A')
            units = float(t.get('units', 0))
            reason = t.get('reason', 'N/A')

            direction = 'LONG' if units > 0 else 'SHORT'

            print(f"{time} | {instrument:10} | {direction:6} | Units: {units:>8.0f} | {reason}")

else:
    print("Error fetching transactions:", data)

print("\n" + "=" * 80)
