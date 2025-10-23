"""
Query Alpaca Options Chain API to get REAL valid option contracts
"""
import requests
from datetime import datetime

API_KEY = 'PKOWU7D6JANXP47ZU72X72757D'
API_SECRET = '52QKewJCoafjLsFPKJJSTZs7BG7XBa6mLwi3e1W3Z7Tq'

headers = {
    'APCA-API-KEY-ID': API_KEY,
    'APCA-API-SECRET-KEY': API_SECRET
}

print("=" * 70)
print("QUERYING ALPACA OPTIONS CHAIN")
print("=" * 70)

# Query options for SPY (most liquid options in the world)
print("\n[1] Getting SPY option contracts...")

# Alpaca Options API endpoint
url = "https://data.alpaca.markets/v1beta1/options/contracts"
params = {
    'underlying_symbols': 'SPY',
    'expiration_date_gte': datetime.now().strftime('%Y-%m-%d'),
    'limit': 10
}

response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    data = response.json()
    contracts = data.get('option_contracts', [])

    print(f"  [OK] Found {len(contracts)} SPY option contracts")
    print("\n[SAMPLE CONTRACTS]")

    for i, contract in enumerate(contracts[:5], 1):
        symbol = contract.get('symbol')
        strike = contract.get('strike_price')
        expiry = contract.get('expiration_date')
        option_type = contract.get('type')

        print(f"\n  Contract #{i}:")
        print(f"    Symbol: {symbol}")
        print(f"    Type: {option_type}")
        print(f"    Strike: ${strike}")
        print(f"    Expiry: {expiry}")

        # Save first valid symbol for testing
        if i == 1:
            test_symbol = symbol

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Valid option symbol format: {test_symbol}")
    print("=" * 70)

    # Now test placing an order with this valid symbol
    print("\n[2] Testing order placement with valid symbol...")

    order_data = {
        'symbol': test_symbol,
        'qty': 1,
        'side': 'buy',
        'type': 'limit',
        'limit_price': 1.00,  # $1.00 per contract
        'time_in_force': 'day'
    }

    order_url = "https://paper-api.alpaca.markets/v2/orders"
    order_response = requests.post(order_url, headers=headers, json=order_data)

    print(f"  Order Symbol: {test_symbol}")
    print(f"  Order Type: BUY 1 @ $1.00 limit")
    print(f"  HTTP Status: {order_response.status_code}")

    if order_response.status_code in [200, 201]:
        order = order_response.json()
        print(f"  [SUCCESS] Order placed!")
        print(f"  Order ID: {order.get('id')}")
        print(f"  Status: {order.get('status')}")
    else:
        print(f"  [INFO] Order response: {order_response.text[:200]}")

else:
    print(f"  [ERROR] HTTP {response.status_code}")
    print(f"  Response: {response.text}")
