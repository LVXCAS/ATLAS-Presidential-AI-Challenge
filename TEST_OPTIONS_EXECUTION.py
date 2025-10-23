"""
SIMPLE OPTIONS EXECUTION TEST
Debug why butterfly trades aren't executing
"""
import os
import requests
from datetime import datetime, timedelta

# API Keys
API_KEY = 'PKOWU7D6JANXP47ZU72X72757D'
API_SECRET = '52QKewJCoafjLsFPKJJSTZs7BG7XBa6mLwi3e1W3Z7Tq'

print("=" * 70)
print("OPTIONS EXECUTION DIAGNOSTIC TEST")
print("=" * 70)

# Test 1: Check account info
print("\n[TEST 1] Checking Alpaca account...")
url = "https://paper-api.alpaca.markets/v2/account"
headers = {
    'APCA-API-KEY-ID': API_KEY,
    'APCA-API-SECRET-KEY': API_SECRET
}

response = requests.get(url, headers=headers)
if response.status_code == 200:
    account = response.json()
    print(f"  [OK] Account Status: {account['status']}")
    print(f"  [OK] Equity: ${float(account['equity']):,.2f}")
    print(f"  [OK] Options Trading: {account.get('options_trading_level', 'NOT FOUND')}")
else:
    print(f"  [ERROR] HTTP {response.status_code}: {response.text}")
    exit(1)

# Test 2: Try to place a simple butterfly option trade
print("\n[TEST 2] Attempting butterfly option trade...")

# Use PM (Philip Morris) - Score 13.3 from scanner
symbol = "PM"
current_price = 146.34

# Get next Friday expiry
today = datetime.now()
days_until_friday = (4 - today.weekday()) % 7
if days_until_friday == 0:
    days_until_friday = 7
expiry = today + timedelta(days=days_until_friday)
expiry_str = expiry.strftime('%Y%m%d')

# Butterfly: ATM call (simplified)
strike = round(current_price)
option_symbol = f"{symbol}{expiry_str}C{strike:05d}000"

print(f"  Symbol: {option_symbol}")
print(f"  Breakdown: {symbol} + {expiry_str} + C + {strike:05d}000")
print(f"  Expiry Date: {expiry.strftime('%Y-%m-%d')} (Next Friday)")

order_data = {
    'symbol': option_symbol,
    'qty': 1,
    'side': 'buy',
    'type': 'market',
    'time_in_force': 'day'
}

print(f"  Order: BUY 1 {option_symbol} @ market")

url = "https://paper-api.alpaca.markets/v2/orders"
response = requests.post(url, headers=headers, json=order_data, timeout=10)

print(f"\n[RESPONSE] HTTP {response.status_code}")
if response.status_code in [200, 201]:
    order = response.json()
    print(f"  [SUCCESS] Order placed!")
    print(f"  Order ID: {order.get('id')}")
    print(f"  Status: {order.get('status')}")
else:
    print(f"  [FAILED] Order rejected!")
    print(f"  Response: {response.text}")

    # Parse error
    try:
        error = response.json()
        print(f"\n  Error Code: {error.get('code')}")
        print(f"  Error Message: {error.get('message')}")
    except:
        pass

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
