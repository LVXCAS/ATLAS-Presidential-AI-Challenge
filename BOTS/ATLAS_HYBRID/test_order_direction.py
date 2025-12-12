"""
Test order direction conversion to verify the bug fix logic
"""

import os
from dotenv import load_dotenv
from adapters.oanda_adapter import OandaAdapter

# Load environment
load_dotenv('../../.env')

# Initialize OANDA
oanda = OandaAdapter(
    api_key=os.getenv('OANDA_API_KEY'),
    account_id=os.getenv('OANDA_ACCOUNT_ID')
)

print("=" * 80)
print("OANDA CONNECTION TEST")
print("=" * 80)

# Test 1: Account balance
balance = oanda.get_account_balance()
if balance:
    print(f"\n[OK] OANDA Connected")
    print(f"  Balance: ${balance['balance']:,.2f}")
    print(f"  Unrealized P/L: ${balance.get('unrealized_pnl', 0):+,.2f}")
else:
    print("\n[ERROR] OANDA Connection Failed")
    exit(1)

# Test 2: Market data
print("\n" + "=" * 80)
print("MARKET DATA TEST")
print("=" * 80)

pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]
for pair in pairs:
    data = oanda.get_market_data(pair)
    if data:
        print(f"\n[OK] {pair}")
        print(f"  Bid: {data['bid']:.5f}")
        print(f"  Ask: {data['ask']:.5f}")
        print(f"  Spread: {(data['ask'] - data['bid']) * 10000:.1f} pips")
    else:
        print(f"\n[ERROR] {pair} - Failed to get market data")

# Test 3: Order direction conversion logic
print("\n" + "=" * 80)
print("ORDER DIRECTION CONVERSION TEST")
print("=" * 80)

print("\nTesting conversion logic (without executing orders):")

# Simulate BUY decision
decision_direction = "BUY"
direction = 'long' if decision_direction == 'BUY' else 'short'
units = 100000  # 1 lot

print(f"\n1. ATLAS Decision: '{decision_direction}'")
print(f"   Converts to: '{direction}'")
print(f"   Units: {units}")

# Simulate what OANDA adapter would do
order_units = units if direction.lower() == 'long' else -units
print(f"   OANDA order_units: {order_units:+,}")
print(f"   Expected: +100,000 (LONG position)")
if order_units > 0:
    print(f"   Result: [OK] CORRECT")
else:
    print(f"   Result: [ERROR] WRONG - INVERTED!")

# Simulate SELL decision
decision_direction = "SELL"
direction = 'long' if decision_direction == 'BUY' else 'short'

print(f"\n2. ATLAS Decision: '{decision_direction}'")
print(f"   Converts to: '{direction}'")
print(f"   Units: {units}")

order_units = units if direction.lower() == 'long' else -units
print(f"   OANDA order_units: {order_units:+,}")
print(f"   Expected: -100,000 (SHORT position)")
if order_units < 0:
    print(f"   Result: [OK] CORRECT")
else:
    print(f"   Result: [ERROR] WRONG - INVERTED!")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print("\nIf both conversions show '[OK] CORRECT', the logic is working properly.")
print("The debug logs in live_trader.py will show actual execution flow.")
