#!/usr/bin/env python3
"""Quick test - submit one real options trade to see if it works"""

from options_executor import AlpacaOptionsExecutor

print("TESTING REAL OPTIONS EXECUTION")
print("=" * 60)

executor = AlpacaOptionsExecutor()

# Check account first
print("\n1. Checking account status...")
account = executor.get_account_status()
print(f"   Buying Power: ${account['buying_power']:,.2f}")
print(f"   Options Level: {account['options_level']}")
print(f"   [OK] Account ready")

# Check current positions
print("\n2. Checking current positions...")
positions = executor.get_positions()
print(f"   Current positions: {len(positions)}")
if positions:
    for pos in positions:
        print(f"     - {pos.symbol}: {pos.qty} @ ${pos.avg_entry_price}")
else:
    print("   No positions (ready to trade)")

# Try to submit a test trade
print("\n3. Testing with AAPL straddle...")
print("   (This will submit REAL paper trading orders)")

try:
    result = executor.execute_straddle(
        symbol='AAPL',
        current_price=256.0,
        contracts=1,
        expiry_days=7  # Next week expiry
    )

    print("\n4. Results:")
    print(f"   Strategy: {result['strategy']}")
    print(f"   Orders submitted: {len(result.get('orders', []))}")

    for order in result.get('orders', []):
        if 'order_id' in order:
            print(f"   [OK] {order['type']}: {order['order_id']}")
        else:
            print(f"   [FAIL] {order['type']}: {order.get('error', 'Unknown error')}")

    print("\n5. Check your Alpaca dashboard now:")
    print("   https://app.alpaca.markets/paper/dashboard/overview")
    print("   You should see AAPL option positions")

except Exception as e:
    print(f"\n[ERROR]: {e}")
    print("\nThis might mean:")
    print("  - Options not enabled on paper account")
    print("  - API credentials issue")
    print("  - No option contracts available")

print("\n" + "=" * 60)
