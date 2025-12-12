"""
Test Trailing Stop Implementation

This script tests the trailing stop functionality without placing real trades.
"""

import sys
sys.path.insert(0, 'adapters')
from oanda_adapter import OandaAdapter

print("="*70)
print("TRAILING STOP TEST")
print("="*70)

oanda = OandaAdapter()

# Get current positions (if any)
positions = oanda.get_open_positions()

print(f"\nOpen Positions: {len(positions)}")

if positions:
    for pos in positions:
        print(f"\nPosition: {pos['instrument']}")
        print(f"  Type: {pos['type']}")
        print(f"  Units: {pos['units']:,.0f}")
        print(f"  Entry: {pos['avg_price']:.5f}")
        print(f"  P/L: ${pos['unrealized_pnl']:+,.2f}")
        print(f"  Trade ID: {pos.get('trade_id', 'NONE')}")

        # Test trailing stop update if in profit
        if pos['unrealized_pnl'] > 0 and pos.get('trade_id'):
            print(f"\n[TEST] Updating trailing stop...")
            result = oanda.update_trailing_stop(
                trade_id=pos['trade_id'],
                trailing_distance_pips=14,
                symbol=pos['instrument']
            )

            if result:
                print(f"[OK] Trailing stop updated successfully")
            else:
                print(f"[FAIL] Could not update trailing stop")
        elif pos['unrealized_pnl'] <= 0:
            print(f"\n[SKIP] Position not in profit, trailing stop not applicable")
        else:
            print(f"\n[SKIP] No trade ID available")
else:
    print("\nNo open positions to test")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print("\nIf you saw '[OK] Trailing stop updated successfully', the system is")
print("ready to protect your profits automatically on every scan!")
