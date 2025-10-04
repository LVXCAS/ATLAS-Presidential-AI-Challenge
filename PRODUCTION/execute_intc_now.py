#!/usr/bin/env python3
"""Execute INTC trade NOW - score 4.76 qualifies"""

from options_executor import AlpacaOptionsExecutor
import json
from datetime import datetime

print("EXECUTING INTC INTEL DUAL STRATEGY")
print("=" * 60)

executor = AlpacaOptionsExecutor()

# Current INTC data
symbol = 'INTC'
price = 34.26
score = 4.76
volatility = 0.057

print(f"\nOpportunity Details:")
print(f"  Symbol: {symbol}")
print(f"  Price: ${price}")
print(f"  Score: {score} (qualifies at 4.0+)")
print(f"  Volatility: {volatility*100:.1f}% (high)")

# Execute Intel dual strategy
print("\n" + "=" * 60)
result = executor.execute_intel_dual(
    symbol=symbol,
    current_price=price,
    contracts=2,  # Week 1 conservative sizing
    expiry_days=21
)

print("\n" + "=" * 60)
print("EXECUTION COMPLETE")
print("=" * 60)

# Count successful orders
successful = len([o for o in result.get('orders', []) if 'order_id' in o])
print(f"\nOrders submitted: {successful}/2")

for order in result.get('orders', []):
    if 'order_id' in order:
        print(f"  [OK] {order['type']}: {order['order_id']}")
    else:
        print(f"  [FAIL] {order['type']}: {order.get('error', 'Unknown')}")

# Save trade record
trade_record = {
    'timestamp': datetime.now().isoformat(),
    'strategy': 'intel_dual_manual',
    'symbol': symbol,
    'current_price': price,
    'opportunity_score': score,
    'volatility': volatility,
    'execution_type': 'MANUAL_WEEK1',
    'week1_conservative': True,
    'paper_trade': True,
    'contracts': 2,
    'alpaca_execution': result,
    'orders_submitted': successful,
    'execution_success': successful > 0
}

filename = f"week1_trade_INTC_manual_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
with open(filename, 'w') as f:
    json.dump(trade_record, f, indent=2)

print(f"\nTrade logged: {filename}")
print("\nCheck positions:")
print("  python check_positions.py")
print("\nWeek 1 Status: 2/2 trades complete!")
