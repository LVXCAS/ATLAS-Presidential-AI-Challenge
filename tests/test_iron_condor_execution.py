#!/usr/bin/env python3
"""
Test why Iron Condors aren't executing in scanner
"""

import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from strategies.iron_condor_engine import IronCondorEngine
from time_series_momentum_strategy import TimeSeriesMomentumStrategy

load_dotenv()

print("="*70)
print("TESTING IRON CONDOR EXECUTION - DIAGNOSTIC")
print("="*70)

# Initialize
api = tradeapi.REST(
    key_id=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    base_url=os.getenv('ALPACA_BASE_URL'),
    api_version='v2'
)

iron_condor_engine = IronCondorEngine()
momentum_strategy = TimeSeriesMomentumStrategy()

# Test on MO (0.02% momentum - perfect Iron Condor candidate)
test_symbol = 'MO'

print(f"\nStep 1: Check {test_symbol} momentum...")
momentum = momentum_strategy.calculate_momentum_signal(test_symbol, lookback_days=21)
if momentum:
    mom_pct = abs(momentum['momentum'])
    print(f"  Momentum: {mom_pct:.2%}")

    if mom_pct < 0.03:
        print(f"  [OK] Qualifies for Iron Condor (< 3%)")
    else:
        print(f"  [FAIL] Too much momentum for Iron Condor")
        exit(1)
else:
    print(f"  [FAIL] Could not calculate momentum")
    exit(1)

print(f"\nStep 2: Get current price...")
try:
    bars = api.get_bars(test_symbol, '1Day', limit=1).df
    current_price = float(bars['close'].iloc[-1])
    print(f"  Current price: ${current_price:.2f}")
except Exception as e:
    print(f"  [FAIL] Could not get price: {e}")
    exit(1)

print(f"\nStep 3: Check buying power...")
try:
    account = api.get_account()
    buying_power = float(account.buying_power)
    print(f"  Buying power: ${buying_power:,.2f}")

    if buying_power < 1000:
        print(f"  [FAIL] Insufficient buying power")
        exit(1)
    else:
        print(f"  [OK] Sufficient buying power")
except Exception as e:
    print(f"  [FAIL] Could not check account: {e}")
    exit(1)

print(f"\nStep 4: Execute Iron Condor...")
try:
    result = iron_condor_engine.execute_iron_condor(
        symbol=test_symbol,
        current_price=current_price,
        contracts=1,
        expiration_days=7
    )

    if result['success']:
        print(f"\n[SUCCESS] Iron Condor executed successfully!")
        print(f"  Orders submitted: {len(result['orders'])}")
    else:
        print(f"\n[FAILED] Iron Condor execution failed")
        print(f"  Error: {result.get('error', 'Unknown')}")

except Exception as e:
    print(f"\n[FAILED] Exception during execution: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
