#!/usr/bin/env python3
"""
DAY 3 READINESS CHECK
====================
Run this before starting the scanner to verify everything is ready
"""

print("="*70)
print("DAY 3 READINESS CHECK - MULTI-STRATEGY MODE")
print("="*70)

# Check 1: Multi-strategy mode activated
print("\n[CHECK 1] Verifying scanner configuration...")
with open('week2_sp500_scanner.py', 'r') as f:
    content = f.read()
    if 'self.multi_strategy_mode = True' in content:
        print("  [OK] Multi-strategy mode: ACTIVATED")
    else:
        print("  [ERROR] Multi-strategy mode: NOT ACTIVATED")
        print("  → Edit line 66: self.multi_strategy_mode = True")

    if 'self.max_trades_per_day = 20' in content:
        print("  [OK] Max trades: 20/day")
    else:
        print("  [WARNING]  Max trades: May not be set to 20")

# Check 2: Strategy engines exist
print("\n[CHECK 2] Verifying strategy engines...")
import os
if os.path.exists('strategies/iron_condor_engine.py'):
    print("  [OK] Iron Condor engine: Found")
else:
    print("  [ERROR] Iron Condor engine: NOT FOUND")

if os.path.exists('strategies/butterfly_spread_engine.py'):
    print("  [OK] Butterfly engine: Found")
else:
    print("  [ERROR] Butterfly engine: NOT FOUND")

# Check 3: Account status
print("\n[CHECK 3] Checking account status...")
try:
    from week1_execution_system import Week1ExecutionSystem
    system = Week1ExecutionSystem()
    account = system.api.get_account()

    equity = float(account.equity)
    cash = float(account.cash)
    buying_power = float(account.buying_power)

    print(f"  Account Equity: ${equity:,.2f}")
    print(f"  Cash Available: ${cash:,.2f}")
    print(f"  Buying Power: ${buying_power:,.2f}")

    if buying_power > 50000:
        print("  [OK] Sufficient buying power for 20+ trades")
    else:
        print("  [WARNING]  Limited buying power - may not execute all 20 trades")

except Exception as e:
    print(f"  [ERROR] Could not check account: {e}")

# Check 4: Open positions
print("\n[CHECK 4] Reviewing open positions...")
try:
    positions = system.api.list_positions()
    print(f"  Open Positions: {len(positions)}")

    if len(positions) > 0:
        total_value = sum(float(p.market_value) for p in positions)
        print(f"  Total Position Value: ${total_value:,.2f}")

        for pos in positions:
            pnl_pct = float(pos.unrealized_plpc) * 100
            symbol = pos.symbol
            print(f"    - {symbol}: {pnl_pct:+.2f}%")
except Exception as e:
    print(f"  [ERROR] Could not check positions: {e}")

# Check 5: Time until market open
print("\n[CHECK 5] Time until market open...")
from datetime import datetime
import pytz

pdt = pytz.timezone('America/Los_Angeles')
now = datetime.now(pdt)
market_open = now.replace(hour=6, minute=30, second=0, microsecond=0)

if now < market_open:
    time_diff = (market_open - now).total_seconds()
    hours = int(time_diff // 3600)
    minutes = int((time_diff % 3600) // 60)
    print(f"  Market opens in: {hours}h {minutes}m")
    print(f"  [OK] Ready for pre-market preparation")
elif now.hour < 13:
    print(f"  🔔 MARKET IS OPEN - START SCANNER NOW!")
else:
    print(f"  Market is closed - opens tomorrow at 6:30 AM PDT")

# Summary
print("\n" + "="*70)
print("DAY 3 READINESS SUMMARY")
print("="*70)
print("\n[OK] = Ready")
print("[WARNING]  = Warning (may still work)")
print("[ERROR] = Critical issue (fix before starting)")

print("\n" + "="*70)
print("TO START SCANNER:")
print("="*70)
print("  python week2_sp500_scanner.py")
print("\nEXPECTED OUTPUT:")
print("  [OK] Advanced strategies loaded: Iron Condor, Butterfly")
print("  Max trades: 20 (starting conservative)")
print("\nDURING TRADES:")
print("  [STRATEGY] Iron Condor - Low momentum (2.1%), high probability")
print("  [OK] PAPER TRADE EXECUTED - IRON_CONDOR")
print("\n" + "="*70)
print("\n🎯 TARGET FOR DAY 3:")
print("  - Execute 15-20 Iron Condors")
print("  - Win rate: 70%+")
print("  - P&L: +1-2% ($1,000-2,000)")
print("  - PROVE THE STRATEGY WORKS")
print("\n" + "="*70)
print("\n💤 Get some sleep. Market opens in ~6 hours.")
print("🚀 Tomorrow changes everything.\n")
