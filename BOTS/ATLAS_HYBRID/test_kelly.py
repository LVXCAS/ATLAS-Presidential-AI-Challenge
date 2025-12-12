"""
Quick test to verify Kelly Criterion position sizing is working.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from live_trader import calculate_kelly_position_size

print("\n" + "="*80)
print("KELLY CRITERION POSITION SIZING TEST")
print("="*80 + "\n")

# Test with current balance
balance = 182999.16
kelly_fraction = 0.10  # 1/10 Kelly = 10% of optimal
stop_loss_pips = 14
symbol = "EUR_USD"
max_lots = 25.0
min_lots = 3.0

print(f"Test Parameters:")
print(f"  Balance: ${balance:,.2f}")
print(f"  Kelly Fraction: {kelly_fraction*100:.1f}% (1/{int(1/kelly_fraction)} Kelly)")
print(f"  Stop Loss: {stop_loss_pips} pips")
print(f"  Lot Range: {min_lots:.1f} - {max_lots:.1f} lots")
print(f"  Symbol: {symbol}\n")

print("-"*80)
print("Calculating position size...\n")

units = calculate_kelly_position_size(
    balance=balance,
    kelly_fraction=kelly_fraction,
    stop_loss_pips=stop_loss_pips,
    symbol=symbol,
    max_lots=max_lots,
    min_lots=min_lots
)

lots = units / 100000

print(f"\n" + "="*80)
print(f"RESULT: {units:,} units ({lots:.2f} lots)")
print(f"="*80)

# Calculate actual risk
pip_value = 10.0
risk_amount = lots * stop_loss_pips * pip_value
risk_pct = (risk_amount / balance) * 100

print(f"\nRisk Analysis:")
print(f"  Risk Amount: ${risk_amount:,.2f}")
print(f"  Risk %: {risk_pct:.2f}%")
print(f"  Leverage: {lots * 100000 / balance:.1f}x")

print("\nExpected Behavior:")
print(f"  ✓ Position size should be ~25 lots (capped at max_lots)")
print(f"  ✓ Risk should be ~3% of balance")
print(f"  ✓ Leverage should be ~14x")

if lots >= max_lots * 0.9:
    print(f"\n[OK] Kelly Criterion is working correctly! Position size at/near cap.")
else:
    print(f"\n[WARNING] Position size seems low. Check calculation logic.")

print("\n" + "="*80 + "\n")
