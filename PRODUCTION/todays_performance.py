#!/usr/bin/env python3
"""Calculate today's trading performance"""

from options_executor import AlpacaOptionsExecutor

executor = AlpacaOptionsExecutor()
positions = executor.get_positions()
account = executor.get_account_status()

print("=" * 60)
print("TODAY'S PERFORMANCE - WEDNESDAY OCT 1, 2025")
print("=" * 60)

print(f"\nAccount Summary:")
print(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")
print(f"  Cash: ${account['cash']:,.2f}")
print(f"  Buying Power: ${account['buying_power']:,.2f}")

print(f"\nPositions: {len(positions)}")
print("-" * 60)

total_value = 0
total_cost = 0
total_pl = 0

for pos in positions:
    value = float(pos.market_value)
    cost = float(pos.cost_basis)
    pl = float(pos.unrealized_pl)
    pl_pct = float(pos.unrealized_plpc) * 100

    total_value += value
    total_cost += abs(cost)
    total_pl += pl

    side = "LONG" if int(pos.qty) > 0 else "SHORT"
    print(f"{pos.symbol[:4]} {side}:")
    print(f"  Qty: {pos.qty} @ ${float(pos.avg_entry_price):.2f}")
    print(f"  Value: ${value:.2f} | P&L: ${pl:.2f} ({pl_pct:+.1f}%)")
    print()

print("=" * 60)
print("TOTAL PERFORMANCE")
print("=" * 60)

print(f"\nCapital Deployed: ${total_cost:,.2f}")
print(f"Current Value: ${total_value:,.2f}")
print(f"Total P&L: ${total_pl:,.2f}")

pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0
account_pct = (total_pl / 100000 * 100)

print(f"\nReturns:")
print(f"  On Deployed Capital: {pl_pct:+.2f}%")
print(f"  On Total Account: {account_pct:+.3f}%")

print(f"\nWeek 1 Assessment:")
if total_pl > 0:
    print(f"  Status: PROFITABLE (+${total_pl:.2f})")
    print(f"  Rating: EXCELLENT")
elif total_pl > -100:
    print(f"  Status: Small loss (${total_pl:.2f})")
    print(f"  Rating: ACCEPTABLE (early positions)")
else:
    print(f"  Status: Loss (${total_pl:.2f})")
    print(f"  Rating: NEEDS REVIEW")

print("\n" + "=" * 60)

# Find best and worst performers
if positions:
    best = max(positions, key=lambda p: float(p.unrealized_plpc))
    worst = min(positions, key=lambda p: float(p.unrealized_plpc))

    print("Best Performer:")
    print(f"  {best.symbol[:4]}: {float(best.unrealized_plpc)*100:+.1f}% (${float(best.unrealized_pl):+.2f})")

    print("\nWorst Performer:")
    print(f"  {worst.symbol[:4]}: {float(worst.unrealized_plpc)*100:+.1f}% (${float(worst.unrealized_pl):+.2f})")

print("\n" + "=" * 60)
