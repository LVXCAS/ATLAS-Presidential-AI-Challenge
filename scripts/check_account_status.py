from HYBRID_OANDA_TRADELOCKER import HybridAdapter

adapter = HybridAdapter()
summary = adapter.get_account_summary()

print("=" * 70)
print("E8 ACCOUNT STATUS")
print("=" * 70)
print(f"Balance: ${summary['balance']:,.2f}")
print(f"Equity (NAV): ${summary['NAV']:,.2f}")
print(f"Unrealized P/L: ${summary['unrealizedPL']:,.2f}")
print()

# Calculate challenge metrics
starting_balance = 200000
current_equity = summary['NAV']
profit = current_equity - starting_balance
profit_pct = (profit / starting_balance) * 100

# Calculate peak balance and trailing drawdown
peak_balance = 208163  # $200k + $8,163 previous wins
trailing_dd = (peak_balance - current_equity) / peak_balance * 100

print("CHALLENGE METRICS:")
print(f"  Starting Balance: ${starting_balance:,.2f}")
print(f"  Current Equity: ${current_equity:,.2f}")
print(f"  Total Profit/Loss: ${profit:,.2f} ({profit_pct:.2f}%)")
print(f"  Peak Balance: ${peak_balance:,.2f}")
print(f"  Trailing Drawdown: {trailing_dd:.2f}% / 6.00% max")
print()

# Check status
if trailing_dd >= 6.0:
    print("❌ CHALLENGE FAILED - Trailing drawdown exceeded 6%")
elif profit_pct >= 10.0:
    print("✅ CHALLENGE PASSED - Made 10% profit!")
else:
    remaining_profit = starting_balance * 0.10 - profit
    print(f"⏳ IN PROGRESS - Need ${remaining_profit:,.2f} more to pass")
print("=" * 70)
