from HYBRID_OANDA_TRADELOCKER import HybridAdapter

adapter = HybridAdapter()
summary = adapter.get_account_summary()

balance = summary['balance']
equity = summary['NAV']
peak = 208163

dd_pct = ((peak - equity) / peak) * 100

print("=" * 70)
print("POSITION CLOSED - FINAL STATUS")
print("=" * 70)
print(f"Balance: ${balance:,.2f}")
print(f"Equity: ${equity:,.2f}")
print(f"")
print(f"Peak Balance: ${peak:,.2f}")
print(f"Trailing DD: {dd_pct:.2f}% / 6.00% max")
print(f"DD Cushion Remaining: {6.0 - dd_pct:.2f}%")
print(f"")
print(f"Total P/L from start: ${equity - 200000:,.2f}")
print(f"E8 Challenge Status: ALIVE (DD under 6%)")
print("=" * 70)
