from options_executor import AlpacaOptionsExecutor

e = AlpacaOptionsExecutor()
pos = e.get_positions()

print(f"CURRENT POSITIONS: {len(pos)}")
for p in pos:
    print(f"  {p.symbol}: {p.qty} @ ${p.avg_entry_price}")
    print(f"    Current Value: ${float(p.market_value):.2f}")
    print(f"    P&L: ${float(p.unrealized_pl):.2f} ({float(p.unrealized_plpc)*100:.2f}%)")
