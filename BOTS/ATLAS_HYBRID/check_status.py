from adapters.oanda_adapter import OandaAdapter

adapter = OandaAdapter()
balance = adapter.get_account_balance()
positions = adapter.get_open_positions()

print(f'Balance: ${balance["balance"]:,.2f}')
print(f'Unrealized P/L: ${balance.get("unrealized_pnl", 0):+,.2f}')
print(f'Open Positions: {len(positions) if positions else 0}')

if positions:
    for p in positions:
        pnl = p.get("unrealized_pnl", 0)
        print(f'  {p["instrument"]}: {p["units"]} units @ ${pnl:+,.2f}')
