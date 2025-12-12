"""Quick script to check ATLAS trading status via OANDA."""
import sys
from pathlib import Path

# Add BOTS directory to path
sys.path.append(str(Path(__file__).parent / "BOTS" / "ATLAS_HYBRID"))

from adapters.oanda_adapter import OandaAdapter

adapter = OandaAdapter()

# Get account info
balance_data = adapter.get_account_balance()
print(f"Account Balance: ${balance_data['balance']:,.2f}")
print(f"Unrealized P/L: ${balance_data.get('unrealized_pnl', 0):+,.2f}")

# Get open positions
positions = adapter.get_open_positions()
print(f"\nOpen Positions: {len(positions) if positions else 0}")
if positions:
    for pos in positions:
        print(f"  {pos['instrument']}: {pos['units']} units @ {pos['avg_price']:.5f}")
        print(f"    P/L: ${pos.get('unrealized_pnl', 0):+,.2f}")
