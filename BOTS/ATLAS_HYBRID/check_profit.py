#!/usr/bin/env python3
"""Quick P/L check"""

from adapters.oanda_adapter import OandaAdapter
from pathlib import Path
from dotenv import load_dotenv

# Load environment
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / '.env')

# Initialize adapter
oanda = OandaAdapter()

# Get balance
bal_data = oanda.get_account_balance()
balance = bal_data.get('balance') if isinstance(bal_data, dict) else bal_data

# Calculate P/L
starting_balance = 182999.16
total_profit = balance - starting_balance
pct = (total_profit / starting_balance) * 100

print(f"ATLAS Performance Summary")
print(f"=" * 50)
print(f"Starting Balance: ${starting_balance:,.2f}")
print(f"Current Balance:  ${balance:,.2f}")
print(f"Total P/L:        ${total_profit:+,.2f} ({pct:+.2f}%)")
print(f"=" * 50)

# Get positions
positions = oanda.get_open_positions()
if positions:
    print(f"\nOpen Positions: {len(positions)}")
    for pos in positions:
        print(f"  {pos['instrument']} {pos['type'].upper()}: {pos['units']:,.0f} units")
        print(f"    Unrealized P/L: ${pos['unrealized_pnl']:,.2f}")
else:
    print("\nNo open positions")
