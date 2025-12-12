#!/usr/bin/env python3
"""Close EUR/USD SHORT position to unlock trading"""

from adapters.oanda_adapter import OandaAdapter
from pathlib import Path
from dotenv import load_dotenv

# Load environment
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / '.env')

# Initialize adapter
oanda = OandaAdapter()

# Get current position
print("Checking current EUR/USD position...")
positions = oanda.get_open_positions()
eur_position = [p for p in positions if p['instrument'] == 'EUR_USD'] if positions else []

if eur_position:
    pos = eur_position[0]
    print(f"Found: {pos['type'].upper()} {pos['units']:,.0f} units")
    print(f"Unrealized P/L: ${pos['unrealized_pnl']:,.2f}")

    # Close the position
    print(f"\nClosing {pos['type'].upper()} position...")
    result = oanda.close_position('EUR_USD', direction=pos['type'])

    if result:
        print("✓ Position closed successfully!")

        # Get new balance
        bal = oanda.get_account_balance()
        balance = bal.get('balance') if isinstance(bal, dict) else bal
        print(f"New balance: ${balance:,.2f}")
    else:
        print("✗ Failed to close position")
else:
    print("No EUR/USD position found")
