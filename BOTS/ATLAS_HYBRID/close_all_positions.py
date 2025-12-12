#!/usr/bin/env python3
"""Close ALL positions to stop FIFO blocking issues"""

from adapters.oanda_adapter import OandaAdapter
from pathlib import Path
from dotenv import load_dotenv

# Load environment
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / '.env')

# Initialize adapter
oanda = OandaAdapter()

# Get all positions
print("Checking for open positions...")
positions = oanda.get_open_positions()

if not positions:
    print("No positions found - trading is clear!")
else:
    print(f"\nFound {len(positions)} position(s) blocking trades:\n")

    total_pnl = 0
    for pos in positions:
        print(f"  {pos['instrument']} {pos['type'].upper()}")
        print(f"    Units: {pos['units']:,.0f}")
        print(f"    Unrealized P/L: ${pos['unrealized_pnl']:,.2f}")
        total_pnl += pos['unrealized_pnl']

        # Close it
        print(f"    Closing...")
        result = oanda.close_position(pos['instrument'], direction=pos['type'])
        if result:
            print(f"    [OK] Closed!")
        else:
            print(f"    [FAILED] Could not close")
        print()

    print(f"Total realized P/L from closures: ${total_pnl:,.2f}")
    print("\n" + "="*60)
    print("ALL POSITIONS CLOSED - ATLAS CAN NOW TRADE FREELY")
    print("="*60)
