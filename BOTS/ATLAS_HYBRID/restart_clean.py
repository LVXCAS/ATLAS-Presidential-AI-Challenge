"""
Clean Restart Script for ATLAS
- Closes all open positions
- Provides restart instructions
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from adapters.oanda_adapter import OandaAdapter

def main():
    print("=" * 70)
    print("ATLAS CLEAN RESTART")
    print("=" * 70)

    # Initialize OANDA
    oanda = OandaAdapter()

    # Get current status
    try:
        balance = oanda.get_account_balance()
        positions = oanda.get_open_positions()

        print(f"\n[CURRENT STATUS]")
        print(f"  Balance: ${balance:,.2f}")
        print(f"  Open Positions: {len(positions)}")

        if positions:
            print(f"\n[POSITIONS TO CLOSE]")
            for pos in positions:
                pair = pos['instrument']
                units = pos['units']
                lots = abs(units) / 100000
                pnl = pos['unrealized_pnl']
                direction = 'LONG' if units > 0 else 'SHORT'
                print(f"  {pair} {direction}: {lots:.1f} lots, P/L: ${pnl:,.2f}")

            # Close all positions
            print(f"\n[CLOSING POSITIONS]")
            for pos in positions:
                pair = pos['instrument']
                direction = 'long' if pos['units'] > 0 else 'short'
                print(f"  Closing {pair} {direction.upper()}...")
                result = oanda.close_position(pair, direction)
                if result:
                    print(f"  ✓ Closed {pair}")
                else:
                    print(f"  ✗ Failed to close {pair}")

            # Check final status
            final_balance = oanda.get_account_balance()
            final_positions = oanda.get_open_positions()

            print(f"\n[FINAL STATUS]")
            print(f"  Balance: ${final_balance:,.2f}")
            print(f"  P/L from closures: ${final_balance - balance:,.2f}")
            print(f"  Open Positions: {len(final_positions)}")
        else:
            print(f"  No positions to close")

        print(f"\n" + "=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print("1. Kill old processes:")
        print("   taskkill /F /IM pythonw.exe")
        print("")
        print("2. Start new ATLAS with Kelly + TradeLogger:")
        print("   cd C:\\Users\\lucas\\PC-HIVE-TRADING\\BOTS\\ATLAS_HYBRID")
        print("   start pythonw live_trader.py")
        print("")
        print("3. Monitor status:")
        print("   python check_live_scores.py")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
