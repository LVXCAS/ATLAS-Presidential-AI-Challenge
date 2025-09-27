#!/usr/bin/env python3
"""
FINAL RECOVERY LIQUIDATION
Liquidate remaining TSLA position with correct quantity to complete cash recovery
"""

import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
import time
from datetime import datetime

class FinalRecoveryLiquidation:
    """Complete the account recovery by liquidating remaining large positions"""

    def __init__(self):
        load_dotenv()

        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL')
        )

    def get_current_positions(self):
        """Get current positions with correct quantities"""
        try:
            positions = self.api.list_positions()

            for pos in positions:
                print(f"{pos.symbol}: {pos.qty} shares, ${float(pos.market_value):,.2f}")

            return positions
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []

    def liquidate_remaining_tsla(self):
        """Liquidate TSLA with correct available quantity"""
        try:
            positions = self.api.list_positions()

            # Find TSLA position
            tsla_pos = None
            for pos in positions:
                if pos.symbol == 'TSLA':
                    tsla_pos = pos
                    break

            if not tsla_pos:
                print("No TSLA position found")
                return False

            available_qty = int(tsla_pos.qty)
            market_value = float(tsla_pos.market_value)

            print(f"LIQUIDATING TSLA:")
            print(f"Available Quantity: {available_qty}")
            print(f"Market Value: ${market_value:,.2f}")

            if available_qty <= 0:
                print("No TSLA shares available to sell")
                return False

            # Submit market sell order with correct quantity
            order = self.api.submit_order(
                symbol='TSLA',
                qty=available_qty,
                side='sell',
                type='market',
                time_in_force='day'
            )

            print(f"ORDER SUBMITTED: {order.id}")
            print(f"Expected Cash Recovery: ${market_value:,.2f}")

            return True

        except Exception as e:
            print(f"Error liquidating TSLA: {e}")
            return False

    def liquidate_remaining_mvis(self):
        """Liquidate MVIS with correct available quantity"""
        try:
            positions = self.api.list_positions()

            # Find MVIS position
            mvis_pos = None
            for pos in positions:
                if pos.symbol == 'MVIS':
                    mvis_pos = pos
                    break

            if not mvis_pos:
                print("No MVIS position found")
                return False

            available_qty = int(mvis_pos.qty)
            market_value = float(mvis_pos.market_value)

            print(f"LIQUIDATING MVIS:")
            print(f"Available Quantity: {available_qty}")
            print(f"Market Value: ${market_value:,.2f}")

            if available_qty <= 0:
                print("No MVIS shares available to sell")
                return False

            # Submit market sell order with correct quantity
            order = self.api.submit_order(
                symbol='MVIS',
                qty=available_qty,
                side='sell',
                type='market',
                time_in_force='day'
            )

            print(f"ORDER SUBMITTED: {order.id}")
            print(f"Expected Cash Recovery: ${market_value:,.2f}")

            return True

        except Exception as e:
            print(f"Error liquidating MVIS: {e}")
            return False

    def check_final_status(self):
        """Check if account is now ready for trading"""
        try:
            account = self.api.get_account()

            print(f"\nFINAL ACCOUNT STATUS:")
            print(f"Cash: ${float(account.cash):,.2f}")
            print(f"Buying Power: ${float(account.buying_power):,.2f}")
            print(f"Equity: ${float(account.equity):,.2f}")

            if float(account.cash) > 0:
                print("[SUCCESS] Account recovered - positive cash balance!")
                print("Ready for Intel-puts-style trades!")
                return True
            else:
                print(f"Still need ${abs(float(account.cash)):,.2f} more cash recovery")
                return False

        except Exception as e:
            print(f"Error checking status: {e}")
            return False

    def run_final_recovery(self):
        """Execute final recovery liquidations"""
        print("FINAL RECOVERY LIQUIDATION")
        print("=" * 60)

        print("\nCurrent positions:")
        self.get_current_positions()

        # Liquidate TSLA
        tsla_success = self.liquidate_remaining_tsla()

        # Liquidate MVIS
        mvis_success = self.liquidate_remaining_mvis()

        if tsla_success or mvis_success:
            print("\nWaiting for orders to fill...")
            time.sleep(10)

            # Check final status
            self.check_final_status()
        else:
            print("No additional liquidations possible")

def main():
    recovery = FinalRecoveryLiquidation()
    recovery.run_final_recovery()

if __name__ == "__main__":
    main()