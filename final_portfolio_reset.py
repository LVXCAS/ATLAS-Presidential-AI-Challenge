#!/usr/bin/env python3
"""
FINAL PORTFOLIO RESET
Reset portfolio to clean slate by closing ALL remaining positions except core winners
Simple and effective approach to free maximum capital
"""

import os
from datetime import datetime
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()

class FinalPortfolioReset:
    """Complete portfolio reset - close everything except core winners"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # Absolute core positions to keep (only the best quality)
        self.core_keeps = {'AAPL', 'MSFT', 'NVDA', 'TSLA', 'SPY', 'QQQ'}

    def execute_final_reset(self):
        """Execute complete portfolio reset"""

        print("FINAL PORTFOLIO RESET")
        print("=" * 60)
        print("Closing ALL positions except core quality winners")
        print("Maximum capital freedom for concentrated strategy")
        print("=" * 60)

        try:
            positions = self.alpaca.list_positions()

            if not positions:
                print("No positions to close")
                return

            print(f"Current positions: {len(positions)}")

            # Categorize positions
            keep_positions = []
            close_positions = []

            for pos in positions:
                symbol = pos.symbol

                # Keep only core quality positions (no options, no junk)
                if symbol in self.core_keeps and not any(x in symbol for x in ['C00', 'P00']):
                    keep_positions.append(pos)
                else:
                    close_positions.append(pos)

            print(f"\nKEEP: {len(keep_positions)} core positions")
            for pos in keep_positions:
                print(f"  KEEP {pos.symbol}: ${float(pos.market_value):,.0f}")

            print(f"\nCLOSE: {len(close_positions)} positions")

            # Execute closures
            successful_closes = 0
            failed_closes = 0
            total_freed_value = 0

            for pos in close_positions:
                symbol = pos.symbol
                qty = int(pos.qty)
                market_value = float(pos.market_value)

                # Determine order side
                side = "sell" if qty > 0 else "buy"
                qty = abs(qty)

                try:
                    # Submit market order to close
                    order = self.alpaca.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side=side,
                        type='market',
                        time_in_force='day'
                    )

                    successful_closes += 1
                    total_freed_value += abs(market_value)

                    print(f"  SUCCESS {symbol}: {side.upper()} {qty} shares (${market_value:,.0f})")

                except Exception as e:
                    failed_closes += 1
                    print(f"  FAILED {symbol}: {str(e)[:50]}...")

            print("-" * 60)
            print(f"RESET RESULTS:")
            print(f"  Successful closures: {successful_closes}")
            print(f"  Failed closures: {failed_closes}")
            print(f"  Capital freed: ${total_freed_value:,.0f}")
            print(f"  Remaining positions: ~{len(keep_positions)}")

            print("\n" + "=" * 60)
            print("PORTFOLIO RESET COMPLETE")
            print("Clean slate ready for concentrated strategy deployment")
            print("=" * 60)

            return {
                'positions_closed': successful_closes,
                'positions_kept': len(keep_positions),
                'capital_freed': total_freed_value,
                'reset_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Error during reset: {e}")
            return {}

def main():
    """Execute final portfolio reset"""
    reset = FinalPortfolioReset()
    result = reset.execute_final_reset()
    return result

if __name__ == "__main__":
    main()