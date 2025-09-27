#!/usr/bin/env python3
"""
PRECISE FINAL LIQUIDATION
Use exact available quantities from position data to liquidate remaining positions
"""

import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
import time

def liquidate_with_exact_quantities():
    """Liquidate using exact available quantities"""
    load_dotenv()

    api = tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        os.getenv('ALPACA_BASE_URL')
    )

    print("PRECISE FINAL LIQUIDATION")
    print("=" * 50)

    try:
        # Get positions and identify exact quantities
        positions = api.list_positions()

        for pos in positions:
            symbol = pos.symbol
            qty = int(pos.qty)
            market_value = float(pos.market_value)

            print(f"\nPosition: {symbol}")
            print(f"Quantity: {qty}")
            print(f"Market Value: ${market_value:,.2f}")

            # Skip options positions (preserve winners)
            if 'P0' in symbol or 'C0' in symbol:
                print(f"[PRESERVE] Keeping options position {symbol}")
                continue

            # Skip worthless positions
            if abs(market_value) < 100:
                print(f"[SKIP] Position too small: ${market_value:.2f}")
                continue

            # Liquidate large positions
            if abs(market_value) > 1000:
                print(f"[LIQUIDATING] {symbol}")
                try:
                    if qty > 0:
                        # Long position - sell
                        order = api.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side='sell',
                            type='market',
                            time_in_force='day'
                        )
                    elif qty < 0:
                        # Short position - buy to cover
                        order = api.submit_order(
                            symbol=symbol,
                            qty=abs(qty),
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )

                    print(f"ORDER SUBMITTED: {order.id}")
                    print(f"Expected cash recovery: ${abs(market_value):,.2f}")

                    # Brief pause between orders
                    time.sleep(2)

                except Exception as e:
                    print(f"ERROR liquidating {symbol}: {e}")

        print("\nWaiting for orders to fill...")
        time.sleep(15)

        # Check final account status
        account = api.get_account()
        print(f"\nFINAL STATUS:")
        print(f"Cash: ${float(account.cash):,.2f}")
        print(f"Buying Power: ${float(account.buying_power):,.2f}")

        if float(account.cash) > 0:
            print("\n[SUCCESS] Account recovered! Positive cash balance achieved!")
        else:
            cash_needed = abs(float(account.cash))
            print(f"\nStill need ${cash_needed:,.2f} more cash recovery")

            # Show remaining positions
            remaining_positions = api.list_positions()
            print(f"\nRemaining positions ({len(remaining_positions)}):")
            for pos in remaining_positions:
                if abs(float(pos.market_value)) > 100:
                    print(f"  {pos.symbol}: ${float(pos.market_value):,.2f}")

    except Exception as e:
        print(f"Error in precise liquidation: {e}")

if __name__ == "__main__":
    liquidate_with_exact_quantities()