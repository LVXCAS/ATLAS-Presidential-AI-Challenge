#!/usr/bin/env python3
"""
COMPLETE ACCOUNT LIQUIDATION
Liquidate all remaining positions including winners to restore positive cash
This is emergency mode - we can rebuild the winners once account is functional
"""

import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
import time

def complete_liquidation():
    """Liquidate everything to get positive cash balance"""
    load_dotenv()

    api = tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        os.getenv('ALPACA_BASE_URL')
    )

    print("EMERGENCY COMPLETE ACCOUNT LIQUIDATION")
    print("=" * 60)
    print("CRITICAL: Liquidating ALL positions to restore account")
    print("We can rebuild winning positions once account is functional")
    print("=" * 60)

    try:
        # Cancel all pending orders first
        print("Canceling all pending orders...")
        orders = api.list_orders(status='open')
        for order in orders:
            try:
                api.cancel_order(order.id)
                print(f"Canceled order: {order.id}")
            except:
                pass

        time.sleep(5)

        # Get all positions and liquidate everything
        positions = api.list_positions()
        print(f"\nLiquidating {len(positions)} positions...")

        total_expected_cash = 0

        for pos in positions:
            symbol = pos.symbol
            qty = int(pos.qty)
            market_value = float(pos.market_value)
            total_expected_cash += abs(market_value)

            print(f"\n[LIQUIDATING] {symbol}")
            print(f"  Quantity: {qty}")
            print(f"  Market Value: ${market_value:,.2f}")

            # Skip if essentially worthless
            if abs(market_value) < 50:
                print(f"  [SKIP] Too small: ${market_value:.2f}")
                continue

            try:
                if qty > 0:
                    # Long position - sell everything
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

                print(f"  [SUCCESS] Order: {order.id}")
                time.sleep(1)

            except Exception as e:
                print(f"  [ERROR] {e}")
                # Try with reduced quantity if insufficient qty error
                if "insufficient qty" in str(e).lower():
                    try:
                        reduced_qty = int(qty * 0.95)  # Try 95% of position
                        if reduced_qty > 0:
                            order = api.submit_order(
                                symbol=symbol,
                                qty=reduced_qty,
                                side='sell' if qty > 0 else 'buy',
                                type='market',
                                time_in_force='day'
                            )
                            print(f"  [RETRY SUCCESS] Reduced qty: {reduced_qty}, Order: {order.id}")
                    except:
                        print(f"  [RETRY FAILED] Could not liquidate {symbol}")

        print(f"\nTotal expected cash recovery: ${total_expected_cash:,.2f}")
        print("\nWaiting for all liquidation orders to fill...")
        time.sleep(30)

        # Check final status
        account = api.get_account()
        final_cash = float(account.cash)
        final_buying_power = float(account.buying_power)

        print(f"\n" + "=" * 60)
        print("FINAL LIQUIDATION RESULTS:")
        print(f"Cash: ${final_cash:,.2f}")
        print(f"Buying Power: ${final_buying_power:,.2f}")
        print(f"Positions remaining: {len(api.list_positions())}")

        if final_cash > 0:
            print("\n🎉 SUCCESS! ACCOUNT RECOVERED!")
            print("✅ Positive cash balance achieved")
            print("✅ Ready for Intel-puts-style trades")
            print("\nNext steps:")
            print("1. Account is now functional")
            print("2. Can rebuild winning positions if desired")
            print("3. Ready to deploy capital for new trades")
        else:
            print(f"\nStill need ${abs(final_cash):,.2f} more cash recovery")
            remaining_positions = api.list_positions()
            if remaining_positions:
                print("Remaining positions that couldn't be liquidated:")
                for pos in remaining_positions:
                    print(f"  {pos.symbol}: {pos.qty} shares, ${float(pos.market_value):,.2f}")

    except Exception as e:
        print(f"Error in complete liquidation: {e}")

if __name__ == "__main__":
    complete_liquidation()