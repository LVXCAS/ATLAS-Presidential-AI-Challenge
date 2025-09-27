#!/usr/bin/env python3
"""
Conservative Rebalancing Script
Trims only the clear losers to free up buying power for autonomous systems
"""

import os
import alpaca_trade_api as tradeapi

# Initialize Alpaca API (using paper trading)
api = tradeapi.REST(
    key_id=os.getenv('ALPACA_API_KEY', 'your_api_key_here'),
    secret_key=os.getenv('ALPACA_SECRET_KEY', 'your_secret_key_here'),
    base_url='https://paper-api.alpaca.markets'
)

def conservative_rebalance():
    """
    Phase 1 Conservative Rebalancing:
    1. Close INTC calls (-138.7% - clear loser)
    2. Close RIVN puts (RIVN up 7.7%, puts not working as hedge)

    Goal: Free up ~$900+ buying power for autonomous systems
    """

    print("=== CONSERVATIVE REBALANCING ===")
    print("Target: Free up buying power for autonomous trading")
    print("Strategy: Trim only clear losers, keep winners intact")
    print()

    # Get current account info
    try:
        account = api.get_account()
        print(f"Current Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"Current Buying Power: ${float(account.buying_power):,.2f}")
        print(f"Day P&L: ${float(account.unrealized_pl):,.2f}")
        print()
    except Exception as e:
        print(f"Account check error: {e}")
        return

    # Get current positions
    try:
        positions = api.list_positions()
        print(f"Current positions: {len(positions)}")

        # Identify positions to trim
        positions_to_close = []

        for position in positions:
            symbol = position.symbol
            qty = float(position.qty)
            market_value = float(position.market_value)
            unrealized_pl = float(position.unrealized_pl) if hasattr(position, 'unrealized_pl') else 0
            pct_change = (unrealized_pl / abs(market_value - unrealized_pl)) * 100 if market_value != unrealized_pl else 0

            print(f"{symbol}: {qty} shares, ${market_value:,.2f} value, {pct_change:+.1f}% P&L")

            # Conservative trimming criteria
            if 'INTC' in symbol and 'C' in symbol and pct_change < -100:
                # INTC calls with massive losses
                positions_to_close.append((symbol, qty, market_value, "Clear loser - cut losses"))

            elif 'RIVN' in symbol and 'P' in symbol and unrealized_pl < 0:
                # RIVN puts losing money while RIVN stock is up
                positions_to_close.append((symbol, qty, market_value, "Hedge not working - RIVN rallying"))

            elif 'SNAP' in symbol and 'P' in symbol and unrealized_pl < 0:
                # SNAP puts losing money while SNAP calls are up 13%
                positions_to_close.append((symbol, qty, market_value, "Hedge not working - SNAP rallying"))

        print(f"\n=== POSITIONS TO CLOSE ===")
        total_freed_capital = 0

        if not positions_to_close:
            print("No positions meet conservative trimming criteria")
            return

        for symbol, qty, value, reason in positions_to_close:
            print(f"{symbol}: Close {qty} shares (${value:,.2f}) - {reason}")
            total_freed_capital += abs(value)

        print(f"\nTotal capital to be freed: ${total_freed_capital:,.2f}")
        print(f"This will increase buying power from $76.50 to ~${76.50 + total_freed_capital:,.2f}")

        # Execute trades
        print(f"\n=== EXECUTING CONSERVATIVE REBALANCE ===")
        successful_closes = 0

        for symbol, qty, value, reason in positions_to_close:
            try:
                # Determine order side
                side = 'sell' if qty > 0 else 'buy'
                abs_qty = abs(qty)

                print(f"Closing {symbol}: {side} {abs_qty} shares...")

                # Place market order to close position
                order = api.submit_order(
                    symbol=symbol,
                    qty=abs_qty,
                    side=side,
                    type='market',
                    time_in_force='day'
                )

                print(f"✅ Order submitted: {order.id}")
                successful_closes += 1

            except Exception as e:
                print(f"❌ Failed to close {symbol}: {e}")

        print(f"\n=== REBALANCING COMPLETE ===")
        print(f"Successfully closed {successful_closes}/{len(positions_to_close)} positions")

        # Check new account status
        try:
            new_account = api.get_account()
            new_buying_power = float(new_account.buying_power)
            print(f"New buying power: ${new_buying_power:,.2f}")
            print(f"Buying power increase: ${new_buying_power - 76.50:,.2f}")

            if new_buying_power > 500:
                print("🚀 SUCCESS: Sufficient buying power for autonomous trading!")
            else:
                print("⚠️  May need additional trimming for full autonomous trading")

        except Exception as e:
            print(f"New account check error: {e}")

    except Exception as e:
        print(f"Position check error: {e}")

if __name__ == "__main__":
    conservative_rebalance()