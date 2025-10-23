#!/usr/bin/env python3
"""Close losing positions to free up capital for Week 2"""

import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

load_dotenv('.env.paper')

api = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True)

print("=== CLOSING LOSING POSITIONS ===\n")

# Get all positions
positions = api.get_all_positions()

# Positions to close (all except INTC which is winning)
to_close = []
to_keep = []

for pos in positions:
    pl = float(pos.unrealized_pl)
    symbol = pos.symbol

    # Keep INTC positions (they're winning +$452)
    if 'INTC' in symbol:
        to_keep.append((symbol, pl))
        print(f"[KEEP] {symbol}: {pos.qty} | P&L: ${pl:+.2f} (winning)")
        continue

    # Close everything else to free up capital
    to_close.append(pos)
    print(f"[CLOSE] {symbol}: {pos.qty} | P&L: ${pl:+.2f}")

print(f"\n{len(to_close)} positions to close, {len(to_keep)} to keep\n")

# Close each position
closed_count = 0
total_freed_pl = 0

for pos in to_close:
    try:
        symbol = pos.symbol
        qty = abs(float(pos.qty))
        pl = float(pos.unrealized_pl)

        # Determine side (if qty negative, we sold, so buy to close; if positive, we bought, so sell to close)
        if float(pos.qty) > 0:
            side = OrderSide.SELL
        else:
            side = OrderSide.BUY

        # Submit market order to close
        order = api.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
        )

        print(f"[OK] Closed {symbol}: {side.value} {qty} | Realized P&L: ${pl:+.2f}")
        closed_count += 1
        total_freed_pl += pl

    except Exception as e:
        print(f"[X] Failed to close {symbol}: {e}")

print(f"\n=== SUMMARY ===")
print(f"Positions closed: {closed_count}/{len(to_close)}")
print(f"Total realized P&L: ${total_freed_pl:+.2f}")
print(f"Positions kept: {len(to_keep)} (INTC winning positions)")

# Check new buying power
account = api.get_account()
print(f"\n=== NEW ACCOUNT STATUS ===")
print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
print(f"Cash: ${float(account.cash):,.2f}")
print(f"Buying Power: ${float(account.buying_power):,.2f}")
print(f"Options Buying Power: ${float(account.options_buying_power):,.2f}")
print(f"\n[OK] Ready for Week 2 scanner!")
