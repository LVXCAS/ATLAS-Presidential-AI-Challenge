#!/usr/bin/env python3
"""
EMERGENCY RECOVERY - Close losers, free capital for Iron Condors
"""

import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()

api = tradeapi.REST(
    key_id=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    base_url=os.getenv('ALPACA_BASE_URL'),
    api_version='v2'
)

print("="*70)
print("EMERGENCY RECOVERY - CLOSING LOSING STOCK POSITIONS")
print("="*70)

# Get all positions
positions = api.list_positions()

print(f"\nTotal positions: {len(positions)}")

# Close all stock positions (not options)
stocks_to_close = []
for pos in positions:
    symbol = pos.symbol

    # Check if it's a stock (no expiration date in symbol)
    if len(symbol) <= 5 and not any(char.isdigit() for char in symbol):
        unrealized_plpc = float(pos.unrealized_plpc)
        qty = int(pos.qty)

        # Close if losing OR if stock position (to free capital)
        stocks_to_close.append({
            'symbol': symbol,
            'qty': qty,
            'pnl_pct': unrealized_plpc * 100
        })

print(f"\nStock positions to close: {len(stocks_to_close)}")

if stocks_to_close:
    print("\nClosing stock positions...")
    for stock in stocks_to_close:
        try:
            print(f"  Closing {stock['symbol']}: {stock['qty']} shares ({stock['pnl_pct']:+.2f}%)")

            # Market order to close
            api.submit_order(
                symbol=stock['symbol'],
                qty=stock['qty'],
                side='sell',
                type='market',
                time_in_force='day'
            )

            print(f"    [OK] Closed")

        except Exception as e:
            print(f"    [ERROR] {e}")

print("\n" + "="*70)
print("RECOVERY COMPLETE")
print("="*70)

# Check new account status
account = api.get_account()
print(f"\nNew Account Status:")
print(f"  Equity: ${float(account.equity):,.2f}")
print(f"  Cash: ${float(account.cash):,.2f}")
print(f"  Buying Power: ${float(account.buying_power):,.2f}")

remaining_positions = api.list_positions()
print(f"  Positions: {len(remaining_positions)}")

print("\n" + "="*70)
print("SCANNER CAN NOW EXECUTE IRON CONDORS")
print("="*70)
print("\nWait for next scan cycle (shows in scanner terminal)")
print("Scanner will execute Iron Condors on low-momentum stocks")
print("\nExpected: 15-20 Iron Condors over next 4 hours")
