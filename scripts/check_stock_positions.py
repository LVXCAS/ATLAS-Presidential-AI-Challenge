import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv(override=True)
api = tradeapi.REST(
    key_id=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    base_url=os.getenv('ALPACA_BASE_URL')
)

positions = api.list_positions()

# Separate stock vs options positions
stock_positions = []
option_positions = []

for p in positions:
    if 'P00' in p.symbol or 'C00' in p.symbol:
        option_positions.append(p)
    else:
        stock_positions.append(p)

print(f"\n{'='*80}")
print(f"CURRENT POSITION BREAKDOWN - {len(positions)} TOTAL POSITIONS")
print(f"{'='*80}\n")

print(f"STOCK POSITIONS: {len(stock_positions)}")
if stock_positions:
    print("\nWARNING: Stock fallback positions detected!\n")
    total_stock_value = 0
    total_stock_pnl = 0
    for p in sorted(stock_positions, key=lambda x: float(x.unrealized_pl)):
        qty = float(p.qty)
        price = float(p.avg_entry_price)
        value = qty * price
        pnl = float(p.unrealized_pl)
        total_stock_value += value
        total_stock_pnl += pnl
        print(f"  {p.symbol:6s}: {int(qty):4d} shares @ ${price:7.2f} | Value: ${value:9,.2f} | P&L: ${pnl:+9,.2f}")

    print(f"\n  Total Stock Value: ${total_stock_value:,.2f}")
    print(f"  Total Stock P&L:   ${total_stock_pnl:+,.2f}")
else:
    print("  ✓ NO STOCK POSITIONS - Stock fallback successfully disabled!\n")

print(f"\nOPTIONS POSITIONS: {len(option_positions)}")
if option_positions:
    total_options_pnl = 0
    for p in sorted(option_positions, key=lambda x: float(x.unrealized_pl)):
        pnl = float(p.unrealized_pl)
        total_options_pnl += pnl
        print(f"  {p.symbol}: {p.qty} | P&L: ${pnl:+,.2f}")
    print(f"\n  Total Options P&L: ${total_options_pnl:+,.2f}")

print(f"\n{'='*80}\n")
