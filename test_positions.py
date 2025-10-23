"""Check current open positions"""

from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
import os

load_dotenv()

client = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True)
positions = client.get_all_positions()

print("=" * 60)
print("CURRENT OPEN POSITIONS")
print("=" * 60)
print(f"Total Positions: {len(positions)}")
print()

if positions:
    for p in positions:
        print(f"  {p.symbol}:")
        print(f"    Quantity: {p.qty}")
        print(f"    Entry Price: ${float(p.avg_entry_price):.2f}")
        print(f"    Current Price: ${float(p.current_price):.2f}")
        print(f"    P/L: ${float(p.unrealized_pl):.2f} ({float(p.unrealized_plpc)*100:.2f}%)")
        print()
else:
    print("  No open positions")
    print()

print("=" * 60)
