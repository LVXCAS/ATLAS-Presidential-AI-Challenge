#!/usr/bin/env python3
"""Quick position and P&L check"""

import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from datetime import datetime

load_dotenv('.env.paper')

api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')

api = TradingClient(api_key, secret_key, paper=True)

print("\n" + "=" * 60)
print("CURRENT POSITIONS & P&L CHECK")
print("=" * 60)
print(f"Time: {datetime.now().strftime('%I:%M %p PDT')}")
print()

# Get account
account = api.get_account()
print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
print(f"Buying Power: ${float(account.buying_power):,.2f}")
print(f"Cash: ${float(account.cash):,.2f}")
print()

# Get positions
positions = api.get_all_positions()

if not positions:
    print("No open positions")
else:
    print(f"Open Positions: {len(positions)}")
    print("-" * 60)

    total_pl = 0
    total_cost = 0

    for pos in positions:
        symbol = pos.symbol
        qty = float(pos.qty)
        entry = float(pos.avg_entry_price)
        current = float(pos.current_price)
        market_value = float(pos.market_value)
        cost_basis = float(pos.cost_basis)
        pl = float(pos.unrealized_pl)
        pl_pct = float(pos.unrealized_plpc) * 100

        total_pl += pl
        total_cost += cost_basis

        status = "[UP]" if pl > 0 else "[DOWN]"
        print(f"\n{status} {symbol}")
        print(f"   Qty: {qty:.0f} | Entry: ${entry:.2f} | Current: ${current:.2f}")
        print(f"   Market Value: ${market_value:,.2f}")
        print(f"   P&L: ${pl:+,.2f} ({pl_pct:+.1f}%)")

    print()
    print("=" * 60)
    print(f"TOTAL P&L: ${total_pl:+,.2f}")
    if total_cost > 0:
        total_pct = (total_pl / total_cost) * 100
        print(f"Total Return: {total_pct:+.2f}%")
    print("=" * 60)
