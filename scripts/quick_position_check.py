#!/usr/bin/env python3
"""
QUICK POSITION CHECK
Fast overview of current positions and P&L
"""

import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from datetime import datetime

load_dotenv('.env.paper')

api = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True)

print("\n" + "=" * 80)
print("QUICK POSITION CHECK")
print("=" * 80)
print(f"Time: {datetime.now().strftime('%I:%M:%S %p')}")
print()

# Get account
account = api.get_account()
print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
print(f"Cash: ${float(account.cash):,.2f}")
print(f"Buying Power: ${float(account.buying_power):,.2f}")
print()

# Get positions
positions = api.get_all_positions()

if not positions:
    print("No open positions")
else:
    # Categorize
    stocks = [p for p in positions if len(p.symbol) < 10]
    options = [p for p in positions if len(p.symbol) > 10]

    losing_positions = [p for p in positions if p.unrealized_pl and float(p.unrealized_pl) < 0]
    winning_positions = [p for p in positions if p.unrealized_pl and float(p.unrealized_pl) > 0]

    print(f"Total Positions: {len(positions)}")
    print(f"  Stocks: {len(stocks)}")
    print(f"  Options: {len(options)}")
    print(f"  Losing: {len(losing_positions)} ({len(losing_positions)/len(positions)*100:.1f}%)")
    print(f"  Winning: {len(winning_positions)} ({len(winning_positions)/len(positions)*100:.1f}%)")
    print()

    # Calculate totals
    total_unrealized_pl = sum(
        float(p.unrealized_pl) if p.unrealized_pl else 0
        for p in positions
    )

    print(f"Total Unrealized P&L: ${total_unrealized_pl:,.2f}")
    print()

    # Show top 5 losers
    if losing_positions:
        print("Top 5 Losers:")
        losing_positions.sort(key=lambda p: float(p.unrealized_pl))
        for i, pos in enumerate(losing_positions[:5], 1):
            pl = float(pos.unrealized_pl)
            plpc = float(pos.unrealized_plpc) * 100 if pos.unrealized_plpc else 0
            print(f"  {i}. {pos.symbol}: ${pl:,.2f} ({plpc:.1f}%)")
        print()

    # Show top 5 winners
    if winning_positions:
        print("Top 5 Winners:")
        winning_positions.sort(key=lambda p: float(p.unrealized_pl), reverse=True)
        for i, pos in enumerate(winning_positions[:5], 1):
            pl = float(pos.unrealized_pl)
            plpc = float(pos.unrealized_plpc) * 100 if pos.unrealized_plpc else 0
            print(f"  {i}. {pos.symbol}: ${pl:,.2f} (+{plpc:.1f}%)")
        print()

    # Check for worthless options
    worthless = [
        p for p in options
        if p.current_price and float(p.current_price) == 0
    ]

    if worthless:
        print(f"[!] ALERT: {len(worthless)} worthless options found!")
        for pos in worthless:
            pl = float(pos.unrealized_pl) if pos.unrealized_pl else 0
            print(f"  - {pos.symbol}: ${pl:.2f}")
        print()

    # Check for critical losses
    critical = [
        p for p in positions
        if p.unrealized_plpc and float(p.unrealized_plpc) * 100 < -30
    ]

    if critical:
        print(f"[X] ALERT: {len(critical)} positions with >30% loss!")
        for pos in critical:
            pl = float(pos.unrealized_pl) if pos.unrealized_pl else 0
            plpc = float(pos.unrealized_plpc) * 100 if pos.unrealized_plpc else 0
            print(f"  - {pos.symbol}: ${pl:.2f} ({plpc:.1f}%)")
        print()

    # Check for oversized positions
    portfolio_value = float(account.portfolio_value)
    oversized = [
        p for p in positions
        if p.market_value and abs(float(p.market_value)) > portfolio_value * 0.10
    ]

    if oversized:
        print(f"[!] WARNING: {len(oversized)} positions >10% of portfolio!")
        for pos in oversized:
            mv = float(pos.market_value) if pos.market_value else 0
            pct = (mv / portfolio_value) * 100
            print(f"  - {pos.symbol}: ${mv:,.2f} ({pct:.1f}% of portfolio)")
        print()

print("=" * 80)
print()
