#!/usr/bin/env python3
"""Weekend Risk Analysis - Friday Close"""

import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

load_dotenv()

api = tradeapi.REST(
    key_id=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    base_url=os.getenv('ALPACA_BASE_URL'),
    api_version='v2'
)

account = api.get_account()
positions = api.list_positions()

# Categorize positions
winners = []
small_losers = []
big_losers = []

for pos in positions:
    pl = float(pos.unrealized_pl)
    pl_pct = float(pos.unrealized_plpc) * 100

    if pl > 0:
        winners.append((pos.symbol, pl, pl_pct))
    elif pl > -100:
        small_losers.append((pos.symbol, pl, pl_pct))
    else:
        big_losers.append((pos.symbol, pl, pl_pct))

print('=' * 70)
print('WEEKEND RISK ANALYSIS - FRIDAY CLOSE')
print('=' * 70)
print()

print(f'Account Value: ${float(account.equity):,.2f}')
print(f'Cash: ${float(account.cash):,.2f}')
print(f'Total Positions: {len(positions)}')
print()

print(f'WINNERS ({len(winners)} positions): +${sum(w[1] for w in winners):.2f}')
for symbol, pl, pl_pct in sorted(winners, key=lambda x: x[1], reverse=True):
    print(f'  {symbol:30s} +${pl:>8.2f} ({pl_pct:>+6.1f}%)')

print()
print(f'SMALL LOSERS (<$100) ({len(small_losers)} positions): ${sum(w[1] for w in small_losers):.2f}')
for symbol, pl, pl_pct in sorted(small_losers, key=lambda x: x[1]):
    print(f'  {symbol:30s} ${pl:>8.2f} ({pl_pct:>+6.1f}%)')

print()
print(f'BIG LOSERS (>$100) ({len(big_losers)} positions): ${sum(w[1] for w in big_losers):.2f}')
for symbol, pl, pl_pct in sorted(big_losers, key=lambda x: x[1]):
    print(f'  {symbol:30s} ${pl:>8.2f} ({pl_pct:>+6.1f}%)')

print()
print('=' * 70)
total_pl = sum(w[1] for w in winners) + sum(w[1] for w in small_losers) + sum(w[1] for w in big_losers)
print(f'TOTAL UNREALIZED P&L: ${total_pl:,.2f}')
print(f'WEEKEND GAP RISK: {"HIGH" if len(big_losers) > 5 else "MODERATE"}')
print('=' * 70)
