#!/usr/bin/env python3
"""Calculate detailed P&L for OPTIONS account"""

import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()
api = tradeapi.REST(
    key_id=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    base_url=os.getenv('ALPACA_BASE_URL')
)

account = api.get_account()
positions = api.list_positions()

print('='*70)
print('OPTIONS ACCOUNT P&L SUMMARY')
print('='*70)
print(f'\nStarting Capital: $100,000.00 (paper trading)')
print(f'Current Portfolio Value: ${float(account.equity):,.2f}')
print(f'Total P&L: ${float(account.equity) - 100000:+,.2f}')
print(f'Return: {((float(account.equity) - 100000) / 100000) * 100:+.2f}%')
print(f'\nCash: ${float(account.cash):,.2f}')
print(f'Buying Power: ${float(account.buying_power):,.2f}')
print(f'Open Positions: {len(positions)}')

total_unrealized = 0
winners = []
losers = []

print(f'\n{"="*70}')
print(f'POSITION BREAKDOWN ({len(positions)} positions)')
print(f'{"="*70}\n')

for pos in positions:
    symbol = pos.symbol
    qty = float(pos.qty)
    avg_price = float(pos.avg_entry_price)
    current_price = float(pos.current_price)
    unrealized_pl = float(pos.unrealized_pl) if pos.unrealized_pl else 0
    unrealized_plpc = float(pos.unrealized_plpc) if pos.unrealized_plpc else 0

    total_unrealized += unrealized_pl

    if unrealized_pl > 0:
        winners.append((symbol, unrealized_pl, unrealized_plpc))
    else:
        losers.append((symbol, unrealized_pl, unrealized_plpc))

    print(f'{symbol:20s} {qty:>6.0f} @ ${avg_price:>8.2f} | P&L: ${unrealized_pl:>+9.2f} ({unrealized_plpc*100:>+6.2f}%)')

print(f'\n{"="*70}')
print(f'WINNERS vs LOSERS')
print(f'{"="*70}\n')

winners.sort(key=lambda x: x[1], reverse=True)
losers.sort(key=lambda x: x[1])

print(f'Top 5 Winners:')
for symbol, pl, plpc in winners[:5]:
    print(f'  {symbol:10s} ${pl:>+9.2f} ({plpc*100:>+6.2f}%)')

print(f'\nTop 5 Losers:')
for symbol, pl, plpc in losers[:5]:
    print(f'  {symbol:10s} ${pl:>+9.2f} ({plpc*100:>+6.2f}%)')

print(f'\nTotal Winners: {len(winners)} positions')
print(f'Total Losers: {len(losers)} positions')
print(f'Win Rate: {len(winners) / len(positions) * 100:.1f}%')
print(f'\nTotal Unrealized P&L: ${total_unrealized:+,.2f}')
