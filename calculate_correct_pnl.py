#!/usr/bin/env python3
"""Calculate CORRECT P&L with proper starting balances"""

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
print('CORRECTED P&L REPORT - ACCOUNT #1')
print('='*70)
print(f'\nAccount ID: {account.account_number}')
print(f'Starting Capital: $1,000,000.00')
print(f'Current Equity: ${float(account.equity):,.2f}')
print(f'Total P&L: ${float(account.equity) - 1000000:+,.2f}')
print(f'Return: {((float(account.equity) - 1000000) / 1000000) * 100:+.2f}%')
print(f'\nCash: ${float(account.cash):,.2f}')
print(f'Buying Power: ${float(account.buying_power):,.2f}')
print(f'Open Positions: {len(positions)}')

total_unrealized = 0
winners = []
losers = []

print(f'\n{"="*70}')
print(f'OPEN POSITIONS ({len(positions)} positions)')
print(f'{"="*70}\n')

for pos in positions:
    symbol = pos.symbol
    qty = float(pos.qty)
    avg_price = float(pos.avg_entry_price)
    unrealized_pl = float(pos.unrealized_pl) if pos.unrealized_pl else 0
    unrealized_plpc = float(pos.unrealized_plpc) if pos.unrealized_plpc else 0

    total_unrealized += unrealized_pl

    if unrealized_pl > 0:
        winners.append((symbol, unrealized_pl, unrealized_plpc))
    else:
        losers.append((symbol, unrealized_pl, unrealized_plpc))

    print(f'{symbol:20s} {qty:>6.0f} @ ${avg_price:>8.2f} | P&L: ${unrealized_pl:>+9.2f} ({unrealized_plpc*100:>+6.2f}%)')

print(f'\n{"="*70}')
print(f'SUMMARY')
print(f'{"="*70}\n')

print(f'Unrealized P&L: ${total_unrealized:+,.2f}')
print(f'Realized P&L (closed trades): ${float(account.equity) - 1000000 - total_unrealized:+,.2f}')

print(f'\nWinning positions: {len(winners)}/{len(positions)} ({len(winners)/len(positions)*100:.1f}%)')
print(f'Losing positions: {len(losers)}/{len(positions)} ({len(losers)/len(positions)*100:.1f}%)')
