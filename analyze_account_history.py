#!/usr/bin/env python3
"""Analyze what happened to Account #1 - where did the gains go?"""

import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from collections import defaultdict

load_dotenv()
api = tradeapi.REST(
    key_id=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    base_url=os.getenv('ALPACA_BASE_URL')
)

print('='*70)
print('ACCOUNT #1 - WHAT HAPPENED TO THE GAINS?')
print('='*70)

# Get portfolio history
portfolio = api.get_portfolio_history(period='3M', timeframe='1D')

starting_equity = portfolio.equity[0] if portfolio.equity[0] > 0 else 1000000
current_equity = portfolio.equity[-1]
peak_equity = max(portfolio.equity)

print(f'\n[PORTFOLIO TIMELINE]')
print(f'Starting (3 months ago): ${starting_equity:,.2f}')
print(f'Peak Equity: ${peak_equity:,.2f}')
print(f'Current Equity: ${current_equity:,.2f}')
print(f'\nDrawdown from peak: ${current_equity - peak_equity:+,.2f} ({((current_equity - peak_equity) / peak_equity) * 100:+.2f}%)')

# Find when we peaked
peak_idx = portfolio.equity.index(peak_equity)
peak_date = datetime.fromtimestamp(portfolio.timestamp[peak_idx])
print(f'\nPeak was on: {peak_date.strftime("%Y-%m-%d")}')
print(f'Days since peak: {(datetime.now() - peak_date).days} days')

# Get all closed positions
print(f'\n{'='*70}')
print('[RECENT TRADE ACTIVITY]')
print('='*70)

try:
    activities = api.get_activities(activity_types='FILL', page_size=200)

    print(f'\nTotal fills in last 90 days: {len(activities)}')
    print(f'\nMost recent 30 trades:')
    print(f'{"Date":<12} {"Action":<6} {"Qty":>8} {"Symbol":<25} {"Price":>12}')
    print('-'*70)

    for activity in activities[:30]:
        date_str = activity.transaction_time.strftime('%Y-%m-%d')
        side = activity.side.upper()
        qty = float(activity.qty)
        symbol = activity.symbol
        price = float(activity.price)

        print(f'{date_str:<12} {side:<6} {qty:>8.0f} {symbol:<25} ${price:>11.2f}')

    # Analyze by month
    print(f'\n{'='*70}')
    print('[MONTHLY ACTIVITY]')
    print('='*70)

    monthly_trades = defaultdict(int)
    for activity in activities:
        month = activity.transaction_time.strftime('%Y-%m')
        monthly_trades[month] += 1

    for month in sorted(monthly_trades.keys(), reverse=True):
        print(f'{month}: {monthly_trades[month]} trades')

except Exception as e:
    print(f'Error getting activities: {e}')

# Check current open positions for losses
print(f'\n{'='*70}')
print('[CURRENT LOSING POSITIONS]')
print('='*70)

positions = api.list_positions()
losing_positions = [p for p in positions if float(p.unrealized_pl or 0) < 0]
losing_positions.sort(key=lambda x: float(x.unrealized_pl), reverse=False)

total_unrealized_loss = sum(float(p.unrealized_pl) for p in losing_positions)

print(f'\nLosing positions: {len(losing_positions)}/{len(positions)}')
print(f'Total unrealized losses: ${total_unrealized_loss:,.2f}')
print(f'\nTop 10 losers:')
print(f'{"Symbol":<25} {"Qty":>8} {"Entry":>10} {"Current":>10} {"P&L":>12}')
print('-'*70)

for pos in losing_positions[:10]:
    print(f'{pos.symbol:<25} {float(pos.qty):>8.0f} ${float(pos.avg_entry_price):>9.2f} ${float(pos.current_price):>9.2f} ${float(pos.unrealized_pl):>11.2f}')

print(f'\n{'='*70}')
print('[SUMMARY]')
print('='*70)

account = api.get_account()
print(f'\nAccount started with: $1,000,000.00')
print(f'Peaked at: ${peak_equity:,.2f}')
print(f'Current equity: ${float(account.equity):,.2f}')
print(f'\nTotal loss from peak: ${float(account.equity) - peak_equity:+,.2f}')
print(f'Realized losses (closed trades): ${float(account.equity) - peak_equity - total_unrealized_loss:+,.2f}')
print(f'Unrealized losses (open positions): ${total_unrealized_loss:+,.2f}')
