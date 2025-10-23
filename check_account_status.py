#!/usr/bin/env python3
"""Quick account status checker"""

import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

load_dotenv('.env.paper')

api = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True)

# Get account
account = api.get_account()
print('=== ACCOUNT STATUS ===')
print(f'Portfolio Value: ${float(account.portfolio_value):,.2f}')
print(f'Cash: ${float(account.cash):,.2f}')
print(f'Buying Power: ${float(account.buying_power):,.2f}')
print(f'Options Buying Power: ${float(account.options_buying_power):,.2f}')
print(f'Options Approved Level: {account.options_approved_level}')
print(f'Options Trading Level: {account.options_trading_level}')

# Get positions
positions = api.get_all_positions()
print(f'\n=== OPEN POSITIONS ({len(positions)}) ===')
for pos in positions:
    print(f'{pos.symbol}: {pos.qty} @ ${float(pos.avg_entry_price):.2f} | P&L: ${float(pos.unrealized_pl):+.2f}')

if not positions:
    print('No open positions')

# Get recent orders (FIXED: Correct API signature)
orders = api.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=10))
print(f'\n=== RECENT ORDERS (last 10) ===')
for order in orders:
    print(f'{order.symbol}: {order.side} {order.qty} @ {order.type} | Status: {order.status}')
