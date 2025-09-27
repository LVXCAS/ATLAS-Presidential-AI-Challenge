#!/usr/bin/env python3
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

load_dotenv()

api = tradeapi.REST(
    os.getenv('ALPACA_API_KEY'),
    os.getenv('ALPACA_SECRET_KEY'),
    os.getenv('ALPACA_BASE_URL')
)

print("=== CURRENT POSITIONS ===")
positions = api.list_positions()
for pos in positions:
    print(f"{pos.symbol}: {pos.qty} shares, ${float(pos.market_value):,.2f}")

print("\n=== RECENT ORDERS ===")
orders = api.list_orders(status='all', limit=10)
for order in orders[:5]:
    print(f"{order.symbol}: {order.side} {order.qty}, Status: {order.status}")