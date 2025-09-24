#!/usr/bin/env python3

import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from datetime import datetime

load_dotenv()

try:
    # Initialize Alpaca
    api = tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    )

    # Get account info
    account = api.get_account()

    print("=== PAPER TRADING ACCOUNT STATUS ===")
    print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
    print(f"Buying Power: ${float(account.buying_power):,.2f}")
    print(f"Cash: ${float(account.cash):,.2f}")
    print(f"Day P&L: ${float(account.equity) - 1000000:+,.2f}")
    print(f"Total Equity: ${float(account.equity):,.2f}")

    # Get positions
    positions = api.list_positions()
    print(f"\n=== CURRENT POSITIONS ({len(positions)}) ===")
    total_position_value = 0
    for pos in positions:
        market_value = float(pos.market_value)
        total_position_value += market_value
        unrealized_pl = float(pos.unrealized_pl)
        print(f"{pos.symbol}: {pos.qty} shares @ ${float(pos.avg_cost_basis):.2f} | Value: ${market_value:.2f} | P&L: ${unrealized_pl:+.2f}")

    print(f"\nTotal Position Value: ${total_position_value:.2f}")

    # Get recent orders
    orders = api.list_orders(status='all', limit=10)
    print(f"\n=== RECENT ORDERS ({len(orders)}) ===")
    for order in orders:
        status = order.status
        filled_price = float(order.filled_avg_price) if order.filled_avg_price else 0
        price_str = f"@ ${filled_price:.2f}" if filled_price > 0 else "PENDING"
        print(f"{str(order.created_at)[:19]} | {order.symbol} {order.side.upper()} {order.qty} {price_str} - {status}")

except Exception as e:
    print(f"Error connecting to Alpaca: {e}")
    print("Running in SIMULATION MODE")