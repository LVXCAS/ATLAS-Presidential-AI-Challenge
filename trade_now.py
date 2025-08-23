#!/usr/bin/env python3
"""
HIVE TRADE - SIMPLE TRADING COMMANDS
Place trades that work right now!
"""

import alpaca_trade_api as tradeapi
import os
import sys
from dotenv import load_dotenv

def trade_now():
    load_dotenv()
    
    api = tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        os.getenv('ALPACA_BASE_URL'),
        api_version='v2'
    )
    
    print("*** HIVE TRADE - LIVE TRADING SYSTEM ***")
    print("=" * 50)
    
    # Show account
    account = api.get_account()
    print(f"[BUYING POWER] ${float(account.buying_power):,.2f}")
    print(f"[PORTFOLIO VALUE] ${float(account.portfolio_value):,.2f}")
    
    # Show current positions
    positions = api.list_positions()
    if positions:
        print(f"\n[CURRENT POSITIONS] ({len(positions)}):")
        for pos in positions:
            pnl = float(pos.unrealized_pl)
            print(f"  {pos.symbol}: {pos.qty} shares | P&L: ${pnl:,.2f}")
    else:
        print("\n[CURRENT POSITIONS] None")
    
    print("\n[MARKET STATUS]")
    clock = api.get_clock()
    print(f"  Market Open: {clock.is_open}")
    if not clock.is_open:
        print(f"  Next Open: {clock.next_open}")
        print("  >> Orders placed now will execute when market opens!")
    
    print("\n[TRADING COMMANDS]")
    print("  python trade_now.py buy SPY 10")
    print("  python trade_now.py sell AAPL 5") 
    print("  python trade_now.py quote TSLA")
    print("  python trade_now.py positions")
    print("  python trade_now.py orders")

def execute_command(command, symbol=None, qty=None):
    load_dotenv()
    
    api = tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        os.getenv('ALPACA_BASE_URL'),
        api_version='v2'
    )
    
    if command == 'quote' and symbol:
        try:
            trade = api.get_latest_trade(symbol)
            quote = api.get_latest_quote(symbol)
            print(f"\n[{symbol} QUOTE]")
            print(f"  Last: ${trade.price}")
            print(f"  Bid: ${quote.bid_price}")
            print(f"  Ask: ${quote.ask_price}")
        except Exception as e:
            print(f"[ERROR] {e}")
    
    elif command in ['buy', 'sell'] and symbol and qty:
        try:
            print(f"\n[PLACING {command.upper()} ORDER]")
            print(f"  Symbol: {symbol}")
            print(f"  Quantity: {qty}")
            
            # Get quote first
            trade = api.get_latest_trade(symbol)
            estimated_cost = float(trade.price) * int(qty)
            print(f"  Estimated Cost: ${estimated_cost:,.2f}")
            
            # Place order
            order = api.submit_order(
                symbol=symbol,
                qty=int(qty),
                side=command,
                type='market',
                time_in_force='day'
            )
            
            print(f"\n[ORDER PLACED!]")
            print(f"  Order ID: {order.id}")
            print(f"  Status: {order.status}")
            print(f"  Time: {order.created_at}")
            
        except Exception as e:
            print(f"[ORDER ERROR] {e}")
    
    elif command == 'positions':
        positions = api.list_positions()
        if positions:
            print(f"\n[CURRENT POSITIONS]")
            total_value = 0
            for pos in positions:
                market_val = float(pos.market_value)
                pnl = float(pos.unrealized_pl)
                total_value += market_val
                print(f"  {pos.symbol}: {pos.qty} shares @ ${pos.avg_cost} | Value: ${market_val:,.2f} | P&L: ${pnl:,.2f}")
            print(f"\n  [TOTAL VALUE] ${total_value:,.2f}")
        else:
            print("\n[CURRENT POSITIONS] None")
    
    elif command == 'orders':
        orders = api.list_orders(status='all', limit=5)
        if orders:
            print(f"\n[RECENT ORDERS]")
            for order in orders:
                status_icon = "[FILLED]" if order.status == "filled" else "[PENDING]" if order.status == "accepted" else "[FAILED]"
                print(f"  {status_icon} {order.symbol} {order.side} {order.qty} - {order.status}")
        else:
            print("\n[RECENT ORDERS] None")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        trade_now()
    else:
        command = sys.argv[1].lower()
        symbol = sys.argv[2].upper() if len(sys.argv) > 2 else None
        qty = sys.argv[3] if len(sys.argv) > 3 else None
        execute_command(command, symbol, qty)