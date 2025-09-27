#!/usr/bin/env python3
"""
PORTFOLIO STATUS CHECKER
Check current portfolio performance and losses
"""

import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

def check_portfolio_status():
    """Check current portfolio status and losses"""
    load_dotenv()

    try:
        api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL')
        )

        account = api.get_account()
        positions = api.list_positions()

        print('CURRENT PORTFOLIO STATUS')
        print('=' * 60)
        print(f'Portfolio Value: ${float(account.portfolio_value):,.2f}')
        print(f'Equity: ${float(account.equity):,.2f}')
        print(f'Cash: ${float(account.cash):,.2f}')
        print(f'Buying Power: ${float(account.buying_power):,.2f}')
        print()
        # Try to get P/L data if available
        try:
            print(f'TODAY P/L: ${float(account.today_pl):,.2f}')
        except:
            print('TODAY P/L: Data not available')

        try:
            print(f'TOTAL UNREALIZED P/L: ${float(account.unrealized_pl):,.2f}')
        except:
            print('TOTAL UNREALIZED P/L: Data not available')

        print(f'\nPOSITIONS ({len(positions)} total):')
        print('Symbol   | Qty    | Avg Cost | Current  | Market Val |    P/L    | P/L%')
        print('-' * 75)

        total_pl = 0
        total_market_value = 0
        for pos in positions:
            qty = int(pos.qty)
            avg_price = float(pos.avg_entry_price)
            current_price = float(pos.market_value) / qty if qty != 0 else 0
            market_val = float(pos.market_value)
            pl = float(pos.unrealized_pl)
            pl_pct = float(pos.unrealized_plpc) * 100

            total_pl += pl
            total_market_value += market_val

            print(f'{pos.symbol:8} | {qty:6} | ${avg_price:7.2f} | ${current_price:7.2f} | ${market_val:9.2f} | ${pl:8.2f} | {pl_pct:5.1f}%')

        print('-' * 75)
        print(f'TOTAL MARKET VALUE: ${total_market_value:,.2f}')
        print(f'TOTAL UNREALIZED P/L: ${total_pl:,.2f}')

        # Calculate percentage of total portfolio value
        if float(account.portfolio_value) > 0:
            loss_pct = (total_pl / float(account.portfolio_value)) * 100
            print(f'TOTAL LOSS PERCENTAGE: {loss_pct:.2f}%')

        print(f'\nACCOUNT STATUS: {account.status}')
        print(f'PATTERN DAY TRADER: {account.pattern_day_trader}')
        print(f'TRADE SUSPENDED: {account.trade_suspended_by_user}')

        return {
            'portfolio_value': float(account.portfolio_value),
            'total_pl': float(account.unrealized_pl),
            'total_pl_pct': float(account.unrealized_plpc) * 100,
            'today_pl': float(account.today_pl),
            'today_pl_pct': float(account.today_plpc) * 100,
            'positions_count': len(positions),
            'buying_power': float(account.buying_power)
        }

    except Exception as e:
        print(f'Error getting account info: {e}')
        return None

if __name__ == "__main__":
    check_portfolio_status()