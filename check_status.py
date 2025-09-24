"""
Check current portfolio status and performance
"""

import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
load_dotenv()

print('CURRENT PORTFOLIO STATUS')
print('=' * 30)

try:
    alpaca = tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        os.getenv('ALPACA_BASE_URL'),
        api_version='v2'
    )

    account = alpaca.get_account()

    print('ACCOUNT OVERVIEW:')
    print(f'Portfolio Value: ${float(account.portfolio_value):,.0f}')
    print(f'Cash: ${float(account.cash):,.0f}')
    print(f'Buying Power: ${float(account.buying_power):,.0f}')
    # Day P&L may not be available in all API versions
    try:
        day_pl = float(account.todays_pl) if hasattr(account, 'todays_pl') else 0
        day_plpc = float(account.todays_plpc) if hasattr(account, 'todays_plpc') else 0
        print(f'Day P&L: ${day_pl:,.0f} ({day_plpc*100:+.2f}%)')
    except:
        print('Day P&L: Not available')
    print()

    # Get current positions
    positions = alpaca.list_positions()

    print(f'CURRENT POSITIONS: {len(positions)}')
    total_market_value = 0
    total_unrealized_pl = 0

    for pos in positions:
        symbol = pos.symbol
        qty = int(pos.qty)
        market_value = float(pos.market_value)
        unrealized_pl = float(pos.unrealized_pl)
        unrealized_plpc = float(pos.unrealized_plpc)

        total_market_value += market_value
        total_unrealized_pl += unrealized_pl

        side = 'LONG' if qty > 0 else 'SHORT'

        # Check if it's an options position
        is_options = len(symbol) > 6 and any(c in symbol for c in ['C', 'P'])
        position_type = 'OPTIONS' if is_options else 'STOCK'

        print(f'{symbol}: {abs(qty)} {position_type} {side} | ${market_value:,.0f} | P&L: ${unrealized_pl:+,.0f} ({unrealized_plpc:+.1%})')

    print()
    print(f'TOTAL POSITIONS VALUE: ${total_market_value:,.0f}')
    print(f'TOTAL UNREALIZED P&L: ${total_unrealized_pl:+,.0f}')

    # Performance since start
    if float(account.portfolio_value) != 1000000:
        total_return = (float(account.portfolio_value) - 1000000) / 1000000 * 100
        print(f'TOTAL RETURN: {total_return:+.2f}% (from $1M start)')

    # Get recent orders
    orders = alpaca.list_orders(status='all', limit=10)

    print()
    print('RECENT ORDERS (Last 10):')
    for i, order in enumerate(orders):
        symbol = order.symbol
        side = order.side.upper()
        qty = int(order.qty)
        status = order.status
        submitted_at = order.submitted_at.strftime('%H:%M:%S')

        # Check if options
        is_options = len(symbol) > 6 and any(c in symbol for c in ['C', 'P'])
        order_type = 'OPTIONS' if is_options else 'STOCK'

        print(f'{i+1}. {submitted_at}: {side} {qty} {symbol} ({order_type}) - {status}')

except Exception as e:
    print(f'Error getting portfolio status: {e}')