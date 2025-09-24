"""
VEGAS HOURS POSITION ADJUSTMENT
Adjust positions for proper Vegas timezone trading
"""

import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import os
from datetime import datetime
import pytz

load_dotenv(override=True)

def adjust_for_vegas_hours():
    """Adjust positions for Vegas trading hours"""

    api = tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        os.getenv('ALPACA_BASE_URL'),
        api_version='v2'
    )

    vegas_tz = pytz.timezone('America/Los_Angeles')
    now_vegas = datetime.now(vegas_tz)

    print('=== ADJUSTING POSITIONS FOR VEGAS TRADING HOURS ===')
    print(f'Current Vegas Time: {now_vegas.strftime("%H:%M:%S")}')

    account = api.get_account()
    positions = api.list_positions()

    print(f'Portfolio Value: ${float(account.portfolio_value):,.0f}')
    print(f'Available Buying Power: ${float(account.buying_power):,.0f}')

    print('\n[CURRENT POSITIONS TO ADJUST]')

    total_value = 0
    total_pl = 0

    for pos in positions:
        market_value = float(pos.market_value)
        unrealized_pl = float(pos.unrealized_pl)

        total_value += abs(market_value)
        total_pl += unrealized_pl

        print(f'{pos.symbol}: {pos.qty} shares | ${abs(market_value):,.0f} | P&L: ${unrealized_pl:+,.0f}')

    print(f'\nTotal Position Value: ${total_value:,.0f}')
    print(f'Total P&L: ${total_pl:+,.0f}')

    # Strategy options for Vegas hours
    print(f'\n[VEGAS HOURS STRATEGY OPTIONS]')
    print(f'1. HOLD current positions and monitor during Vegas day')
    print(f'2. SELL all and redeploy with better options timing')
    print(f'3. PARTIAL adjustment - sell losers, keep winners')
    print(f'4. ADD to positions if market recovers')

    print(f'\nWhat would you like to do with your ${total_value:,.0f} in positions?')

    return {
        'total_value': total_value,
        'total_pl': total_pl,
        'positions': len(positions),
        'vegas_time': now_vegas.strftime("%H:%M:%S")
    }

if __name__ == "__main__":
    adjust_for_vegas_hours()