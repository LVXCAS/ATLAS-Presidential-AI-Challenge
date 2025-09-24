import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import os

load_dotenv(override=True)

api = tradeapi.REST(
    os.getenv('ALPACA_API_KEY'),
    os.getenv('ALPACA_SECRET_KEY'),
    os.getenv('ALPACA_BASE_URL'),
    api_version='v2'
)

account = api.get_account()
positions = api.list_positions()

portfolio_value = float(account.portfolio_value)
total_pl_today = 0
total_position_value = 0

print('=== DAY 1 SYNTHETIC OPTIONS PERFORMANCE ===')
print(f'Portfolio Value: ${portfolio_value:,.0f}')

if positions:
    print('\n[TODAYS RESULTS]')

    for pos in positions:
        market_value = float(pos.market_value)
        unrealized_pl = float(pos.unrealized_pl)
        cost_basis = market_value - unrealized_pl
        pl_percent = (unrealized_pl / cost_basis) * 100 if cost_basis > 0 else 0

        synthetic_exposure = abs(market_value) * 3

        total_pl_today += unrealized_pl
        total_position_value += abs(market_value)

        print(f'{pos.symbol}: ${abs(market_value):,.0f} | P&L: ${unrealized_pl:+.0f} ({pl_percent:+.2f}%)')
        print(f'  3x Exposure: ${synthetic_exposure:,.0f}')

    total_cost_basis = total_position_value - total_pl_today
    total_pl_percent = (total_pl_today / total_cost_basis) * 100 if total_cost_basis > 0 else 0

    print(f'\nTOTAL P&L DAY 1: ${total_pl_today:+.0f} ({total_pl_percent:+.2f}%)')
    print(f'Total Synthetic Exposure: ${total_position_value * 3:,.0f}')

    if total_pl_percent > 0:
        daily_return = total_pl_percent / 100
        monthly_projection = ((1 + daily_return) ** 21 - 1) * 100

        print(f'\nIf this rate continues monthly: {monthly_projection:.1f}%')
        print(f'Target: 41.67% monthly')

        if monthly_projection >= 41.67:
            print('SUCCESS: ON TRACK for monthly target!')
        else:
            gap = 41.67 - monthly_projection
            print(f'NEED: {gap:.1f}% more monthly performance')

print('\nSYNTHETIC OPTIONS STATUS: ACTIVE')