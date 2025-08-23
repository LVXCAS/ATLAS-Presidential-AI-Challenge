import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

load_dotenv()
api = tradeapi.REST(
    os.getenv('ALPACA_API_KEY'),
    os.getenv('ALPACA_SECRET_KEY'),
    os.getenv('ALPACA_BASE_URL'),
    api_version='v2'
)

print('[CURRENT POSITIONS]')
positions = api.list_positions()
for pos in positions:
    pnl = float(pos.unrealized_pl)
    market_val = float(pos.market_value)
    # Try different attribute names for average cost
    try:
        avg_cost = pos.avg_cost
    except:
        try:
            avg_cost = pos.cost_basis
        except:
            avg_cost = "N/A"
    
    print(f'  {pos.symbol}: {pos.qty} @ ${avg_cost} | Value: ${market_val:,.2f} | P&L: ${pnl:,.2f}')

if not positions:
    print('  No positions')

print()
print('[ACCOUNT STATUS]')
account = api.get_account()
print(f'Portfolio Value: ${float(account.portfolio_value):,.2f}')
print(f'Buying Power: ${float(account.buying_power):,.2f}')
print(f'Cash: ${float(account.cash):,.2f}')
print(f'Day Trade Count: {account.daytrade_count}')

print()
print('[RECENT CRYPTO ORDER]')
orders = api.list_orders(status='all', limit=5)
for order in orders:
    if 'USD' in order.symbol and len(order.symbol) > 5:  # Crypto pairs
        print(f'  {order.symbol}: {order.side} {order.qty or order.notional} - {order.status}')