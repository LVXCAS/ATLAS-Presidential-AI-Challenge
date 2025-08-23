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

print('WHAT CAN WE TRADE RIGHT NOW?')
print('=' * 50)

# Check major assets
symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'QQQ', 'BTCUSD', 'ETHUSD']

print('ASSET PRICES & STATUS:')
for symbol in symbols:
    try:
        asset = api.get_asset(symbol)
        latest = api.get_latest_trade(symbol)
        print(f'{symbol:8} - ${latest.price:8.2f} | Tradable: {asset.tradable} | Class: {asset.asset_class}')
    except Exception as e:
        print(f'{symbol:8} - ERROR: {str(e)[:50]}')

print('\nMARKET STATUS:')
clock = api.get_clock()
print(f'Market Open: {clock.is_open}')
print(f'Current Time: {clock.timestamp}')

print('\nCAN WE PLACE A TEST ORDER?')
print('Let\'s try a small test order...')

try:
    # Try to place a very small test order for 1 share of SPY
    print('Testing order placement (will cancel immediately)...')
    
    test_order = api.submit_order(
        symbol='SPY',
        qty=1,
        side='buy',
        type='market',
        time_in_force='day'
    )
    
    print(f'TEST ORDER PLACED: {test_order.id}')
    print(f'Status: {test_order.status}')
    
    # Cancel the test order immediately
    api.cancel_order(test_order.id)
    print('TEST ORDER CANCELLED - System working!')
    
except Exception as e:
    print(f'ORDER ERROR: {e}')
    if 'market is closed' in str(e).lower():
        print('MARKET IS CLOSED - But we can place orders for when it opens!')
    elif 'extended hours' in str(e).lower():
        print('EXTENDED HOURS TRADING AVAILABLE!')