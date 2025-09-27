
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
load_dotenv()
api = tradeapi.REST(key_id=os.getenv('ALPACA_API_KEY'), secret_key=os.getenv('ALPACA_SECRET_KEY'), base_url=os.getenv('ALPACA_BASE_URL'), api_version='v2')
orders = api.list_orders(status='all', limit=10)
print('RECENT ORDERS:')
for order in orders:
    print(f'{order.created_at}: {order.side} {order.qty} {order.symbol} - {order.status}')

