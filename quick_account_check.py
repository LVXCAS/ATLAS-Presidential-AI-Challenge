import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
load_dotenv()
api = tradeapi.REST(key_id=os.getenv('ALPACA_API_KEY'), secret_key=os.getenv('ALPACA_SECRET_KEY'), base_url=os.getenv('ALPACA_BASE_URL'), api_version='v2')
account = api.get_account()
print(f'Buying Power: ${float(account.buying_power):,.0f}')
print(f'Cash: ${float(account.cash):,.0f}')
print(f'Portfolio Value: ${float(account.portfolio_value):,.0f}')