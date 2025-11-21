import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()
api = tradeapi.REST(
    key_id=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    base_url=os.getenv('ALPACA_BASE_URL')
)

account = api.get_account()
print(f'Account: {account.account_number}')
print(f'Equity: ${float(account.equity):,.2f}')
print(f'Options BP: ${float(account.options_buying_power):,.2f}')
print(f'Regular BP: ${float(account.buying_power):,.2f}')
print(f'Cash: ${float(account.cash):,.2f}')
positions = api.list_positions()
print(f'Open Positions: {len(positions)}')
