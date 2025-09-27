
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
load_dotenv()
api = tradeapi.REST(key_id=os.getenv('ALPACA_API_KEY'), secret_key=os.getenv('ALPACA_SECRET_KEY'), base_url=os.getenv('ALPACA_BASE_URL'), api_version='v2')
positions = api.list_positions()
print('CURRENT POSITIONS:')
for pos in positions:
    print(pos.symbol, pos.qty, pos.unrealized_pl)
if not positions:
    print('No positions held')

