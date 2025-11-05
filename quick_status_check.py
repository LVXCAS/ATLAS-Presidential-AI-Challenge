import os
from dotenv import load_dotenv
load_dotenv()

import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts

client = API(access_token=os.getenv('OANDA_API_KEY'), environment='practice')
r = accounts.AccountSummary(accountID=os.getenv('OANDA_ACCOUNT_ID'))
resp = client.request(r)

print("OANDA Account Status:")
print(f"  Balance: ${float(resp['account']['balance']):,.2f}")
print(f"  Unrealized P/L: ${float(resp['account']['unrealizedPL']):,.2f}")
print(f"  Open Positions: {resp['account']['openPositionCount']}")
print(f"  Open Trades: {resp['account']['openTradeCount']}")
