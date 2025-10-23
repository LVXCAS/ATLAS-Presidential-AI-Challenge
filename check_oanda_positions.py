import os
from dotenv import load_dotenv
load_dotenv()

import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts

oanda_token = os.getenv('OANDA_API_KEY')
oanda_account_id = '101-001-37330890-001'

client = API(access_token=oanda_token, environment='practice')

# Account
r = accounts.AccountSummary(accountID=oanda_account_id)
response = client.request(r)
balance = float(response['account']['balance'])
unrealized_pl = float(response['account'].get('unrealizedPL', 0))

print(f"Account: {oanda_account_id}")
print(f"Balance: ${balance:,.2f}")
print(f"Unrealized P/L: ${unrealized_pl:,.2f}")

# Positions
r = positions.OpenPositions(accountID=oanda_account_id)
response = client.request(r)
positions_list = response.get('positions', [])

active = []
for pos in positions_list:
    long_units = float(pos.get('long', {}).get('units', 0))
    short_units = float(pos.get('short', {}).get('units', 0))
    if long_units != 0 or short_units != 0:
        active.append(pos)

print(f"\n=== {len(active)} ACTIVE POSITIONS ===")

for pos in active:
    instrument = pos['instrument']
    long_units = float(pos.get('long', {}).get('units', 0))
    short_units = float(pos.get('short', {}).get('units', 0))

    if long_units != 0:
        long_pl = float(pos['long'].get('unrealizedPL', 0))
        print(f"{instrument} LONG: {int(long_units)} units | P/L: ${long_pl:.2f}")
    if short_units != 0:
        short_pl = float(pos['short'].get('unrealizedPL', 0))
        print(f"{instrument} SHORT: {int(short_units)} units | P/L: ${short_pl:.2f}")
