import os
from dotenv import load_dotenv
load_dotenv()

import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.pricing as pricing

oanda_token = os.getenv('OANDA_API_KEY')
oanda_account_id = '101-001-37330890-001'

client = API(access_token=oanda_token, environment='practice')

# Account
r = accounts.AccountSummary(accountID=oanda_account_id)
response = client.request(r)
balance = float(response['account']['balance'])
unrealized_pl = float(response['account'].get('unrealizedPL', 0))

# Overall percentage gain/loss
overall_pct = (unrealized_pl / balance) * 100

print(f"Account: {oanda_account_id}")
print(f"Balance: ${balance:,.2f}")
print(f"Unrealized P/L: ${unrealized_pl:,.2f} ({overall_pct:+.3f}%)")

# Get detailed trade information (not just positions)
r = trades.TradesList(accountID=oanda_account_id)
response = client.request(r)
trades_list = response.get('trades', [])

print(f"\n=== {len(trades_list)} ACTIVE POSITIONS ===")

if trades_list:
    for trade in trades_list:
        instrument = trade['instrument']
        units = int(trade['currentUnits'])
        direction = "LONG" if units > 0 else "SHORT"
        entry_price = float(trade['price'])
        unrealized = float(trade.get('unrealizedPL', 0))

        # Get current price
        params = {"instruments": instrument}
        r = pricing.PricingInfo(accountID=oanda_account_id, params=params)
        price_data = client.request(r)

        if price_data.get('prices'):
            current_price = float(price_data['prices'][0]['closeoutBid'])

            # Calculate percentage gain/loss on the trade
            # For LONG: (current - entry) / entry * 100
            # For SHORT: (entry - current) / entry * 100
            if units > 0:  # LONG
                price_change_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # SHORT
                price_change_pct = ((entry_price - current_price) / entry_price) * 100

            # Calculate percentage of account balance
            account_pct = (unrealized / balance) * 100

            # Calculate pips (for display)
            if 'JPY' in instrument:
                pip_value = 0.01
            else:
                pip_value = 0.0001

            pips = abs(current_price - entry_price) / pip_value
            if units < 0:  # SHORT
                pips = abs(entry_price - current_price) / pip_value

            # Determine color indicator
            indicator = "+" if unrealized > 0 else ""

            print(f"\n{instrument} {direction}:")
            print(f"  Units: {abs(units):,}")
            print(f"  Entry: {entry_price:.5f}")
            print(f"  Current: {current_price:.5f}")
            print(f"  Pips: {indicator}{pips:.1f}")
            print(f"  P/L: ${indicator}{unrealized:.2f} ({indicator}{price_change_pct:.3f}%)")
            print(f"  % of Account: {indicator}{account_pct:.3f}%")
else:
    print("No active positions")
