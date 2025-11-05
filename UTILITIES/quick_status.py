#!/usr/bin/env python3
"""Quick status check with P/L calculation"""
import os
from dotenv import load_dotenv
import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.trades as trades

load_dotenv()

client = oandapyV20.API(access_token=os.getenv('OANDA_API_KEY'))
account_id = '101-001-37330890-001'

# Starting balance (after you closed bad trades)
STARTING_BALANCE = 198899.52

# Get current account info
r = accounts.AccountDetails(accountID=account_id)
account_data = client.request(r)
account = account_data['account']

current_balance = float(account['balance'])
unrealized_pl = float(account.get('unrealizedPL', 0))
total_pl = current_balance - STARTING_BALANCE

print("="*70)
print("FOREX BOT PERFORMANCE - DAY 2")
print("="*70)
print(f"\nStarting Balance: ${STARTING_BALANCE:,.2f}")
print(f"Current Balance:  ${current_balance:,.2f}")
print(f"Unrealized P/L:   ${unrealized_pl:,.2f}")
print(f"\n{'='*70}")
print(f"TOTAL P/L: ${total_pl:,.2f} ({total_pl/STARTING_BALANCE*100:.3f}%)")
print(f"{'='*70}")

# Weekly target progress
WEEKLY_TARGET = 500.00
remaining = WEEKLY_TARGET - total_pl
print(f"\nWeekly Target: ${WEEKLY_TARGET:.2f}")
print(f"Progress: ${total_pl:.2f} / ${WEEKLY_TARGET:.2f}")
print(f"Remaining: ${remaining:.2f} ({remaining/WEEKLY_TARGET*100:.1f}% to go)")

# Get active trades
r = trades.TradesList(accountID=account_id)
trades_data = client.request(r)
active_trades = trades_data.get('trades', [])

print(f"\n{'='*70}")
print(f"ACTIVE POSITIONS: {len(active_trades)}")
print(f"{'='*70}")

for t in active_trades:
    pair = t['instrument']
    units = int(t['currentUnits'])
    direction = "LONG" if units > 0 else "SHORT"
    entry_price = float(t['price'])
    unrealized = float(t.get('unrealizedPL', 0))

    print(f"\n{pair} {direction}")
    print(f"  Units: {abs(units):,}")
    print(f"  Entry: {entry_price:.5f}")
    print(f"  P/L: ${unrealized:,.2f}")
