"""
COMMAND CENTER - One-stop dashboard for forex empire
Run this anytime you want a complete status update
"""

import os
from dotenv import load_dotenv
load_dotenv()

import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.trades as trades
import subprocess
from datetime import datetime

oanda_token = os.getenv('OANDA_API_KEY')
oanda_account = os.getenv('OANDA_ACCOUNT_ID')
client = API(access_token=oanda_token, environment='practice')

print("\n" + "="*80)
print(" " * 25 + "FOREX EMPIRE COMMAND CENTER")
print("="*80)
print(f"Time: {datetime.now().strftime('%I:%M %p, %A %B %d, %Y')}")
print("="*80)

# 1. BOT STATUS
print("\n[1] BOT STATUS")
print("-" * 80)
try:
    result = subprocess.run(['tasklist'], capture_output=True, text=True, shell=True)
    pythonw_count = result.stdout.count('pythonw.exe')

    if pythonw_count >= 1:
        print(f"Status: RUNNING ({pythonw_count} pythonw.exe processes)")
        print("Bot: WORKING_FOREX_OANDA.py")
        print("Pairs: EUR_USD, USD_JPY, GBP_USD, GBP_JPY")
        print("Scan Interval: 1 hour")
    else:
        print("Status: NOT RUNNING")
        print("Action: Run START_FOREX.bat or start pythonw WORKING_FOREX_OANDA.py")
except:
    print("Status: Unable to check")

# 2. ACCOUNT STATUS
print("\n[2] ACCOUNT STATUS")
print("-" * 80)

r = accounts.AccountSummary(accountID=oanda_account)
resp = client.request(r)

balance = float(resp['account']['balance'])
unrealized_pl = float(resp['account']['unrealizedPL'])
total_equity = balance + unrealized_pl
margin_used = float(resp['account']['marginUsed'])
margin_available = float(resp['account']['marginAvailable'])

start_balance = 187190  # Sunday start
total_profit = balance - start_balance
profit_pct = (total_profit / start_balance) * 100

print(f"Balance: ${balance:,.2f}")
print(f"Unrealized P/L: ${unrealized_pl:,.2f}")
print(f"Total Equity: ${total_equity:,.2f}")
print(f"Margin Used: ${margin_used:,.2f} / Available: ${margin_available:,.2f}")
print(f"\nWeekend Profit: ${total_profit:,.2f} ({profit_pct:+.2f}%)")
print(f"Started: $187,190 on Sunday Nov 3, 8 PM")

# 3. OPEN POSITIONS
print("\n[3] OPEN POSITIONS")
print("-" * 80)

r = trades.TradesList(accountID=oanda_account)
resp = client.request(r)
open_trades = resp.get('trades', [])

if open_trades:
    for i, trade in enumerate(open_trades, 1):
        pair = trade['instrument']
        trade_id = trade['id']
        units = int(trade['currentUnits'])
        direction = "LONG" if units > 0 else "SHORT"
        entry = float(trade['price'])
        unrealized = float(trade['unrealizedPL'])
        pct = (unrealized / balance) * 100

        print(f"\nPosition #{i}: {pair} {direction} (Trade #{trade_id})")
        print(f"  Entry: {entry}")
        print(f"  Units: {abs(units):,}")
        print(f"  P/L: ${unrealized:,.2f} ({pct:+.3f}%)")

        if 'stopLossOrder' in trade:
            stop = float(trade['stopLossOrder']['price'])
            stop_pct = abs((stop - entry) / entry * 100)
            print(f"  Stop Loss: {stop} ({stop_pct:.2f}% from entry)")

        if 'takeProfitOrder' in trade:
            target = float(trade['takeProfitOrder']['price'])
            target_pct = abs((target - entry) / entry * 100)
            print(f"  Take Profit: {target} ({target_pct:.2f}% from entry)")
else:
    print("No open positions (bot scanning for opportunities)")

# 4. E8 CHALLENGE STATUS
print("\n[4] E8 CHALLENGE READINESS")
print("-" * 80)

e8_250k = 1227
e8_500k = 1627

print(f"$250K Challenge: ${e8_250k}")
if total_profit >= e8_250k:
    print(f"  STATUS: READY (${total_profit:,.2f} available)")
    print(f"  Surplus: ${total_profit - e8_250k:,.2f}")
else:
    print(f"  Remaining: ${e8_250k - total_profit:,.2f}")

print(f"\n$500K Challenge: ${e8_500k}")
if total_profit >= e8_500k:
    print(f"  STATUS: READY (${total_profit:,.2f} available)")
    print(f"  Surplus: ${total_profit - e8_500k:,.2f}")
else:
    print(f"  Remaining: ${e8_500k - total_profit:,.2f}")

print(f"\nRECOMMENDATION: Wait until Nov 17 to validate consistency")
print(f"Target: 7-10 completed trades with 38-45% win rate")

# 5. WEEK 1 PROGRESS
print("\n[5] WEEK 1 PROGRESS (Nov 4-10)")
print("-" * 80)

week1_target = 2000
week1_progress = total_profit

print(f"Target: ${week1_target:,.2f}")
print(f"Current: ${week1_progress:,.2f}")
if week1_progress >= week1_target:
    print(f"STATUS: WEEK 1 TARGET HIT ON DAY 1! (+${week1_progress - week1_target:,.2f})")
    print(f"Progress: {week1_progress/week1_target*100:.1f}%")
else:
    print(f"Remaining: ${week1_target - week1_progress:,.2f}")
    print(f"Progress: {week1_progress/week1_target*100:.1f}%")

# 6. QUICK ACTIONS
print("\n[6] QUICK ACTIONS")
print("-" * 80)
print("python POSITION_SUMMARY.py     - Quick position check")
print("python DAILY_TRACKER.py        - Daily performance report")
print("python VIEW_FOREX_POSITIONS.py - Detailed position view")
print("python COMMAND_CENTER.py       - This dashboard")
print("\ntaskkill /F /IM pythonw.exe    - Stop bot (if needed)")
print("start pythonw WORKING_FOREX_OANDA.py - Start bot")

# 7. NEXT MILESTONES
print("\n[7] NEXT MILESTONES")
print("-" * 80)
print("Nov 10:  Complete Week 1 validation")
print("Nov 17:  Purchase first E8 challenge ($1,227)")
print("Dec 15:  Complete 8% challenge target ($20,000)")
print("Dec 22:  Receive $250K funded account")
print("Jan 2026: First $16K payout -> Buy 10 more challenges")

print("\n" + "="*80)
print(" " * 30 + "STATUS: OPERATIONAL")
print("="*80 + "\n")
