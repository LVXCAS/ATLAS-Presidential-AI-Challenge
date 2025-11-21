"""
Daily Performance Tracker
Run this once per day to track progress toward E8 challenge
"""

import os
from dotenv import load_dotenv
load_dotenv()

import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
import json
from datetime import datetime

oanda_token = os.getenv('OANDA_API_KEY')
oanda_account = os.getenv('OANDA_ACCOUNT_ID')
client = API(access_token=oanda_token, environment='practice')

# Get current balance
r = accounts.AccountSummary(accountID=oanda_account)
resp = client.request(r)
balance = float(resp['account']['balance'])
unrealized_pl = float(resp['account']['unrealizedPL'])

# Load tracking history
history_file = 'daily_tracker_history.json'
try:
    with open(history_file, 'r') as f:
        history = json.load(f)
except:
    history = {
        "start_date": "2025-11-04",
        "start_balance": 187190,
        "records": []
    }

# Add today's record
today = datetime.now().strftime('%Y-%m-%d')
record = {
    "date": today,
    "balance": balance,
    "unrealized_pl": unrealized_pl,
    "total_equity": balance + unrealized_pl
}

# Don't duplicate if already recorded today
if not history['records'] or history['records'][-1]['date'] != today:
    history['records'].append(record)

# Save history
with open(history_file, 'w') as f:
    json.dump(history, f, indent=2)

# Calculate metrics
start_balance = history['start_balance']
total_profit = balance - start_balance
profit_pct = (total_profit / start_balance) * 100
days_trading = len(history['records'])
daily_avg = total_profit / days_trading if days_trading > 0 else 0

# E8 Challenge targets
e8_250k_cost = 1227
e8_500k_cost = 1627
days_to_250k = (e8_250k_cost - total_profit) / daily_avg if daily_avg > 0 and total_profit < e8_250k_cost else 0
days_to_500k = (e8_500k_cost - total_profit) / daily_avg if daily_avg > 0 and total_profit < e8_500k_cost else 0

print("\n" + "="*70)
print(f"DAILY PERFORMANCE TRACKER - {today}")
print("="*70)

print(f"\nCURRENT STATUS:")
print(f"  Balance: ${balance:,.2f}")
print(f"  Unrealized P/L: ${unrealized_pl:,.2f}")
print(f"  Total Equity: ${balance + unrealized_pl:,.2f}")

print(f"\nPROGRESS SINCE {history['start_date']}:")
print(f"  Starting Balance: ${start_balance:,.2f}")
print(f"  Total Profit: ${total_profit:,.2f} ({profit_pct:+.2f}%)")
print(f"  Days Trading: {days_trading}")
print(f"  Daily Average: ${daily_avg:,.2f}")

print(f"\nE8 CHALLENGE PROGRESS:")
print(f"  $250K Challenge Cost: ${e8_250k_cost}")
if total_profit >= e8_250k_cost:
    print(f"  STATUS: READY TO BUY! You have ${total_profit:,.2f}")
    print(f"  Surplus: ${total_profit - e8_250k_cost:,.2f}")
else:
    print(f"  Remaining: ${e8_250k_cost - total_profit:,.2f}")
    print(f"  Days to Target: {days_to_250k:.1f} days at current rate")
    print(f"  Target Date: ~{datetime.now().strftime('%b')} {int(datetime.now().strftime('%d')) + int(days_to_250k)}")

print(f"\n  $500K Challenge Cost: ${e8_500k_cost}")
if total_profit >= e8_500k_cost:
    print(f"  STATUS: READY TO BUY! You have ${total_profit:,.2f}")
    print(f"  Surplus: ${total_profit - e8_500k_cost:,.2f}")
else:
    print(f"  Remaining: ${e8_500k_cost - total_profit:,.2f}")
    print(f"  Days to Target: {days_to_500k:.1f} days at current rate")

# Show last 7 days
print(f"\nLAST 7 DAYS PERFORMANCE:")
print(f"{'Date':<12} {'Balance':<15} {'Daily Change':<15} {'Change %'}")
print("-" * 70)

recent = history['records'][-7:]
for i, rec in enumerate(recent):
    bal = rec['balance']
    if i > 0:
        prev_bal = recent[i-1]['balance']
        change = bal - prev_bal
        change_pct = (change / prev_bal) * 100
        print(f"{rec['date']:<12} ${bal:>13,.2f} ${change:>13,.2f} {change_pct:>+6.2f}%")
    else:
        print(f"{rec['date']:<12} ${bal:>13,.2f} {'--':>14} {'--':>6}")

# Weekly target
print(f"\nWEEK 1 TARGET (Nov 4-10):")
target_weekly = 2000
weekly_progress = total_profit
print(f"  Target: $2,000 profit")
print(f"  Current: ${weekly_progress:,.2f}")
if weekly_progress >= target_weekly:
    print(f"  STATUS: TARGET HIT! (+${weekly_progress - target_weekly:,.2f})")
else:
    print(f"  Remaining: ${target_weekly - weekly_progress:,.2f}")
    print(f"  Progress: {weekly_progress/target_weekly*100:.1f}%")

print("\n" + "="*70)
print("Run: python DAILY_TRACKER.py (once per day)")
print("="*70 + "\n")
