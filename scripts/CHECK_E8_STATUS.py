"""
E8 CHALLENGE STATUS CHECKER
Shows: Account balance, positions, P/L, time until profit target
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from tradelocker import TLAPI

load_dotenv()

# Connect to E8 account
tl = TLAPI(
    environment=os.getenv('TRADELOCKER_ENV'),
    username=os.getenv('TRADELOCKER_EMAIL'),
    password=os.getenv('TRADELOCKER_PASSWORD'),
    server=os.getenv('TRADELOCKER_SERVER')
)

print("=" * 70)
print("E8 CHALLENGE STATUS - LIVE ACCOUNT CHECK")
print("=" * 70)

# Get current time
now = datetime.now()
print(f"\n[SYSTEM TIME]")
print(f"Date: {now.strftime('%Y-%m-%d')}")
print(f"Time: {now.strftime('%I:%M %p %Z')}")
print(f"Day of Week: {now.strftime('%A')}")

# Get account info
accounts = tl.get_all_accounts()
if not accounts.empty:
    account = accounts.iloc[0]
    account_id = account['id']
    balance = account['accountBalance']

    print(f"\n[E8 ACCOUNT]")
    print(f"Account ID: {account_id}")
    print(f"Status: {account.get('status', 'UNKNOWN')}")
    print(f"Current Balance: ${balance:,.2f}")

    # Calculate progress
    STARTING_BALANCE = 200000
    PROFIT_TARGET = 18000  # 9% (CORRECTED)
    current_profit = balance - STARTING_BALANCE
    progress_pct = (current_profit / PROFIT_TARGET) * 100

    print(f"\n[CHALLENGE PROGRESS]")
    print(f"Starting Balance: ${STARTING_BALANCE:,.2f}")
    print(f"Profit Target: ${PROFIT_TARGET:,.2f} (9%)")
    print(f"Current Profit: ${current_profit:,.2f}")
    print(f"Progress: {progress_pct:.1f}%")
    print(f"Remaining: ${PROFIT_TARGET - current_profit:,.2f}")

    if current_profit >= PROFIT_TARGET:
        print(f"\n*** CHALLENGE PASSED! ***")
    elif current_profit > 0:
        days_running = max(1, (current_profit / STARTING_BALANCE) / 0.0025)  # 0.25% daily
        print(f"Days trading: ~{days_running:.0f}")

else:
    print("[ERROR] Could not fetch account info")
    exit(1)

# Get open positions
positions = tl.get_all_positions()

print(f"\n[OPEN POSITIONS]")
if positions.empty:
    print("No open positions")
else:
    print(f"Total Positions: {len(positions)}")

    total_pnl = 0
    for _, pos in positions.iterrows():
        pair_id = pos.get('tradableInstrumentId', 'Unknown')
        side = pos.get('side', 'Unknown')
        qty = pos.get('qty', 0)
        entry = pos.get('avgPrice', 0)
        pnl = pos.get('unrealizedPl', 0)
        total_pnl += pnl

        print(f"\n  Position:")
        print(f"    Pair ID: {pair_id}")
        print(f"    Side: {side}")
        print(f"    Quantity: {qty:,.0f}")
        print(f"    Entry Price: {entry:.5f}")
        print(f"    Unrealized P/L: ${pnl:,.2f}")

    print(f"\n  Total Unrealized P/L: ${total_pnl:,.2f}")

    # Calculate total balance including unrealized
    total_balance = balance + total_pnl
    total_profit = total_balance - STARTING_BALANCE
    print(f"  Total Balance (with unrealized): ${total_balance:,.2f}")
    print(f"  Total Profit (with unrealized): ${total_profit:,.2f}")

# Check drawdown
print(f"\n[DRAWDOWN CHECK]")
MAX_DRAWDOWN = 0.06  # 6%
max_dd_amount = STARTING_BALANCE * MAX_DRAWDOWN

if current_profit < 0:
    current_dd_pct = abs(current_profit / STARTING_BALANCE) * 100
    print(f"Current Drawdown: {current_dd_pct:.2f}%")
    print(f"Max Allowed: {MAX_DRAWDOWN*100}%")

    if current_dd_pct >= MAX_DRAWDOWN * 100:
        print(f"*** WARNING: NEAR DRAWDOWN LIMIT ***")
    else:
        remaining_dd = (MAX_DRAWDOWN * 100) - current_dd_pct
        print(f"Remaining buffer: {remaining_dd:.2f}%")
else:
    print(f"No drawdown (in profit)")
    print(f"Max allowed loss: ${max_dd_amount:,.2f} (6%)")

# Trading hours check
hour = now.hour
is_trading_hours = False

if now.weekday() < 5:  # Monday-Friday
    # London/NY overlap: 8 AM - 12 PM EST
    if 8 <= hour <= 12:
        is_trading_hours = True
        session = "London/NY Overlap"
    # Tokyo session: 8 PM - 11 PM EST
    elif 20 <= hour <= 23:
        is_trading_hours = True
        session = "Tokyo Session"

print(f"\n[TRADING STATUS]")
if is_trading_hours:
    print(f"Status: ACTIVE TRADING HOURS ({session})")
    print(f"Bots: Scanning for opportunities")
else:
    if now.weekday() >= 5:
        print(f"Status: WEEKEND (Market Closed)")
        print(f"Next trading: Monday 8:00 AM EST")
    else:
        if hour < 8:
            print(f"Status: Pre-market (waiting for 8:00 AM)")
        elif 12 < hour < 20:
            print(f"Status: Between sessions (next: 8:00 PM)")
        else:
            print(f"Status: After hours (next: Monday 8:00 AM)")

# Check bot processes
import subprocess
try:
    result = subprocess.run(
        ['powershell', 'Get-Process pythonw -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count'],
        capture_output=True,
        text=True,
        timeout=5
    )
    bot_count = int(result.stdout.strip() or 0)

    print(f"\n[BOT STATUS]")
    print(f"Running bots: {bot_count}")

    if bot_count >= 3:
        print(f"Status: All 3 bots running (EUR/USD, GBP/USD, USD/JPY)")
    elif bot_count > 0:
        print(f"WARNING: Only {bot_count}/3 bots running")
    else:
        print(f"ERROR: No bots running!")

except Exception as e:
    print(f"\n[BOT STATUS] Could not check: {e}")

print("\n" + "=" * 70)
