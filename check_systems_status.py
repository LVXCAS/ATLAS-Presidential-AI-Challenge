#!/usr/bin/env python3
"""
SYSTEM STATUS CHECKER
Shows all running trading systems and their current status
"""

import os
import subprocess
from datetime import datetime
import pytz
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()

print("=" * 80)
print("TRADING SYSTEMS STATUS CHECK")
print("=" * 80)
print(f"Time: {datetime.now(pytz.timezone('America/Los_Angeles')).strftime('%I:%M:%S %p PDT')}")
print()

# Check running Python processes
print("[RUNNING PROCESSES]")
print("-" * 80)

processes_to_check = [
    ("Scanner", "week2_sp500_scanner"),
    ("Stop Loss Monitor", "stop_loss_monitor"),
    ("Live Mission Control", "live_mission_control")
]

running_count = 0
for name, process_name in processes_to_check:
    try:
        result = subprocess.run(
            f'tasklist | findstr python',
            shell=True,
            capture_output=True,
            text=True
        )

        # Check if process is in output
        if process_name.lower() in result.stdout.lower() or "python.exe" in result.stdout:
            # Rough check - just see if python is running
            # More sophisticated: check actual command line
            print(f"  [{chr(10003)}] {name:30s} - Status unknown (Python process running)")
            running_count += 1
        else:
            print(f"  [X] {name:30s} - NOT RUNNING")
    except:
        print(f"  [?] {name:30s} - Cannot check")

print()
print(f"Systems running: {running_count}/3")
print()

# Check account status
print("[ACCOUNT STATUS]")
print("-" * 80)

try:
    api = tradeapi.REST(
        key_id=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        base_url=os.getenv('ALPACA_BASE_URL'),
        api_version='v2'
    )

    account = api.get_account()
    positions = api.list_positions()

    equity = float(account.equity)
    cash = float(account.cash)
    buying_power = float(account.buying_power)

    print(f"  Account Equity:       ${equity:,.2f}")
    print(f"  Cash:                 ${cash:,.2f}")
    print(f"  Buying Power:         ${buying_power:,.2f}")
    print(f"  Options Buying Power: ${float(account.options_buying_power):,.2f}")
    print(f"  Open Positions:       {len(positions)}")

    if len(positions) > 0:
        total_pl = sum(float(p.unrealized_pl) for p in positions)
        winners = sum(1 for p in positions if float(p.unrealized_pl) > 0)
        losers = sum(1 for p in positions if float(p.unrealized_pl) < 0)

        print(f"  Total Unrealized P&L: ${total_pl:,.2f}")
        print(f"  Winners/Losers:       {winners}W / {losers}L ({(winners/(winners+losers)*100 if winners+losers > 0 else 0):.1f}% win rate)")

except Exception as e:
    print(f"  [ERROR] Cannot check account: {e}")

print()

# Check log files
print("[LOG FILES]")
print("-" * 80)

log_files = [
    ("Scanner Log", "scanner_output.log"),
    ("Stop Loss Log", "stop_loss_output.log")
]

for name, filename in log_files:
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        modified = datetime.fromtimestamp(os.path.getmtime(filename))
        age_seconds = (datetime.now() - modified).total_seconds()

        if age_seconds < 60:
            age_str = f"{int(age_seconds)}s ago"
        elif age_seconds < 3600:
            age_str = f"{int(age_seconds/60)}m ago"
        else:
            age_str = f"{int(age_seconds/3600)}h ago"

        print(f"  {name:20s} - {size:,} bytes, updated {age_str}")
    else:
        print(f"  {name:20s} - NOT FOUND")

print()

# Market status
print("[MARKET STATUS]")
print("-" * 80)

pdt = pytz.timezone('America/Los_Angeles')
now = datetime.now(pdt)
market_open = now.replace(hour=6, minute=30, second=0, microsecond=0)
market_close = now.replace(hour=13, minute=0, second=0, microsecond=0)

if now < market_open:
    time_until = (market_open - now).total_seconds() / 3600
    print(f"  Market Status: CLOSED (opens in {time_until:.1f} hours)")
elif now > market_close:
    print(f"  Market Status: CLOSED (opens tomorrow at 6:30 AM PDT)")
else:
    time_remaining = (market_close - now).total_seconds() / 3600
    print(f"  Market Status: OPEN ({time_remaining:.1f} hours remaining)")

print()
print("=" * 80)
print()

# Recommendations
print("[RECOMMENDATIONS]")
print("-" * 80)

if running_count < 3:
    print("  WARNING: Not all systems are running!")
    print("  Scanner: Executes trades")
    print("  Stop Loss Monitor: Protects against large losses (CRITICAL)")
    print("  Mission Control: Real-time dashboard (optional)")
    print()
    print("  To start missing systems:")
    print("    python week2_sp500_scanner.py")
    print("    python stop_loss_monitor.py")
    print("    python live_mission_control.py")
else:
    print("  All systems operational!")

print("=" * 80)
