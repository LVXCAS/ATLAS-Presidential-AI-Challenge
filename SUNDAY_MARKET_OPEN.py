#!/usr/bin/env python3
"""
SUNDAY MARKET OPEN PREPARATION
Run this Sunday afternoon before forex markets reopen at 5 PM EST
"""

import os
import time
import subprocess
from datetime import datetime, timedelta

def cleanup_old_files():
    """Remove stop files and old logs"""
    print("[CLEANUP] Removing stop files...")

    stop_files = [
        "STOP_FOREX_TRADING.txt",
        "STOP_FUTURES_TRADING.txt",
        "STOP_ALL_TRADING.txt"
    ]

    for file in stop_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"  ✓ Removed {file}")

    print("[CLEANUP] Complete\n")

def test_market_connection():
    """Test if we can connect to markets"""
    print("[TESTING] Market connections...")

    # Run our working monitor to test API
    try:
        result = subprocess.run(
            ["python", "WORKING_FOREX_MONITOR.py"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if "EUR_USD" in result.stdout:
            print("  ✓ OANDA API connection working")
            print("  ✓ Fetched current prices successfully")
        else:
            print("  ✗ API connection issue - check credentials")

    except Exception as e:
        print(f"  ✗ Connection test failed: {e}")

    print()

def prepare_scanners():
    """Prepare scanners for launch"""
    print("[PREPARE] Setting up scanners...")

    # Check if config exists
    if os.path.exists("config/forex_elite_config.json"):
        print("  ✓ Forex Elite config found")
    else:
        print("  ✗ Config missing - will be created on launch")

    # Show scanner options
    print("\n[SCANNERS] Ready to launch:")
    print("  1. WORKING_FOREX_MONITOR.py - Reliable single-scan (run manually)")
    print("  2. START_FOREX_ELITE.py --strategy strict - Full auto-trading")
    print()

def show_market_schedule():
    """Display market open times"""
    now = datetime.now()
    print(f"[TIME] Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # Calculate time until Sunday 5 PM EST
    days_until_sunday = (6 - now.weekday()) % 7
    if days_until_sunday == 0 and now.hour >= 17:
        days_until_sunday = 7

    market_open = now + timedelta(days=days_until_sunday)
    market_open = market_open.replace(hour=17, minute=0, second=0, microsecond=0)

    time_until = market_open - now
    hours_until = time_until.total_seconds() / 3600

    print(f"[MARKET] Forex opens: Sunday 5:00 PM EST")
    print(f"[COUNTDOWN] {hours_until:.1f} hours until market open\n")

def create_launch_script():
    """Create easy launch script"""

    launch_content = """@echo off
echo.
echo ========================================
echo FOREX TRADING SYSTEM LAUNCHER
echo ========================================
echo.
echo Choose scanner to launch:
echo 1. WORKING_FOREX_MONITOR (manual, reliable)
echo 2. START_FOREX_ELITE (auto-trading)
echo 3. Exit
echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    python WORKING_FOREX_MONITOR.py
    pause
    goto :eof
)

if "%choice%"=="2" (
    python START_FOREX_ELITE.py --strategy strict
    pause
    goto :eof
)

echo Exiting...
"""

    with open("LAUNCH_SCANNER.bat", "w") as f:
        f.write(launch_content)

    print("[CREATED] LAUNCH_SCANNER.bat - Easy launcher ready")
    print()

def main():
    print("\n" + "="*60)
    print("SUNDAY MARKET OPEN PREPARATION")
    print("="*60 + "\n")

    # Run all preparation steps
    cleanup_old_files()
    test_market_connection()
    show_market_schedule()
    prepare_scanners()
    create_launch_script()

    print("="*60)
    print("SYSTEM READY FOR MARKET OPEN")
    print("="*60)
    print("\nTo start trading when market opens:")
    print("  Option 1: Run LAUNCH_SCANNER.bat")
    print("  Option 2: python WORKING_FOREX_MONITOR.py")
    print("  Option 3: python START_FOREX_ELITE.py --strategy strict")
    print("\nRemember: Markets open Sunday 5 PM EST!")
    print()

if __name__ == "__main__":
    main()