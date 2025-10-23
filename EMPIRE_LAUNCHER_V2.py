#!/usr/bin/env python3
"""
TRADING EMPIRE LAUNCHER V2
==========================
Launches Forex Elite + Adaptive Options systems using subprocess
to avoid asyncio conflicts.

TARGET: 30%+ Monthly Combined Returns
MODE: Paper Trading Only (Safe)
"""

import subprocess
import time
import os
import sys
from datetime import datetime

def print_banner():
    print("\n" + "="*80)
    print("TRADING EMPIRE LAUNCHER V2")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")
    print("Mode: PAPER TRADING (No real money)")
    print("Target: 30%+ monthly combined")
    print("="*80)
    print("\nSYSTEMS TO LAUNCH:")
    print("  1. Forex Elite System (71-75% WR, 3-5% monthly)")
    print("  2. Adaptive Options System (68% ROI, 4-6% monthly)")
    print("\nPress Ctrl+C at any time to view status")
    print("="*80 + "\n")

def launch_forex_elite():
    """Launch Forex Elite system"""
    print("[FOREX] Launching Forex Elite System...")
    print("[FOREX] Strategy: Strict (71-75% Win Rate)")
    print("[FOREX] Pairs: EUR/USD, USD/JPY")
    print("[FOREX] Mode: PAPER TRADING")

    try:
        # Launch in subprocess
        process = subprocess.Popen(
            [sys.executable, "START_FOREX_ELITE.py", "--strategy", "strict"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        print(f"[FOREX] Launched with PID: {process.pid}")
        print("[FOREX] Status: RUNNING")
        return process

    except Exception as e:
        print(f"[FOREX ERROR] Failed to launch: {e}")
        return None

def launch_adaptive_options():
    """Launch Adaptive Options system"""
    print("\n[OPTIONS] Launching Adaptive Options System...")
    print("[OPTIONS] Strategy: Dual Options (Cash-Secured Puts + Long Calls)")
    print("[OPTIONS] Target: 4-6% monthly")
    print("[OPTIONS] Mode: PAPER TRADING")

    try:
        # Launch in subprocess
        process = subprocess.Popen(
            [sys.executable, "START_ADAPTIVE_OPTIONS.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        print(f"[OPTIONS] Launched with PID: {process.pid}")
        print("[OPTIONS] Status: RUNNING")
        return process

    except Exception as e:
        print(f"[OPTIONS ERROR] Failed to launch: {e}")
        return None

def check_process_status(process, name):
    """Check if process is still running"""
    if process is None:
        return f"[{name}] NOT RUNNING"

    poll = process.poll()
    if poll is None:
        return f"[{name}] RUNNING (PID: {process.pid})"
    else:
        return f"[{name}] STOPPED (Exit code: {poll})"

def main():
    """Main launcher"""

    print_banner()

    # Ensure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    os.makedirs('forex_trades', exist_ok=True)

    print("[PRE-FLIGHT] Checking environment...")

    # Check credentials
    if not os.getenv('OANDA_API_KEY'):
        print("[WARNING] OANDA_API_KEY not found in environment")
    else:
        print("[OK] OANDA credentials found")

    if not os.getenv('ALPACA_API_KEY'):
        print("[WARNING] ALPACA_API_KEY not found in environment")
    else:
        print("[OK] Alpaca credentials found")

    print()
    time.sleep(2)

    # Launch systems
    print("="*80)
    print("LAUNCHING SYSTEMS")
    print("="*80 + "\n")

    forex_process = launch_forex_elite()
    time.sleep(3)  # Wait 3 seconds between launches

    options_process = launch_adaptive_options()
    time.sleep(2)

    print("\n" + "="*80)
    print("LAUNCH COMPLETE")
    print("="*80)

    # Monitor loop
    print("\n[MONITOR] Monitoring systems (Press Ctrl+C to stop)...")
    print("[MONITOR] Checking status every 30 seconds...")
    print()

    check_count = 0
    try:
        while True:
            check_count += 1

            # Check every 30 seconds
            time.sleep(30)

            print(f"\n[STATUS CHECK #{check_count}] {datetime.now().strftime('%I:%M:%S %p')}")
            print(check_process_status(forex_process, "FOREX"))
            print(check_process_status(options_process, "OPTIONS"))

            # Check if any process died
            if forex_process and forex_process.poll() is not None:
                print("[ALERT] Forex Elite process stopped!")
                stderr = forex_process.stderr.read() if forex_process.stderr else ""
                if stderr:
                    print(f"[FOREX ERROR] {stderr[:500]}")

            if options_process and options_process.poll() is not None:
                print("[ALERT] Adaptive Options process stopped!")
                stderr = options_process.stderr.read() if options_process.stderr else ""
                if stderr:
                    print(f"[OPTIONS ERROR] {stderr[:500]}")

            # If both stopped, exit
            if (forex_process is None or forex_process.poll() is not None) and \
               (options_process is None or options_process.poll() is not None):
                print("\n[STOPPED] All systems stopped")
                break

    except KeyboardInterrupt:
        print("\n\n[SHUTDOWN] Stopping all systems...")

        # Terminate processes
        if forex_process and forex_process.poll() is None:
            print("[FOREX] Terminating...")
            forex_process.terminate()
            try:
                forex_process.wait(timeout=5)
                print("[FOREX] Stopped")
            except:
                forex_process.kill()
                print("[FOREX] Force killed")

        if options_process and options_process.poll() is None:
            print("[OPTIONS] Terminating...")
            options_process.terminate()
            try:
                options_process.wait(timeout=5)
                print("[OPTIONS] Stopped")
            except:
                options_process.kill()
                print("[OPTIONS] Force killed")

    finally:
        print("\n" + "="*80)
        print("TRADING EMPIRE STOPPED")
        print("="*80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")
        print("\nSession complete.")
        print("Check logs/ and forex_trades/ directories for detailed logs.")
        print("="*80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
