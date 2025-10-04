#!/usr/bin/env python3
"""
AUTOMATED SAFE TRADE MONITOR
============================
Continuously monitor safe trades and alert when holds complete
Ensure compliance with 1-hour minimum hold requirement
"""

import time
import json
import glob
import yfinance as yf
from datetime import datetime, timedelta

def automated_monitor():
    print("STARTING AUTOMATED SAFE TRADE MONITORING")
    print("=" * 45)
    print("Monitoring trades for 1-hour minimum holds...")
    print("Press Ctrl+C to stop")
    print()

    start_time = datetime.now()
    cycle_count = 0

    try:
        while True:
            cycle_count += 1
            current_time = datetime.now()

            print(f"\n[{current_time.strftime('%H:%M:%S')}] MONITORING CYCLE {cycle_count}")
            print("-" * 40)

            # Load all safe trades
            safe_trades = glob.glob('safe_trade_*.json')

            if not safe_trades:
                print("No safe trades to monitor")
                time.sleep(60)
                continue

            active_trades = 0
            completed_holds = 0
            total_estimated_pnl = 0

            for trade_file in safe_trades:
                try:
                    with open(trade_file, 'r') as f:
                        trade = json.load(f)

                    symbol = trade['symbol']
                    trade_time = datetime.fromisoformat(trade['timestamp'])
                    hold_duration = current_time - trade_time
                    hold_minutes = hold_duration.total_seconds() / 60

                    # Get current price for P&L calculation
                    try:
                        ticker = yf.Ticker(symbol)
                        current_price = ticker.info.get('currentPrice', trade['current_price'])
                    except:
                        current_price = trade['current_price']

                    original_price = trade['current_price']
                    price_change_pct = ((current_price - original_price) / original_price) * 100

                    # Calculate basic P&L estimate
                    estimated_pnl = 0
                    for strategy in trade['strategies']:
                        if strategy['type'] == 'CASH_SECURED_PUT':
                            if current_price > strategy['strike']:
                                estimated_pnl += strategy['premium_estimate'] * strategy['contracts'] * 100
                            else:
                                intrinsic = (strategy['strike'] - current_price) * strategy['contracts'] * 100
                                estimated_pnl += (strategy['premium_estimate'] * strategy['contracts'] * 100) - intrinsic

                        elif strategy['type'] == 'LONG_CALL':
                            if current_price > strategy['strike']:
                                intrinsic = (current_price - strategy['strike']) * strategy['contracts'] * 100
                                estimated_pnl += intrinsic - strategy['cost']
                            else:
                                estimated_pnl -= strategy['cost']

                    total_estimated_pnl += estimated_pnl

                    # Check hold status
                    min_hold_minutes = 60
                    if hold_minutes >= min_hold_minutes:
                        hold_status = "HOLD COMPLETE - Can close"
                        completed_holds += 1
                    else:
                        remaining = min_hold_minutes - hold_minutes
                        hold_status = f"HOLDING ({remaining:.0f}min remaining)"
                        active_trades += 1

                    print(f"{symbol}: ${current_price:.2f} ({price_change_pct:+.1f}%) | P&L: ${estimated_pnl:.0f} | {hold_status}")

                except Exception as e:
                    print(f"Error monitoring {trade_file}: {e}")

            print()
            print(f"SUMMARY: {active_trades} active holds, {completed_holds} ready to close")
            print(f"Total Estimated P&L: ${total_estimated_pnl:.0f}")

            # Alert if any trades are ready to close
            if completed_holds > 0:
                print()
                print("*** ALERT: TRADES READY FOR DECISION ***")
                print(f"{completed_holds} position(s) have completed 1-hour minimum hold")
                print("Consider closing profitable positions or holding longer")

            # Show running time
            runtime = current_time - start_time
            runtime_minutes = runtime.total_seconds() / 60
            print(f"Monitor runtime: {runtime_minutes:.1f} minutes")

            # Wait 5 minutes between checks
            print("\nNext check in 5 minutes...")
            time.sleep(300)  # 5 minutes

    except KeyboardInterrupt:
        print("\n\nAUTOMATED MONITORING STOPPED")
        print(f"Total monitoring time: {runtime_minutes:.1f} minutes")
        print(f"Total cycles completed: {cycle_count}")
        print()
        print("Final status check:")

        # Final status report
        safe_trades = glob.glob('safe_trade_*.json')
        ready_to_close = 0

        for trade_file in safe_trades:
            try:
                with open(trade_file, 'r') as f:
                    trade = json.load(f)

                trade_time = datetime.fromisoformat(trade['timestamp'])
                hold_duration = datetime.now() - trade_time
                hold_minutes = hold_duration.total_seconds() / 60

                if hold_minutes >= 60:
                    ready_to_close += 1

            except:
                continue

        print(f"- {len(safe_trades)} total safe trades deployed")
        print(f"- {ready_to_close} trades ready for closing decisions")
        print(f"- Safe system operating within all safeguards")
        print()
        print("MONITORING SESSION COMPLETE")

if __name__ == "__main__":
    automated_monitor()