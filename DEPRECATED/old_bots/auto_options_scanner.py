#!/usr/bin/env python3
"""
AUTOMATIC OPTIONS SCANNER
Runs continuously, scanning and executing trades automatically
Perfect for autonomous 24/7 operation
"""

import time
import schedule
import json
from datetime import datetime, timedelta
from MONDAY_AI_TRADING import MondayAITrading
import os
from pathlib import Path

class AutoOptionsScanner:
    """
    Automatic options scanner that runs on schedule

    Features:
    - Scans market every N hours
    - Auto-executes high-scoring opportunities
    - Rate limiting (max trades per day)
    - Smart scheduling (only during market hours)
    - Position tracking and reporting
    """

    def __init__(
        self,
        scan_interval_hours: int = 4,
        max_trades_per_day: int = 4,
        min_score: float = 8.0,
        market_open_hour: int = 6,  # 6:30 AM PT = 9:30 AM ET
        market_close_hour: int = 13  # 1:00 PM PT = 4:00 PM ET
    ):
        self.scan_interval_hours = scan_interval_hours
        self.max_trades_per_day = max_trades_per_day
        self.min_score = min_score
        self.market_open_hour = market_open_hour
        self.market_close_hour = market_close_hour

        # Track daily stats
        self.trades_today = 0
        self.last_scan_date = None
        self.scan_count = 0
        self.total_trades = 0

        # Status file
        self.status_file = Path('auto_scanner_status.json')
        self.load_status()

        print("\n" + "="*70)
        print("AUTOMATIC OPTIONS SCANNER - INITIALIZED")
        print("="*70)
        print(f"Scan Interval: Every {scan_interval_hours} hours")
        print(f"Max Trades/Day: {max_trades_per_day}")
        print(f"Min Score: {min_score}")
        print(f"Market Hours: {market_open_hour}:30 AM - {market_close_hour}:00 PM PT")
        print(f"Total Trades Executed: {self.total_trades}")
        print("="*70 + "\n")

    def load_status(self):
        """Load scanner status from file"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    status = json.load(f)
                    self.total_trades = status.get('total_trades', 0)
                    self.last_scan_date = status.get('last_scan_date')
                    print(f"[STATUS] Loaded: {self.total_trades} total trades")
            except:
                pass

    def save_status(self):
        """Save scanner status to file"""
        status = {
            'total_trades': self.total_trades,
            'trades_today': self.trades_today,
            'last_scan_date': datetime.now().strftime('%Y-%m-%d'),
            'last_scan_time': datetime.now().isoformat(),
            'scan_count': self.scan_count
        }
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)

    def is_market_hours(self) -> bool:
        """Check if current time is within market hours"""
        now = datetime.now()
        current_hour = now.hour
        current_minute = now.minute

        # Check day (Monday=0, Friday=4)
        if now.weekday() > 4:  # Saturday or Sunday
            return False

        # Check time (6:30 AM - 1:00 PM PT)
        if current_hour < self.market_open_hour:
            return False
        if current_hour == self.market_open_hour and current_minute < 30:
            return False
        if current_hour >= self.market_close_hour:
            return False

        return True

    def reset_daily_counter(self):
        """Reset daily trade counter at start of new day"""
        today = datetime.now().strftime('%Y-%m-%d')
        if self.last_scan_date != today:
            print(f"\n[NEW DAY] Resetting daily trade counter")
            print(f"Yesterday: {self.trades_today} trades")
            self.trades_today = 0
            self.last_scan_date = today
            self.save_status()

    def run_scan(self):
        """Run a single scan and execute trades"""
        self.scan_count += 1

        print("\n" + "="*70)
        print(f"AUTO SCAN #{self.scan_count}")
        print("="*70)
        print(f"Time: {datetime.now().strftime('%A, %B %d, %Y - %I:%M %p')}")

        # Reset daily counter if new day
        self.reset_daily_counter()

        # Check if within market hours
        if not self.is_market_hours():
            print("\n[SKIP] Outside market hours")
            print(f"Next scan: {datetime.now() + timedelta(hours=self.scan_interval_hours)}")
            print("="*70 + "\n")
            return

        # Check if hit daily limit
        if self.trades_today >= self.max_trades_per_day:
            print(f"\n[LIMIT] Daily trade limit reached ({self.trades_today}/{self.max_trades_per_day})")
            print("No more trades today. Scanning resumes tomorrow.")
            print("="*70 + "\n")
            return

        try:
            # Calculate remaining trades for today
            remaining_trades = self.max_trades_per_day - self.trades_today

            print(f"\n[SCANNING] Looking for {remaining_trades} opportunities...")
            print(f"Today: {self.trades_today}/{self.max_trades_per_day} trades executed\n")

            # Initialize trading system
            system = MondayAITrading(
                auto_execute=True,
                max_trades=remaining_trades,
                enable_futures=False  # Options only for now
            )

            # Run scan (will auto-execute if opportunities found)
            opportunities = system.run_morning_scan()

            # Count executed trades
            executed = [opp for opp in opportunities if opp.get('final_score', 0) >= self.min_score]
            num_executed = min(len(executed), remaining_trades)

            self.trades_today += num_executed
            self.total_trades += num_executed

            # Save status
            self.save_status()

            print("\n" + "="*70)
            print("SCAN COMPLETE")
            print("="*70)
            print(f"Trades executed this scan: {num_executed}")
            print(f"Trades today: {self.trades_today}/{self.max_trades_per_day}")
            print(f"Total trades: {self.total_trades}")
            print(f"Next scan: {datetime.now() + timedelta(hours=self.scan_interval_hours)}")
            print("="*70 + "\n")

        except Exception as e:
            print(f"\n[ERROR] Scan failed: {e}")
            print("Will retry on next scheduled scan")
            print("="*70 + "\n")

    def run_once(self):
        """Run scanner once and exit"""
        print("\n[MODE] Single scan mode\n")
        self.run_scan()

    def run_continuous(self):
        """Run scanner continuously on schedule"""
        print("\n[MODE] Continuous mode - Running 24/7\n")
        print(f"Scanner will run every {self.scan_interval_hours} hours")
        print("Press Ctrl+C to stop\n")

        # Schedule the scan
        schedule.every(self.scan_interval_hours).hours.do(self.run_scan)

        # Run first scan immediately
        print("[STARTUP] Running initial scan...\n")
        self.run_scan()

        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def run_daily_morning(self):
        """Run scanner once per day at market open"""
        print("\n[MODE] Daily morning mode\n")
        print(f"Scanner will run daily at {self.market_open_hour}:30 AM PT")
        print("Press Ctrl+C to stop\n")

        # Schedule daily at 6:30 AM PT
        schedule.every().day.at(f"{self.market_open_hour:02d}:30").do(self.run_scan)

        # Check if we should run now
        if self.is_market_hours():
            now = datetime.now()
            if now.hour == self.market_open_hour and now.minute >= 30:
                print("[STARTUP] Market just opened! Running scan...\n")
                self.run_scan()

        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)


def main():
    """Main entry point with command-line options"""
    import sys

    print("\n" + "="*70)
    print("AUTOMATIC OPTIONS SCANNER v1.0")
    print("AI-Enhanced Autonomous Trading")
    print("="*70)

    # Parse command-line arguments
    mode = 'daily'  # Default mode

    if '--continuous' in sys.argv:
        mode = 'continuous'
    elif '--once' in sys.argv:
        mode = 'once'
    elif '--daily' in sys.argv:
        mode = 'daily'

    # Get scan interval
    scan_interval = 4  # Default: every 4 hours
    if '--interval' in sys.argv:
        idx = sys.argv.index('--interval')
        if idx + 1 < len(sys.argv):
            scan_interval = int(sys.argv[idx + 1])

    # Get max trades
    max_trades = 4  # Default: 4 per day
    if '--max-trades' in sys.argv:
        idx = sys.argv.index('--max-trades')
        if idx + 1 < len(sys.argv):
            max_trades = int(sys.argv[idx + 1])

    # Initialize scanner
    scanner = AutoOptionsScanner(
        scan_interval_hours=scan_interval,
        max_trades_per_day=max_trades,
        min_score=8.0
    )

    # Run in selected mode
    try:
        if mode == 'once':
            scanner.run_once()
        elif mode == 'continuous':
            scanner.run_continuous()
        elif mode == 'daily':
            scanner.run_daily_morning()
    except KeyboardInterrupt:
        print("\n\n[STOPPED] Scanner stopped by user")
        print(f"Total trades executed: {scanner.total_trades}")
        print("Scanner status saved.\n")


if __name__ == "__main__":
    main()
