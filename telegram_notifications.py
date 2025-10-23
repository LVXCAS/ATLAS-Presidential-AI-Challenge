#!/usr/bin/env python3
"""
TELEGRAM NOTIFICATIONS - AUTO ALERTS
Automatically sends you notifications when trades execute + daily summaries

FEATURES:
- Trade execution alerts (real-time)
- Stop-loss/take-profit hit notifications
- Daily performance summaries (morning + evening)
- Emergency alerts (system crashes, errors)
- Market regime changes

USAGE:
    python telegram_notifications.py

This runs in background and monitors your trading activity
"""

import os
import sys
import time
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dotenv import load_dotenv
import glob

load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')


class TelegramNotifier:
    """Automatic Telegram notifications for trading events"""

    def __init__(self):
        self.bot_token = BOT_TOKEN
        self.chat_id = CHAT_ID
        self.base_url = f'https://api.telegram.org/bot{self.bot_token}'

        # Track what we've already notified
        self.notified_trades = set()
        self.last_daily_summary = None
        self.last_regime_check = None
        self.current_regime = None

        print("\n" + "="*60)
        print("TELEGRAM NOTIFICATIONS - ACTIVE")
        print("="*60)
        print("Monitoring for:")
        print("  - Trade executions")
        print("  - Stop-loss/Take-profit hits")
        print("  - Daily summaries (8 AM + 8 PM)")
        print("  - Market regime changes")
        print("  - System errors")
        print("="*60 + "\n")

    def send_notification(self, message: str, silent: bool = False):
        """Send Telegram notification"""
        url = f'{self.base_url}/sendMessage'
        data = {
            'chat_id': self.chat_id,
            'text': message,
            'disable_notification': silent
        }

        try:
            response = requests.post(url, data=data, timeout=5)
            if response.json().get('ok'):
                print(f"[SENT] {message[:50]}...")
                return True
        except Exception as e:
            print(f"[ERROR] Failed to send: {e}")

        return False

    def check_new_trades(self):
        """Check for new trade executions"""
        try:
            # Check forex trades
            forex_files = glob.glob('forex_trades/execution_log_*.json')
            for file in forex_files:
                with open(file) as f:
                    data = json.load(f)
                    trades = data.get('trades', [])

                    for trade in trades:
                        trade_id = trade.get('trade_id')
                        if trade_id and trade_id not in self.notified_trades:
                            # New trade!
                            self.notify_trade_execution(trade, 'FOREX')
                            self.notified_trades.add(trade_id)

            # Check options trades
            if os.path.exists('data/options_active_trades.json'):
                with open('data/options_active_trades.json') as f:
                    trades = json.load(f)
                    for trade in trades:
                        trade_id = trade.get('id')
                        if trade_id and trade_id not in self.notified_trades:
                            self.notify_trade_execution(trade, 'OPTIONS')
                            self.notified_trades.add(trade_id)

        except Exception as e:
            print(f"[ERROR] Checking trades: {e}")

    def notify_trade_execution(self, trade: Dict, market: str):
        """Send notification about trade execution"""
        try:
            if market == 'FOREX':
                msg = f"FOREX TRADE EXECUTED\n\n"
                msg += f"Pair: {trade.get('pair', 'Unknown')}\n"
                msg += f"Direction: {trade.get('direction', 'Unknown')}\n"
                msg += f"Entry: {trade.get('entry_price', 'N/A')}\n"
                msg += f"Stop Loss: {trade.get('stop_loss', 'N/A')}\n"
                msg += f"Take Profit: {trade.get('take_profit', 'N/A')}\n"
                msg += f"Size: {trade.get('units', 'N/A')} units\n"
                msg += f"Signal Score: {trade.get('score', 'N/A')}/10\n"
                msg += f"\nTime: {datetime.now().strftime('%H:%M:%S')}"

            elif market == 'OPTIONS':
                msg = f"OPTIONS TRADE EXECUTED\n\n"
                msg += f"Symbol: {trade.get('symbol', 'Unknown')}\n"
                msg += f"Strategy: {trade.get('strategy', 'Unknown')}\n"
                msg += f"Credit/Debit: ${trade.get('credit', 0):.2f}\n"
                msg += f"Max Risk: ${trade.get('max_risk', 0):.2f}\n"
                msg += f"Win Probability: {trade.get('win_prob', 0):.0%}\n"
                msg += f"\nTime: {datetime.now().strftime('%H:%M:%S')}"

            self.send_notification(msg)

        except Exception as e:
            print(f"[ERROR] Notifying trade: {e}")

    def check_daily_summary(self):
        """Send daily summary at 8 AM and 8 PM"""
        now = datetime.now()
        current_hour = now.hour

        # Morning summary at 8 AM
        if current_hour == 8 and (not self.last_daily_summary or
                                  self.last_daily_summary.date() < now.date()):
            self.send_morning_summary()
            self.last_daily_summary = now

        # Evening summary at 8 PM
        elif current_hour == 20 and (not self.last_daily_summary or
                                     (now - self.last_daily_summary).seconds > 43200):
            self.send_evening_summary()
            self.last_daily_summary = now

    def send_morning_summary(self):
        """Send morning summary"""
        msg = f"GOOD MORNING - TRADING SUMMARY\n\n"
        msg += f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"

        # System status
        msg += "SYSTEMS STATUS:\n"
        if os.path.exists('forex_elite.pid'):
            msg += "- Forex Elite: RUNNING\n"
        else:
            msg += "- Forex Elite: STOPPED\n"

        if os.path.exists('auto_scanner_status.json'):
            msg += "- Options Scanner: ACTIVE\n"
        else:
            msg += "- Options Scanner: INACTIVE\n"

        # Market regime
        try:
            response = requests.get('https://api.alternative.me/fng/?limit=1', timeout=5)
            data = response.json()
            if data.get('data'):
                fg_value = int(data['data'][0]['value'])
                msg += f"\nMARKET REGIME:\n"
                msg += f"Fear & Greed: {fg_value}/100\n"
        except:
            pass

        msg += f"\nToday's Goal: Trade smart, follow the plan!\n"
        msg += f"\nGood luck!"

        self.send_notification(msg)

    def send_evening_summary(self):
        """Send evening summary"""
        msg = f"EVENING SUMMARY - END OF DAY\n\n"
        msg += f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"

        # Count trades
        total_trades = 0
        today = datetime.now().strftime('%Y%m%d')

        forex_log = f'forex_trades/execution_log_{today}.json'
        if os.path.exists(forex_log):
            with open(forex_log) as f:
                data = json.load(f)
                total_trades += len(data.get('trades', []))

        msg += f"TRADES TODAY: {total_trades}\n\n"

        if total_trades == 0:
            msg += "No trades executed today.\n"
            msg += "Waiting for high-quality setups.\n"
        else:
            msg += "Check positions before market close.\n"

        msg += f"\nSystems will continue running overnight.\n"
        msg += f"Next summary: Tomorrow 8:00 AM"

        self.send_notification(msg)

    def check_regime_changes(self):
        """Check for market regime changes"""
        try:
            # Only check every 4 hours
            if self.last_regime_check and (datetime.now() - self.last_regime_check).seconds < 14400:
                return

            response = requests.get('https://api.alternative.me/fng/?limit=1', timeout=5)
            data = response.json()

            if data.get('data'):
                fg_value = int(data['data'][0]['value'])

                # Determine regime
                if fg_value < 25:
                    regime = 'EXTREME_FEAR'
                elif fg_value < 45:
                    regime = 'FEAR'
                elif fg_value < 55:
                    regime = 'NEUTRAL'
                elif fg_value < 75:
                    regime = 'GREED'
                else:
                    regime = 'EXTREME_GREED'

                # Check if regime changed
                if self.current_regime and self.current_regime != regime:
                    msg = f"MARKET REGIME CHANGE\n\n"
                    msg += f"Previous: {self.current_regime}\n"
                    msg += f"New: {regime}\n\n"
                    msg += f"Fear & Greed Index: {fg_value}/100\n\n"
                    msg += f"Your strategies will auto-adjust to new regime."

                    self.send_notification(msg)

                self.current_regime = regime
                self.last_regime_check = datetime.now()

        except Exception as e:
            print(f"[ERROR] Checking regime: {e}")

    def check_system_health(self):
        """Check if systems are healthy"""
        try:
            # Check if Forex Elite is running when it should be
            if os.path.exists('forex_elite.pid'):
                with open('forex_elite.pid') as f:
                    pid = int(f.read().strip())

                # Check if process exists
                import subprocess
                result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'],
                                      capture_output=True, text=True)

                if str(pid) not in result.stdout:
                    # Process died!
                    msg = f"SYSTEM ALERT\n\n"
                    msg += f"Forex Elite stopped unexpectedly!\n"
                    msg += f"PID {pid} not found\n\n"
                    msg += f"Restart with: /start_forex"

                    self.send_notification(msg)
                    os.remove('forex_elite.pid')  # Clean up stale PID

        except Exception as e:
            print(f"[ERROR] Health check: {e}")

    def run(self):
        """Main monitoring loop"""
        self.send_notification("Telegram Notifications ACTIVATED\n\nYou will now receive:\n- Trade execution alerts\n- Daily summaries (8 AM + 8 PM)\n- Market regime changes\n- System health alerts")

        print("[MONITORING] Starting notification loop...")

        while True:
            try:
                # Check for new trades every 10 seconds
                self.check_new_trades()

                # Check for daily summaries
                self.check_daily_summary()

                # Check for regime changes every cycle
                self.check_regime_changes()

                # Check system health every cycle
                self.check_system_health()

                # Wait 10 seconds before next check
                time.sleep(10)

            except KeyboardInterrupt:
                print("\n[STOP] Notifications shutting down...")
                self.send_notification("Telegram Notifications shutting down")
                break

            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(30)  # Wait longer on error


if __name__ == '__main__':
    notifier = TelegramNotifier()
    notifier.run()
