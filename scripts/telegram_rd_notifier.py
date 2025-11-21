#!/usr/bin/env python3
"""
TELEGRAM R&D NOTIFIER
Monitors R&D discoveries and sends Telegram notifications when new strategies are found

Notifies you about:
- New Forex/Futures strategy discoveries
- New Stock/Options strategy discoveries
- Top strategies ranked by Sharpe ratio
- Ready-to-deploy recommendations
"""

import os
import sys
import time
import json
import requests
import glob
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')


class TelegramRDNotifier:
    """Monitors R&D discoveries and sends Telegram notifications"""

    def __init__(self):
        self.bot_token = BOT_TOKEN
        self.chat_id = CHAT_ID
        self.base_url = f'https://api.telegram.org/bot{self.bot_token}'

        # Track what we've already notified
        self.notified_files = set()

        print("\n" + "="*60)
        print("TELEGRAM R&D NOTIFIER - ACTIVE")
        print("="*60)
        print("Monitoring for R&D discoveries...")
        print("="*60 + "\n")

    def send_notification(self, message: str):
        """Send Telegram notification"""
        url = f'{self.base_url}/sendMessage'
        data = {'chat_id': self.chat_id, 'text': message}

        try:
            response = requests.post(url, data=data, timeout=5)
            if response.json().get('ok'):
                print(f"[SENT] R&D notification")
                return True
        except Exception as e:
            print(f"[ERROR] Failed to send: {e}")

        return False

    def check_forex_futures_discoveries(self):
        """Check for new Forex/Futures R&D discoveries"""
        try:
            files = glob.glob('logs/forex_futures_strategies_*.json')
            if not files:
                return

            latest = max(files, key=os.path.getctime)

            if latest not in self.notified_files:
                # New discoveries!
                with open(latest) as f:
                    data = json.load(f)

                if len(data) > 0:
                    top_strategies = sorted(data, key=lambda x: x.get('expected_sharpe', 0), reverse=True)[:3]

                    msg = f"R&D DISCOVERY - FOREX/FUTURES\n\n"
                    msg += f"Found {len(data)} new strategies!\n\n"
                    msg += f"TOP 3:\n\n"

                    for i, s in enumerate(top_strategies, 1):
                        msg += f"{i}. {s['name']}\n"
                        msg += f"   Sharpe: {s['expected_sharpe']:.2f}\n"
                        msg += f"   Win Rate: {s.get('expected_win_rate', 0):.1%}\n"
                        msg += f"   Market: {s['type']} ({s['market']})\n\n"

                    msg += f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
                    msg += f"Ready to backtest and deploy!"

                    self.send_notification(msg)
                    self.notified_files.add(latest)

                elif len(data) == 0:
                    # No discoveries - notify about that too
                    msg = f"R&D UPDATE - FOREX/FUTURES\n\n"
                    msg += f"Research cycle completed.\n"
                    msg += f"No strategies met criteria (Sharpe > 1.0)\n\n"
                    msg += f"This is normal - not every cycle finds alpha.\n"
                    msg += f"Next cycle will run overnight."

                    self.send_notification(msg)
                    self.notified_files.add(latest)

        except Exception as e:
            print(f"[ERROR] Checking Forex/Futures R&D: {e}")

    def check_stock_options_discoveries(self):
        """Check for new Stock/Options R&D discoveries"""
        try:
            files = glob.glob('logs/mega_elite_strategies_*.json')
            if not files:
                return

            # Only check files from today/last 24 hours
            recent_files = [f for f in files if (datetime.now() - datetime.fromtimestamp(os.path.getctime(f))).days < 1]

            if not recent_files:
                return

            latest = max(recent_files, key=os.path.getctime)

            if latest not in self.notified_files:
                msg = f"R&D DISCOVERY - STOCK/OPTIONS\n\n"
                msg += f"New strategies discovered!\n"
                msg += f"File: {os.path.basename(latest)}\n\n"
                msg += f"Check discoveries:\n"
                msg += f"python PRODUCTION/check_rd_progress.py\n\n"
                msg += f"Time: {datetime.now().strftime('%H:%M:%S')}"

                self.send_notification(msg)
                self.notified_files.add(latest)

        except Exception as e:
            print(f"[ERROR] Checking Stock/Options R&D: {e}")

    def run(self):
        """Main monitoring loop"""
        self.send_notification("R&D Notifier ONLINE\n\nYou will receive alerts when R&D discovers new strategies!")

        print("[MONITORING] Starting R&D notification loop...")

        while True:
            try:
                # Check for new discoveries every 30 seconds
                self.check_forex_futures_discoveries()
                self.check_stock_options_discoveries()

                # Wait 30 seconds before next check
                time.sleep(30)

            except KeyboardInterrupt:
                print("\n[STOP] R&D Notifier shutting down...")
                self.send_notification("R&D Notifier shutting down")
                break

            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(60)


if __name__ == '__main__':
    notifier = TelegramRDNotifier()
    notifier.run()
