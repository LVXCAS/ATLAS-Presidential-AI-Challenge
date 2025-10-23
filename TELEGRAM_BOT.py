"""
Telegram Trading Bot - Remote Control & Notifications
"""

import os
import json
import time
import requests
from datetime import datetime
from pathlib import Path

class TelegramTradingBot:
    def __init__(self):
        # Your Telegram bot credentials
        self.bot_token = "8203125300:AAE1FTiXQALCFh8cX9lKWhq8arEB2yvUGfQ"
        self.chat_id = "7606409012"
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

        # Track last update
        self.last_update_id = 0

        print("=" * 70)
        print("TELEGRAM TRADING BOT ACTIVE")
        print("=" * 70)
        print(f"Bot Token: {self.bot_token[:10]}...")
        print(f"Chat ID: {self.chat_id}")
        print("Commands: /status, /positions, /forex, /options, /help")
        print("=" * 70)

        # Send startup message
        self.send_message("🚀 Trading Bot Started!\n\nAvailable commands:\n/status - System status\n/positions - Open positions\n/forex - Forex status\n/options - Options scanner\n/help - Show commands")

    def send_message(self, text):
        """Send message to Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': 'Markdown'
            }
            response = requests.post(url, data=data, timeout=5)
            return response.json().get('ok', False)
        except Exception as e:
            print(f"[ERROR sending message] {e}")
            return False

    def get_updates(self):
        """Get new messages from Telegram"""
        try:
            url = f"{self.base_url}/getUpdates"
            params = {'offset': self.last_update_id + 1, 'timeout': 5}
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get('ok') and data.get('result'):
                    return data['result']
        except Exception as e:
            print(f"[ERROR getting updates] {e}")
        return []

    def check_forex_status(self):
        """Check forex trading status"""
        status = "📊 *FOREX STATUS*\n\n"

        # Check for forex position files
        forex_dir = Path("forex_trades")
        if forex_dir.exists():
            position_files = list(forex_dir.glob("positions_*.json"))
            if position_files:
                latest = max(position_files, key=lambda x: x.stat().st_mtime)
                with open(latest) as f:
                    data = json.load(f)
                    if data:
                        status += f"✅ Active Positions: {len(data)}\n"
                        for pos in data[:3]:  # Show first 3
                            status += f"• {pos.get('symbol', 'N/A')}: {pos.get('side', 'N/A')}\n"
                    else:
                        status += "📍 No open positions\n"
            else:
                status += "📍 No position data found\n"
        else:
            status += "⚠️ Forex system not initialized\n"

        # Check if scanner is running
        try:
            with open("forex_paper_trading_log.json") as f:
                log = json.load(f)
                last_scan = log.get('last_scan', 'Never')
                status += f"\n⏰ Last scan: {last_scan}"
        except:
            status += "\n⏰ Scanner status unknown"

        return status

    def check_options_status(self):
        """Check options scanner status"""
        status = "📈 *OPTIONS STATUS*\n\n"

        # Check for recent option signals
        signal_files = list(Path(".").glob("options_signals_*.json"))
        if signal_files:
            latest = max(signal_files, key=lambda x: x.stat().st_mtime)
            with open(latest) as f:
                signals = json.load(f)
                status += f"✅ Found {len(signals)} opportunities\n\n"
                for sig in signals[:3]:  # Show first 3
                    status += f"*{sig['symbol']}* - {sig['strategy']}\n"
                    status += f"  Score: {sig['score']}/10\n"
                    status += f"  IV Rank: {sig['iv_rank']}%\n\n"
        else:
            status += "📍 No recent signals\n"

        return status

    def check_system_status(self):
        """Check overall system status"""
        status = "🖥️ *SYSTEM STATUS*\n\n"

        # Check Python processes
        try:
            import subprocess
            result = subprocess.run(
                ['powershell', '-Command',
                 'Get-Process python* | Measure-Object | Select-Object -ExpandProperty Count'],
                capture_output=True, text=True, timeout=5
            )
            count = int(result.stdout.strip()) if result.stdout.strip() else 0
            status += f"🔧 Python processes: {count}\n"
        except:
            status += "🔧 Process count unavailable\n"

        # Check market hours
        now = datetime.now()
        if now.weekday() < 5:
            if 9 <= now.hour < 16:
                status += "🟢 Markets: OPEN\n"
            elif 16 <= now.hour < 20:
                status += "🟡 Markets: AFTER-HOURS\n"
            else:
                status += "🔴 Markets: CLOSED\n"
        else:
            status += "🔴 Markets: WEEKEND\n"

        status += f"\n⏰ Time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

        return status

    def handle_command(self, command):
        """Handle incoming commands"""
        command = command.lower().strip()

        if command == '/status':
            return self.check_system_status()
        elif command == '/forex':
            return self.check_forex_status()
        elif command == '/options':
            return self.check_options_status()
        elif command == '/positions':
            forex = self.check_forex_status()
            options = self.check_options_status()
            return f"{forex}\n\n{options}"
        elif command == '/help':
            return ("📚 *AVAILABLE COMMANDS*\n\n"
                   "/status - System status\n"
                   "/positions - All open positions\n"
                   "/forex - Forex trading status\n"
                   "/options - Options scanner status\n"
                   "/help - Show this message")
        else:
            return f"❓ Unknown command: {command}\nType /help for available commands"

    def run(self):
        """Main bot loop"""
        print("\n[BOT] Listening for commands...")

        while True:
            try:
                # Get new messages
                updates = self.get_updates()

                for update in updates:
                    # Update the last ID
                    self.last_update_id = update['update_id']

                    # Process message
                    if 'message' in update and 'text' in update['message']:
                        text = update['message']['text']
                        chat_id = update['message']['chat']['id']

                        # Only respond to our chat
                        if str(chat_id) == self.chat_id:
                            print(f"[COMMAND] {text}")
                            response = self.handle_command(text)
                            self.send_message(response)

                # Small delay
                time.sleep(1)

            except KeyboardInterrupt:
                self.send_message("🛑 Bot stopped by user")
                print("\n[BOT] Stopped")
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(5)

if __name__ == "__main__":
    bot = TelegramTradingBot()
    bot.run()