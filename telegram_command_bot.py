#!/usr/bin/env python3
"""
TELEGRAM COMMAND BOT
Allows you to control your trading empire via Telegram commands

COMMANDS:
/status - Get current system status
/positions - View all open positions
/pnl - Check profit/loss today
/stop - Emergency stop all trading
/start_forex - Start Forex Elite
/start_options - Start Options Scanner
/regime - Check current market regime
/help - Show all commands

USAGE:
    python telegram_command_bot.py
"""

import os
import sys
import time
import json
import subprocess
import requests
from datetime import datetime
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')


class TelegramCommandBot:
    """Telegram bot that accepts commands to control trading systems"""

    def __init__(self):
        self.bot_token = BOT_TOKEN
        self.chat_id = CHAT_ID
        self.base_url = f'https://api.telegram.org/bot{self.bot_token}'
        self.last_update_id = 0

        print("\n" + "="*60)
        print("TELEGRAM COMMAND BOT STARTING")
        print("="*60)
        print(f"Bot Token: {self.bot_token[:20]}...")
        print(f"Chat ID: {self.chat_id}")
        print(f"\nListening for commands...")
        print("="*60 + "\n")

    def send_message(self, text: str):
        """Send message to Telegram"""
        url = f'{self.base_url}/sendMessage'
        data = {'chat_id': self.chat_id, 'text': text}
        requests.post(url, data=data)

    def get_updates(self) -> list:
        """Get new messages from Telegram"""
        url = f'{self.base_url}/getUpdates'
        params = {'offset': self.last_update_id + 1, 'timeout': 30}

        try:
            response = requests.get(url, params=params, timeout=35)
            data = response.json()

            if data['ok'] and data['result']:
                self.last_update_id = data['result'][-1]['update_id']
                return data['result']
        except Exception as e:
            print(f"[ERROR] Getting updates: {e}")

        return []

    def get_system_status(self) -> str:
        """Get current system status"""
        try:
            # Check running processes
            result = subprocess.run(['tasklist'], capture_output=True, text=True, shell=True)
            python_procs = [line for line in result.stdout.split('\n') if 'python' in line.lower()]

            status = "SYSTEM STATUS\n\n"

            # Check PID files
            if os.path.exists('forex_elite.pid'):
                with open('forex_elite.pid') as f:
                    pid = f.read().strip()
                    status += f"Forex Elite: RUNNING (PID: {pid})\n"
            else:
                status += "Forex Elite: STOPPED\n"

            # Check scanner status
            if os.path.exists('auto_scanner_status.json'):
                with open('auto_scanner_status.json') as f:
                    scanner_data = json.load(f)
                    status += f"Options Scanner: RUNNING ({scanner_data.get('trades_today', 0)} trades today)\n"
            else:
                status += "Options Scanner: STOPPED\n"

            # Account info
            status += f"\nTotal Python processes: {len(python_procs)}\n"
            status += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            return status
        except Exception as e:
            return f"Error getting status: {e}"

    def get_positions(self) -> str:
        """Get open positions"""
        try:
            # Check for position files
            positions = []

            # Check forex positions
            forex_dir = 'forex_trades'
            if os.path.exists(forex_dir):
                for file in os.listdir(forex_dir):
                    if file.startswith('positions_') and file.endswith('.json'):
                        with open(os.path.join(forex_dir, file)) as f:
                            pos_data = json.load(f)
                            if pos_data:
                                positions.extend(pos_data)

            if positions:
                msg = "OPEN POSITIONS\n\n"
                for pos in positions:
                    msg += f"{pos.get('pair', 'Unknown')}: {pos.get('side', 'Unknown')}\n"
                    msg += f"  Entry: {pos.get('entry_price', 'N/A')}\n"
                    msg += f"  Current P/L: {pos.get('pnl', 'N/A')}\n\n"
                return msg
            else:
                return "No open positions"
        except Exception as e:
            return f"Error getting positions: {e}"

    def get_regime(self) -> str:
        """Get current market regime"""
        try:
            # Try to fetch Fear & Greed Index
            response = requests.get('https://api.alternative.me/fng/?limit=1', timeout=5)
            data = response.json()

            if data.get('data'):
                fg_value = int(data['data'][0]['value'])
                fg_class = data['data'][0]['value_classification']

                msg = f"MARKET REGIME\n\n"
                msg += f"Fear & Greed Index: {fg_value}/100\n"
                msg += f"Classification: {fg_class}\n\n"

                if fg_value < 25:
                    msg += "Regime: EXTREME FEAR (Defensive)\n"
                    msg += "Strategy: Selling options, cash positions"
                elif fg_value < 45:
                    msg += "Regime: BEARISH (Cautious)\n"
                    msg += "Strategy: Neutral spreads, limited risk"
                elif fg_value < 55:
                    msg += "Regime: NEUTRAL (Balanced)\n"
                    msg += "Strategy: Iron Condors, range strategies"
                elif fg_value < 75:
                    msg += "Regime: BULLISH (Aggressive)\n"
                    msg += "Strategy: Bull spreads, long positions"
                else:
                    msg += "Regime: EXTREME GREED (Very Aggressive)\n"
                    msg += "Strategy: Max risk, all strategies active"

                return msg
            else:
                return "Could not fetch market regime data"
        except Exception as e:
            return f"Error getting regime: {e}"

    def handle_command(self, command: str, message_id: int):
        """Handle incoming command"""
        print(f"[COMMAND] {command}")

        if command == '/status':
            response = self.get_system_status()

        elif command == '/positions':
            response = self.get_positions()

        elif command == '/regime':
            response = self.get_regime()

        elif command == '/help':
            response = """TRADING EMPIRE COMMANDS

/status - System status
/positions - Open positions
/regime - Market regime
/pnl - Today's profit/loss
/stop - Emergency stop
/start_forex - Start Forex Elite
/start_options - Start Options Scanner
/help - This message

More commands coming soon!"""

        elif command == '/pnl':
            response = "P/L tracking coming soon! Check the web dashboard at http://localhost:8501"

        elif command == '/stop':
            response = "Emergency stop feature coming soon! Use EMERGENCY_STOP.bat for now"

        elif command == '/start_forex':
            response = "Remote start feature coming soon! Use START_FOREX_ELITE.bat for now"

        elif command == '/start_options':
            response = "Remote start feature coming soon! Use auto_options_scanner.py for now"

        else:
            response = f"Unknown command: {command}\nSend /help for available commands"

        self.send_message(response)

    def run(self):
        """Main bot loop"""
        self.send_message("Telegram Command Bot ONLINE\n\nSend /help for available commands")

        while True:
            try:
                updates = self.get_updates()

                for update in updates:
                    if 'message' in update and 'text' in update['message']:
                        text = update['message']['text']
                        message_id = update['message']['message_id']

                        # Only handle commands (starting with /)
                        if text.startswith('/'):
                            self.handle_command(text, message_id)

                time.sleep(1)  # Small delay to avoid hammering API

            except KeyboardInterrupt:
                print("\n[STOP] Bot shutting down...")
                self.send_message("Command Bot shutting down")
                break

            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(5)


if __name__ == '__main__':
    bot = TelegramCommandBot()
    bot.run()
