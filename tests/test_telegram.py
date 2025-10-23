"""Test Telegram notification"""
import requests
import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'

message = """TRADING EMPIRE ACTIVATED!

Telegram Notifications: LIVE
Forex Elite: RUNNING (PID: 58416)
Options Scanner: RUNNING (PID: 69800)
R&D Department: RUNNING (PID: 75140)

You will now receive:
- Trade execution alerts
- Daily performance summaries
- Risk warnings
- System status updates

Your autonomous trading empire is fully operational!

Account: $912,745 (paper trading)
Current Regime: BEARISH (Fear & Greed: 23/100)
"""

data = {
    'chat_id': CHAT_ID,
    'text': message
}

response = requests.post(url, data=data)

if response.json()['ok']:
    print("SUCCESS! Check your Telegram - you should have received a notification!")
else:
    print(f"Failed: {response.json()}")
