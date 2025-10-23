#!/usr/bin/env python3
"""
SIMULATE TELEGRAM COMMAND
Simulate what happens when you send a command from your phone
"""

import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

token = os.getenv('TELEGRAM_BOT_TOKEN')
chat_id = os.getenv('TELEGRAM_CHAT_ID')

print("="*70)
print("SIMULATING TELEGRAM COMMAND FROM PHONE")
print("="*70)

# Simulate sending /status command
print("\n1. Simulating: You send '/status' from your phone...")

# This simulates what Telegram API does when you send a message
url = f'https://api.telegram.org/bot{token}/sendMessage'
data = {
    'chat_id': chat_id,
    'text': '/status'
}

# Don't actually send (bot would see it as from itself)
# Instead, let's check if bot can see pending messages

print("\n2. Checking if bot is polling for messages...")
url2 = f'https://api.telegram.org/bot{token}/getUpdates?timeout=5'
response = requests.get(url2, timeout=10)
updates = response.json()

if updates.get('ok'):
    print(f"[OK] Bot can poll Telegram API")
    print(f"Pending messages: {len(updates['result'])}")

    if updates['result']:
        for update in updates['result'][-5:]:  # Last 5
            if 'message' in update:
                msg = update['message']
                text = msg.get('text', 'N/A')
                from_user = msg.get('from', {}).get('first_name', 'Unknown')
                print(f"  Message: '{text}' from {from_user}")
    else:
        print("  No pending messages")
else:
    print("[ERROR] Bot cannot poll Telegram")
    print(updates)

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("\n1. From your phone, open Telegram")
print("2. Search for: @LVXCAS_bot")
print("3. Send: /start")
print("4. Then send: /status")
print("\n5. Within 30 seconds, the bot should respond")
print("\nIf bot doesn't respond, it means:")
print("  - Wrong chat_id in .env")
print("  - Bot blocked by you")
print("  - Bot not approved by @BotFather")
