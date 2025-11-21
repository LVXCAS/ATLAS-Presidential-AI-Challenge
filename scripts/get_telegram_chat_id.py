"""
Quick script to get your Telegram Chat ID
Run this, then send a message to your bot, then run it again
"""

import requests
import json

BOT_TOKEN = "8203125300:AAE1FTiXQALCFh8cX9lKWhq8arEB2yvUGfQ"

def get_updates():
    """Get updates from Telegram to find your chat ID"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"

    try:
        response = requests.get(url)
        data = response.json()

        if data['ok'] and data['result']:
            print("\n" + "="*60)
            print("FOUND YOUR CHAT ID!")
            print("="*60)

            for update in data['result']:
                if 'message' in update:
                    chat_id = update['message']['chat']['id']
                    username = update['message']['chat'].get('username', 'Unknown')
                    first_name = update['message']['chat'].get('first_name', 'Unknown')

                    print(f"\nChat ID: {chat_id}")
                    print(f"Username: @{username}")
                    print(f"Name: {first_name}")
                    print("\n" + "="*60)
                    print("Copy this Chat ID to your .env file!")
                    print("="*60)

                    return chat_id
        else:
            print("\n" + "="*60)
            print("NO MESSAGES YET")
            print("="*60)
            print("\nSteps:")
            print("1. Open Telegram on your phone")
            print("2. Search for: @LVXCAS_bot")
            print("3. Send any message (like 'hello')")
            print("4. Run this script again")
            print("\nYour bot link: https://t.me/LVXCAS_bot")
            print("="*60)

    except Exception as e:
        print(f"[ERROR] Failed to get updates: {e}")
        return None

if __name__ == "__main__":
    get_updates()
