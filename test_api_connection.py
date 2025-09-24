#!/usr/bin/env python3
"""
Quick API connection test for new Alpaca keys
"""
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv('.env')

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets/v2')

def test_alpaca_connection():
    """Test Alpaca API connection"""
    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY,
        'Content-Type': 'application/json'
    }
    
    try:
        # Test account endpoint
        response = requests.get(f"{ALPACA_BASE_URL}/account", headers=headers, timeout=10)
        
        if response.status_code == 200:
            account_data = response.json()
            print("SUCCESS: API Connection Successful!")
            print(f"Account ID: {account_data.get('id', 'N/A')}")
            print(f"Buying Power: ${float(account_data.get('buying_power', 0)):,.2f}")
            print(f"Portfolio Value: ${float(account_data.get('portfolio_value', 0)):,.2f}")
            print(f"Pattern Day Trader: {account_data.get('pattern_day_trader', False)}")
            return True
        else:
            print("ERROR: API Connection Failed!")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"ERROR: Connection Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Alpaca API Connection...")
    print(f"API Key: {ALPACA_API_KEY}")
    print(f"Base URL: {ALPACA_BASE_URL}")
    print("-" * 50)
    
    success = test_alpaca_connection()
    
    if success:
        print("\nReady to run OPTIONS_BOT!")
    else:
        print("\nFix API connection before running OPTIONS_BOT")