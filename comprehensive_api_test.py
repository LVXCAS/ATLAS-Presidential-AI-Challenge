#!/usr/bin/env python3
"""
Comprehensive Alpaca API test to diagnose connection issues
"""
import os
import sys
from dotenv import load_dotenv
import requests
import json

# Add current directory to path
sys.path.append('.')

# Load environment variables
load_dotenv('.env')

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets/v2')

def test_endpoints():
    """Test multiple Alpaca API endpoints"""
    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY,
        'Content-Type': 'application/json'
    }
    
    endpoints = [
        ('/account', 'Account Info'),
        ('/positions', 'Positions'),
        ('/orders', 'Orders'),
        ('/assets', 'Assets/Instruments'),
        ('/portfolio/history', 'Portfolio History')
    ]
    
    print("Testing Multiple API Endpoints:")
    print("=" * 60)
    
    for endpoint, description in endpoints:
        try:
            url = f"{ALPACA_BASE_URL}{endpoint}"
            print(f"\nTesting: {description}")
            print(f"URL: {url}")
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                print(f"Status: SUCCESS (200)")
                
                if endpoint == '/account':
                    print(f"  Account ID: {data.get('id')}")
                    print(f"  Status: {data.get('status')}")
                    print(f"  Trading Blocked: {data.get('trading_blocked', False)}")
                    print(f"  Pattern Day Trader: {data.get('pattern_day_trader', False)}")
                elif endpoint == '/positions':
                    print(f"  Open Positions: {len(data) if isinstance(data, list) else 'Unknown'}")
                elif endpoint == '/orders':
                    print(f"  Active Orders: {len(data) if isinstance(data, list) else 'Unknown'}")
                
            else:
                print(f"Status: ERROR ({response.status_code})")
                print(f"Response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"Status: EXCEPTION - {e}")
    
    print("\n" + "=" * 60)

def test_options_capabilities():
    """Test options-specific functionality"""
    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY,
        'Content-Type': 'application/json'
    }
    
    print("\nTesting Options Trading Capabilities:")
    print("=" * 40)
    
    # Test if account can trade options
    try:
        account_url = f"{ALPACA_BASE_URL}/account"
        response = requests.get(account_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            account_data = response.json()
            
            print(f"Options Trading Enabled: {account_data.get('options_trading_level', 'Not specified')}")
            print(f"Crypto Trading: {account_data.get('crypto_status', 'Not specified')}")
            print(f"Max Options Trading Level: {account_data.get('max_options_trading_level', 'Not specified')}")
            
            # Check for any trading restrictions
            if account_data.get('trading_blocked'):
                print("WARNING: Trading is currently BLOCKED on this account")
            
            if account_data.get('account_blocked'):
                print("WARNING: Account is BLOCKED")
                
            if account_data.get('transfers_blocked'):
                print("INFO: Transfers are blocked (normal for paper trading)")
                
        else:
            print(f"Failed to get account details: {response.status_code}")
            
    except Exception as e:
        print(f"Error checking options capabilities: {e}")

def test_broker_integration():
    """Test the same way OPTIONS_BOT connects"""
    print("\nTesting OPTIONS_BOT Connection Method:")
    print("=" * 45)
    
    try:
        # Try importing the broker integration the same way OPTIONS_BOT does
        from agents.broker_integration import AlpacaBrokerIntegration
        
        print("SUCCESS: Imported AlpacaBrokerIntegration")
        
        # Try initializing broker
        broker = AlpacaBrokerIntegration()
        print("SUCCESS: Initialized AlpacaBrokerIntegration")
        
        # Test connection
        account_info = broker.get_account()
        print(f"SUCCESS: Retrieved account info via broker integration")
        print(f"Account Status: {account_info.get('status', 'Unknown')}")
        
    except ImportError as e:
        print(f"IMPORT ERROR: Could not import broker integration - {e}")
    except Exception as e:
        print(f"BROKER ERROR: {e}")

if __name__ == "__main__":
    print("COMPREHENSIVE ALPACA API DIAGNOSTIC TEST")
    print("=" * 60)
    print(f"API Key: {ALPACA_API_KEY}")
    print(f"Base URL: {ALPACA_BASE_URL}")
    print(f"Paper Trading: {os.getenv('PAPER_TRADING', 'true')}")
    
    test_endpoints()
    test_options_capabilities()
    test_broker_integration()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")