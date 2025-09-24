#!/usr/bin/env python3
"""
Check what trading capabilities are actually available
"""
import requests
import os
from dotenv import load_dotenv
import json

load_dotenv('.env')

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets/v2')

def check_full_account_details():
    """Get complete account information"""
    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get(f"{ALPACA_BASE_URL}/account", headers=headers, timeout=10)
        
        if response.status_code == 200:
            account = response.json()
            
            print("COMPLETE ACCOUNT DETAILS")
            print("=" * 50)
            
            # Print all account fields
            for key, value in account.items():
                print(f"{key}: {value}")
                
            print("\n" + "=" * 50)
            
            # Check specific trading capabilities
            print("TRADING CAPABILITIES:")
            print(f"Status: {account.get('status')}")
            print(f"Trading Blocked: {account.get('trading_blocked', False)}")
            print(f"Pattern Day Trader: {account.get('pattern_day_trader', False)}")
            print(f"Day Trading Buying Power: ${float(account.get('daytrading_buying_power', 0)):,.2f}")
            print(f"Regt Buying Power: ${float(account.get('regt_buying_power', 0)):,.2f}")
            print(f"Options Buying Power: ${float(account.get('options_buying_power', 0)):,.2f}")
            
            # Check for options-related fields
            options_fields = [key for key in account.keys() if 'option' in key.lower()]
            if options_fields:
                print(f"\nOPTIONS FIELDS FOUND:")
                for field in options_fields:
                    print(f"  {field}: {account[field]}")
            else:
                print(f"\nNO OPTIONS-SPECIFIC FIELDS FOUND")
                
            return account
        else:
            print(f"Error getting account: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def check_available_assets():
    """Check what types of assets can be traded"""
    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY,
        'Content-Type': 'application/json'
    }
    
    try:
        # Check assets endpoint with different asset classes
        print("\nCHECKING AVAILABLE ASSET TYPES")
        print("=" * 40)
        
        # Check stocks
        response = requests.get(f"{ALPACA_BASE_URL}/assets?status=active&asset_class=us_equity", headers=headers, timeout=10)
        if response.status_code == 200:
            stocks = response.json()
            print(f"US Equity assets available: {len(stocks)}")
        
        # Try to check options
        response = requests.get(f"{ALPACA_BASE_URL}/assets?status=active&asset_class=us_option", headers=headers, timeout=10)
        if response.status_code == 200:
            options = response.json()
            print(f"US Options assets available: {len(options)}")
        elif response.status_code == 404:
            print("Options assets: Not available (404)")
        else:
            print(f"Options assets: Error {response.status_code}")
            
        # Try crypto
        response = requests.get(f"{ALPACA_BASE_URL}/assets?status=active&asset_class=crypto", headers=headers, timeout=10)
        if response.status_code == 200:
            crypto = response.json()
            print(f"Crypto assets available: {len(crypto)}")
        elif response.status_code == 404:
            print("Crypto assets: Not available (404)")
        else:
            print(f"Crypto assets: Error {response.status_code}")
            
    except Exception as e:
        print(f"Error checking assets: {e}")

if __name__ == "__main__":
    print("ALPACA ACCOUNT CAPABILITIES CHECK")
    print("=" * 60)
    
    account = check_full_account_details()
    check_available_assets()
    
    if account:
        print("\n" + "=" * 60)
        print("SUMMARY:")
        if account.get('trading_blocked'):
            print("❌ TRADING IS BLOCKED")
        elif 'options' not in str(account).lower():
            print("⚠️  OPTIONS TRADING MAY NOT BE AVAILABLE")
            print("   This appears to be a stock-only paper trading account")
        else:
            print("✅ ACCOUNT APPEARS READY FOR TRADING")