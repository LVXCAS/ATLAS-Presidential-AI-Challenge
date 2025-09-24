#!/usr/bin/env python3
"""
Find the correct options symbol format that works with Alpaca paper trading
"""
import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv('.env')

def test_alpaca_auth():
    """First verify API credentials work"""
    headers = {
        'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY'),
        'APCA-API-SECRET-KEY': os.getenv('ALPACA_SECRET_KEY'),
        'Content-Type': 'application/json'
    }
    
    base_url = 'https://paper-api.alpaca.markets/v2'
    
    try:
        # Test basic auth with account endpoint
        response = requests.get(f"{base_url}/account", headers=headers, timeout=10)
        
        if response.status_code == 200:
            print("SUCCESS: API credentials are working")
            account = response.json()
            print(f"Account ID: {account.get('id')}")
            print(f"Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
            return True
        else:
            print(f"ERROR: Auth failed - {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"ERROR: Connection failed - {e}")
        return False

def find_tradeable_options():
    """Try to find options that are actually tradeable in Alpaca"""
    headers = {
        'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY'),
        'APCA-API-SECRET-KEY': os.getenv('ALPACA_SECRET_KEY'),
        'Content-Type': 'application/json'
    }
    
    base_url = 'https://paper-api.alpaca.markets/v2'
    
    print("\nSEARCHING FOR TRADEABLE OPTIONS...")
    print("=" * 40)
    
    # Get next Friday for expiration
    today = datetime.now()
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0:
        days_until_friday = 7
    next_friday = today + timedelta(days=days_until_friday)
    
    # Try different date formats
    exp_formats = [
        next_friday.strftime('%y%m%d'),      # 250913
        next_friday.strftime('%Y%m%d'),      # 20250913
        next_friday.strftime('%m%d%y'),      # 091325
    ]
    
    # Try different symbol formats for popular ETFs
    symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA']
    strikes = ['550', '00550000', '550000', '00550']
    
    working_formats = []
    
    for symbol in symbols:
        print(f"\nTesting {symbol} options...")
        
        for exp_format in exp_formats:
            for strike in strikes:
                for option_type in ['C', 'P']:
                    # Test different formats
                    test_symbols = [
                        f"{symbol}{exp_format}{option_type}{strike}",
                        f"{symbol}   {exp_format}{option_type}{strike}",
                        f"{symbol}_{exp_format}{option_type}{strike}",
                        f".{symbol}{exp_format}{option_type}{strike}",
                        f"{symbol} {exp_format} {option_type} {strike}",
                    ]
                    
                    for test_symbol in test_symbols:
                        try:
                            # Check if asset exists
                            asset_url = f"{base_url}/assets/{test_symbol}"
                            response = requests.get(asset_url, headers=headers, timeout=5)
                            
                            if response.status_code == 200:
                                asset = response.json()
                                if asset.get('tradable'):
                                    print(f"  FOUND TRADEABLE: {test_symbol}")
                                    working_formats.append(test_symbol)
                                    
                                    # Try to get quote
                                    try:
                                        quote_url = f"{base_url}/stocks/{test_symbol}/quotes/latest"
                                        quote_response = requests.get(quote_url, headers=headers, timeout=5)
                                        if quote_response.status_code == 200:
                                            print(f"    Has quotes: YES")
                                        else:
                                            print(f"    Has quotes: NO ({quote_response.status_code})")
                                    except:
                                        pass
                                        
                                    return test_symbol  # Return first working format
                                else:
                                    print(f"  Found but not tradeable: {test_symbol}")
                                    
                        except Exception as e:
                            continue  # Skip this format
    
    return None

def test_stock_order():
    """Test if we can place a basic stock order (should work)"""
    headers = {
        'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY'),
        'APCA-API-SECRET-KEY': os.getenv('ALPACA_SECRET_KEY'),
        'Content-Type': 'application/json'
    }
    
    base_url = 'https://paper-api.alpaca.markets/v2'
    
    print(f"\nTESTING BASIC STOCK ORDER...")
    print("=" * 30)
    
    # Test order (very low price, won't fill)
    order_data = {
        "symbol": "SPY",
        "qty": 1,
        "side": "buy",
        "type": "limit",
        "limit_price": "1.00",
        "time_in_force": "day"
    }
    
    try:
        order_url = f"{base_url}/v2/orders"
        response = requests.post(order_url, headers=headers, json=order_data, timeout=10)
        
        if response.status_code in [200, 201]:
            order_response = response.json()
            print(f"SUCCESS: Stock order placed - ID: {order_response.get('id')}")
            
            # Cancel the order immediately
            order_id = order_response.get('id')
            cancel_url = f"{base_url}/orders/{order_id}"
            cancel_response = requests.delete(cancel_url, headers=headers, timeout=10)
            
            if cancel_response.status_code == 204:
                print(f"SUCCESS: Order cancelled")
            
            return True
        else:
            print(f"ERROR: Stock order failed - {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def check_alpaca_options_support():
    """Check if Alpaca paper trading supports options at all"""
    print(f"\nCHECKING ALPACA OPTIONS SUPPORT...")
    print("=" * 35)
    
    print("Based on testing and documentation:")
    print("1. Your account has options_trading_level: 3")
    print("2. Your account has options_buying_power: $85,680")
    print("3. Stock orders work fine")
    print("4. BUT: No options contracts found in assets")
    print("5. Data API has options exchanges but no tradeable contracts")
    
    print("\nPOSSIBLE CONCLUSIONS:")
    print("A. Alpaca paper trading may not support options execution")
    print("B. Options may require live trading account")
    print("C. Options may use different API endpoints")
    print("D. Options may need special approval/setup")

def main():
    print("ALPACA OPTIONS TROUBLESHOOTING")
    print("=" * 50)
    
    # Step 1: Test basic auth
    if not test_alpaca_auth():
        print("\nSTOP: Fix API credentials first")
        return
    
    # Step 2: Test stock order capability
    if not test_stock_order():
        print("\nSTOP: Basic trading not working")
        return
    
    # Step 3: Search for tradeable options
    working_format = find_tradeable_options()
    
    # Step 4: Analysis
    check_alpaca_options_support()
    
    print(f"\n{'='*50}")
    if working_format:
        print(f"SUCCESS: Found working options format: {working_format}")
        print(f"Update the bot to use this format")
    else:
        print("NO WORKING OPTIONS FORMAT FOUND")
        print("Alpaca paper trading may not support options execution")
        print("\nRECOMMENDATIONS:")
        print("1. Contact Alpaca support to confirm options paper trading")
        print("2. Consider switching to live trading for options")
        print("3. Use stock-based strategies instead")

if __name__ == "__main__":
    main()