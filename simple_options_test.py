#!/usr/bin/env python3
"""
Simple test for Alpaca options trading capability
"""
import requests
import os
from dotenv import load_dotenv

load_dotenv('.env')

def test_options():
    headers = {
        'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY'),
        'APCA-API-SECRET-KEY': os.getenv('ALPACA_SECRET_KEY'),
        'Content-Type': 'application/json'
    }
    
    base_url = 'https://paper-api.alpaca.markets/v2'
    
    print("Testing Alpaca Options Support")
    print("=" * 35)
    
    # Test 1: Basic API connectivity
    try:
        response = requests.get(f"{base_url}/account", headers=headers, timeout=10)
        if response.status_code == 200:
            print("SUCCESS: API credentials work")
            account = response.json()
            print(f"Options Level: {account.get('options_trading_level', 'Not found')}")
        else:
            print(f"FAILED: API test - {response.status_code}")
            return
    except Exception as e:
        print(f"FAILED: Connection - {e}")
        return
    
    # Test 2: Try placing an options order
    option_symbols = [
        "SPY250912C00550000",   # Standard OCC format
        "SPY250912C550",        # Short format  
        "SPY",                  # Just underlying (fallback test)
    ]
    
    for symbol in option_symbols:
        print(f"\nTesting order for: {symbol}")
        
        order_data = {
            "symbol": symbol,
            "qty": 1,
            "side": "buy",
            "type": "limit",
            "limit_price": "0.01",  # Very low price
            "time_in_force": "day"
        }
        
        try:
            response = requests.post(f"{base_url}/v2/orders", headers=headers, json=order_data, timeout=10)
            
            print(f"Response: {response.status_code}")
            
            if response.status_code in [200, 201]:
                order_response = response.json()
                print(f"SUCCESS: Order placed - ID: {order_response.get('id')}")
                
                # Cancel immediately to clean up
                order_id = order_response.get('id')
                cancel_response = requests.delete(f"{base_url}/orders/{order_id}", headers=headers, timeout=5)
                print(f"Cancelled: {cancel_response.status_code == 204}")
                
                print(f"\nFOUND WORKING FORMAT: {symbol}")
                return symbol
                
            else:
                error_text = response.text
                print(f"Failed: {error_text[:100]}")
                
                if "unauthorized" in error_text.lower():
                    print("  -> Authentication issue")
                elif "not found" in error_text.lower():
                    print("  -> Symbol not found")
                elif "unprocessable" in error_text.lower():
                    print("  -> Invalid order format")
        
        except Exception as e:
            print(f"Exception: {e}")
    
    print(f"\nCONCLUSION:")
    print("No working options symbol format found")
    print("This suggests Alpaca paper trading may not support options execution")
    print("\nTo fix this, you can:")
    print("1. Contact Alpaca support about options in paper trading")
    print("2. Use live trading for options (real money)")
    print("3. Switch to stock-based strategies")
    
    return None

if __name__ == "__main__":
    working_format = test_options()
    
    if working_format:
        print(f"\nUse this format in your bot: {working_format}")
    else:
        print(f"\nNo options trading available in paper account")