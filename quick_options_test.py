#!/usr/bin/env python3
"""
Quick test to verify if Alpaca paper trading supports options
"""
import requests
import os
from dotenv import load_dotenv

load_dotenv('.env')

def quick_alpaca_options_check():
    """Quick check of Alpaca's options support"""
    
    headers = {
        'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY'),
        'APCA-API-SECRET-KEY': os.getenv('ALPACA_SECRET_KEY'),
        'Content-Type': 'application/json'
    }
    
    base_url = 'https://paper-api.alpaca.markets/v2'
    
    print("QUICK ALPACA OPTIONS CHECK")
    print("=" * 30)
    
    # Test 1: Basic auth
    try:
        response = requests.get(f"{base_url}/account", headers=headers, timeout=10)
        if response.status_code == 200:
            print("✅ API credentials work")
        else:
            print(f"❌ API credentials failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Test 2: Check if ANY options assets exist
    try:
        # Search for assets containing common option patterns
        response = requests.get(f"{base_url}/assets?status=active", headers=headers, timeout=15)
        if response.status_code == 200:
            assets = response.json()
            
            options_count = 0
            stock_count = 0
            
            # Check first 100 assets for options patterns
            for asset in assets[:100]:
                symbol = asset.get('symbol', '')
                asset_class = asset.get('class', '')
                
                if 'option' in asset_class.lower() or len(symbol) > 10:
                    options_count += 1
                    if options_count <= 3:  # Show first 3 examples
                        print(f"Found potential option: {symbol} (class: {asset_class})")
                elif asset_class == 'us_equity':
                    stock_count += 1
            
            print(f"Assets scanned: 100")
            print(f"Potential options found: {options_count}")
            print(f"Stocks found: {stock_count}")
            
        else:
            print(f"❌ Assets endpoint failed: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Assets check failed: {e}")
    
    # Test 3: Try placing a stock order (should work)
    print(f"\nTesting stock order...")
    try:
        order_data = {
            "symbol": "SPY",
            "qty": 1,
            "side": "buy",
            "type": "limit",
            "limit_price": "1.00",
            "time_in_force": "day"
        }
        
        response = requests.post(f"{base_url}/v2/orders", headers=headers, json=order_data, timeout=10)
        
        if response.status_code in [200, 201]:
            order_response = response.json()
            order_id = order_response.get('id')
            print(f"✅ Stock order works - ID: {order_id}")
            
            # Cancel immediately
            cancel_response = requests.delete(f"{base_url}/orders/{order_id}", headers=headers, timeout=10)
            if cancel_response.status_code == 204:
                print(f"✅ Order cancelled successfully")
        else:
            print(f"❌ Stock order failed: {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"❌ Stock order test failed: {e}")
    
    # Test 4: Try a simple options symbol
    print(f"\nTesting simple options order...")
    try:
        # Try the most common format
        option_symbol = "SPY250912C00550000"
        
        order_data = {
            "symbol": option_symbol,
            "qty": 1,
            "side": "buy",
            "type": "limit",
            "limit_price": "0.01",
            "time_in_force": "day"
        }
        
        response = requests.post(f"{base_url}/v2/orders", headers=headers, json=order_data, timeout=10)
        
        if response.status_code in [200, 201]:
            print(f"✅ OPTIONS ORDER WORKS! Symbol: {option_symbol}")
            order_response = response.json()
            order_id = order_response.get('id')
            
            # Cancel immediately
            cancel_response = requests.delete(f"{base_url}/orders/{order_id}", headers=headers, timeout=10)
            print(f"Order cancelled: {cancel_response.status_code == 204}")
            
        elif response.status_code == 422:
            print(f"⚠️  Options symbol format issue: {response.text}")
        elif response.status_code == 403:
            print(f"⚠️  Options not permitted: {response.text}")
        else:
            print(f"❌ Options order failed: {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"❌ Options order test failed: {e}")

if __name__ == "__main__":
    quick_alpaca_options_check()