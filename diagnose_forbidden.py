#!/usr/bin/env python3
"""
Diagnose the forbidden error when getting account info
"""
import alpaca_trade_api as tradeapi
import requests
import json
from datetime import datetime

def diagnose_forbidden_error():
    """Comprehensive diagnosis of the forbidden error"""
    
    api_key = "PKQ5FPEZS2ZY13C0Q9QX"
    secret_key = "UdTRNXTgGYBxjF1NDcVeZZo6mZb2S9UdDJ6fN4JD"
    
    print("ALPACA API DIAGNOSTIC")
    print("=" * 60)
    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"Time: {datetime.now()}")
    print()
    
    # Test different base URLs
    base_urls = [
        "https://paper-api.alpaca.markets",
        "https://paper-api.alpaca.markets/v2", 
        "https://api.alpaca.markets",
    ]
    
    for i, base_url in enumerate(base_urls, 1):
        print(f"[TEST {i}] Testing base URL: {base_url}")
        
        try:
            api = tradeapi.REST(
                key_id=api_key,
                secret_key=secret_key,
                base_url=base_url
            )
            
            # Test account endpoint
            account = api.get_account()
            print(f"  ✅ SUCCESS: Account access working")
            print(f"     Account: {account.account_number}")
            print(f"     Status: {account.status}")
            print(f"     Equity: ${float(account.equity):,.2f}")
            
        except Exception as e:
            print(f"  ❌ ERROR: {str(e)}")
            
            # Check if it's a specific HTTP error
            if "403" in str(e) or "forbidden" in str(e).lower():
                print(f"     → This is a 403 Forbidden error")
            elif "401" in str(e) or "unauthorized" in str(e).lower():
                print(f"     → This is a 401 Unauthorized error") 
            elif "404" in str(e):
                print(f"     → This is a 404 Not Found error")
        print()
    
    # Test raw HTTP requests
    print("[TEST 4] Testing raw HTTP request to account endpoint")
    try:
        import base64
        
        # Create basic auth header
        credentials = f"{api_key}:{secret_key}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        
        headers = {
            'Authorization': f'Basic {encoded_credentials}',
            'Content-Type': 'application/json'
        }
        
        url = "https://paper-api.alpaca.markets/v2/account"
        response = requests.get(url, headers=headers)
        
        print(f"  HTTP Status Code: {response.status_code}")
        
        if response.status_code == 200:
            account_data = response.json()
            print(f"  ✅ SUCCESS: Raw HTTP request working")
            print(f"     Account: {account_data.get('account_number', 'N/A')}")
        else:
            print(f"  ❌ ERROR: HTTP {response.status_code}")
            print(f"     Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"  ❌ HTTP ERROR: {e}")
    
    print()
    
    # Test API key format
    print("[TEST 5] API Key Format Check")
    if len(api_key) == 20 and api_key.startswith('PK'):
        print("  ✅ API key format looks correct (20 chars, starts with PK)")
    else:
        print(f"  ❌ API key format issue: length={len(api_key)}, starts with: {api_key[:2]}")
    
    if len(secret_key) == 40:
        print("  ✅ Secret key length looks correct (40 chars)")
    else:
        print(f"  ❌ Secret key length issue: {len(secret_key)} chars")
    
    print()
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    diagnose_forbidden_error()
    input("Press Enter to close...")