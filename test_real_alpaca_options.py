#!/usr/bin/env python3
"""
Test real Alpaca options trading to understand the correct approach
"""
import asyncio
import sys
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

sys.path.append('.')
load_dotenv('.env')

def test_manual_option_order():
    """Test placing a manual option order through Alpaca API"""
    
    headers = {
        'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY'),
        'APCA-API-SECRET-KEY': os.getenv('ALPACA_SECRET_KEY'),
        'Content-Type': 'application/json'
    }
    
    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets/v2')
    
    print("TESTING MANUAL ALPACA OPTIONS ORDER")
    print("=" * 45)
    
    # Try different option symbol formats that Alpaca might accept
    # Get next Friday for option expiration
    today = datetime.now()
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0:
        days_until_friday = 7
    next_friday = today + timedelta(days=days_until_friday)
    
    # Try different symbol formats
    exp_date_yymmdd = next_friday.strftime('%y%m%d')
    exp_date_yyyymmdd = next_friday.strftime('%Y%m%d')
    
    option_symbols_to_test = [
        f"SPY{exp_date_yymmdd}C00550000",   # Standard OCC format
        f"SPY{exp_date_yyyymmdd}C00550000", # Full year format
        f"SPY   {exp_date_yymmdd}C00550000", # With padding
        f"SPY{exp_date_yymmdd}C550",        # Short format
        f".SPY{exp_date_yymmdd}C550",       # With dot
    ]
    
    for symbol in option_symbols_to_test:
        print(f"\nTesting option symbol: {symbol}")
        
        # First check if asset exists
        try:
            asset_url = f"{base_url}/assets/{symbol}"
            asset_response = requests.get(asset_url, headers=headers, timeout=10)
            
            print(f"  Asset check: {asset_response.status_code}")
            
            if asset_response.status_code == 200:
                asset_data = asset_response.json()
                print(f"  Found: {asset_data.get('symbol')} - Tradable: {asset_data.get('tradable')}")
                
                if asset_data.get('tradable'):
                    # Try to create a test order (very low price, won't fill)
                    print("  Testing order creation...")
                    
                    order_data = {
                        "symbol": symbol,
                        "qty": 1,
                        "side": "buy",
                        "type": "limit",
                        "limit_price": "0.01",  # Very low price
                        "time_in_force": "day"
                    }
                    
                    order_url = f"{base_url}/v2/orders"
                    # DON'T actually submit for now
                    print(f"  Would submit: {order_data}")
                    print("  (Dry run - order not submitted)")
                    
                    return symbol  # Found working symbol
            
            elif asset_response.status_code == 404:
                print("  Asset not found")
            else:
                print(f"  Error: {asset_response.text[:100]}")
                
        except Exception as e:
            print(f"  Exception: {e}")
    
    print("\nNo valid option symbols found")
    return None

def test_alpaca_options_api_documentation():
    """Check what Alpaca's API documentation says about options"""
    
    print(f"\n\nALPACA OPTIONS API RESEARCH")
    print("=" * 35)
    
    # Based on previous testing, let's check what we know:
    print("FINDINGS FROM TESTING:")
    print("1. Account has options_trading_level: 3")
    print("2. Account has options_buying_power: $85,680")
    print("3. Regular stock trading works fine")
    print("4. Options assets endpoint returns 404 for all option symbols")
    print("5. Data API has options exchanges but no contracts")
    
    print("\nPOSSIBLE REASONS:")
    print("A. Paper trading might not support options execution")
    print("B. Options might require different API endpoints")
    print("C. Options might need specific symbol format we haven't tried")
    print("D. Options might be available but not discoverable via assets API")

def test_alpaca_trading_interface():
    """Test what we can actually trade via Alpaca"""
    
    headers = {
        'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY'),
        'APCA-API-SECRET-KEY': os.getenv('ALPACA_SECRET_KEY'),
        'Content-Type': 'application/json'
    }
    
    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets/v2')
    
    print(f"\n\nTESTING ACTUAL ALPACA TRADING CAPABILITY")
    print("=" * 45)
    
    # Test 1: Check what assets are actually available
    try:
        print("1. Checking available asset classes...")
        assets_url = f"{base_url}/assets?status=active"
        response = requests.get(assets_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            assets = response.json()
            
            # Count by asset class
            asset_classes = {}
            for asset in assets[:1000]:  # Check first 1000
                asset_class = asset.get('class', 'unknown')
                asset_classes[asset_class] = asset_classes.get(asset_class, 0) + 1
            
            print(f"  Found {len(assets)} total assets")
            print("  Asset classes:")
            for class_name, count in asset_classes.items():
                print(f"    {class_name}: {count}")
                
                # Show sample symbols for options if any exist
                if 'option' in class_name.lower():
                    sample_options = [asset['symbol'] for asset in assets[:10] 
                                    if asset.get('class') == class_name]
                    print(f"      Sample symbols: {sample_options}")
        else:
            print(f"  Error getting assets: {response.status_code}")
    
    except Exception as e:
        print(f"  Exception: {e}")
    
    # Test 2: Try placing a simple stock order to confirm trading works
    print("\n2. Testing basic stock order capability...")
    
    try:
        # Create a very conservative stock order
        order_data = {
            "symbol": "SPY",
            "qty": 1,
            "side": "buy", 
            "type": "limit",
            "limit_price": "1.00",  # Very low price, won't fill
            "time_in_force": "day"
        }
        
        print(f"  Test order: {order_data}")
        print("  (This is a dry run - no actual order submitted)")
        
        # order_url = f"{base_url}/v2/orders"
        # response = requests.post(order_url, headers=headers, json=order_data)
        # print(f"  Order result: {response.status_code}")
        
    except Exception as e:
        print(f"  Exception: {e}")

if __name__ == "__main__":
    print("REAL ALPACA OPTIONS TRADING TEST")
    print("=" * 60)
    
    working_symbol = test_manual_option_order()
    test_alpaca_options_api_documentation()
    test_alpaca_trading_interface()
    
    print(f"\n{'='*60}")
    print("CONCLUSIONS:")
    
    if working_symbol:
        print(f"✅ Found working option symbol: {working_symbol}")
        print("   The bot can be fixed to use this format")
    else:
        print("❌ No working option symbols found")
        print("   Alpaca paper trading may not support options execution")
        
    print("\nRECOMMENDATION:")
    print("Based on your experience placing options manually,")
    print("can you share what symbol format worked for you?")
    print("This will help fix the bot's options trading logic.")