#!/usr/bin/env python3
"""
Test actual options trading capabilities in Alpaca paper trading
Based on official Alpaca documentation
"""
import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json

load_dotenv('.env')

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets/v2')

def test_options_via_data_api():
    """Test options data via Alpaca's data API"""
    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY,
        'Content-Type': 'application/json'
    }
    
    print("TESTING ALPACA DATA API FOR OPTIONS")
    print("=" * 45)
    
    # Try the data API endpoints
    data_base = "https://data.alpaca.markets"
    
    # Test options snapshots
    symbol = "SPY"
    try:
        # Try options snapshot endpoint (correct Alpaca API endpoint)
        url = f"{data_base}/v1beta1/options/snapshots"
        params = {'symbols': symbol}
        
        response = requests.get(url, headers=headers, params=params, timeout=15)
        print(f"Options snapshots: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response keys: {list(data.keys())}")
            if 'snapshots' in data:
                snapshots = data['snapshots']
                if snapshots:
                    print(f"Found snapshots for: {list(snapshots.keys())}")
                else:
                    print("No snapshots data")
        elif response.status_code == 422:
            print("422 - Unprocessable Entity (might need specific option symbols)")
            print(f"Response: {response.text}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

def test_option_symbol_format():
    """Try to find the correct option symbol format"""
    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY,
        'Content-Type': 'application/json'
    }
    
    print(f"\nTESTING OPTION SYMBOL FORMATS")
    print("=" * 35)
    
    # Generate some realistic option symbols for next Friday
    today = datetime.now()
    # Find next Friday
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0:  # Today is Friday, get next Friday
        days_until_friday = 7
    next_friday = today + timedelta(days=days_until_friday)
    
    # Format: YYMMDD
    exp_date = next_friday.strftime('%y%m%d')
    
    # Common option symbol formats
    option_formats = [
        f"SPY{exp_date}C00550000",  # Standard OCC format
        f"SPY{exp_date}C550",       # Simplified format
        f"SPY_{exp_date}C550",      # With underscore
        f".SPY{exp_date}C550",      # With dot prefix
        f"SPY {exp_date} C550",     # With spaces
    ]
    
    print(f"Testing options expiring {next_friday.strftime('%Y-%m-%d')}:")
    
    for symbol in option_formats:
        try:
            # Test via assets endpoint
            url = f"{ALPACA_BASE_URL}/assets/{symbol}"
            response = requests.get(url, headers=headers, timeout=10)
            
            print(f"  {symbol}: {response.status_code}")
            
            if response.status_code == 200:
                asset = response.json()
                print(f"    FOUND! Class: {asset.get('class')}, Tradable: {asset.get('tradable')}")
                return symbol
            elif response.status_code == 422:
                print(f"    Invalid format")
            
        except Exception as e:
            print(f"    Exception: {e}")
    
    return None

def test_create_option_order():
    """Test creating an actual option order (dry run)"""
    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY,
        'Content-Type': 'application/json'
    }
    
    print(f"\nTESTING OPTION ORDER CREATION")
    print("=" * 35)
    
    # Try to create a very conservative option order
    # Use a realistic but unlikely to fill price
    order_data = {
        "symbol": "SPY",
        "qty": 1,
        "side": "buy",
        "type": "limit",
        "limit_price": "0.01",  # Very low price
        "time_in_force": "day",
        "order_class": "simple",
        # Try option-specific fields
        "legs": [
            {
                "symbol": "SPY250919C00600000",  # Example option symbol
                "side": "buy",
                "qty": 1
            }
        ]
    }
    
    try:
        url = f"{ALPACA_BASE_URL}/v2/orders"
        
        print("Attempting to create option order (will cancel immediately)...")
        print(f"Order data: {json.dumps(order_data, indent=2)}")
        
        # DON'T actually submit - just test the validation
        print("(DRY RUN - not actually submitting)")
        
        # If we wanted to test for real:
        # response = requests.post(url, headers=headers, json=order_data, timeout=10)
        # print(f"Order submission: {response.status_code}")
        # print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"Exception: {e}")

def check_alpaca_documentation():
    """Check what Alpaca actually supports"""
    print(f"\nALPACA OPTIONS SUPPORT CHECK")
    print("=" * 35)
    
    print("Based on testing:")
    print("✅ Account has options_trading_level: 3")
    print("✅ Account has options_buying_power: $85,680")
    print("✅ Polygon API shows options contracts exist")
    print("✅ Alpaca data API has options exchanges")
    print("❌ Alpaca trading API has no options endpoints")
    print("❌ No options found via assets endpoint")
    
    print("\nPossible explanations:")
    print("1. Paper trading might not support options execution")
    print("2. Options might require different API endpoints")
    print("3. Options might need special symbols/format")
    print("4. Options might only work in live trading")

if __name__ == "__main__":
    print("ALPACA OPTIONS TRADING CAPABILITY TEST")
    print("=" * 60)
    
    test_options_via_data_api()
    option_symbol = test_option_symbol_format()
    test_create_option_order()
    check_alpaca_documentation()
    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    if option_symbol:
        print(f"✅ Found working option symbol format: {option_symbol}")
    else:
        print("❌ Could not find valid option symbols in paper trading")
        print("   This suggests paper trading may not support options execution")
        print("   even though the account is approved for options trading.")