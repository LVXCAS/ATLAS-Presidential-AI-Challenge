#!/usr/bin/env python3
"""
Test options chain availability in Alpaca paper trading
"""
import requests
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta

load_dotenv('.env')

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets/v2')

def test_options_chain_endpoints():
    """Test different options chain API endpoints"""
    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY,
        'Content-Type': 'application/json'
    }
    
    test_symbols = ['SPY', 'IWM', 'QQQ', 'AAPL', 'TSLA']
    
    print("TESTING OPTIONS CHAIN AVAILABILITY")
    print("=" * 50)
    
    for symbol in test_symbols:
        print(f"\nTesting {symbol} options chain...")
        
        # Test correct Alpaca API endpoints (avoid 404 errors)
        endpoints_to_try = [
            f"/assets/{symbol}",  # Check if underlying asset exists
        ]
        
        # Test data API for options (correct endpoint)
        data_api_endpoints = [
            f"https://data.alpaca.markets/v1beta1/options/snapshots?symbols={symbol}",
        ]
        
        for endpoint in endpoints_to_try:
            try:
                url = f"{ALPACA_BASE_URL}{endpoint}"
                response = requests.get(url, headers=headers, timeout=10)
                
                print(f"  {endpoint}: Status {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        print(f"    Found {len(data)} options contracts")
                        if len(data) > 0:
                            print(f"    Sample: {data[0]}")
                    elif isinstance(data, dict):
                        print(f"    Response keys: {list(data.keys())}")
                        if 'chains' in data:
                            chains = data['chains']
                            if isinstance(chains, list):
                                print(f"    Found {len(chains)} option chains")
                elif response.status_code == 404:
                    print(f"    Not Found (404)")
                else:
                    print(f"    Error: {response.text[:100]}...")
                    
            except Exception as e:
                print(f"  {endpoint}: Exception - {e}")
        
        # Test data API endpoints (correct way to get options data)
        print(f"  Testing data API for {symbol} options...")
        for endpoint in data_api_endpoints:
            try:
                response = requests.get(endpoint, headers=headers, timeout=10)
                print(f"  Data API: Status {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"    Options data available for {symbol}")
                elif response.status_code == 400:
                    print(f"    Data API available but requires parameters")
                elif response.status_code == 404:
                    print(f"    Options data not available")
                    
            except Exception as e:
                print(f"  Data API: Exception - {e}")

def test_options_via_polygon():
    """Test if we can get options data via Polygon API"""
    print(f"\n\nTESTING POLYGON OPTIONS DATA")
    print("=" * 40)
    
    polygon_key = os.getenv('POLYGON_API_KEY')
    if not polygon_key:
        print("No Polygon API key found")
        return
    
    # Test Polygon options endpoint
    symbol = "SPY"
    try:
        # Get current date for options expiration
        today = datetime.now()
        friday = today + timedelta(days=(4 - today.weekday()) % 7)
        exp_date = friday.strftime('%Y-%m-%d')
        
        url = f"https://api.polygon.io/v3/reference/options/contracts"
        params = {
            'underlying_ticker': symbol,
            'expiration_date': exp_date,
            'limit': 10,
            'apikey': polygon_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        print(f"Polygon API Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                print(f"Found {len(data['results'])} options contracts via Polygon")
                sample = data['results'][0]
                print(f"Sample contract: {sample.get('ticker', 'N/A')}")
            else:
                print("No options found via Polygon")
        else:
            print(f"Polygon error: {response.text[:200]}")
            
    except Exception as e:
        print(f"Polygon test error: {e}")

def test_alpaca_options_format():
    """Test the specific format Alpaca might use for options"""
    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY,
        'Content-Type': 'application/json'
    }
    
    print(f"\n\nTESTING ALPACA OPTIONS FORMAT")
    print("=" * 40)
    
    # Try searching for specific option symbols
    # Alpaca might use formats like: SPY240913C00550000
    test_option_symbols = [
        'SPY240913C00550000',  # SPY call option
        'IWM240913C00220000',  # IWM call option  
        'QQQ240913C00480000',  # QQQ call option
    ]
    
    for option_symbol in test_option_symbols:
        try:
            url = f"{ALPACA_BASE_URL}/assets/{option_symbol}"
            response = requests.get(url, headers=headers, timeout=10)
            
            print(f"Option {option_symbol}: Status {response.status_code}")
            
            if response.status_code == 200:
                asset_data = response.json()
                print(f"  Found option asset: {asset_data.get('symbol')}")
                print(f"  Asset class: {asset_data.get('class')}")
                print(f"  Tradable: {asset_data.get('tradable')}")
            elif response.status_code == 404:
                print(f"  Option not found")
            else:
                print(f"  Error: {response.text[:100]}")
                
        except Exception as e:
            print(f"  Exception: {e}")

def test_market_data_endpoints():
    """Test market data endpoints that might show options"""
    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY,
        'Content-Type': 'application/json'
    }
    
    print(f"\n\nTESTING MARKET DATA ENDPOINTS")
    print("=" * 40)
    
    # Check if there's a different market data API
    data_endpoints = [
        "/v1beta1/options/meta/exchanges",
        "/v1beta1/options/chains",
        "/v2/options/trades",
        "/v2/stocks/SPY/bars",
    ]
    
    # Try data API base URL
    data_base_url = "https://data.alpaca.markets"
    
    for endpoint in data_endpoints:
        try:
            url = f"{data_base_url}{endpoint}"
            response = requests.get(url, headers=headers, timeout=10)
            
            print(f"{endpoint}: Status {response.status_code}")
            
            if response.status_code == 200:
                print(f"  SUCCESS: Found endpoint")
                data = response.json()
                print(f"  Keys: {list(data.keys()) if isinstance(data, dict) else 'List data'}")
            elif response.status_code == 404:
                print(f"  Not found")
            else:
                print(f"  Error: {response.status_code}")
                
        except Exception as e:
            print(f"  Exception: {e}")

if __name__ == "__main__":
    print("COMPREHENSIVE OPTIONS AVAILABILITY TEST")
    print("=" * 60)
    
    test_options_chain_endpoints()
    test_options_via_polygon()
    test_alpaca_options_format()
    test_market_data_endpoints()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE - Check results above to see where options data is available")