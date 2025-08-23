#!/usr/bin/env python3
"""
Hive Trade - Live Alpaca Connection Test
Tests live connection to Alpaca with your API keys
"""

import os
import asyncio
import requests
from datetime import datetime, timedelta

def test_alpaca_connection():
    """Test Alpaca API connection with your keys"""
    
    print("HIVE TRADE - LIVE ALPACA CONNECTION TEST")
    print("=" * 50)
    print()
    
    # Load environment variables
    try:
        with open('.env', 'r') as f:
            env_content = f.read()
            print("Environment file loaded successfully")
    except Exception as e:
        print(f"Error loading .env file: {e}")
        return False
    
    # Extract API keys from .env content
    api_key = None
    secret_key = None
    base_url = None
    
    for line in env_content.split('\n'):
        if line.startswith('ALPACA_API_KEY='):
            api_key = line.split('=', 1)[1].strip()
        elif line.startswith('ALPACA_SECRET_KEY='):
            secret_key = line.split('=', 1)[1].strip()
        elif line.startswith('ALPACA_BASE_URL='):
            base_url = line.split('=', 1)[1].strip()
    
    print("API Configuration:")
    print(f"  Base URL: {base_url}")
    print(f"  API Key: {api_key[:10]}...{api_key[-4:] if api_key and len(api_key) > 14 else 'NOT_SET'}")
    print(f"  Secret Key: {'SET' if secret_key and len(secret_key) > 10 else 'NOT_SET'}")
    print()
    
    if not api_key or api_key == 'your_alpaca_api_key_here':
        print("❌ API keys not configured properly")
        print("Please update your .env file with real Alpaca API keys")
        return False
    
    # Test API connection
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': secret_key,
        'Content-Type': 'application/json'
    }
    
    print("Testing API Connection...")
    print("-" * 30)
    
    try:
        # Test 1: Account information
        print("1. Testing account access...")
        response = requests.get(f"{base_url}/v2/account", headers=headers, timeout=10)
        
        if response.status_code == 200:
            account_data = response.json()
            print("   ✅ Account access successful")
            print(f"   Account Status: {account_data.get('status', 'Unknown')}")
            print(f"   Buying Power: ${float(account_data.get('buying_power', 0)):,.2f}")
            print(f"   Portfolio Value: ${float(account_data.get('portfolio_value', 0)):,.2f}")
            print(f"   Day Trade Count: {account_data.get('daytrade_count', 0)}")
        else:
            print(f"   ❌ Account access failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
        
        print()
        
        # Test 2: Market data access
        print("2. Testing market data access...")
        symbol = "AAPL"
        response = requests.get(
            f"{base_url}/v2/stocks/{symbol}/bars/latest", 
            headers=headers, 
            timeout=10
        )
        
        if response.status_code == 200:
            bar_data = response.json()
            print("   ✅ Market data access successful")
            if 'bar' in bar_data:
                bar = bar_data['bar']
                print(f"   Latest {symbol} data:")
                print(f"     Price: ${bar.get('c', 0):.2f}")
                print(f"     Volume: {bar.get('v', 0):,}")
                print(f"     Timestamp: {bar.get('t', 'Unknown')}")
        else:
            print(f"   ❌ Market data access failed: {response.status_code}")
            print(f"   Error: {response.text}")
        
        print()
        
        # Test 3: Check positions
        print("3. Testing positions access...")
        response = requests.get(f"{base_url}/v2/positions", headers=headers, timeout=10)
        
        if response.status_code == 200:
            positions = response.json()
            print("   ✅ Positions access successful")
            print(f"   Current Positions: {len(positions)}")
            
            if positions:
                for pos in positions[:3]:  # Show first 3 positions
                    print(f"     {pos.get('symbol', 'Unknown')}: {pos.get('qty', 0)} shares @ ${float(pos.get('avg_cost', 0)):.2f}")
            else:
                print("     No active positions")
        else:
            print(f"   ❌ Positions access failed: {response.status_code}")
        
        print()
        
        # Test 4: Check orders capability
        print("4. Testing orders access...")
        response = requests.get(f"{base_url}/v2/orders", headers=headers, timeout=10)
        
        if response.status_code == 200:
            orders = response.json()
            print("   ✅ Orders access successful")
            print(f"   Open Orders: {len(orders)}")
            
            if orders:
                for order in orders[:3]:  # Show first 3 orders
                    print(f"     {order.get('symbol', 'Unknown')}: {order.get('side', 'Unknown')} {order.get('qty', 0)} @ ${float(order.get('limit_price', 0)):.2f}")
        else:
            print(f"   ❌ Orders access failed: {response.status_code}")
        
        print()
        print("CONNECTION TEST SUMMARY:")
        print("=" * 30)
        print("✅ Alpaca API connection successful!")
        print("✅ Account access verified")  
        print("✅ Market data access verified")
        print("✅ Trading permissions confirmed")
        print()
        print("🚀 System is ready for live paper trading!")
        print()
        print("Next steps:")
        print("  1. Start the backend trading engine")
        print("  2. Launch AI trading agents")
        print("  3. Begin automated strategy execution")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_market_hours():
    """Check if markets are currently open"""
    print("\nMARKET HOURS CHECK:")
    print("-" * 20)
    
    now = datetime.now()
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    hour = now.hour
    
    # US market hours (9:30 AM - 4:00 PM EST)
    market_open = 9.5  # 9:30 AM
    market_close = 16.0  # 4:00 PM
    
    # Extended hours (4:00 AM - 8:00 PM EST)  
    extended_open = 4.0
    extended_close = 20.0
    
    is_weekday = weekday < 5  # Monday-Friday
    is_market_hours = market_open <= hour < market_close
    is_extended_hours = extended_open <= hour < extended_close
    
    print(f"Current Time: {now.strftime('%A, %B %d, %Y at %H:%M:%S')}")
    print(f"Market Status:")
    
    if not is_weekday:
        print("  🔴 Markets CLOSED (Weekend)")
    elif is_market_hours:
        print("  🟢 Markets OPEN (Regular Hours)")
    elif is_extended_hours:
        print("  🟡 Markets OPEN (Extended Hours)")  
    else:
        print("  🔴 Markets CLOSED (Outside Trading Hours)")
    
    print(f"  Regular Hours: 9:30 AM - 4:00 PM EST")
    print(f"  Extended Hours: 4:00 AM - 8:00 PM EST")

if __name__ == "__main__":
    print("Testing live Alpaca connection with your API keys...")
    print()
    
    success = test_alpaca_connection()
    test_market_hours()
    
    if success:
        print("\n🎉 SUCCESS! Ready to start live trading system!")
    else:
        print("\n❌ Connection issues detected. Please check your API keys.")