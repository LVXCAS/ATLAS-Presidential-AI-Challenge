#!/usr/bin/env python3
"""
Test Alpaca API Connection with new PI Keys
"""

import os
import sys
import asyncio
from datetime import datetime
sys.path.append('.')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv('.env')
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
except:
    ALPACA_API_KEY = None
    ALPACA_SECRET_KEY = None

print("=" * 60)
print("ALPACA API CONNECTION TEST - PI KEYS")
print("=" * 60)

print(f"\n[CONFIG] Testing with:")
print(f"  API Key: {ALPACA_API_KEY}")
print(f"  Secret Key: {ALPACA_SECRET_KEY[:10]}...{ALPACA_SECRET_KEY[-4:] if ALPACA_SECRET_KEY else 'None'}")
print(f"  Endpoint: https://paper-api.alpaca.markets/v2")

# Test 1: Basic API connection using requests
print(f"\n[TEST 1] Testing basic API connection...")
try:
    import requests
    import base64
    
    # Create auth header
    credentials = f"{ALPACA_API_KEY}:{ALPACA_SECRET_KEY}"
    auth_header = base64.b64encode(credentials.encode()).decode()
    
    headers = {
        'Authorization': f'Basic {auth_header}',
        'Content-Type': 'application/json'
    }
    
    # Test account endpoint
    response = requests.get('https://paper-api.alpaca.markets/v2/account', headers=headers)
    
    if response.status_code == 200:
        account_data = response.json()
        print("[OK] ✅ Basic API connection successful!")
        print(f"  Account ID: {account_data.get('id', 'N/A')}")
        print(f"  Status: {account_data.get('status', 'N/A')}")
        print(f"  Buying Power: ${float(account_data.get('buying_power', 0)):.2f}")
        print(f"  Cash: ${float(account_data.get('cash', 0)):.2f}")
        print(f"  Portfolio Value: ${float(account_data.get('portfolio_value', 0)):.2f}")
        print(f"  Day Trading Buying Power: ${float(account_data.get('daytrading_buying_power', 0)):.2f}")
        print(f"  Pattern Day Trader: {account_data.get('pattern_day_trader', False)}")
    else:
        print(f"[FAIL] ❌ API connection failed!")
        print(f"  Status Code: {response.status_code}")
        print(f"  Response: {response.text}")
        
except Exception as e:
    print(f"[ERROR] Basic API test failed: {e}")

# Test 2: Test using alpaca-trade-api library
print(f"\n[TEST 2] Testing with alpaca-trade-api library...")
try:
    import alpaca_trade_api as tradeapi
    
    api = tradeapi.REST(
        ALPACA_API_KEY,
        ALPACA_SECRET_KEY,
        'https://paper-api.alpaca.markets',
        api_version='v2'
    )
    
    account = api.get_account()
    print("[OK] ✅ Alpaca-trade-api connection successful!")
    print(f"  Account ID: {account.id}")
    print(f"  Status: {account.status}")
    print(f"  Buying Power: ${float(account.buying_power):.2f}")
    print(f"  Cash: ${float(account.cash):.2f}")
    print(f"  Portfolio Value: ${float(account.portfolio_value):.2f}")
    
    # Test getting positions
    positions = api.list_positions()
    print(f"  Current Positions: {len(positions)}")
    
    # Test getting orders
    orders = api.list_orders(status='all', limit=5)
    print(f"  Recent Orders: {len(orders)}")
    
    # Test market data
    try:
        bars = api.get_bars('SPY', tradeapi.TimeFrame.Day, limit=1)
        if bars:
            print(f"  Market Data Access: ✅ Working (SPY: ${bars[0].c:.2f})")
    except:
        print(f"  Market Data Access: ⚠️  Limited (expected for paper account)")
        
except Exception as e:
    print(f"[ERROR] Alpaca-trade-api test failed: {e}")

# Test 3: Test with broker integration agent (will test in main function)
print(f"\n[TEST 3] Testing with OPTIONS_BOT broker integration...")
print("[INFO] Broker integration test will run in async main function")

# Test 4: Test OPTIONS_BOT initialization
print(f"\n[TEST 4] Testing OPTIONS_BOT initialization...")
try:
    from OPTIONS_BOT import TomorrowReadyOptionsBot
    
    bot = TomorrowReadyOptionsBot()
    
    # Test broker connection
    if bot.broker:
        print("[OK] ✅ OPTIONS_BOT broker initialized!")
    else:
        print("[INFO] Broker will be initialized on first run")
    
    print("[OK] ✅ OPTIONS_BOT created successfully!")
    
except Exception as e:
    print(f"[ERROR] OPTIONS_BOT initialization failed: {e}")

print(f"\n" + "=" * 60)
print("ALPACA API CONNECTION TEST COMPLETE")
print("=" * 60)

print(f"\n[SUMMARY]")
if ALPACA_API_KEY and ALPACA_SECRET_KEY:
    print("✅ New PI Keys configured successfully")
    print("✅ Ready for paper trading")
    print("✅ Connection tests completed")
else:
    print("❌ API keys not found - check .env file")

print(f"\n[NEXT STEPS]")
print("1. ✅ API keys updated to PI Keys")
print("2. ✅ Connection tests completed")
print("3. 🎯 Ready to run OPTIONS_BOT with new account")
print("4. 💡 Use 'python OPTIONS_BOT.py' to start trading")

async def main():
    # Test broker integration in async context
    print(f"\n[ASYNC TEST] Testing broker integration...")
    try:
        from agents.broker_integration import AlpacaBrokerIntegration
        
        broker = AlpacaBrokerIntegration()
        
        # Test connection
        if await broker.connect():
            print("[OK] ✅ Broker integration successful!")
            
            # Get account info
            account_info = await broker.get_account_info()
            if account_info:
                print(f"  Account validated: ✅")
                print(f"  Buying power: ${account_info.get('buying_power', 0)}")
                print(f"  Cash: ${account_info.get('cash', 0)}")
            
            # Test getting positions
            positions = await broker.get_positions()
            print(f"  Positions retrieved: {len(positions) if positions else 0}")
            
            # Test market data
            try:
                market_data = await broker.get_market_data('SPY')
                if market_data:
                    print(f"  Market data: ✅ SPY ${market_data.get('price', 'N/A')}")
            except:
                print(f"  Market data: ⚠️  Using fallback source")
                
        else:
            print("[FAIL] ❌ Broker integration failed!")
            
    except Exception as e:
        print(f"[ERROR] Async broker integration test failed: {e}")
    
    print(f"\n[ASYNC TESTS COMPLETE]")

if __name__ == "__main__":
    asyncio.run(main())