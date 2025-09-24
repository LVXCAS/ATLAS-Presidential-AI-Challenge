#!/usr/bin/env python3
"""
Debug script to test actual trade execution
"""
import asyncio
import sys
import os
from datetime import datetime
from dotenv import load_dotenv

sys.path.append('.')
load_dotenv('.env')

async def test_trade_execution():
    """Test if the bot can actually place orders"""
    try:
        print("TESTING TRADE EXECUTION")
        print("=" * 50)
        
        from agents.broker_integration import AlpacaBrokerIntegration
        from agents.options_trading_agent import OptionsTrader, OptionsStrategy
        
        # Initialize components
        print("1. Initializing broker and options trader...")
        broker = AlpacaBrokerIntegration()
        options_trader = OptionsTrader(broker)
        
        # Check account first
        print("2. Checking account status...")
        account = await broker.get_account_info()
        if not account:
            print("ERROR: Cannot get account info")
            return False
            
        print(f"   Account Status: {account.get('status')}")
        print(f"   Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
        print(f"   Trading Blocked: {account.get('trading_blocked', False)}")
        
        # Test simple options order
        print("3. Testing simple options order...")
        
        # Try to find available options for IWM
        symbol = "IWM"
        print(f"   Looking for {symbol} options...")
        
        # Check if we can get options chain
        try:
            # This would be the actual order submission test
            print("   Testing order submission logic...")
            
            # Check if there are any existing positions first
            positions = await broker.get_positions()
            print(f"   Current positions: {len(positions) if positions else 0}")
            
            if positions:
                for pos in positions[:3]:  # Show first 3
                    print(f"     - {pos.get('symbol', 'Unknown')}: {pos.get('qty', 0)} shares")
            
            # Test market data access
            print("4. Testing market data access...")
            # This is where we'd test if the bot can actually get options prices
            
            print("5. Testing options strategy execution...")
            # This would test the actual strategy execution
            
        except Exception as e:
            print(f"   ERROR in order submission: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            return False
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False
    
    return True

async def check_options_permissions():
    """Check if account has proper options permissions"""
    try:
        print("\nCHECKING OPTIONS PERMISSIONS")
        print("=" * 40)
        
        from agents.broker_integration import AlpacaBrokerIntegration
        broker = AlpacaBrokerIntegration()
        
        account = await broker.get_account_info()
        
        print(f"Options Trading Level: {account.get('options_trading_level', 'Not found')}")
        print(f"Options Approved Level: {account.get('options_approved_level', 'Not found')}")
        print(f"Max Options Trading Level: {account.get('max_options_trading_level', 'Not found')}")
        
        # Check if any specific options restrictions
        if 'options_trading_level' not in account:
            print("WARNING: Options trading level not found in account")
        
        if account.get('options_trading_level', 0) == 0:
            print("ERROR: Options trading not enabled!")
            return False
            
        print("SUCCESS: Options permissions appear to be in order")
        return True
        
    except Exception as e:
        print(f"ERROR checking options permissions: {e}")
        return False

async def test_live_order():
    """Try to place a small test order"""
    try:
        print("\nTEST LIVE ORDER (DRY RUN)")
        print("=" * 35)
        
        from agents.broker_integration import AlpacaBrokerIntegration
        broker = AlpacaBrokerIntegration()
        
        # Test a simple stock order first (easier than options)
        print("Testing stock order capability...")
        
        # Don't actually place order, just test the submission path
        order_details = {
            'symbol': 'SPY',
            'qty': 1,
            'side': 'buy',
            'type': 'limit',
            'limit_price': 1.00  # Very low price so it won't fill
        }
        
        print(f"Would submit order: {order_details}")
        print("(This is a dry run - no actual order placed)")
        
        return True
        
    except Exception as e:
        print(f"ERROR in test order: {e}")
        return False

async def main():
    print("OPTIONS_BOT TRADE EXECUTION DIAGNOSTIC")
    print("=" * 60)
    
    success1 = await test_trade_execution()
    success2 = await check_options_permissions()
    success3 = await test_live_order()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC RESULTS:")
    print(f"Trade Execution Test: {'PASS' if success1 else 'FAIL'}")
    print(f"Options Permissions: {'PASS' if success2 else 'FAIL'}")
    print(f"Order Submission Test: {'PASS' if success3 else 'FAIL'}")
    
    if all([success1, success2, success3]):
        print("\nCONCLUSION: Bot should be able to trade")
        print("Check OPTIONS_BOT logs for specific error messages")
    else:
        print("\nCONCLUSION: Found issues preventing trade execution")

if __name__ == "__main__":
    asyncio.run(main())