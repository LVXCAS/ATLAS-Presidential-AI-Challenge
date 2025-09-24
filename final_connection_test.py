#!/usr/bin/env python3
"""
Final connection test using OPTIONS_BOT's exact connection method
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append('.')

# Load environment variables
load_dotenv('.env')

async def test_options_bot_connection():
    """Test using the same connection method as OPTIONS_BOT"""
    try:
        print("Testing OPTIONS_BOT Connection Method...")
        print("=" * 50)
        
        # Import the same way OPTIONS_BOT does
        from agents.broker_integration import AlpacaBrokerIntegration
        
        # Initialize broker the same way
        print("1. Initializing AlpacaBrokerIntegration...")
        broker = AlpacaBrokerIntegration()
        
        # Test account info
        print("2. Getting account information...")
        account_info = await broker.get_account_info()
        
        if account_info:
            print("SUCCESS: Account connection working!")
            print(f"   Account ID: {account_info.get('id')}")
            print(f"   Status: {account_info.get('status')}")
            print(f"   Buying Power: ${float(account_info.get('buying_power', 0)):,.2f}")
            print(f"   Portfolio Value: ${float(account_info.get('portfolio_value', 0)):,.2f}")
            print(f"   Options Level: {account_info.get('options_trading_level')}")
            print(f"   Trading Blocked: {account_info.get('trading_blocked', False)}")
            
            if account_info.get('trading_blocked'):
                print("WARNING: Trading is blocked on this account!")
                return False
            else:
                print("READY: Account is ready for trading!")
                return True
        else:
            print("ERROR: Could not retrieve account information")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

async def main():
    print("FINAL OPTIONS_BOT CONNECTION TEST")
    print("=" * 60)
    
    # Test connection
    success = await test_options_bot_connection()
    
    print("\n" + "=" * 60)
    if success:
        print("RESULT: OPTIONS_BOT should work with these credentials!")
        print("You can now run: python OPTIONS_BOT.py")
    else:
        print("RESULT: There may be issues with OPTIONS_BOT connection")
        print("Check the error messages above for details")

if __name__ == "__main__":
    asyncio.run(main())