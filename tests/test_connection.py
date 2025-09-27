#!/usr/bin/env python3
"""
Test Alpaca API connection with current credentials
"""
import os
import sys
from config.settings import get_settings

def test_alpaca_connection():
    """Test connection to Alpaca API"""
    try:
        print("Testing Alpaca API connection...")
        print("=" * 50)
        
        # Load settings
        settings = get_settings()
        
        # Check if credentials are loaded
        api_key = settings.ALPACA_API_KEY
        secret_key = settings.ALPACA_SECRET_KEY
        base_url = settings.ALPACA_PAPER_BASE_URL
        
        print(f"API Key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else 'NOT FOUND'}")
        print(f"Secret Key: {'*' * 8}...{secret_key[-4:] if len(secret_key) > 12 else 'NOT FOUND'}")
        print(f"Base URL: {base_url}")
        print(f"Paper Trading: {settings.trading.paper_trading}")
        print(f"Initial Capital: ${settings.trading.initial_capital:,.2f}")
        
        if not api_key or not secret_key:
            print("\n[ERROR] API credentials not found in environment!")
            print("Please check your .env file contains:")
            print("ALPACA_API_KEY=your_key_here")
            print("ALPACA_SECRET_KEY=your_secret_here")
            return False
        
        # Test actual API connection
        try:
            import alpaca_trade_api as tradeapi
            
            api = tradeapi.REST(
                key_id=api_key,
                secret_key=secret_key,
                base_url=base_url
            )
            
            # Test account access
            account = api.get_account()
            print(f"\n[SUCCESS] CONNECTION ESTABLISHED!")
            print(f"Account Status: {account.status}")
            print(f"Account Equity: ${float(account.equity):,.2f}")
            print(f"Buying Power: ${float(account.buying_power):,.2f}")
            print(f"Day Trading Buying Power: ${float(account.daytrading_buying_power):,.2f}")
            
            # Test positions
            positions = api.list_positions()
            print(f"Current Positions: {len(positions)}")
            
            # Test orders
            orders = api.list_orders(status='all', limit=5)
            print(f"Recent Orders: {len(orders)}")
            
            print(f"\n[SUCCESS] Your trades WILL appear on the Alpaca website!")
            print(f"Login at: https://app.alpaca.markets/paper/dashboard/overview")
            
            return True
            
        except Exception as api_error:
            print(f"\n[ERROR] API CONNECTION FAILED: {api_error}")
            print("Please check your API credentials and try again.")
            return False
        
    except Exception as e:
        print(f"[ERROR]: {e}")
        return False

if __name__ == "__main__":
    success = test_alpaca_connection()
    print("\n" + "=" * 50)
    if success:
        print("[SUCCESS] Setup complete! Bot is ready for trading.")
    else:
        print("[ERROR] Setup incomplete. Please fix the issues above.")
    
    input("Press Enter to close...")
    sys.exit(0 if success else 1)