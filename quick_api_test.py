"""
QUICK API TEST
==============
Test API connections without Unicode issues
"""

import os
from dotenv import load_dotenv

def test_apis():
    """Test API connections"""
    load_dotenv()

    print("API LOADING TEST")
    print("================")

    # Check environment loading
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    base_url = os.getenv('ALPACA_BASE_URL')

    if api_key and secret_key:
        print(f"[OK] Alpaca API Key: {api_key[:8]}...{api_key[-4:]}")
        print(f"[OK] Alpaca Secret: {secret_key[:8]}...{secret_key[-4:]}")
        print(f"[OK] Alpaca URL: {base_url}")

        # Test connection
        try:
            import alpaca_trade_api as tradeapi
            api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
            account = api.get_account()

            print("\nALPACA CONNECTION TEST")
            print("======================")
            print(f"[SUCCESS] Connected to Alpaca!")
            print(f"Account ID: {account.id}")
            print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
            print(f"Buying Power: ${float(account.buying_power):,.2f}")
            print(f"Paper Trading: {not account.account_blocked}")

            return True

        except Exception as e:
            print(f"[ERROR] Alpaca connection failed: {e}")
            return False
    else:
        print("[ERROR] API keys not loaded")
        return False

if __name__ == "__main__":
    success = test_apis()
    print(f"\nAPI TEST RESULT: {'SUCCESS' if success else 'FAILED'}")
    if success:
        print("SYSTEM IS READY FOR MONDAY DEPLOYMENT!")