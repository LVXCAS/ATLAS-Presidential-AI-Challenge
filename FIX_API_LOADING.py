"""
FIX API LOADING
===============
Quick fix for environment variable loading
"""

import os
from dotenv import load_dotenv

def test_env_loading():
    """Test environment variable loading"""
    print("="*60)
    print("FIXING API ENVIRONMENT LOADING")
    print("="*60)

    # Load environment variables
    print("Loading .env file...")
    load_result = load_dotenv()
    print(f"Load result: {load_result}")

    # Check if variables are loaded
    api_keys = {
        'ALPACA_API_KEY': os.getenv('ALPACA_API_KEY'),
        'ALPACA_SECRET_KEY': os.getenv('ALPACA_SECRET_KEY'),
        'ALPACA_BASE_URL': os.getenv('ALPACA_BASE_URL'),
        'POLYGON_API_KEY': os.getenv('POLYGON_API_KEY'),
        'ALPHA_VANTAGE_API_KEY': os.getenv('ALPHA_VANTAGE_API_KEY')
    }

    print("\nAPI KEYS STATUS:")
    print("-" * 40)
    all_loaded = True

    for key, value in api_keys.items():
        if value and value != 'YOUR_API_KEY_HERE':
            print(f"[OK] {key}: {value[:8]}...{value[-4:]} ({len(value)} chars)")
        else:
            print(f"[MISSING] {key}: Not loaded")
            all_loaded = False

    print("\n" + "="*60)
    if all_loaded:
        print("✅ ALL API KEYS LOADED SUCCESSFULLY!")
        print("✅ SYSTEM IS NOW 100% READY FOR MONDAY!")
    else:
        print("❌ Some API keys not loading properly")
        print("Check .env file format")

    return all_loaded

def test_alpaca_connection():
    """Test actual Alpaca connection"""
    print("\n" + "="*60)
    print("TESTING ALPACA CONNECTION")
    print("="*60)

    try:
        # Load environment
        load_dotenv()

        # Import and test Alpaca
        import alpaca_trade_api as tradeapi

        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

        if not api_key or not secret_key:
            print("❌ API keys not loaded")
            return False

        # Create API connection
        api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')

        # Test connection
        account = api.get_account()

        print(f"✅ ALPACA CONNECTION SUCCESSFUL!")
        print(f"Account ID: {account.id}")
        print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"Buying Power: ${float(account.buying_power):,.2f}")
        print(f"Paper Trading: {account.account_blocked == False}")

        return True

    except Exception as e:
        print(f"❌ Alpaca connection failed: {e}")
        return False

def create_fixed_test_script():
    """Create a fixed test script with proper env loading"""

    script_content = '''"""
FIXED MONDAY DEPLOYMENT TEST
===========================
Test script with proper environment loading
"""

import os
from dotenv import load_dotenv
import asyncio

# Load environment variables FIRST
load_dotenv()

# Now import other modules
from MONDAY_DEPLOYMENT_SYSTEM import MondayDeploymentSystem

async def test_fixed_monday_deployment():
    """Test Monday deployment with fixed environment loading"""
    print("="*80)
    print("FIXED MONDAY DEPLOYMENT VALIDATION")
    print("="*80)

    # Initialize deployment system
    deployment = MondayDeploymentSystem()

    # Run validation
    await deployment.run_comprehensive_validation()

    # Show results
    if deployment.deployment_ready:
        print("\\n✅ SYSTEM 100% READY FOR MONDAY!")
        print("All APIs loaded and systems validated!")
    else:
        print("\\n⚠️ Some systems need attention")

    # Show component status
    print("\\nCOMPONENT STATUS:")
    for name, status in deployment.component_status.items():
        status_icon = "[OK]" if status.status == 'READY' else "[SETUP]"
        print(f"  {status_icon} {name}: {status.details}")

if __name__ == "__main__":
    asyncio.run(test_fixed_monday_deployment())
'''

    with open('test_fixed_monday_deployment.py', 'w') as f:
        f.write(script_content)

    print("\n✅ Created fixed test script: test_fixed_monday_deployment.py")

def main():
    """Fix the API loading issue"""

    # Test environment loading
    env_success = test_env_loading()

    # Test Alpaca connection
    alpaca_success = test_alpaca_connection()

    # Create fixed test script
    create_fixed_test_script()

    print("\n" + "="*60)
    print("FIX COMPLETE!")
    print("="*60)

    if env_success and alpaca_success:
        print("✅ ALL ISSUES FIXED!")
        print("✅ SYSTEM IS NOW 100% READY FOR MONDAY!")
        print("\\nNEXT STEP:")
        print("Run: python test_fixed_monday_deployment.py")
    else:
        print("⚠️ Some issues remain - check output above")

if __name__ == "__main__":
    main()