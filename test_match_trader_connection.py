"""
TEST MATCH TRADER CONNECTION

Verify your Match Trader demo credentials work before starting the bot.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_connection():
    """Test Match Trader connection"""

    print("=" * 70)
    print("MATCH TRADER CONNECTION TEST")
    print("=" * 70)

    # Check if credentials exist in .env
    print("\n[1/3] Checking .env credentials...")

    e8_account = os.getenv('E8_ACCOUNT')
    e8_password = os.getenv('E8_PASSWORD')
    e8_server = os.getenv('E8_SERVER', 'match-trader-demo')

    if not e8_account:
        print("  [ERROR] E8_ACCOUNT not found in .env")
        print("\nAdd to your .env file:")
        print("E8_ACCOUNT=your_account_number")
        print("E8_PASSWORD=your_password")
        print("E8_SERVER=match-trader-demo")
        return False

    if not e8_password:
        print("  [ERROR] E8_PASSWORD not found in .env")
        return False

    print(f"  ✓ E8_ACCOUNT: {e8_account}")
    print(f"  ✓ E8_PASSWORD: {'*' * len(e8_password)}")
    print(f"  ✓ E8_SERVER: {e8_server}")

    # Test connection via hybrid adapter
    print("\n[2/3] Testing connection to Match Trader...")

    try:
        from HYBRID_OANDA_TRADELOCKER import HybridAdapter

        client = HybridAdapter()

        print("  ✓ HybridAdapter initialized")

        # Get account balance
        balance_data = client.get_account_balance()

        if balance_data:
            equity = balance_data.get('equity', 0)

            print(f"  ✓ Successfully connected!")
            print(f"  Account: {e8_account}")
            print(f"  Balance: ${equity:,.2f}")
            print(f"  Server: {e8_server}")

            # Check if it's the expected demo balance
            if abs(equity - 200000) < 100:  # Within $100 of $200k
                print("\n  ✓ This looks like an E8 $200K demo account")
            else:
                print(f"\n  [WARN] Balance is ${equity:,.2f}, expected ~$200,000")
                print("  Make sure this is your E8 demo account, not a different account")

            return True
        else:
            print("  [ERROR] Failed to get account balance")
            return False

    except ImportError:
        print("  [ERROR] HYBRID_OANDA_TRADELOCKER.py not found")
        print("  Make sure the hybrid adapter is in the same directory")
        return False
    except Exception as e:
        print(f"  [ERROR] Connection failed: {e}")
        print("\nPossible issues:")
        print("  1. Wrong account number or password")
        print("  2. Server name incorrect (should be match-trader-demo)")
        print("  3. Account expired or terminated")
        print("  4. Network/firewall blocking connection")
        return False

    # Test data retrieval
    print("\n[3/3] Testing data retrieval...")

    try:
        # Try to get candle data for EUR/USD
        candles = client.get_candles('EUR_USD', 'H1', count=10)

        if candles and len(candles) > 0:
            print(f"  ✓ Retrieved {len(candles)} candles for EUR/USD")
            print(f"  Latest price: {candles[-1]['close']:.5f}")

            print("\n" + "=" * 70)
            print("[SUCCESS] Match Trader connection working!")
            print("=" * 70)
            print("\nYou're ready to start the ultra-conservative bot:")
            print("  cd BOTS")
            print("  pythonw E8_ULTRA_CONSERVATIVE_BOT.py")
            print("\nOr run in foreground to see output:")
            print("  cd BOTS")
            print("  python E8_ULTRA_CONSERVATIVE_BOT.py")
            print("=" * 70)

            return True
        else:
            print("  [ERROR] Failed to retrieve candle data")
            return False

    except Exception as e:
        print(f"  [ERROR] Data retrieval failed: {e}")
        return False


if __name__ == "__main__":
    success = test_connection()

    if not success:
        print("\n" + "=" * 70)
        print("[SETUP NEEDED]")
        print("=" * 70)
        print("\nTo get your Match Trader demo credentials:")
        print("  1. Log into your E8 account at e8funding.com")
        print("  2. Go to 'My Challenges' or 'Accounts'")
        print("  3. Look for 'Match Trader Demo' section")
        print("  4. Copy account number and password")
        print("  5. Add to your .env file:")
        print("\n     E8_ACCOUNT=your_account_number")
        print("     E8_PASSWORD=your_password")
        print("     E8_SERVER=match-trader-demo")
        print("\n  6. Run this test again:")
        print("     python test_match_trader_connection.py")
        print("=" * 70)
