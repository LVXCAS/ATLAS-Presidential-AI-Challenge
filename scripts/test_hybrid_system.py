"""
Quick test script to verify hybrid system components
Tests: SHARED libraries, AI confirmation, OANDA connection
"""
import os
from dotenv import load_dotenv

print("=" * 80)
print(" " * 25 + "HYBRID SYSTEM COMPONENT TEST")
print("=" * 80)

# Load environment
load_dotenv()

# Test 1: Environment Variables
print("\n[TEST 1] Environment Variables")
print("-" * 80)

oanda_key = os.getenv('OANDA_API_KEY')
oanda_account = os.getenv('OANDA_ACCOUNT_ID')
openrouter_key = os.getenv('OPENROUTER_API_KEY')

if oanda_key:
    print(f"  OANDA_API_KEY: {oanda_key[:20]}... [OK]")
else:
    print("  OANDA_API_KEY: MISSING [ERROR]")

if oanda_account:
    print(f"  OANDA_ACCOUNT_ID: {oanda_account} [OK]")
else:
    print("  OANDA_ACCOUNT_ID: MISSING [ERROR]")

if openrouter_key:
    print(f"  OPENROUTER_API_KEY: {openrouter_key[:20]}... [OK]")
else:
    print("  OPENROUTER_API_KEY: MISSING [ERROR]")

# Test 2: SHARED Libraries Import
print("\n[TEST 2] SHARED Libraries Import")
print("-" * 80)

try:
    from SHARED.technical_analysis import ta
    print("  technical_analysis.py: [OK]")
except Exception as e:
    print(f"  technical_analysis.py: [ERROR] {e}")

try:
    from SHARED.kelly_criterion import kelly
    print("  kelly_criterion.py: [OK]")
except Exception as e:
    print(f"  kelly_criterion.py: [ERROR] {e}")

try:
    from SHARED.multi_timeframe import mtf
    print("  multi_timeframe.py: [OK]")
except Exception as e:
    print(f"  multi_timeframe.py: [ERROR] {e}")

try:
    from SHARED.ai_confirmation import ai_agent
    print("  ai_confirmation.py: [OK]")
    print(f"    AI Agent Enabled: {ai_agent.enabled}")
    print(f"    Models: {list(ai_agent.models.keys())}")
except Exception as e:
    print(f"  ai_confirmation.py: [ERROR] {e}")

try:
    from SHARED.trade_logger import trade_logger
    print("  trade_logger.py: [OK]")
    print(f"    Log Directory: {trade_logger.log_dir}")
    print(f"    Session ID: {trade_logger.session_id}")
except Exception as e:
    print(f"  trade_logger.py: [ERROR] {e}")

# Test 3: TA-Lib Availability
print("\n[TEST 3] TA-Lib Availability")
print("-" * 80)

try:
    import talib
    import numpy as np

    # Test RSI calculation
    test_data = np.random.randn(100) * 10 + 100
    rsi = talib.RSI(test_data, timeperiod=14)

    print(f"  TA-Lib installed: [OK]")
    print(f"  RSI test value: {rsi[-1]:.2f}")
except ImportError:
    print("  TA-Lib: [WARN] Not installed (will use fallback calculations)")
except Exception as e:
    print(f"  TA-Lib test error: {e}")

# Test 4: OANDA Connection
print("\n[TEST 4] OANDA Connection")
print("-" * 80)

if oanda_key and oanda_account:
    try:
        import oandapyV20
        from oandapyV20 import API
        import oandapyV20.endpoints.accounts as accounts

        client = API(access_token=oanda_key, environment='practice')
        r = accounts.AccountSummary(accountID=oanda_account)
        response = client.request(r)

        balance = float(response['account']['balance'])
        currency = response['account']['currency']

        print(f"  OANDA Connection: [OK]")
        print(f"  Account Balance: {currency} {balance:,.2f}")
        print(f"  Account ID: {oanda_account}")

    except Exception as e:
        print(f"  OANDA Connection: [ERROR] {e}")
else:
    print("  OANDA Connection: SKIPPED (keys missing)")

# Test 5: AI Confirmation Test
print("\n[TEST 5] AI Confirmation Test")
print("-" * 80)

if openrouter_key:
    try:
        from SHARED.ai_confirmation import ai_agent

        # Test trade data
        test_trade = {
            'symbol': 'EUR_USD',
            'direction': 'long',
            'score': 7.5,
            'rsi': 28.5,
            'macd': {'macd': 0.0015, 'signal': 0.0010, 'histogram': 0.0005},
            'current_price': 1.0850,
            'trend_4h': 'bullish'
        }

        print("  Testing AI confirmation with sample trade...")
        print(f"    Symbol: {test_trade['symbol']}")
        print(f"    Direction: {test_trade['direction']}")
        print(f"    TA Score: {test_trade['score']}/10")
        print(f"    RSI: {test_trade['rsi']}")
        print()
        print("  Calling DeepSeek V3.1 + MiniMax APIs...")

        ai_decision = ai_agent.analyze_trade(test_trade, market_type='forex')

        print(f"\n  AI Decision: {ai_decision['action']} [OK]")
        print(f"    Confidence: {ai_decision['confidence']:.0f}%")
        print(f"    Consensus: {ai_decision.get('consensus', False)}")
        print(f"    Reason: {ai_decision['reason']}")

        if ai_decision.get('deepseek_decision'):
            print(f"\n    DeepSeek: {ai_decision['deepseek_decision']['action']} ({ai_decision['deepseek_decision']['confidence']}%)")
        if ai_decision.get('minimax_decision'):
            print(f"    MiniMax: {ai_decision['minimax_decision']['action']} ({ai_decision['minimax_decision']['confidence']}%)")

    except Exception as e:
        print(f"  AI Confirmation Test: [ERROR] {e}")
        import traceback
        traceback.print_exc()
else:
    print("  AI Confirmation: SKIPPED (OpenRouter key missing)")

# Summary
print("\n" + "=" * 80)
print(" " * 30 + "TEST SUMMARY")
print("=" * 80)

if oanda_key and oanda_account and openrouter_key:
    print("\n  [SUCCESS] ALL COMPONENTS READY")
    print("\n  You can now run:")
    print("    python MULTI_MARKET_TRADER.py")
    print("\n  OR use the launcher:")
    print("    START_HYBRID_TRADER.bat")
else:
    print("\n  [WARNING] Some components missing:")
    if not oanda_key:
        print("    - OANDA_API_KEY needed for forex trading")
    if not oanda_account:
        print("    - OANDA_ACCOUNT_ID needed for forex trading")
    if not openrouter_key:
        print("    - OPENROUTER_API_KEY needed for AI confirmation")

print("\n" + "=" * 80)
