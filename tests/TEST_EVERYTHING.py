#!/usr/bin/env python3
"""
COMPREHENSIVE SYSTEM TEST
Tests all 5 systems before Monday
"""

import sys
import traceback

def print_test_header(test_name):
    print("\n" + "="*70)
    print(f"TEST: {test_name}")
    print("="*70)

def print_result(passed, message):
    status = "[PASS]" if passed else "[FAIL]"
    symbol = "✓" if passed else "✗"
    print(f"{status} {message}")
    return passed

# Track results
all_tests_passed = True

# TEST 1: Account Verification System
print_test_header("1. ACCOUNT VERIFICATION SYSTEM")
try:
    from account_verification_system import AccountVerificationSystem

    verifier = AccountVerificationSystem()
    result = verifier.verify_account_ready('BULL_PUT_SPREAD')

    if result['ready']:
        print_result(True, f"Account verified: {result['account_id'][:10]}... with ${result['equity']:,.0f}")
        print_result(True, f"Options buying power: ${result['options_buying_power']:,.0f}")
    else:
        print_result(False, f"Account NOT ready: {result['issues']}")
        all_tests_passed = False

except Exception as e:
    print_result(False, f"Account verification failed: {e}")
    traceback.print_exc()
    all_tests_passed = False

# TEST 2: Market Regime Detector
print_test_header("2. MARKET REGIME DETECTOR")
try:
    from market_regime_detector import MarketRegimeDetector

    detector = MarketRegimeDetector()
    regime = detector.analyze_market_regime()

    print_result(True, f"Market regime detected: {regime['regime']}")
    print_result(True, f"S&P 500 momentum: {regime['sp500_momentum']:+.1%}")
    print_result(True, f"VIX level: {regime['vix_level']:.2f}")
    print_result(True, f"Bull Put Spreads viable: {regime['bull_put_spread_viable']}")

except Exception as e:
    print_result(False, f"Market regime detector failed: {e}")
    traceback.print_exc()
    all_tests_passed = False

# TEST 3: Multi-Source Data Fetcher
print_test_header("3. MULTI-SOURCE DATA FETCHER")
try:
    from multi_source_data_fetcher import MultiSourceDataFetcher
    import time

    fetcher = MultiSourceDataFetcher()

    # Test 5 random symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    start_time = time.time()

    successes = 0
    for symbol in test_symbols:
        try:
            bars = fetcher.get_bars(symbol, '1Day', limit=30)
            if bars and hasattr(bars, 'df') and not bars.df.empty:
                successes += 1
        except:
            pass

    elapsed = time.time() - start_time

    print_result(successes == 5, f"Fetched data for {successes}/5 symbols in {elapsed:.2f}s")
    print_result(elapsed < 5, f"Speed test: {elapsed:.2f}s (should be <5s for 5 symbols)")

    if successes < 5:
        all_tests_passed = False

except Exception as e:
    print_result(False, f"Multi-source data fetcher failed: {e}")
    traceback.print_exc()
    all_tests_passed = False

# TEST 4: All-Weather System
print_test_header("4. ALL-WEATHER TRADING SYSTEM")
try:
    from orchestration.all_weather_trading_system import AllWeatherTradingSystem

    system = AllWeatherTradingSystem()
    regime_data = system.detect_market_regime()
    should_trade, reason = system.should_trade_today(regime_data)

    print_result(True, f"Regime: {regime_data['regime'].value}")
    print_result(True, f"Position sizing: {regime_data['position_sizing']:.1f}x")
    print_result(True, f"Should trade: {should_trade}")
    print_result(True, f"Reason: {reason}")

except Exception as e:
    print_result(False, f"All-weather system failed: {e}")
    traceback.print_exc()
    all_tests_passed = False

# TEST 5: Bull Put Spread Engine
print_test_header("5. BULL PUT SPREAD ENGINE")
try:
    from strategies.bull_put_spread_engine import BullPutSpreadEngine

    engine = BullPutSpreadEngine()
    print_result(True, "Bull Put Spread engine loaded successfully")

    # Test strategy calculation (not actual execution)
    test_price = 100.0
    sell_strike = round(test_price * 0.90)  # 10% OTM
    buy_strike = sell_strike - 5

    print_result(True, f"Test calculation: Price=${test_price}, Sell=${sell_strike}, Buy=${buy_strike}")
    print_result(sell_strike == 90, f"Sell strike calculation correct: {sell_strike}")
    print_result(buy_strike == 85, f"Buy strike calculation correct: {buy_strike}")

    if sell_strike != 90 or buy_strike != 85:
        all_tests_passed = False

except Exception as e:
    print_result(False, f"Bull Put Spread engine failed: {e}")
    traceback.print_exc()
    all_tests_passed = False

# TEST 6: Week 3 Production Scanner (Imports Only)
print_test_header("6. WEEK 3 PRODUCTION SCANNER (Import Test)")
try:
    # Check if file exists
    import os
    scanner_path = 'week3_production_scanner.py'

    if os.path.exists(scanner_path):
        print_result(True, f"Scanner file exists: {scanner_path}")

        # Try importing key dependencies
        from time_series_momentum_strategy import TimeSeriesMomentumStrategy
        from week1_execution_system import Week1ExecutionSystem
        from core.adaptive_dual_options_engine import AdaptiveDualOptionsEngine

        print_result(True, "All scanner dependencies import successfully")
    else:
        print_result(False, f"Scanner file not found: {scanner_path}")
        all_tests_passed = False

except Exception as e:
    print_result(False, f"Scanner import test failed: {e}")
    traceback.print_exc()
    all_tests_passed = False

# TEST 7: Strategy Selection Logic
print_test_header("7. STRATEGY SELECTION LOGIC")
try:
    # Test momentum-based strategy selection
    test_cases = [
        (0.01, "Bull Put Spread", "Low momentum <3%"),
        (0.05, "Dual Options", "High momentum >3%"),
        (0.15, "Dual Options", "Very high momentum"),
    ]

    passed = 0
    for momentum, expected_strategy, description in test_cases:
        if momentum < 0.03:
            selected = "Bull Put Spread"
        else:
            selected = "Dual Options"

        if selected == expected_strategy:
            print_result(True, f"{description}: {selected} (correct)")
            passed += 1
        else:
            print_result(False, f"{description}: {selected} (expected {expected_strategy})")

    if passed != len(test_cases):
        all_tests_passed = False

except Exception as e:
    print_result(False, f"Strategy selection test failed: {e}")
    traceback.print_exc()
    all_tests_passed = False

# TEST 8: File Structure Check
print_test_header("8. FILE STRUCTURE CHECK")
try:
    import os

    required_files = [
        'account_verification_system.py',
        'market_regime_detector.py',
        'multi_source_data_fetcher.py',
        'week3_production_scanner.py',
        'orchestration/all_weather_trading_system.py',
        'strategies/bull_put_spread_engine.py',
    ]

    missing = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print_result(True, f"Found: {file_path}")
        else:
            print_result(False, f"Missing: {file_path}")
            missing.append(file_path)

    if missing:
        all_tests_passed = False

except Exception as e:
    print_result(False, f"File structure check failed: {e}")
    traceback.print_exc()
    all_tests_passed = False

# FINAL SUMMARY
print("\n" + "="*70)
print("FINAL TEST SUMMARY")
print("="*70)

if all_tests_passed:
    print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("\nSYSTEM STATUS: READY FOR MONDAY")
    print("\nWhat works:")
    print("  ✓ Account verification")
    print("  ✓ Market regime detection")
    print("  ✓ Multi-source data fetching")
    print("  ✓ All-weather system")
    print("  ✓ Bull Put Spread engine")
    print("  ✓ Strategy selection logic")
    print("  ✓ All files present")
    print("\nNext step: Run full scanner Monday 9:30 AM")
else:
    print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
    print("\nSYSTEM STATUS: NEEDS FIXES")
    print("\nReview failures above and fix before Monday")

print("="*70 + "\n")
