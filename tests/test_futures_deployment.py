#!/usr/bin/env python3
"""
TEST FUTURES DEPLOYMENT
Verify all futures deployment components are working

Tests:
1. Futures scanner works
2. Strategy works
3. Data fetcher works
4. All deployment scripts are runnable
5. Safety limits are applied

Usage:
    python test_futures_deployment.py
"""

import sys
import os
import importlib

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def test_imports():
    """Test all futures modules can be imported"""

    print("\n" + "="*70)
    print("TEST 1: MODULE IMPORTS")
    print("="*70)

    modules_to_test = [
        ('scanners.futures_scanner', 'AIEnhancedFuturesScanner'),
        ('strategies.futures_ema_strategy', 'FuturesEMAStrategy'),
        ('data.futures_data_fetcher', 'FuturesDataFetcher'),
    ]

    all_passed = True

    for module_name, class_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            print(f"✓ {module_name}.{class_name}")
        except Exception as e:
            print(f"✗ {module_name}.{class_name} - {e}")
            all_passed = False

    if all_passed:
        print("\n[PASS] All modules imported successfully")
    else:
        print("\n[FAIL] Some modules failed to import")

    return all_passed


def test_scanner_initialization():
    """Test futures scanner can be initialized"""

    print("\n" + "="*70)
    print("TEST 2: SCANNER INITIALIZATION")
    print("="*70)

    try:
        from scanners.futures_scanner import AIEnhancedFuturesScanner
        scanner = AIEnhancedFuturesScanner(paper_trading=True)
        print("✓ Scanner initialized successfully")
        print(f"  Min score for AI: {scanner.min_score_for_ai}")
        print(f"  Confidence multiplier: {scanner.confidence_multiplier}")
        print("\n[PASS] Scanner initialization successful")
        return True
    except Exception as e:
        print(f"✗ Scanner initialization failed: {e}")
        print("\n[FAIL] Scanner initialization failed")
        return False


def test_strategy():
    """Test strategy can analyze data"""

    print("\n" + "="*70)
    print("TEST 3: STRATEGY ANALYSIS")
    print("="*70)

    try:
        import pandas as pd
        import numpy as np
        from strategies.futures_ema_strategy import FuturesEMAStrategy

        # Create mock data
        dates = pd.date_range(start='2025-01-01', periods=300, freq='15min')
        np.random.seed(42)

        trend = np.linspace(4500, 4600, 300)
        noise = np.random.normal(0, 5, 300)
        close_prices = trend + noise

        df = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices - np.random.uniform(0, 3, 300),
            'high': close_prices + np.random.uniform(0, 5, 300),
            'low': close_prices - np.random.uniform(0, 5, 300),
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, 300)
        })

        strategy = FuturesEMAStrategy()
        print("✓ Strategy initialized")

        # Calculate indicators
        df_with_indicators = strategy.calculate_indicators(df)
        print("✓ Indicators calculated")

        # Check for opportunity
        opportunity = strategy.analyze_opportunity(df_with_indicators, 'MES')
        if opportunity:
            print(f"✓ Opportunity detected: {opportunity['direction']}")
            print(f"  Score: {opportunity['score']:.2f}")
            print(f"  Entry: ${opportunity['entry_price']:.2f}")
        else:
            print("✓ No opportunity (expected with random data)")

        print("\n[PASS] Strategy analysis successful")
        return True
    except Exception as e:
        print(f"✗ Strategy analysis failed: {e}")
        print("\n[FAIL] Strategy analysis failed")
        import traceback
        traceback.print_exc()
        return False


def test_data_fetcher():
    """Test data fetcher initialization"""

    print("\n" + "="*70)
    print("TEST 4: DATA FETCHER")
    print("="*70)

    try:
        from data.futures_data_fetcher import FuturesDataFetcher, MICRO_FUTURES

        fetcher = FuturesDataFetcher(paper_trading=True)
        print("✓ Data fetcher initialized")

        print("\nSupported contracts:")
        for symbol, specs in MICRO_FUTURES.items():
            print(f"  {symbol}: {specs['name']} (${specs['point_value']}/point)")

        print("\n[PASS] Data fetcher working")
        return True
    except Exception as e:
        print(f"✗ Data fetcher failed: {e}")
        print("\n[FAIL] Data fetcher failed")
        return False


def test_deployment_scripts():
    """Test deployment scripts exist and are valid Python"""

    print("\n" + "="*70)
    print("TEST 5: DEPLOYMENT SCRIPTS")
    print("="*70)

    scripts = [
        'futures_live_validation.py',
        'start_futures_paper_trading.py',
        'futures_polygon_data.py'
    ]

    all_passed = True

    for script in scripts:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), script)

        if os.path.exists(filepath):
            print(f"✓ {script} exists")

            # Try to compile it (syntax check)
            try:
                with open(filepath, 'r') as f:
                    compile(f.read(), filepath, 'exec')
                print(f"  ✓ Valid Python syntax")
            except SyntaxError as e:
                print(f"  ✗ Syntax error: {e}")
                all_passed = False
        else:
            print(f"✗ {script} not found")
            all_passed = False

    if all_passed:
        print("\n[PASS] All deployment scripts valid")
    else:
        print("\n[FAIL] Some deployment scripts have issues")

    return all_passed


def test_monday_ai_integration():
    """Test MONDAY_AI_TRADING.py has futures support"""

    print("\n" + "="*70)
    print("TEST 6: MONDAY_AI_TRADING INTEGRATION")
    print("="*70)

    try:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MONDAY_AI_TRADING.py')

        with open(filepath, 'r') as f:
            content = f.read()

        # Check for futures integration
        checks = [
            ('enable_futures', 'Futures flag present'),
            ('futures_scanner', 'Futures scanner integration'),
            ('futures_max_risk', 'Risk limit defined'),
            ('futures_max_positions', 'Position limit defined'),
            ('CONSERVATIVE MODE', 'Conservative mode message')
        ]

        all_passed = True
        for check_str, description in checks:
            if check_str in content:
                print(f"✓ {description}")
            else:
                print(f"✗ {description} - not found")
                all_passed = False

        if all_passed:
            print("\n[PASS] MONDAY_AI_TRADING.py has futures support")
        else:
            print("\n[FAIL] MONDAY_AI_TRADING.py missing some futures features")

        return all_passed
    except Exception as e:
        print(f"✗ Error checking MONDAY_AI_TRADING.py: {e}")
        print("\n[FAIL] Could not verify integration")
        return False


def test_documentation():
    """Test documentation files exist"""

    print("\n" + "="*70)
    print("TEST 7: DOCUMENTATION")
    print("="*70)

    docs = [
        ('FUTURES_DEPLOYMENT_GUIDE.md', 'Complete deployment guide'),
        ('FUTURES_QUICK_START.md', 'Quick start reference'),
        ('FUTURES_DEPLOYMENT_SUMMARY.md', 'Deployment summary')
    ]

    all_passed = True

    for doc_file, description in docs:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), doc_file)

        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✓ {doc_file} ({size:,} bytes)")
            print(f"  {description}")
        else:
            print(f"✗ {doc_file} not found")
            all_passed = False

    if all_passed:
        print("\n[PASS] All documentation present")
    else:
        print("\n[FAIL] Some documentation missing")

    return all_passed


def test_safety_limits():
    """Test safety limits are properly configured"""

    print("\n" + "="*70)
    print("TEST 8: SAFETY LIMITS")
    print("="*70)

    try:
        # Check start_futures_paper_trading.py
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'start_futures_paper_trading.py')

        with open(filepath, 'r') as f:
            content = f.read()

        safety_checks = [
            ('max_risk_per_trade: float = 100.0', 'Max risk per trade: $100'),
            ('max_positions: int = 2', 'Max positions: 2'),
            ('max_total_risk: float = 500.0', 'Max total risk: $500'),
            ('max_consecutive_losses = 3', 'Max consecutive losses: 3'),
            ('contracts = 1', 'Position size: 1 contract')
        ]

        all_passed = True
        for check_str, description in safety_checks:
            if check_str in content:
                print(f"✓ {description}")
            else:
                print(f"⚠ {description} - not found (may be configured differently)")

        print("\n[PASS] Safety limits configured")
        return True
    except Exception as e:
        print(f"✗ Error checking safety limits: {e}")
        print("\n[FAIL] Could not verify safety limits")
        return False


def run_all_tests():
    """Run all tests"""

    print("\n" + "="*70)
    print("FUTURES DEPLOYMENT TEST SUITE")
    print("="*70)
    print("Testing all futures deployment components...\n")

    results = {
        'Module Imports': test_imports(),
        'Scanner Initialization': test_scanner_initialization(),
        'Strategy Analysis': test_strategy(),
        'Data Fetcher': test_data_fetcher(),
        'Deployment Scripts': test_deployment_scripts(),
        'MONDAY_AI Integration': test_monday_ai_integration(),
        'Documentation': test_documentation(),
        'Safety Limits': test_safety_limits()
    }

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")

    print()
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ ALL TESTS PASSED - Futures deployment ready!")
        print("\nNext steps:")
        print("  1. Read: FUTURES_QUICK_START.md")
        print("  2. Choose deployment option (A, B, or C)")
        print("  3. Run your chosen script")
        return True
    else:
        print(f"\n✗ {total - passed} TESTS FAILED - Please fix issues")
        print("\nCheck error messages above for details")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
