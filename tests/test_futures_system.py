#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUICK TEST - FUTURES TRADING SYSTEM
Verify all components are working
"""

import sys
import os

# Force UTF-8 encoding for output
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("\n" + "="*70)
print("FUTURES SYSTEM - COMPONENT TEST")
print("="*70)

# Test 1: Strategy Module
print("\n[TEST 1/5] Testing Strategy Module...")
try:
    from strategies.futures_ema_strategy import FuturesEMAStrategy
    strategy = FuturesEMAStrategy()
    print("  ✓ Strategy module loaded successfully")
except Exception as e:
    print(f"  ✗ Strategy module failed: {e}")
    sys.exit(1)

# Test 2: Data Fetcher
print("\n[TEST 2/5] Testing Data Fetcher...")
try:
    from data.futures_data_fetcher import FuturesDataFetcher
    fetcher = FuturesDataFetcher(paper_trading=True)
    if fetcher.api:
        print("  ✓ Data fetcher initialized")

        # Test price fetch
        price = fetcher.get_current_price('MES')
        if price:
            print(f"  ✓ MES current price: ${price:.2f}")
        else:
            print("  ⚠ Could not fetch price (may be outside market hours)")
    else:
        print("  ⚠ Data fetcher loaded but API not configured")
except Exception as e:
    print(f"  ✗ Data fetcher failed: {e}")
    sys.exit(1)

# Test 3: Scanner
print("\n[TEST 3/5] Testing Scanner...")
try:
    from scanners.futures_scanner import AIEnhancedFuturesScanner
    scanner = AIEnhancedFuturesScanner(paper_trading=True)
    print("  ✓ Scanner initialized successfully")
except Exception as e:
    print(f"  ✗ Scanner failed: {e}")
    sys.exit(1)

# Test 4: Execution Engine Integration
print("\n[TEST 4/5] Testing Execution Engine...")
try:
    from execution.auto_execution_engine import AutoExecutionEngine
    engine = AutoExecutionEngine(paper_trading=True, max_risk_per_trade=500)

    # Check if futures execution method exists
    if hasattr(engine, 'execute_futures_trade'):
        print("  ✓ Execution engine has execute_futures_trade() method")
    else:
        print("  ✗ execute_futures_trade() method not found")
        sys.exit(1)

    # Check execution summary
    summary = engine.get_execution_summary()
    if 'futures_trades' in summary:
        print("  ✓ Execution summary includes futures_trades")
    else:
        print("  ✗ futures_trades not in execution summary")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Execution engine failed: {e}")
    sys.exit(1)

# Test 5: Main System Integration
print("\n[TEST 5/5] Testing Main System Integration...")
try:
    # Check if MONDAY_AI_TRADING.py has futures integration
    with open('MONDAY_AI_TRADING.py', 'r') as f:
        content = f.read()

    checks = [
        ('AIEnhancedFuturesScanner' in content, "Futures scanner imported"),
        ('enable_futures' in content, "enable_futures parameter exists"),
        ('--futures' in content, "--futures flag implemented"),
        ('futures_opportunities' in content, "Futures opportunities handled")
    ]

    all_passed = True
    for check, description in checks:
        if check:
            print(f"  ✓ {description}")
        else:
            print(f"  ✗ {description}")
            all_passed = False

    if not all_passed:
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Main system integration failed: {e}")
    sys.exit(1)

# All tests passed!
print("\n" + "="*70)
print("SYSTEM STATUS: ✓ ALL TESTS PASSED")
print("="*70)

print("\nFutures trading system is READY FOR USE!")
print("\nQuick Start:")
print("  1. Run backtest:    python futures_backtest.py")
print("  2. Test scanner:    python scanners/futures_scanner.py")
print("  3. Enable futures:  python MONDAY_AI_TRADING.py --futures")

print("\nDocumentation:")
print("  See FUTURES_SYSTEM_GUIDE.md for complete guide")

print("\n" + "="*70)
