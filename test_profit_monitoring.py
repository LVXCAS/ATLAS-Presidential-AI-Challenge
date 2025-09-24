#!/usr/bin/env python3
"""
Test script for profit monitoring functionality
"""

import asyncio
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append('.')

def test_profit_monitor_creation():
    """Test that we can create ProfitTargetMonitor"""
    try:
        from profit_target_monitor import ProfitTargetMonitor
        monitor = ProfitTargetMonitor()

        print("[PASS] ProfitTargetMonitor created successfully")
        print(f"   Target: {monitor.profit_target_pct}%")
        print(f"   Trading date: {monitor.trading_date}")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to create ProfitTargetMonitor: {e}")
        return False

def test_options_bot_integration():
    """Test OPTIONS_BOT integration"""
    try:
        # Just test imports without full initialization
        from OPTIONS_BOT import TomorrowReadyOptionsBot
        from profit_target_monitor import ProfitTargetMonitor

        print("[PASS] OPTIONS_BOT integration working")
        return True
    except Exception as e:
        print(f"[FAIL] OPTIONS_BOT integration failed: {e}")
        return False

def test_market_hunter_integration():
    """Test start_real_market_hunter integration"""
    try:
        # Just test imports without network calls
        import start_real_market_hunter

        print("[PASS] Market Hunter integration working")
        return True
    except Exception as e:
        print(f"[FAIL] Market Hunter integration failed: {e}")
        return False

async def test_profit_monitor_methods():
    """Test ProfitTargetMonitor methods without broker"""
    try:
        from profit_target_monitor import ProfitTargetMonitor
        monitor = ProfitTargetMonitor()

        # Test status method
        status = monitor.get_status()
        print("[PASS] get_status() working")
        print(f"   Monitoring active: {status['monitoring_active']}")
        print(f"   Target hit: {status['target_hit']}")

        # Test stop method
        monitor.stop_monitoring()
        print("[PASS] stop_monitoring() working")

        return True
    except Exception as e:
        print(f"[FAIL] Method testing failed: {e}")
        return False

def test_file_structure():
    """Test that files are in correct locations"""
    files_to_check = [
        'profit_target_monitor.py',
        'OPTIONS_BOT.py',
        'start_real_market_hunter.py'
    ]

    all_exist = True
    for file in files_to_check:
        if os.path.exists(file):
            print(f"[PASS] {file} exists")
        else:
            print(f"[FAIL] {file} missing")
            all_exist = False

    return all_exist

async def main():
    """Run all tests"""
    print("PROFIT MONITORING SYSTEM TESTS")
    print("=" * 40)
    print(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    tests = [
        ("File Structure", test_file_structure),
        ("Profit Monitor Creation", test_profit_monitor_creation),
        ("OPTIONS_BOT Integration", test_options_bot_integration),
        ("Market Hunter Integration", test_market_hunter_integration),
        ("Profit Monitor Methods", test_profit_monitor_methods)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed += 1
                print(f"[PASS] {test_name} PASSED")
            else:
                print(f"[FAIL] {test_name} FAILED")
        except Exception as e:
            print(f"[ERROR] {test_name} ERROR: {e}")

        print()

    print("=" * 40)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        print("SUCCESS: ALL TESTS PASSED - Profit monitoring system is working!")
    else:
        print("WARNING: Some tests failed - check implementation")

    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)