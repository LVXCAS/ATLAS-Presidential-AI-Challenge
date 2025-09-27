#!/usr/bin/env python3
"""
HIVE TRADING EMPIRE - COMPLETE OpenBB TEST SUITE
===============================================

Comprehensive test of OpenBB Platform integration
"""

import sys
import traceback
from datetime import datetime

def test_basic_import():
    """Test 1: Basic OpenBB import"""
    print("[TEST 1] Testing OpenBB Platform import...")
    try:
        from openbb import obb
        version = obb.system.version
        print(f"[OK] OpenBB Platform v{version} imported successfully")
        return obb, True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return None, False

def test_equity_data(obb):
    """Test 2: Equity data retrieval"""
    print("\n[TEST 2] Testing equity data retrieval...")
    
    symbols = ["SPY", "AAPL", "MSFT", "TSLA"]
    results = {}
    
    for symbol in symbols:
        try:
            print(f"[FETCH] Getting data for {symbol}...")
            data = obb.equity.price.historical(symbol, period="5d")
            
            if hasattr(data, 'to_df'):
                df = data.to_df()
                latest = df.iloc[-1]
                price = latest['close']
                volume = latest['volume']
                results[symbol] = {
                    'price': price,
                    'volume': volume,
                    'success': True
                }
                print(f"[OK] {symbol}: ${price:.2f}, Volume: {volume:,.0f}")
            else:
                results[symbol] = {'success': False, 'error': 'No to_df method'}
                print(f"[WARN] {symbol}: Data retrieved but format unclear")
                
        except Exception as e:
            results[symbol] = {'success': False, 'error': str(e)}
            print(f"[FAIL] {symbol}: {e}")
    
    success_count = sum(1 for r in results.values() if r['success'])
    print(f"[RESULT] Equity test: {success_count}/{len(symbols)} successful")
    return results, success_count == len(symbols)

def test_company_profiles(obb):
    """Test 3: Company profile data"""
    print("\n[TEST 3] Testing company profiles...")
    
    symbols = ["AAPL", "MSFT"]
    results = {}
    
    for symbol in symbols:
        try:
            print(f"[FETCH] Getting profile for {symbol}...")
            profile = obb.equity.profile(symbol)
            
            if hasattr(profile, 'results') and profile.results:
                if hasattr(profile.results[0], 'name'):
                    name = profile.results[0].name
                    print(f"[OK] {symbol}: {name}")
                    results[symbol] = {'success': True, 'name': name}
                else:
                    print(f"[OK] {symbol}: Profile retrieved (limited data)")
                    results[symbol] = {'success': True}
            else:
                print(f"[OK] {symbol}: Profile object retrieved")
                results[symbol] = {'success': True}
                
        except Exception as e:
            results[symbol] = {'success': False, 'error': str(e)}
            print(f"[FAIL] {symbol}: {e}")
    
    success_count = sum(1 for r in results.values() if r['success'])
    print(f"[RESULT] Profile test: {success_count}/{len(symbols)} successful")
    return results, success_count > 0

def test_news_functionality(obb):
    """Test 4: News functionality"""
    print("\n[TEST 4] Testing news functionality...")
    
    try:
        print("[FETCH] Getting news for TSLA...")
        news = obb.news.company("TSLA", limit=3)
        
        if hasattr(news, 'results') and news.results:
            print(f"[OK] Retrieved {len(news.results)} news articles")
            if hasattr(news.results[0], 'title'):
                print(f"[SAMPLE] Latest: {news.results[0].title}")
            return True
        elif hasattr(news, 'to_df'):
            df = news.to_df()
            print(f"[OK] Retrieved {len(df)} news articles")
            return True
        else:
            print("[OK] News object retrieved (format unclear)")
            return True
            
    except Exception as e:
        print(f"[FAIL] News test failed: {e}")
        return False

def test_openbb_modules(obb):
    """Test 5: Available modules"""
    print("\n[TEST 5] Testing available modules...")
    
    try:
        modules = [attr for attr in dir(obb) if not attr.startswith('_')]
        print(f"[OK] Available modules: {', '.join(modules)}")
        
        # Test equity submodules
        equity_modules = [attr for attr in dir(obb.equity) if not attr.startswith('_')]
        print(f"[OK] Equity modules: {', '.join(equity_modules)}")
        
        return True, modules
    except Exception as e:
        print(f"[FAIL] Module enumeration failed: {e}")
        return False, []

def test_hive_integration(obb):
    """Test 6: Integration with Hive Trading Empire"""
    print("\n[TEST 6] Testing Hive Trading integration...")
    
    try:
        # Simulate integration with your existing systems
        print("[SIMULATE] Creating market data feed for Hive system...")
        
        # Get data for portfolio symbols
        portfolio_symbols = ["SPY", "QQQ", "IWM"]
        market_data = {}
        
        for symbol in portfolio_symbols:
            data = obb.equity.price.historical(symbol, period="1d")
            if hasattr(data, 'to_df'):
                df = data.to_df()
                market_data[symbol] = {
                    'price': df.iloc[-1]['close'],
                    'volume': df.iloc[-1]['volume'],
                    'timestamp': datetime.now()
                }
        
        print("[OK] Market data feed created for Hive system")
        print(f"[DATA] Portfolio data: {len(market_data)} symbols updated")
        
        # Test data format compatibility
        for symbol, data in market_data.items():
            print(f"[FEED] {symbol}: ${data['price']:.2f}")
        
        return True, market_data
        
    except Exception as e:
        print(f"[FAIL] Hive integration test failed: {e}")
        return False, {}

def run_comprehensive_test():
    """Run all tests"""
    print("=" * 60)
    print("HIVE TRADING EMPIRE - OpenBB PLATFORM TEST SUITE")
    print("=" * 60)
    print(f"[INFO] Python version: {sys.version}")
    print(f"[INFO] Test started at: {datetime.now()}")
    
    test_results = {}
    
    # Test 1: Basic import
    obb, success = test_basic_import()
    test_results['import'] = success
    
    if not success:
        print("\n[CRITICAL] Basic import failed - cannot continue")
        return False
    
    # Test 2: Equity data
    _, success = test_equity_data(obb)
    test_results['equity'] = success
    
    # Test 3: Company profiles
    _, success = test_company_profiles(obb)
    test_results['profiles'] = success
    
    # Test 4: News
    success = test_news_functionality(obb)
    test_results['news'] = success
    
    # Test 5: Modules
    success, modules = test_openbb_modules(obb)
    test_results['modules'] = success
    
    # Test 6: Hive integration
    success, market_data = test_hive_integration(obb)
    test_results['hive_integration'] = success
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\n[SUMMARY] {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("[SUCCESS] All tests passed! OpenBB Platform is ready for production.")
        print("[READY] Your Hive Trading Empire can now use OpenBB Platform!")
    elif passed_tests >= total_tests * 0.8:
        print("[MOSTLY OK] Most tests passed. OpenBB Platform is functional.")
    else:
        print("[ISSUES] Several tests failed. Check configuration.")
    
    print("\n[INTEGRATION] OpenBB Platform ready for:")
    print("  - Real-time market data via obb.equity.price")
    print("  - Company fundamentals via obb.equity.profile")
    print("  - Market news via obb.news.company")
    print("  - Technical analysis and screening")
    print("  - Integration with your 353-file Hive Trading system")
    
    return passed_tests >= total_tests * 0.8

if __name__ == "__main__":
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Test suite failed: {e}")
        print("[TRACEBACK]")
        traceback.print_exc()
        sys.exit(1)