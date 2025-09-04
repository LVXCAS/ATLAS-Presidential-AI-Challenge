#!/usr/bin/env python3
"""
Test the trading loop to see where it's getting stuck
"""
import asyncio
import sys
from datetime import datetime
import traceback

# Add current directory to path
sys.path.append('.')

async def test_trading_functionality():
    """Test each component of the trading system"""
    
    print("=" * 60)
    print("TRADING SYSTEM DIAGNOSTIC TEST")
    print("=" * 60)
    print(f"Time: {datetime.now()}")
    print()
    
    try:
        # Test 1: Import the trading class
        print("[TEST 1] Importing RealMarketDataHunter...")
        from start_real_market_hunter import RealMarketDataHunter
        hunter = RealMarketDataHunter()
        print("✅ SUCCESS: Class imported and initialized")
        print()
        
        # Test 2: Check broker connection
        print("[TEST 2] Testing broker connection...")
        if hasattr(hunter, 'broker') and hunter.broker:
            print("✅ SUCCESS: Broker object exists")
            # Try to get account info
            try:
                account_info = await hunter.get_account_info()
                if account_info:
                    print(f"✅ SUCCESS: Account connected - {account_info.get('account', 'N/A')}")
                    print(f"   Buying Power: ${account_info.get('buying_power', 0):,.2f}")
                else:
                    print("❌ ERROR: No account info returned")
            except Exception as e:
                print(f"❌ ERROR: Account access failed - {e}")
        else:
            print("❌ ERROR: No broker connection")
        print()
        
        # Test 3: Test data retrieval for one stock
        print("[TEST 3] Testing data retrieval...")
        test_symbols = ['SPY', 'AAPL', 'QQQ']
        
        for symbol in test_symbols:
            try:
                print(f"  Testing {symbol}...")
                data = await hunter.get_polygon_data(symbol)
                if data:
                    print(f"    ✅ Polygon data: ${data.get('price', 'N/A'):.2f}")
                else:
                    print(f"    ⚠️  No Polygon data, trying Alpaca...")
                    data = await hunter.get_alpaca_data(symbol)
                    if data:
                        print(f"    ✅ Alpaca data: ${data.get('price', 'N/A'):.2f}")
                    else:
                        print(f"    ⚠️  No Alpaca data, trying Yahoo...")
                        data = await hunter.get_yahoo_data(symbol)
                        if data:
                            print(f"    ✅ Yahoo data: ${data.get('price', 'N/A'):.2f}")
                        else:
                            print(f"    ❌ No data sources working for {symbol}")
                
                # Test signal generation
                if data:
                    print(f"    📊 Testing signal analysis...")
                    signal_result = await hunter.analyze_signals(symbol, data)
                    if signal_result:
                        signal, confidence, reason = signal_result
                        print(f"    🎯 Signal: {signal} (confidence: {confidence:.1%})")
                        print(f"    📝 Reason: {reason}")
                    else:
                        print(f"    📊 No signals generated (normal - waiting for opportunity)")
                        
            except Exception as e:
                print(f"    ❌ ERROR testing {symbol}: {e}")
                print(f"    Stack trace: {traceback.format_exc()}")
        
        print()
        
        # Test 4: Check if main loop would run
        print("[TEST 4] Testing main loop structure...")
        try:
            # Check if the hunt method exists
            if hasattr(hunter, 'hunt'):
                print("✅ SUCCESS: hunt() method exists")
            else:
                print("❌ ERROR: No hunt() method found")
                
            # Check stock list
            if hasattr(hunter, 'stock_sectors'):
                total_stocks = sum(len(stocks) for stocks in hunter.stock_sectors.values())
                print(f"✅ SUCCESS: {total_stocks} stocks configured for scanning")
            else:
                print("❌ ERROR: No stock list configured")
                
        except Exception as e:
            print(f"❌ ERROR in main loop test: {e}")
        
        print()
        print("=" * 60)
        print("DIAGNOSTIC COMPLETE")
        print("=" * 60)
        
        # Try to run one cycle manually
        print("\n[MANUAL TEST] Running one trading cycle...")
        try:
            # This should show us where it gets stuck
            await hunter.hunt()
        except Exception as e:
            print(f"❌ ERROR in hunt() method: {e}")
            print(f"Stack trace: {traceback.format_exc()}")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        print(f"Stack trace: {traceback.format_exc()}")

if __name__ == "__main__":
    print("Starting trading system diagnostic...")
    try:
        asyncio.run(test_trading_functionality())
    except KeyboardInterrupt:
        print("\nDiagnostic interrupted by user")
    except Exception as e:
        print(f"Diagnostic failed: {e}")
    
    input("Press Enter to close...")