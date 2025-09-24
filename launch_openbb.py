#!/usr/bin/env python3
"""
HIVE TRADING EMPIRE - OpenBB Platform Launcher
==============================================

Launch OpenBB Platform with Hive Trading integrations
"""

import sys
import os

def launch_openbb():
    """Launch OpenBB Platform"""
    print("[HIVE] HIVE TRADING EMPIRE - OpenBB Platform")
    print("=" * 50)
    print("[INFO] Starting OpenBB Platform with Python 3.10...")
    print(f"[VERSION] Python version: {sys.version}")
    
    try:
        from openbb import obb
        print("[OK] OpenBB Platform loaded successfully!")
        
        # Get some basic data to test
        print("\n[TEST] Testing OpenBB equity data...")
        spy_data = obb.equity.price.historical("SPY", period="1mo")
        
        # Convert to DataFrame for easier handling
        if hasattr(spy_data, 'to_df'):
            df = spy_data.to_df()
            print(f"[OK] Retrieved {len(df)} days of SPY data")
            print(f"[PRICE] Latest SPY close: ${df.iloc[-1]['close']:.2f}")
        elif hasattr(spy_data, 'results'):
            print(f"[OK] Retrieved SPY data (results object)")
            print(f"[DATA] Data type: {type(spy_data.results)}")
        else:
            print(f"[OK] Retrieved SPY data object: {type(spy_data)}")
        
        print("\n[MODULES] Available OpenBB modules:")
        modules = [attr for attr in dir(obb) if not attr.startswith('_')]
        print(f"[LIST] {', '.join(modules)}")
        
        print("\n[EQUITY] Available equity functions:")
        equity_funcs = [attr for attr in dir(obb.equity) if not attr.startswith('_')]
        print(f"[LIST] {', '.join(equity_funcs)}")
        
        print("\n" + "=" * 50)
        print("[SUCCESS] OpenBB Platform is ready!")
        print("[READY] You can now use:")
        print("  - obb.equity.price.historical(symbol)")
        print("  - obb.equity.profile(symbol)")  
        print("  - obb.news.company(symbol)")
        print("  - And many more OpenBB functions!")
        print("")
        print("[INTEGRATION] Ready for Hive Trading Empire integration")
        
        return obb
        
    except Exception as e:
        print(f"[ERROR] Failed to launch OpenBB: {e}")
        return None

def interactive_demo(obb):
    """Simple interactive demo"""
    print("\n[DEMO] Interactive OpenBB Demo")
    print("Enter a stock symbol (or 'quit' to exit):")
    
    while True:
        try:
            symbol = input("> ").strip().upper()
            
            if symbol in ['QUIT', 'EXIT', 'Q']:
                break
                
            if not symbol:
                continue
                
            print(f"[FETCHING] Getting data for {symbol}...")
            
            # Get price data
            try:
                price_data = obb.equity.price.historical(symbol, period="5d")
                if hasattr(price_data, 'to_df'):
                    df = price_data.to_df()
                    latest = df.iloc[-1]
                    print(f"[PRICE] {symbol}: ${latest['close']:.2f}")
                    print(f"[VOLUME] Volume: {latest['volume']:,.0f}")
                else:
                    print(f"[DATA] Retrieved price data for {symbol}")
            except Exception as e:
                print(f"[ERROR] Could not get price for {symbol}: {e}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[ERROR] {e}")
    
    print("[BYE] Thanks for using OpenBB Platform!")

if __name__ == "__main__":
    obb = launch_openbb()
    
    if obb:
        print("\n[OPTION] Want to try interactive demo? (y/n)")
        if input("> ").lower().startswith('y'):
            interactive_demo(obb)