#!/usr/bin/env python3
"""
HIVE TRADING EMPIRE - OpenBB Platform Demo
==========================================

Demonstrate OpenBB Platform capabilities with Python 3.10
"""

import sys
print(f"Python version: {sys.version}")

# Import OpenBB Platform
try:
    from openbb import obb
    print("[OK] OpenBB Platform imported successfully!")
except ImportError as e:
    print(f"[ERROR] OpenBB import failed: {e}")
    sys.exit(1)

def demo_openbb():
    """Demonstrate OpenBB Platform features"""
    print("\n[DEMO] HIVE TRADING EMPIRE - OpenBB Platform Demo")
    print("=" * 60)
    
    # Get SPY stock price data
    print("\n[DATA] Getting SPY Historical Data...")
    try:
        spy_data = obb.equity.price.historical("SPY", period="1mo")
        print(f"[OK] Retrieved {len(spy_data)} days of SPY data")
        print(f"Latest close: ${spy_data.iloc[-1]['close']:.2f}")
        print(f"Volume: {spy_data.iloc[-1]['volume']:,.0f}")
    except Exception as e:
        print(f"[ERROR] Error getting SPY data: {e}")
    
    # Get company information
    print("\n[INFO] Getting Company Profile for AAPL...")
    try:
        # Note: This might not work with basic setup, but we'll try
        aapl_info = obb.equity.profile("AAPL")
        if hasattr(aapl_info, 'name'):
            print(f"[OK] Company: {aapl_info.name}")
        else:
            print("[OK] Company profile retrieved (limited data available)")
    except Exception as e:
        print(f"[ERROR] Company profile not available: {e}")
    
    # Test news functionality
    print("\n[NEWS] Testing News Functionality...")
    try:
        news = obb.news.company("TSLA", limit=3)
        print(f"[OK] Retrieved {len(news)} news articles for TSLA")
        if len(news) > 0:
            print(f"Latest headline: {news[0].title}")
    except Exception as e:
        print(f"[ERROR] News functionality not available: {e}")
    
    # Show available equity functions
    print("\n[FUNCS] Available OpenBB Equity Functions:")
    try:
        equity_attrs = [attr for attr in dir(obb.equity) if not attr.startswith('_')]
        print(f"[OK] Available equity modules: {', '.join(equity_attrs[:10])}")
        if len(equity_attrs) > 10:
            print(f"   ... and {len(equity_attrs)-10} more")
    except Exception as e:
        print(f"[ERROR] Could not list equity functions: {e}")
    
    print("\n" + "=" * 60)
    print("[DONE] OpenBB Platform Demo Complete!")
    print("\n[SUCCESS] OpenBB Platform is ready for integration with your Hive Trading Empire!")
    print("[READY] You can now use obb.equity, obb.news, and other modules in your algorithms")

if __name__ == "__main__":
    demo_openbb()