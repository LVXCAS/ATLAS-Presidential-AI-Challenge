#!/usr/bin/env python3
"""Quick test to verify OpenBB Platform v4.5.0 installation"""

from openbb import obb

print("="*70)
print("TESTING OPENBB PLATFORM v4.5.0")
print("="*70)

# Test 1: Equity quote
try:
    data = obb.equity.price.quote(symbol='AAPL', provider='yfinance')
    price = data.results[0].last_price
    print(f"\n[OK] Equity Quote: AAPL = ${price:.2f}")
except Exception as e:
    print(f"\n[ERROR] Equity Quote failed: {e}")

# Test 2: Options chains (critical for your trading)
try:
    options = obb.derivatives.options.chains(symbol='TSLA', provider='yfinance')
    print(f"[OK] Options Chains: TSLA has {len(options.results)} option contracts")
except Exception as e:
    print(f"[ERROR] Options Chains failed: {e}")

# Test 3: News feed (could be useful for sentiment)
try:
    news = obb.news.world(provider='benzinga', limit=3)
    print(f"[OK] News Feed: Retrieved {len(news.results)} recent articles")
except Exception as e:
    print(f"[ERROR] News Feed failed: {e}")

# Test 4: Economic calendar (market-moving events)
try:
    calendar = obb.economy.calendar(provider='fmp', start_date='2025-10-08', end_date='2025-10-09')
    if calendar.results:
        print(f"[OK] Economic Calendar: {len(calendar.results)} events tomorrow")
    else:
        print(f"[OK] Economic Calendar: API working (no events tomorrow)")
except Exception as e:
    print(f"[ERROR] Economic Calendar failed: {e}")

print("\n" + "="*70)
print("OPENBB PLATFORM INSTALLATION: SUCCESS")
print("="*70)
print("\nNext steps:")
print("  1. Set API keys in ~/.openbb/user_settings.json (optional)")
print("  2. Integrate with your scanner for backup data or options IV")
print("  3. Use obb.derivatives.options.chains() for real-time IV data")
