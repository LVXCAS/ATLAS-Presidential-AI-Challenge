"""Quick test to see what candle data we're getting"""
from E8_TRADELOCKER_ADAPTER import E8TradeLockerAdapter

print("Connecting to E8...")
adapter = E8TradeLockerAdapter()

print("\n" + "="*70)
print("TESTING CANDLE DATA")
print("="*70)

pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']

for pair in pairs:
    print(f"\n[{pair}] Fetching last 100 1H candles...")
    candles = adapter.get_candles(pair, count=100, granularity='H1')
    print(f"  Received: {len(candles)} candles")

    if len(candles) > 0:
        latest = candles[-1]
        print(f"  Latest: {latest['time']}")
        print(f"  OHLC: O={latest['mid']['o']:.5f} H={latest['mid']['h']:.5f} L={latest['mid']['l']:.5f} C={latest['mid']['c']:.5f}")
    else:
        print(f"  [ERROR] No candle data received!")

print("\n" + "="*70)
print("Test complete.")
