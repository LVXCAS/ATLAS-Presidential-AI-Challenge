"""
Quick test of hybrid adapter
"""

from HYBRID_OANDA_TRADELOCKER import HybridAdapter

print("[TEST] Creating hybrid adapter...")
adapter = HybridAdapter()

print("[TEST] Fetching EUR_USD candles from OANDA...")
candles = adapter.get_candles('EUR_USD', count=10, granularity='H1')
print(f"[TEST] Received {len(candles)} candles")

if candles:
    print(f"[TEST] Latest candle: {candles[-1]}")

print("[TEST] Getting TradeLocker account...")
account = adapter.get_account_summary()
print(f"[TEST] Account balance: ${account['balance']:,.2f}")

print("[TEST] ✅ HYBRID ADAPTER WORKS!")
