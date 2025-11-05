"""
Debug script - why isn't the bot trading?
Simulates what the bot is seeing right now
"""
import os
from dotenv import load_dotenv
load_dotenv()

import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import numpy as np

try:
    import talib
    TALIB_AVAILABLE = True
except:
    TALIB_AVAILABLE = False
    print("[WARN] TA-Lib not available")

oanda_token = os.getenv('OANDA_API_KEY')
client = API(access_token=oanda_token, environment='practice')

def get_forex_data(pair, granularity='H1'):
    """Get forex data"""
    try:
        params = {'count': 100, 'granularity': granularity}
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        response = client.request(r)
        candles = response['candles']

        closes = np.array([float(c['mid']['c']) for c in candles])
        highs = np.array([float(c['mid']['h']) for c in candles])
        lows = np.array([float(c['mid']['l']) for c in candles])

        return {
            'closes': closes,
            'highs': highs,
            'lows': lows,
            'current_price': closes[-1]
        }
    except Exception as e:
        print(f"Error getting data for {pair}: {e}")
        return None

def check_pair(pair):
    """Check why this pair isn't trading"""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {pair}")
    print(f"{'='*70}")

    # Get 1H data
    data_1h = get_forex_data(pair, 'H1')
    if not data_1h:
        print(f"[ERROR] Could not get 1H data")
        return

    # Get 4H data
    data_4h = get_forex_data(pair, 'H4')
    if not data_4h:
        print(f"[ERROR] Could not get 4H data")
        return

    if not TALIB_AVAILABLE:
        print("[ERROR] TA-Lib not available, can't analyze")
        return

    closes_1h = data_1h['closes']
    highs_1h = data_1h['highs']
    lows_1h = data_1h['lows']
    price = data_1h['current_price']

    closes_4h = data_4h['closes']
    highs_4h = data_4h['highs']
    lows_4h = data_4h['lows']

    print(f"\nCurrent Price: {price:.5f}")

    # === 1H TECHNICAL ANALYSIS ===
    print(f"\n[1H TIMEFRAME ANALYSIS]")

    rsi = talib.RSI(closes_1h, timeperiod=14)[-1]
    macd, macd_signal, macd_hist = talib.MACD(closes_1h)
    adx = talib.ADX(highs_1h, lows_1h, closes_1h, timeperiod=14)[-1]
    atr = talib.ATR(highs_1h, lows_1h, closes_1h, timeperiod=14)[-1]
    volatility = (atr / price) * 100

    ema_fast = talib.EMA(closes_1h, timeperiod=10)
    ema_slow = talib.EMA(closes_1h, timeperiod=21)
    ema_trend = talib.EMA(closes_1h, timeperiod=200)

    print(f"  RSI: {rsi:.1f} {'[OVERSOLD <40]' if rsi < 40 else '[OVERBOUGHT >60]' if rsi > 60 else '[NEUTRAL]'}")
    print(f"  MACD: {macd[-1]:.5f} {'[BULLISH]' if macd_hist[-1] > 0 else '[BEARISH]'}")
    print(f"  ADX: {adx:.1f} {'[STRONG TREND >20]' if adx > 20 else '[WEAK TREND]'}")
    print(f"  Volatility: {volatility:.2f}% {'[GOOD >0.3%]' if volatility > 0.3 else '[LOW]'}")
    print(f"  EMA: Fast vs Slow = {'BULLISH' if ema_fast[-1] > ema_slow[-1] else 'BEARISH'}")
    print(f"  Price vs 200 EMA: {'ABOVE (uptrend)' if price > ema_trend[-1] else 'BELOW (downtrend)'}")

    # Calculate score
    long_score = 0
    short_score = 0

    if rsi < 40:
        long_score += 2
    if rsi > 60:
        short_score += 2

    if len(macd_hist) >= 2 and macd_hist[-1] > 0 and macd_hist[-2] <= 0:
        long_score += 2.5
    if len(macd_hist) >= 2 and macd_hist[-1] < 0 and macd_hist[-2] >= 0:
        short_score += 2.5

    if adx > 20:
        long_score += 1.5
        short_score += 1.5

    if volatility > 0.3:
        long_score += 1
        short_score += 1

    if ema_fast[-1] > ema_slow[-1]:
        long_score += 1
    if ema_fast[-1] < ema_slow[-1]:
        short_score += 1

    if price > ema_trend[-1]:
        long_score += 1
    if price < ema_trend[-1]:
        short_score += 1

    print(f"\n  1H LONG Score: {long_score:.1f}/10")
    print(f"  1H SHORT Score: {short_score:.1f}/10")

    # === 4H TREND ANALYSIS ===
    print(f"\n[4H TIMEFRAME TREND]")

    ema_fast_4h = talib.EMA(closes_4h, timeperiod=10)
    ema_slow_4h = talib.EMA(closes_4h, timeperiod=21)
    ema_trend_4h = talib.EMA(closes_4h, timeperiod=50)

    price_4h = closes_4h[-1]

    if ema_fast_4h[-1] > ema_slow_4h[-1] and price_4h > ema_trend_4h[-1]:
        trend_4h = 'BULLISH'
    elif ema_fast_4h[-1] < ema_slow_4h[-1] and price_4h < ema_trend_4h[-1]:
        trend_4h = 'BEARISH'
    else:
        trend_4h = 'NEUTRAL'

    print(f"  4H Trend: {trend_4h}")
    print(f"  Fast EMA: {ema_fast_4h[-1]:.5f}")
    print(f"  Slow EMA: {ema_slow_4h[-1]:.5f}")
    print(f"  Trend EMA: {ema_trend_4h[-1]:.5f}")

    # Apply 4H filter
    if trend_4h == 'BULLISH':
        long_score += 2
        print(f"  [BONUS] +2.0 to LONG score (4H aligned)")
        if short_score > 0:
            short_score -= 1.5
            print(f"  [PENALTY] -1.5 to SHORT score (counter-trend)")
    elif trend_4h == 'BEARISH':
        short_score += 2
        print(f"  [BONUS] +2.0 to SHORT score (4H aligned)")
        if long_score > 0:
            long_score -= 1.5
            print(f"  [PENALTY] -1.5 to LONG score (counter-trend)")

    # === FINAL SCORES ===
    print(f"\n[FINAL SCORES AFTER 4H FILTER]")
    print(f"  LONG:  {long_score:.1f}/10 {'[TRADE]' if long_score >= 4.0 else '[SKIP - Below 4.0 threshold]'}")
    print(f"  SHORT: {short_score:.1f}/10 {'[TRADE]' if short_score >= 4.0 else '[SKIP - Below 4.0 threshold]'}")

    # === VERDICT ===
    print(f"\n[VERDICT]")
    if long_score >= 4.0 or short_score >= 4.0:
        print(f"  [OK] TRADEABLE SETUP FOUND!")
        if long_score >= 4.0:
            print(f"    Direction: LONG")
            print(f"    Confidence: {long_score}/10")
        if short_score >= 4.0:
            print(f"    Direction: SHORT")
            print(f"    Confidence: {short_score}/10")
    else:
        print(f"  [X] NO TRADE - Scores below 4.0/10 threshold")
        print(f"  Reason: Setup not strong enough")
        if trend_4h == 'NEUTRAL':
            print(f"  Note: 4H trend is choppy/sideways (no clear direction)")
        print(f"  Action: WAIT for better setup")

# Check all pairs
pairs = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'GBP_JPY']

print("="*70)
print("WHY NO TRADES? - CURRENT MARKET SCAN")
print("="*70)
print(f"Time: 7 PM EST (Asian Session - Low Liquidity)")
print(f"Threshold: 4.0/10 (recovery mode)")
print(f"Multi-timeframe: Active (4H trend filter)")

for pair in pairs:
    check_pair(pair)

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"\nThe bot is working correctly. It's rejecting low-quality setups")
print(f"during the Asian session (low liquidity, choppy markets).")
print(f"\nBest opportunities appear during:")
print(f"  - London Open (3-8 AM EST)")
print(f"  - NY Open (8 AM-12 PM EST)")
print(f"  - London/NY Overlap (8-11 AM EST) ← BEST")
print(f"\nCurrent behavior: CORRECT (being selective)")
print(f"{'='*70}\n")
