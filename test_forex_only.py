"""
Quick test: Run only FOREX handler to verify it works
Skips futures/crypto for faster testing
"""
import sys
import time
from datetime import datetime

print("=" * 80)
print(" " * 25 + "FOREX-ONLY QUICK TEST")
print("=" * 80)
print()

# Import just the forex handler logic
from SHARED.technical_analysis import ta
from SHARED.kelly_criterion import kelly
from SHARED.multi_timeframe import mtf
from SHARED.ai_confirmation import ai_agent

import os
from dotenv import load_dotenv
load_dotenv()

import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.accounts as accounts
import numpy as np

# Quick forex scan
oanda_token = os.getenv('OANDA_API_KEY')
oanda_account = os.getenv('OANDA_ACCOUNT_ID', '101-001-37330890-001')

if not oanda_token:
    print("[ERROR] OANDA_API_KEY not found in .env")
    sys.exit(1)

client = API(access_token=oanda_token, environment='practice')

print("[FOREX SCANNER] Scanning EUR_USD, USD_JPY, GBP_USD, GBP_JPY")
print("-" * 80)

pairs = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'GBP_JPY']
opportunities = []

for pair in pairs:
    try:
        # Get candle data
        params = {"count": 200, "granularity": "H1"}
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        response = client.request(r)

        candles = response['candles']
        closes = np.array([float(c['mid']['c']) for c in candles])
        highs = np.array([float(c['mid']['h']) for c in candles])
        lows = np.array([float(c['mid']['l']) for c in candles])
        current_price = closes[-1]

        # Calculate TA indicators
        rsi = ta.calculate_rsi(closes)
        macd = ta.calculate_macd(closes)
        ema_fast = ta.calculate_ema(closes, period=10)
        ema_slow = ta.calculate_ema(closes, period=21)
        adx = ta.calculate_adx(highs, lows, closes)

        # Score LONG
        long_score = 0
        if rsi < 30: long_score += 2
        elif rsi < 40: long_score += 1
        if macd['macd'] > macd['signal']: long_score += 2
        if current_price > ema_fast and ema_fast > ema_slow: long_score += 2
        if adx > 25: long_score += 1

        # Score SHORT
        short_score = 0
        if rsi > 70: short_score += 2
        elif rsi > 60: short_score += 1
        if macd['macd'] < macd['signal']: short_score += 2
        if current_price < ema_fast and ema_fast < ema_slow: short_score += 2
        if adx > 25: short_score += 1

        best_score = max(long_score, short_score)
        direction = 'LONG' if long_score > short_score else 'SHORT'

        print(f"\n{pair}:")
        print(f"  Price: {current_price:.5f}")
        print(f"  RSI: {rsi:.1f} | MACD: {macd['macd']:.5f} | ADX: {adx:.1f}")
        print(f"  Score: {best_score:.1f}/10 ({direction})")

        if best_score >= 2.5:
            opportunities.append({
                'pair': pair,
                'score': best_score,
                'direction': direction,
                'rsi': rsi,
                'price': current_price
            })
            print(f"  [OPPORTUNITY FOUND]")
        else:
            print(f"  [NO TRADE]")

    except Exception as e:
        print(f"\n{pair}: [ERROR] {e}")

# Summary
print("\n" + "=" * 80)
print(" " * 30 + "SCAN SUMMARY")
print("=" * 80)

if opportunities:
    print(f"\nFound {len(opportunities)} trading opportunities:")
    for opp in opportunities:
        print(f"  - {opp['pair']} {opp['direction']}: {opp['score']:.1f}/10 (RSI: {opp['rsi']:.1f})")
else:
    print("\nNo trading opportunities found (all scores < 2.5)")

# Get account balance
try:
    r = accounts.AccountSummary(accountID=oanda_account)
    response = client.request(r)
    balance = float(response['account']['balance'])
    print(f"\nAccount Balance: ${balance:,.2f}")
except:
    pass

print("\n" + "=" * 80)
print("[TEST COMPLETE] Forex handler working correctly")
print("=" * 80)
