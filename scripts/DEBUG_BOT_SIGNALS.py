"""
DEBUG BOT SIGNALS - See why no trades are being made
Shows current market conditions and scores for all 3 pairs
"""

import os
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from tradelocker import TLAPI

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("[WARNING] TA-Lib not available")

load_dotenv()

# Connect to E8
tl = TLAPI(
    environment=os.getenv('TRADELOCKER_ENV'),
    username=os.getenv('TRADELOCKER_EMAIL'),
    password=os.getenv('TRADELOCKER_PASSWORD'),
    server=os.getenv('TRADELOCKER_SERVER')
)

# Get instruments mapping
instruments = tl.get_all_instruments()

def get_instrument_id(symbol):
    """Get TradeLocker instrument ID from symbol"""
    matches = instruments[instruments['name'].str.contains(symbol, case=False, na=False)]
    if not matches.empty:
        return matches.iloc[0]['tradableInstrumentId']
    return None

def calculate_score(candles):
    """Calculate entry score for a pair"""
    if len(candles) < 200:
        return 0, [], 'none', "Insufficient data"

    closes = np.array([c['c'] for c in candles])
    highs = np.array([c['h'] for c in candles])
    lows = np.array([c['l'] for c in candles])

    if not TALIB_AVAILABLE:
        return 0, [], 'none', "TA-Lib not available"

    try:
        rsi = talib.RSI(closes, timeperiod=14)
        macd, signal, _ = talib.MACD(closes)
        adx = talib.ADX(highs, lows, closes, timeperiod=14)
        ema_fast = talib.EMA(closes, timeperiod=10)
        ema_slow = talib.EMA(closes, timeperiod=21)
        ema_trend = talib.EMA(closes, timeperiod=200)

        score = 0
        signals = []
        reasons = []

        # Current values
        curr_rsi = rsi[-1]
        curr_macd = macd[-1]
        curr_signal = signal[-1]
        curr_adx = adx[-1]

        # LONG signals
        if curr_rsi < 40:
            score += 2
            signals.append('rsi_oversold')
            reasons.append(f"RSI oversold ({curr_rsi:.1f} < 40)")
        elif curr_rsi < 50:
            reasons.append(f"RSI neutral-low ({curr_rsi:.1f})")
        else:
            reasons.append(f"RSI overbought ({curr_rsi:.1f} > 50)")

        if curr_macd > curr_signal and macd[-2] <= signal[-2]:
            score += 2
            signals.append('macd_bull_cross')
            reasons.append(f"MACD bullish crossover")
        elif curr_macd > curr_signal:
            reasons.append(f"MACD above signal (no cross)")
        else:
            reasons.append(f"MACD below signal")

        if curr_adx > 25:
            score += 1
            signals.append('strong_trend')
            reasons.append(f"Strong trend (ADX {curr_adx:.1f} > 25)")
        else:
            reasons.append(f"Weak trend (ADX {curr_adx:.1f} < 25)")

        if ema_fast[-1] > ema_slow[-1] > ema_trend[-1]:
            score += 1
            signals.append('ema_bullish')
            reasons.append(f"EMA alignment bullish")
        elif ema_fast[-1] < ema_slow[-1] < ema_trend[-1]:
            score += 1
            signals.append('ema_bearish')
            reasons.append(f"EMA alignment bearish")
        else:
            reasons.append(f"EMA mixed/neutral")

        # SHORT signals
        if curr_rsi > 60:
            score += 2
            signals.append('rsi_overbought')
            reasons.append(f"RSI overbought ({curr_rsi:.1f} > 60)")

        if curr_macd < curr_signal and macd[-2] >= signal[-2]:
            score += 2
            signals.append('macd_bear_cross')
            reasons.append(f"MACD bearish crossover")

        # Determine direction
        if 'rsi_oversold' in signals or 'macd_bull_cross' in signals or 'ema_bullish' in signals:
            direction = 'LONG'
        elif 'rsi_overbought' in signals or 'macd_bear_cross' in signals or 'ema_bearish' in signals:
            direction = 'SHORT'
        else:
            direction = 'NONE'

        return score, signals, direction, reasons

    except Exception as e:
        return 0, [], 'none', [f"Error: {e}"]

print("=" * 70)
print("BOT SIGNAL DEBUGGER - WHY NO TRADES?")
print("=" * 70)

now = datetime.now()
print(f"\nCurrent Time: {now.strftime('%Y-%m-%d %I:%M %p')} ({now.strftime('%A')})")
print(f"Current Hour: {now.hour}")

# Check trading hours
is_trading = False
if now.weekday() < 5:  # Monday-Friday
    if 8 <= now.hour <= 12:
        is_trading = True
        print("Trading Status: ACTIVE (London/NY overlap 8 AM-12 PM)")
    elif 20 <= now.hour <= 23:
        is_trading = True
        print("Trading Status: ACTIVE (Tokyo session 8 PM-11 PM)")
    else:
        print("Trading Status: OUTSIDE TRADING HOURS")
        print("  Next window: 8 PM tonight or 8 AM tomorrow")
else:
    print("Trading Status: WEEKEND (Market closed)")

if not is_trading:
    print("\n*** REASON: Bots only trade during specific hours ***")
    print("  - Monday-Friday 8 AM - 12 PM EST (London/NY)")
    print("  - Monday-Friday 8 PM - 11 PM EST (Tokyo for JPY)")
    print("\nWait for next trading window to see trades.")
    exit(0)

# Analyze each pair
pairs = [
    {'name': 'EUR/USD', 'symbol': 'EURUSD', 'min_score': 2.5},
    {'name': 'GBP/USD', 'symbol': 'GBPUSD', 'min_score': 2.0},
    {'name': 'USD/JPY', 'symbol': 'USDJPY', 'min_score': 2.0}
]

print("\n" + "=" * 70)
print("ANALYZING CURRENT MARKET CONDITIONS")
print("=" * 70)

for pair in pairs:
    print(f"\n{'='*70}")
    print(f"{pair['name']} ({pair['symbol']})")
    print(f"{'='*70}")
    print(f"Minimum Score Required: {pair['min_score']}")

    # Get instrument ID
    inst_id = get_instrument_id(pair['symbol'])
    if not inst_id:
        print(f"[ERROR] Could not find instrument ID for {pair['symbol']}")
        continue

    print(f"Instrument ID: {inst_id}")

    # Get historical data
    try:
        bars = tl.get_price_history(
            tradable_instrument_id=int(inst_id),
            resolution='60',
            lookback=250
        )

        if bars.empty:
            print("[ERROR] No price data received")
            continue

        print(f"Data Points: {len(bars)} candles")

        # Convert to dict format for score calculation
        candles = []
        for _, row in bars.iterrows():
            candles.append({
                'o': row['o'],
                'h': row['h'],
                'l': row['l'],
                'c': row['c']
            })

        # Calculate score
        score, signals, direction, reasons = calculate_score(candles)

        current_price = candles[-1]['c']

        print(f"\nCurrent Price: {current_price:.5f}")
        print(f"Score: {score:.1f} / {pair['min_score']} needed")
        print(f"Direction: {direction}")
        print(f"Signals: {', '.join(signals) if signals else 'None'}")

        print(f"\nDETAILED BREAKDOWN:")
        for reason in reasons:
            print(f"  - {reason}")

        # Decision
        print(f"\n{'='*70}")
        if score >= pair['min_score']:
            print(f"*** TRADE SIGNAL DETECTED! ***")
            print(f"  Bot SHOULD enter {direction} position")
            print(f"  Check bot logs to see if order was placed")
        else:
            print(f"*** NO TRADE ***")
            print(f"  Score {score:.1f} is below threshold {pair['min_score']}")
            print(f"  Waiting for stronger setup...")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if is_trading:
    print("\nBots ARE active and scanning.")
    print("If no trades above, market conditions don't meet entry criteria.")
    print("\nThis is NORMAL - profitable trading requires patience!")
    print("Bots wait for high-conviction setups (score >= threshold).")
    print("\nYour backtests showed ~3-12 trades per 90 days per pair.")
    print("That's 1 trade every 7-30 days per pair on average.")
else:
    print("\nBots are idle (outside trading hours).")
    print("Wait for next trading window.")

print("\n" + "=" * 70)
