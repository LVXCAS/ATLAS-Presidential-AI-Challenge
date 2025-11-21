"""
FIXED FOREX STRATEGY
Corrects the 3 fatal flaws causing 16.7% win rate:
1. Trade WITH RSI momentum, not against it
2. Enter on MACD momentum, not lagging crossovers
3. Only trade WITH the 4H trend, never counter-trend
"""

import talib
import numpy as np

def calculate_entry_signal_FIXED(closes, highs, lows, current_price):
    """
    CORRECTED entry logic that should achieve 50%+ win rate

    Key Changes:
    - RSI: Buy when RSI > 50 AND rising (momentum), not oversold
    - MACD: Enter when histogram expanding, not on crossover
    - Trend: ONLY trade with 4H EMA direction, reject counter-trend
    """

    if len(closes) < 200:
        return {'direction': None, 'score': 0, 'reason': 'Insufficient data'}

    # Calculate indicators
    rsi = talib.RSI(closes, timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
    adx = talib.ADX(highs, lows, closes, timeperiod=14)

    # EMAs for trend
    ema_20 = talib.EMA(closes, timeperiod=20)   # Short-term
    ema_50 = talib.EMA(closes, timeperiod=50)   # Medium-term
    ema_200 = talib.EMA(closes, timeperiod=200) # Long-term trend

    # Current values
    rsi_now = rsi[-1]
    rsi_prev = rsi[-2]

    macd_hist_now = macd_hist[-1]
    macd_hist_prev = macd_hist[-2]
    macd_hist_2bars = macd_hist[-3]

    adx_now = adx[-1]

    price_now = current_price
    ema_20_now = ema_20[-1]
    ema_50_now = ema_50[-1]
    ema_200_now = ema_200[-1]

    # ============================================================================
    # STEP 1: DETERMINE 4H TREND (NEVER TRADE AGAINST THIS)
    # ============================================================================

    # Simplified: Use EMA alignment on 1H data as proxy
    # In production, fetch actual 4H candles
    if ema_20_now > ema_50_now > ema_200_now:
        trend_4h = 'BULLISH'
    elif ema_20_now < ema_50_now < ema_200_now:
        trend_4h = 'BEARISH'
    else:
        trend_4h = 'NEUTRAL'

    # ============================================================================
    # STEP 2: CHECK TREND STRENGTH (Only trade strong trends)
    # ============================================================================

    if adx_now < 20:
        return {
            'direction': None,
            'score': 0,
            'reason': f'Weak trend (ADX {adx_now:.1f} < 20)'
        }

    # ============================================================================
    # STEP 3: LONG SIGNAL (CORRECTED LOGIC)
    # ============================================================================

    long_score = 0
    long_signals = []

    if trend_4h == 'BULLISH':  # MUST be in uptrend

        # RSI: Buy when RSI > 50 AND rising (WITH momentum)
        if rsi_now > 50 and rsi_now > rsi_prev:
            long_score += 3
            long_signals.append(f'RSI_MOMENTUM_UP ({rsi_now:.1f})')

        # RSI: Extra points if bouncing from 40-50 zone (pullback in uptrend)
        elif 40 <= rsi_now <= 50 and rsi_now > rsi_prev:
            long_score += 2
            long_signals.append(f'RSI_PULLBACK_BOUNCE ({rsi_now:.1f})')

        # MACD: Histogram expanding (momentum building)
        if macd_hist_now > 0 and macd_hist_now > macd_hist_prev > macd_hist_2bars:
            long_score += 3
            long_signals.append('MACD_MOMENTUM_BUILDING')

        # MACD: Fresh bullish cross (bonus, but not required)
        if macd_hist_now > 0 and macd_hist_prev <= 0:
            long_score += 2
            long_signals.append('MACD_FRESH_CROSS')

        # Price: Above key EMAs (confirmation)
        if price_now > ema_20_now > ema_50_now:
            long_score += 2
            long_signals.append('PRICE_ABOVE_EMAS')

        # Trend strength bonus
        if adx_now > 25:
            long_score += 2
            long_signals.append(f'STRONG_TREND (ADX {adx_now:.1f})')

    # ============================================================================
    # STEP 4: SHORT SIGNAL (CORRECTED LOGIC)
    # ============================================================================

    short_score = 0
    short_signals = []

    if trend_4h == 'BEARISH':  # MUST be in downtrend

        # RSI: Sell when RSI < 50 AND falling (WITH momentum)
        if rsi_now < 50 and rsi_now < rsi_prev:
            short_score += 3
            short_signals.append(f'RSI_MOMENTUM_DOWN ({rsi_now:.1f})')

        # RSI: Extra points if rejecting from 50-60 zone (pullback in downtrend)
        elif 50 <= rsi_now <= 60 and rsi_now < rsi_prev:
            short_score += 2
            short_signals.append(f'RSI_PULLBACK_REJECT ({rsi_now:.1f})')

        # MACD: Histogram expanding negative (momentum building)
        if macd_hist_now < 0 and macd_hist_now < macd_hist_prev < macd_hist_2bars:
            short_score += 3
            short_signals.append('MACD_MOMENTUM_BUILDING')

        # MACD: Fresh bearish cross (bonus, but not required)
        if macd_hist_now < 0 and macd_hist_prev >= 0:
            short_score += 2
            short_signals.append('MACD_FRESH_CROSS')

        # Price: Below key EMAs (confirmation)
        if price_now < ema_20_now < ema_50_now:
            short_score += 2
            short_signals.append('PRICE_BELOW_EMAS')

        # Trend strength bonus
        if adx_now > 25:
            short_score += 2
            short_signals.append(f'STRONG_TREND (ADX {adx_now:.1f})')

    # ============================================================================
    # STEP 5: RETURN BEST SIGNAL
    # ============================================================================

    # Minimum score threshold: 6.0 (need strong confluence)
    MIN_SCORE = 6.0

    if long_score >= MIN_SCORE and long_score > short_score:
        return {
            'direction': 'LONG',
            'score': long_score,
            'signals': long_signals,
            'reason': f'{trend_4h} trend + {len(long_signals)} confirmations',
            'details': {
                'rsi': rsi_now,
                'macd_hist': macd_hist_now,
                'adx': adx_now,
                'trend': trend_4h
            }
        }

    elif short_score >= MIN_SCORE and short_score > long_score:
        return {
            'direction': 'SHORT',
            'score': short_score,
            'signals': short_signals,
            'reason': f'{trend_4h} trend + {len(short_signals)} confirmations',
            'details': {
                'rsi': rsi_now,
                'macd_hist': macd_hist_now,
                'adx': adx_now,
                'trend': trend_4h
            }
        }

    else:
        return {
            'direction': None,
            'score': max(long_score, short_score),
            'reason': f'Score {max(long_score, short_score):.1f} < {MIN_SCORE} threshold',
            'details': {
                'long_score': long_score,
                'short_score': short_score,
                'trend': trend_4h
            }
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    # Test with sample data
    closes = np.array([1.1000 + i*0.0001 for i in range(250)])  # Uptrend
    highs = closes + 0.0005
    lows = closes - 0.0005
    current_price = closes[-1]

    signal = calculate_entry_signal_FIXED(closes, highs, lows, current_price)

    print('=' * 70)
    print('FIXED FOREX STRATEGY - TEST')
    print('=' * 70)
    print(f"\nDirection: {signal.get('direction')}")
    print(f"Score: {signal.get('score'):.1f}")
    print(f"Reason: {signal.get('reason')}")

    if signal.get('signals'):
        print(f"\nSignals ({len(signal['signals'])}):")
        for s in signal['signals']:
            print(f"  - {s}")

    if signal.get('details'):
        print(f"\nDetails:")
        for k, v in signal['details'].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")

    print('=' * 70)
