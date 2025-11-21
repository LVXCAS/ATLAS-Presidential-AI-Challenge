"""
GET LIVE SCORES - Calculate current trading signals for all E8 pairs
"""
from E8_TRADELOCKER_ADAPTER import E8TradeLockerAdapter
import os
try:
    import talib
    import numpy as np
except ImportError:
    print("[ERROR] TA-Lib not installed")
    exit(1)

def calculate_score(candles):
    """Calculate trading score from candles (same logic as bot)"""
    if len(candles) < 50:
        return None, "Insufficient data"

    # Extract price data
    closes = np.array([float(c) for c in candles['close']])
    highs = np.array([float(c) for c in candles['high']])
    lows = np.array([float(c) for c in candles['low']])

    # Calculate indicators
    rsi = talib.RSI(closes, timeperiod=14)
    macd, signal, hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
    adx = talib.ADX(highs, lows, closes, timeperiod=14)
    ema10 = talib.EMA(closes, timeperiod=10)
    ema21 = talib.EMA(closes, timeperiod=21)
    ema200 = talib.EMA(closes, timeperiod=200)

    # Get latest values
    current_price = closes[-1]
    current_rsi = rsi[-1]
    current_macd = macd[-1]
    current_signal = signal[-1]
    current_hist = hist[-1]
    current_adx = adx[-1]
    current_ema10 = ema10[-1]
    current_ema21 = ema21[-1]
    current_ema200 = ema200[-1]

    # Calculate score (0-8 scale)
    score = 0
    reasons = []

    # RSI signals (0-2 points)
    if current_rsi < 30:
        score += 2
        reasons.append(f"RSI oversold ({current_rsi:.1f})")
    elif current_rsi < 40:
        score += 1
        reasons.append(f"RSI low ({current_rsi:.1f})")
    elif current_rsi > 70:
        score -= 2
        reasons.append(f"RSI overbought ({current_rsi:.1f})")
    elif current_rsi > 60:
        score -= 1
        reasons.append(f"RSI high ({current_rsi:.1f})")

    # MACD signals (0-2 points)
    if current_macd > current_signal and current_hist > 0:
        score += 2
        reasons.append("MACD bullish cross")
    elif current_macd > current_signal:
        score += 1
        reasons.append("MACD above signal")
    elif current_macd < current_signal and current_hist < 0:
        score -= 2
        reasons.append("MACD bearish cross")
    elif current_macd < current_signal:
        score -= 1
        reasons.append("MACD below signal")

    # ADX trend strength (0-2 points)
    if current_adx > 25:
        if current_ema10 > current_ema21:
            score += 2
            reasons.append(f"Strong uptrend (ADX {current_adx:.1f})")
        else:
            score -= 2
            reasons.append(f"Strong downtrend (ADX {current_adx:.1f})")
    elif current_adx > 20:
        if current_ema10 > current_ema21:
            score += 1
            reasons.append(f"Uptrend (ADX {current_adx:.1f})")
        else:
            score -= 1
            reasons.append(f"Downtrend (ADX {current_adx:.1f})")

    # EMA trend (0-2 points)
    if current_price > current_ema10 > current_ema21 > current_ema200:
        score += 2
        reasons.append("All EMAs aligned bullish")
    elif current_price > current_ema10 > current_ema21:
        score += 1
        reasons.append("Short-term bullish")
    elif current_price < current_ema10 < current_ema21 < current_ema200:
        score -= 2
        reasons.append("All EMAs aligned bearish")
    elif current_price < current_ema10 < current_ema21:
        score -= 1
        reasons.append("Short-term bearish")

    direction = "LONG" if score > 0 else "SHORT" if score < 0 else "NEUTRAL"

    return {
        'score': abs(score),
        'direction': direction,
        'rsi': current_rsi,
        'macd': current_macd,
        'signal': current_signal,
        'adx': current_adx,
        'price': current_price,
        'ema10': current_ema10,
        'ema21': current_ema21,
        'ema200': current_ema200,
        'reasons': reasons
    }

def main():
    # Connect
    client = E8TradeLockerAdapter(environment=os.getenv('TRADELOCKER_ENV', 'https://demo.tradelocker.com'))

    pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']

    print('='*70)
    print('LIVE TRADING SCORES - E8 CHALLENGE')
    print('='*70)
    print(f'Min Score to Trade: 2.5')
    print('='*70)

    for pair in pairs:
        print(f'\n[{pair}]')
        try:
            candles = client.get_candles(pair, count=100, granularity='H1')

            if len(candles) < 50:
                print(f'  Status: Insufficient data ({len(candles)} candles)')
                continue

            result = calculate_score(candles)

            if result:
                print(f'  Score: {result["score"]:.2f} ({result["direction"]})')
                print(f'  Price: {result["price"]:.5f}')
                print(f'  RSI: {result["rsi"]:.1f}')
                print(f'  MACD: {result["macd"]:.5f} (Signal: {result["signal"]:.5f})')
                print(f'  ADX: {result["adx"]:.1f}')
                print(f'  EMA10: {result["ema10"]:.5f}')
                print(f'  EMA21: {result["ema21"]:.5f}')
                print(f'  EMA200: {result["ema200"]:.5f}')

                if result['score'] >= 2.5:
                    print(f'  *** TRADE SIGNAL: {result["direction"]} ***')
                    for reason in result['reasons']:
                        print(f'    - {reason}')
                else:
                    print(f'  Status: Score too low (need 2.5+)')

        except Exception as e:
            print(f'  Error: {e}')

    print('\n' + '='*70)

if __name__ == '__main__':
    main()
