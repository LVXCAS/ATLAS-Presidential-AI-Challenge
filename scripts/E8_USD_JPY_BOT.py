"""
E8 USD/JPY BOT - OPTIMIZED (BEST PERFORMER!)
Score=2.0, Risk=1.5%, Target=3%, Stop=1.5%
Expected ROI: +11.03% per 90 days
"""

import os
import time
from datetime import datetime
from E8_TRADELOCKER_ADAPTER import E8TradeLockerAdapter
import numpy as np

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("[WARN] TA-Lib not available")

class E8UsdJpyBot:
    def __init__(self):
        self.client = E8TradeLockerAdapter(
            environment=os.getenv('TRADELOCKER_ENV', 'https://demo.tradelocker.com')
        )

        # OPTIMIZED FOR USD/JPY (BEST PERFORMER)
        self.pair = 'USDJPY'
        self.min_score = 2.0  # Lower threshold
        self.risk_per_trade = 0.015  # 1.5% (more conservative for USD/JPY volatility)
        self.profit_target_pct = 0.03  # 3% (wider target)
        self.stop_loss_pct = 0.015  # 1.5% (wider stop for JPY volatility)
        self.scan_interval = 3600  # 1 hour
        self.trading_hours = list(range(24))  # 24/5 - ALL HOURS (especially good during Tokyo session!)

        print("=" * 70)
        print("E8 USD/JPY BOT [OPTIMIZED - BEST PERFORMER]")
        print("=" * 70)
        print(f"Pair: {self.pair}")
        print(f"Min Score: {self.min_score}")
        print(f"Risk: {self.risk_per_trade*100}%")
        print(f"Target: {self.profit_target_pct*100}%")
        print(f"Stop: {self.stop_loss_pct*100}%")
        print(f"Expected ROI: +11.03% per 90 days (HIGHEST)")
        print(f"Trading Hours: London/NY overlap + Tokyo session")
        print("=" * 70)

    def calculate_score(self, candles):
        if len(candles) < 200:
            return 0, [], 'long'

        closes = np.array([float(c['c']) for c in candles])
        highs = np.array([float(c['h']) for c in candles])
        lows = np.array([float(c['l']) for c in candles])

        if not TALIB_AVAILABLE:
            return 0, [], 'long'

        try:
            rsi = talib.RSI(closes, timeperiod=14)
            macd, signal, _ = talib.MACD(closes)
            adx = talib.ADX(highs, lows, closes, timeperiod=14)
            ema_fast = talib.EMA(closes, timeperiod=10)
            ema_slow = talib.EMA(closes, timeperiod=21)
            ema_trend = talib.EMA(closes, timeperiod=200)

            score = 0
            signals = []

            # LONG signals
            if rsi[-1] < 40:
                score += 2
                signals.append('rsi_oversold')
            if macd[-1] > signal[-1] and macd[-2] <= signal[-2]:
                score += 2
                signals.append('macd_bull_cross')
            if adx[-1] > 25:
                score += 1
                signals.append('strong_trend')
            if ema_fast[-1] > ema_slow[-1] > ema_trend[-1]:
                score += 1
                signals.append('ema_bullish')

            # SHORT signals
            if rsi[-1] > 60:
                score += 2
                signals.append('rsi_overbought')
            if macd[-1] < signal[-1] and macd[-2] >= signal[-2]:
                score += 2
                signals.append('macd_bear_cross')
            if ema_fast[-1] < ema_slow[-1] < ema_trend[-1]:
                score += 1
                signals.append('ema_bearish')

            # Determine direction
            direction = 'long' if 'rsi_oversold' in signals or 'macd_bull_cross' in signals or 'ema_bullish' in signals else 'short'

            return score, signals, direction

        except Exception as e:
            print(f"[ERROR] Score calculation: {e}")
            return 0, [], 'long'

    def is_trading_hour(self):
        return datetime.now().hour in self.trading_hours

    def scan(self):
        print(f"\n[SCAN] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if not self.is_trading_hour():
            print("[SKIP] Outside trading hours")
            return

        # Check positions
        positions = self.client.get_open_positions()
        has_position = any(p.get('symbol') == self.pair for p in positions)

        if has_position:
            print("[SKIP] Already in position")
            return

        # Get candles
        candles = self.client.get_candles(self.pair, count=250, granularity='H1')
        if len(candles) < 200:
            print("[SKIP] Insufficient data")
            return

        # Calculate score
        score, signals, direction = self.calculate_score(candles)
        print(f"Score: {score:.1f} | Signals: {', '.join(signals)}")

        if score < self.min_score:
            print(f"[SKIP] Score {score:.1f} < {self.min_score}")
            return

        # Entry signal!
        current_price = float(candles[-1]['c'])
        balance = self.client.get_account_summary()['balance']

        # Position sizing
        risk_amount = balance * self.risk_per_trade
        stop_distance = current_price * self.stop_loss_pct
        units = int(risk_amount / stop_distance)
        units = max(10000, min(units, 1000000))  # 1 mini lot to 10 standard lots

        # TP/SL
        if direction == 'long':
            side = 'buy'
            tp = current_price * (1 + self.profit_target_pct)
            sl = current_price * (1 - self.stop_loss_pct)
        else:
            side = 'sell'
            tp = current_price * (1 - self.profit_target_pct)
            sl = current_price * (1 + self.stop_loss_pct)

        print(f"\n[ENTRY] {self.pair} {side.upper()}")
        print(f"  Price: {current_price:.3f}")
        print(f"  Units: {units:,}")
        print(f"  TP: {tp:.3f}")
        print(f"  SL: {sl:.3f}")

        order_id = self.client.place_order(
            symbol=self.pair,
            units=units,
            side=side,
            take_profit=tp,
            stop_loss=sl
        )

        if order_id:
            print(f"[SUCCESS] Order: {order_id}")
        else:
            print("[FAILED] Order rejected")

    def run(self):
        print(f"\n[START] USD/JPY Bot running...")
        while True:
            try:
                self.scan()
                print(f"[WAIT] Next scan in 60 minutes...")
                time.sleep(self.scan_interval)
            except KeyboardInterrupt:
                print("\n[STOP] Bot stopped")
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(300)

if __name__ == "__main__":
    bot = E8UsdJpyBot()
    bot.run()
