"""
FOREX TRADING BOT - E8 MARKETS
Separated architecture using SHARED/ libraries
Target: E8 One $500K account (+$30K profit to pass)
"""
import os
import sys
import time
import threading
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared libraries
from SHARED.technical_analysis import ta
from SHARED.kelly_criterion import kelly
from SHARED.multi_timeframe import mtf

# OANDA API
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np


class ForexTraderE8:
    """
    Forex trading bot for E8 Markets prop firm
    Uses shared technical analysis, Kelly sizing, and multi-timeframe confirmation
    """

    def __init__(self):
        print("=" * 70)
        print("FOREX TRADER - E8 MARKETS (SEPARATED ARCHITECTURE)")
        print("=" * 70)

        # OANDA API
        self.oanda_token = os.getenv('OANDA_API_KEY')
        self.oanda_account_id = os.getenv('OANDA_ACCOUNT_ID', '101-001-37330890-001')

        if not self.oanda_token:
            raise ValueError("Missing OANDA_API_KEY in .env")

        self.client = API(access_token=self.oanda_token, environment='practice')

        # Trading pairs
        self.forex_pairs = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'GBP_JPY']

        # Risk parameters (E8-compatible)
        self.min_score = 2.5  # Quality filter
        self.risk_per_trade = 0.01  # 1% risk per trade
        self.max_positions = 3
        self.leverage = 5  # 5x leverage (conservative)

        # Time-based entry filters (avoid choppy hours)
        self.AVOID_HOURS = {
            'EUR_USD': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            'GBP_USD': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            'USD_JPY': [0, 1, 2, 3, 4, 16, 17, 18],
            'GBP_JPY': [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17, 18],
        }

        # Scanning interval
        self.scan_interval = 3600  # 1 hour

        print(f"Pairs: {', '.join(self.forex_pairs)}")
        print(f"Leverage: {self.leverage}x")
        print(f"Min Score: {self.min_score}/10")
        print(f"Risk Per Trade: {self.risk_per_trade * 100}%")
        print(f"Max Positions: {self.max_positions}")
        print(f"Shared Libraries: TA-Lib, Kelly, MTF")
        print("=" * 70)

    def get_forex_data(self, pair, granularity='H1', count=200):
        """Fetch historical forex data from OANDA"""
        try:
            params = {"count": count, "granularity": granularity}
            r = instruments.InstrumentsCandles(instrument=pair, params=params)
            response = self.client.request(r)

            candles = response['candles']
            closes = [float(c['mid']['c']) for c in candles]
            highs = [float(c['mid']['h']) for c in candles]
            lows = [float(c['mid']['l']) for c in candles]

            return {
                'closes': np.array(closes),
                'highs': np.array(highs),
                'lows': np.array(lows),
                'current_price': closes[-1]
            }
        except Exception as e:
            print(f"  [ERROR] Getting data for {pair}: {e}")
            return None

    def calculate_score(self, pair, data):
        """
        Calculate trading score using SHARED technical analysis
        Returns: dict with score, direction, signals
        """
        closes = data['closes']
        highs = data['highs']
        lows = data['lows']
        current_price = data['current_price']

        # Use shared TA library
        rsi = ta.calculate_rsi(closes)
        macd = ta.calculate_macd(closes)
        ema_fast = ta.calculate_ema(closes, period=10)
        ema_slow = ta.calculate_ema(closes, period=21)
        adx = ta.calculate_adx(highs, lows, closes)
        atr = ta.calculate_atr(highs, lows, closes)

        # Calculate 4H trend using shared MTF library
        data_4h = self.get_forex_data(pair, granularity='H4', count=100)
        if data_4h:
            trend_4h = mtf.get_higher_timeframe_trend(data_4h['closes'], current_price)
        else:
            trend_4h = 'neutral'

        # Score LONG signals
        long_score = 0
        long_signals = []

        if rsi < 30:
            long_score += 2
            long_signals.append("RSI_OVERSOLD")
        elif rsi < 40:
            long_score += 1
            long_signals.append("RSI_LOW")

        if macd['macd'] > macd['signal'] and macd['histogram'] > 0:
            long_score += 2
            long_signals.append("MACD_BULL_CROSS")

        if current_price > ema_fast and ema_fast > ema_slow:
            long_score += 2
            long_signals.append("EMA_BULLISH")

        if adx > 25:
            long_score += 1
            long_signals.append("STRONG_TREND")

        if trend_4h == 'bullish':
            long_score += 2
            long_signals.append("4H_BULLISH_TREND")
        elif trend_4h == 'bearish':
            long_score -= 1.5
            long_signals.append("COUNTER_4H_TREND")

        # Score SHORT signals
        short_score = 0
        short_signals = []

        if rsi > 70:
            short_score += 2
            short_signals.append("RSI_OVERBOUGHT")
        elif rsi > 60:
            short_score += 1
            short_signals.append("RSI_HIGH")

        if macd['macd'] < macd['signal'] and macd['histogram'] < 0:
            short_score += 2
            short_signals.append("MACD_BEAR_CROSS")

        if current_price < ema_fast and ema_fast < ema_slow:
            short_score += 2
            short_signals.append("EMA_BEARISH")

        if adx > 25:
            short_score += 1
            short_signals.append("STRONG_TREND")

        if trend_4h == 'bearish':
            short_score += 2
            short_signals.append("4H_BEARISH_TREND")
        elif trend_4h == 'bullish':
            short_score -= 1.5
            short_signals.append("COUNTER_4H_TREND")

        # Determine best direction
        if long_score > short_score and long_score >= self.min_score:
            return {
                'score': long_score,
                'direction': 'long',
                'signals': long_signals,
                'rsi': rsi,
                'adx': adx,
                'pair': pair
            }
        elif short_score >= self.min_score:
            return {
                'score': short_score,
                'direction': 'short',
                'signals': short_signals,
                'rsi': rsi,
                'adx': adx,
                'pair': pair
            }
        else:
            return {'score': 0, 'direction': None, 'pair': pair}

    def scan_forex(self):
        """Scan all forex pairs for opportunities"""
        print(f"\n[FOREX SCAN] {datetime.now().strftime('%H:%M:%S')}")

        opportunities = []
        current_hour = datetime.now().hour

        for pair in self.forex_pairs:
            # Time filter
            if current_hour in self.AVOID_HOURS.get(pair, []):
                print(f"  [SKIP] {pair} - Avoiding hour {current_hour}")
                continue

            data = self.get_forex_data(pair)
            if data is None:
                continue

            result = self.calculate_score(pair, data)

            if result['score'] >= self.min_score:
                print(f"  [FOUND] {pair} {result['direction'].upper()}: {result['score']:.1f}/10")
                print(f"          Signals: {', '.join(result['signals'])}")
                opportunities.append(result)
            else:
                print(f"  [SKIP] {pair}: Score {result['score']:.1f}/10")

        return opportunities

    def get_account_balance(self):
        """Get current account balance"""
        try:
            r = accounts.AccountSummary(accountID=self.oanda_account_id)
            response = self.client.request(r)
            return float(response['account']['balance'])
        except Exception as e:
            print(f"[ERROR] Getting balance: {e}")
            return 0

    def run(self):
        """Main trading loop"""
        print("\n[STARTING FOREX SYSTEM]")
        print(f"Scanning every {self.scan_interval/60:.0f} minutes")
        print("Press Ctrl+C to stop\n")

        iteration = 0

        while True:
            iteration += 1
            print(f"\n{'='*70}")
            print(f"ITERATION #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}")

            balance = self.get_account_balance()
            print(f"[ACCOUNT] Balance: ${balance:,.2f}")

            opportunities = self.scan_forex()

            if opportunities:
                print(f"\n[FOUND {len(opportunities)} opportunities]")
                # TODO: Execute trades (implement in next step)
            else:
                print("\n[No opportunities found]")

            print(f"\n[Next scan in {self.scan_interval/60:.0f} minutes...]")
            time.sleep(self.scan_interval)


if __name__ == "__main__":
    trader = ForexTraderE8()
    trader.run()
