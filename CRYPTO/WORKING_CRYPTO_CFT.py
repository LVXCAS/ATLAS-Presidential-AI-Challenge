"""
CRYPTO TRADING BOT - CRYPTO FUND TRADER
Separated architecture using SHARED/ libraries
Target: CFT $200K account (+$12K profit to pass, 6% target)
Markets: BTC/USD, ETH/USD
Trading: 24/7/365 (crypto never closes)
"""
import os
import sys
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared libraries
from SHARED.technical_analysis import ta
from SHARED.kelly_criterion import kelly
from SHARED.multi_timeframe import mtf

import numpy as np


class CryptoTraderCFT:
    """
    Crypto trading bot for Crypto Fund Trader
    Uses shared technical analysis, Kelly sizing, and multi-timeframe confirmation
    Trades BTC/USD and ETH/USD 24/7
    """

    def __init__(self):
        print("=" * 70)
        print("CRYPTO TRADER - CRYPTO FUND TRADER (SEPARATED ARCHITECTURE)")
        print("=" * 70)

        # Crypto pairs
        self.crypto_pairs = {
            'BTCUSD': {
                'name': 'Bitcoin',
                'symbol': 'BTCUSD',
                'min_size': 0.001,  # Minimum trade size in BTC
                'precision': 8,  # Decimal places
            },
            'ETHUSD': {
                'name': 'Ethereum',
                'symbol': 'ETHUSD',
                'min_size': 0.01,  # Minimum trade size in ETH
                'precision': 8,
            }
        }

        # Risk parameters (CFT-compatible)
        self.account_size = 200000  # CFT account size
        self.min_score = 2.5  # Quality filter
        self.risk_per_trade = 0.008  # 0.8% risk (crypto is more volatile)
        self.max_positions = 2  # Max 2 crypto positions

        # Volatility adjustment for crypto (wider stops needed)
        self.volatility_multiplier = 2.0

        # Time-based filters (crypto trades 24/7 but best during US hours)
        self.BEST_HOURS = list(range(9, 17))  # 9 AM - 5 PM EST (highest volume)
        self.AVOID_HOURS = list(range(0, 5))  # 12 AM - 5 AM EST (low liquidity)

        # Scanning interval
        self.scan_interval = 1800  # 30 minutes (crypto moves fast)

        print(f"Pairs: {', '.join(self.crypto_pairs.keys())}")
        print(f"Account Size: ${self.account_size:,}")
        print(f"Min Score: {self.min_score}/10")
        print(f"Risk Per Trade: {self.risk_per_trade * 100}%")
        print(f"Max Positions: {self.max_positions}")
        print(f"Trading Hours: 24/7 (best 9AM-5PM EST)")
        print(f"Volatility Multiplier: {self.volatility_multiplier}x")
        print(f"Shared Libraries: TA-Lib, Kelly, MTF")
        print("=" * 70)

    def get_crypto_data(self, pair, granularity='1H', count=200):
        """
        Fetch historical crypto data

        NOTE: This is a placeholder. In production, you would connect to:
        - Crypto Fund Trader's data feed
        - Or use APIs like: Binance, Bybit, Coinbase, CryptoCompare

        For now, returns simulated data structure
        """
        # TODO: Replace with actual crypto exchange API
        # Could use: ccxt library for multi-exchange support
        #   pip install ccxt
        #   import ccxt
        #   exchange = ccxt.binance()
        #   ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h')

        # Simulated data for now
        base_price = 65000 if 'BTC' in pair else 3500
        return {
            'closes': np.random.randn(count) * 1000 + base_price,
            'highs': np.random.randn(count) * 1000 + base_price + 500,
            'lows': np.random.randn(count) * 1000 + base_price - 500,
            'current_price': base_price,
            'volume': np.random.randint(100, 1000, count)
        }

    def calculate_score(self, pair, data):
        """
        Calculate trading score using SHARED technical analysis
        Returns: dict with score, direction, signals

        Uses IDENTICAL logic to forex/futures bots
        """
        closes = data['closes']
        highs = data['highs']
        lows = data['lows']
        current_price = data['current_price']

        # Use shared TA library (SAME as forex and futures)
        rsi = ta.calculate_rsi(closes)
        macd = ta.calculate_macd(closes)
        ema_fast = ta.calculate_ema(closes, period=10)
        ema_slow = ta.calculate_ema(closes, period=21)
        adx = ta.calculate_adx(highs, lows, closes)
        atr = ta.calculate_atr(highs, lows, closes)
        bb = ta.calculate_bollinger_bands(closes)

        # Calculate 4H trend using shared MTF library (SAME as forex/futures)
        data_4h = self.get_crypto_data(pair, granularity='4H', count=100)
        trend_4h = mtf.get_higher_timeframe_trend(data_4h['closes'], current_price)

        # Score LONG signals (IDENTICAL to forex/futures logic)
        long_score = 0
        long_signals = []

        if rsi < 30:
            long_score += 2
            long_signals.append("RSI_OVERSOLD")
        elif rsi < 40:
            long_score += 1
            long_signals.append("RSI_LOW")

        if macd['macd'] > macd['signal']:
            long_score += 2
            long_signals.append("MACD_BULL_CROSS")

        if current_price > ema_fast and ema_fast > ema_slow:
            long_score += 2
            long_signals.append("EMA_BULLISH")

        if current_price < bb['lower']:  # Bollinger bounce
            long_score += 1
            long_signals.append("BB_BOUNCE")

        if adx > 25:
            long_score += 1
            long_signals.append("STRONG_TREND")

        if trend_4h == 'bullish':
            long_score += 2
            long_signals.append("4H_BULLISH_TREND")

        # Score SHORT signals (IDENTICAL to forex/futures logic)
        short_score = 0
        short_signals = []

        if rsi > 70:
            short_score += 2
            short_signals.append("RSI_OVERBOUGHT")
        elif rsi > 60:
            short_score += 1
            short_signals.append("RSI_HIGH")

        if macd['macd'] < macd['signal']:
            short_score += 2
            short_signals.append("MACD_BEAR_CROSS")

        if current_price < ema_fast and ema_fast < ema_slow:
            short_score += 2
            short_signals.append("EMA_BEARISH")

        if current_price > bb['upper']:  # Bollinger rejection
            short_score += 1
            short_signals.append("BB_REJECTION")

        if adx > 25:
            short_score += 1
            short_signals.append("STRONG_TREND")

        if trend_4h == 'bearish':
            short_score += 2
            short_signals.append("4H_BEARISH_TREND")

        # Determine best direction
        if long_score > short_score and long_score >= self.min_score:
            return {
                'score': long_score,
                'direction': 'long',
                'signals': long_signals,
                'rsi': rsi,
                'adx': adx,
                'pair': pair,
                'current_price': current_price
            }
        elif short_score >= self.min_score:
            return {
                'score': short_score,
                'direction': 'short',
                'signals': short_signals,
                'rsi': rsi,
                'adx': adx,
                'pair': pair,
                'current_price': current_price
            }
        else:
            return {'score': 0, 'direction': None, 'pair': pair}

    def calculate_position_size(self, pair_name, score, current_price):
        """
        Calculate crypto position size using SHARED Kelly Criterion
        """
        # Use shared Kelly library
        position_data = kelly.calculate_position_size(
            technical_score=score,
            fundamental_score=0,  # No fundamentals for crypto yet
            account_balance=self.account_size,
            risk_per_trade=self.risk_per_trade
        )

        # Convert to crypto units using shared Kelly function
        crypto_units = kelly.calculate_crypto_units(
            position_size_dollars=position_data['final_size'],
            current_price=current_price
        )

        return {
            'crypto_units': crypto_units,
            'dollar_value': crypto_units * current_price,
            'kelly_multiplier': position_data['kelly_multiplier'],
            'confidence': position_data['confidence']
        }

    def scan_crypto(self):
        """Scan all crypto pairs for opportunities"""
        print(f"\n[CRYPTO SCAN] {datetime.now().strftime('%H:%M:%S')}")

        opportunities = []
        current_hour = datetime.now().hour

        # Time filter (avoid dead hours but allow most times since 24/7)
        if current_hour in self.AVOID_HOURS:
            print(f"  [INFO] Hour {current_hour} - Low liquidity period (still scanning)")

        for pair_name in self.crypto_pairs.keys():
            data = self.get_crypto_data(pair_name)
            result = self.calculate_score(pair_name, data)

            if result['score'] >= self.min_score:
                print(f"  [FOUND] {pair_name} {result['direction'].upper()}: {result['score']:.1f}/10")
                print(f"          Signals: {', '.join(result['signals'])}")

                # Calculate position size
                position = self.calculate_position_size(
                    pair_name,
                    result['score'],
                    result['current_price']
                )

                result['crypto_units'] = position['crypto_units']
                result['dollar_value'] = position['dollar_value']
                result['confidence'] = position['confidence']

                opportunities.append(result)
            else:
                print(f"  [SKIP] {pair_name}: Score {result['score']:.1f}/10")

        return opportunities

    def run(self):
        """Main trading loop - runs 24/7"""
        print("\n[STARTING CRYPTO SYSTEM - 24/7 TRADING]")
        print(f"Scanning every {self.scan_interval/60:.0f} minutes")
        print("Press Ctrl+C to stop\n")

        iteration = 0

        while True:
            iteration += 1
            print(f"\n{'='*70}")
            print(f"ITERATION #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}")

            opportunities = self.scan_crypto()

            if opportunities:
                print(f"\n[FOUND {len(opportunities)} opportunities]")
                for opp in opportunities:
                    print(f"  {opp['pair']} {opp['direction'].upper()}: {opp['crypto_units']:.4f} units (${opp['dollar_value']:,.2f})")
                # TODO: Execute trades (connect to CFT platform or crypto exchange)
            else:
                print("\n[No opportunities found]")

            print(f"\n[Next scan in {self.scan_interval/60:.0f} minutes...]")
            time.sleep(self.scan_interval)


if __name__ == "__main__":
    trader = CryptoTraderCFT()
    trader.run()
