"""
REAL OPTIONS TRADER - Alpaca Paper Trading
Actually trades options (calls/puts/spreads) not stocks
Runs alongside FOREX for diversification
"""
import os
import time
import requests
from datetime import datetime, timedelta
from options_executor import OptionsExecutor

try:
    import talib
    import numpy as np
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

class RealOptionsTrader:
    def __init__(self):
        # Alpaca Paper Trading (supports options!)
        self.api_key = os.getenv('ALPACA_API_KEY', 'PKZ7F4B26EOEZ8UN8G8U')
        self.api_secret = os.getenv('ALPACA_SECRET_KEY', 'B1aTbyUpEUsCF1CpxsyshsdUXvGZBqoYEfORpLok')
        self.base_url = "https://paper-api.alpaca.markets"

        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }

        # High-volatility stocks for options
        self.symbols = [
            'TSLA', 'NVDA', 'AMD', 'AMZN', 'GOOGL',
            'META', 'AAPL', 'MSFT', 'SPY', 'QQQ'
        ]

        # CONSERVATIVE parameters for options
        self.min_score = 6.0  # Options need higher conviction (more expensive)
        self.max_positions = 2  # Only 2 options positions at once
        self.position_size = 500  # $500 per position (1-5 contracts)
        self.scan_interval = 300  # 5 minutes (options move fast)

        # Options executor
        self.executor = OptionsExecutor()

        print("=" * 70)
        print("REAL OPTIONS TRADER - ALPACA PAPER")
        print("=" * 70)
        print(f"Broker: Alpaca Paper (Options Enabled)")
        print(f"Symbols: {len(self.symbols)} high-volatility stocks")
        print(f"Min Score: {self.min_score} (HIGH CONVICTION)")
        print(f"Max Positions: {self.max_positions}")
        print(f"Position Size: ${self.position_size}")
        print(f"Strategies: Long Calls, Long Puts, Bull Put Spreads")
        if TALIB_AVAILABLE:
            print("Quant Libraries: TA-Lib ENABLED")
        print("=" * 70)

    def is_market_open(self):
        """Check if market is open"""
        try:
            url = f"{self.base_url}/v2/clock"
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                clock = response.json()
                return clock['is_open']
        except:
            pass
        return False

    def get_stock_data(self, symbol):
        """Get stock data for analysis"""
        try:
            # Get 100 bars of 5-minute data
            url = f"{self.base_url}/v2/stocks/{symbol}/bars"
            params = {
                'timeframe': '5Min',
                'limit': 100
            }

            response = requests.get(url, headers=self.headers, params=params, timeout=10)

            if response.status_code != 200:
                return None

            data = response.json()
            bars = data.get('bars', [])

            if len(bars) < 50:
                return None

            closes = np.array([float(b['c']) for b in bars])
            highs = np.array([float(b['h']) for b in bars])
            lows = np.array([float(b['l']) for b in bars])
            volumes = np.array([float(b['v']) for b in bars])

            return {
                'closes': closes,
                'highs': highs,
                'lows': lows,
                'volumes': volumes,
                'current_price': closes[-1]
            }

        except Exception as e:
            print(f"  [ERROR] {symbol}: {e}")
            return None

    def calculate_score(self, symbol, data):
        """Calculate options trading score using TA-Lib"""
        closes = data['closes']
        highs = data['highs']
        lows = data['lows']
        current_price = data['current_price']

        score = 0
        signals = []
        strategy = None

        if TALIB_AVAILABLE and len(closes) >= 50:
            # RSI
            rsi = talib.RSI(closes, timeperiod=14)[-1]

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(closes)

            # ATR (volatility - critical for options)
            atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
            volatility = (atr / current_price) * 100

            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(closes)
            bb_position = (current_price - lower[-1]) / (upper[-1] - lower[-1]) if upper[-1] != lower[-1] else 0.5

            # ADX (trend strength)
            adx = talib.ADX(highs, lows, closes, timeperiod=14)[-1]

            # SCORING FOR OPTIONS (STRICTER THAN FOREX/STOCKS)

            # 1. Volatility (OPTIONS LOVE VOLATILITY)
            if volatility > 3.0:  # 3%+ ATR
                score += 3
                signals.append("HIGH_VOLATILITY")
            elif volatility > 2.0:
                score += 1.5
                signals.append("MEDIUM_VOLATILITY")

            # 2. RSI extremes (reversal plays)
            if rsi < 30:
                score += 2.5
                signals.append("RSI_OVERSOLD")
                strategy = 'LONG_CALL'  # Expecting bounce
            elif rsi > 70:
                score += 2.5
                signals.append("RSI_OVERBOUGHT")
                strategy = 'LONG_PUT'  # Expecting pullback

            # 3. MACD momentum
            if len(macd_hist) >= 2:
                if macd_hist[-1] > 0 and macd_hist[-2] <= 0:
                    score += 3
                    signals.append("MACD_BULLISH_CROSS")
                    if not strategy:
                        strategy = 'LONG_CALL'
                elif macd_hist[-1] < 0 and macd_hist[-2] >= 0:
                    score += 3
                    signals.append("MACD_BEARISH_CROSS")
                    if not strategy:
                        strategy = 'LONG_PUT'

            # 4. Bollinger Band extremes
            if bb_position < 0.1:  # Near lower band
                score += 2
                signals.append("BB_OVERSOLD")
                if not strategy:
                    strategy = 'LONG_CALL'
            elif bb_position > 0.9:  # Near upper band
                score += 2
                signals.append("BB_OVERBOUGHT")
                if not strategy:
                    strategy = 'LONG_PUT'

            # 5. Strong trend confirmation
            if adx > 25:
                score += 1.5
                signals.append("STRONG_TREND")

            # Default strategy if signals exist but no direction determined
            if not strategy and score >= 4:
                # Use MACD direction
                if macd_hist[-1] > 0:
                    strategy = 'LONG_CALL'
                else:
                    strategy = 'LONG_PUT'

            return {
                'symbol': symbol,
                'score': score,
                'price': current_price,
                'strategy': strategy,
                'rsi': rsi,
                'volatility': volatility,
                'adx': adx,
                'signals': signals
            }
        else:
            return {
                'symbol': symbol,
                'score': 0,
                'price': current_price,
                'strategy': None,
                'signals': []
            }

    def get_current_positions(self):
        """Get open options positions"""
        try:
            url = f"{self.base_url}/v2/positions"
            response = requests.get(url, headers=self.headers, timeout=5)

            if response.status_code == 200:
                positions = response.json()
                # Filter for options only (symbol contains expiry date pattern)
                options_positions = [p for p in positions if len(p['symbol']) > 10]
                return len(options_positions)
        except:
            pass
        return 0

    def scan_opportunities(self):
        """Scan for options trading opportunities"""
        opportunities = []

        for symbol in self.symbols:
            data = self.get_stock_data(symbol)

            if not data:
                continue

            result = self.calculate_score(symbol, data)

            # DEBUG: Show all scores
            print(f"  [DEBUG] {symbol}: Score {result['score']:.1f}/10 | Strategy: {result['strategy']}", end="")
            if TALIB_AVAILABLE:
                print(f" | RSI: {result.get('rsi', 0):.1f} | Vol: {result['volatility']:.1f}% | ADX: {result.get('adx', 0):.1f}")
            else:
                print()

            if result['score'] >= self.min_score and result['strategy']:
                opportunities.append(result)
                print(f"  >>> [FOUND] {symbol}: Score {result['score']:.1f}/10 - {result['strategy']}")
                print(f"          Signals: {', '.join(result['signals'])}")

        return opportunities

    def run(self):
        """Main trading loop"""
        print("\n[STARTING REAL OPTIONS TRADER]")
        print("Scanning every 5 minutes during market hours")
        print("Press Ctrl+C to stop\n")

        iteration = 0

        while True:
            iteration += 1

            print("\n" + "=" * 70)
            print(f"ITERATION #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)

            if not self.is_market_open():
                print("[MARKET CLOSED] Waiting...")
                time.sleep(self.scan_interval)
                continue

            print(f"\n[OPTIONS SCAN] {datetime.now().strftime('%H:%M:%S')}")

            # Check current positions
            current_positions = self.get_current_positions()
            print(f"\n[POSITIONS] Currently holding: {current_positions}/{self.max_positions}")

            if current_positions >= self.max_positions:
                print("  [INFO] Max positions reached")
                print(f"\n[Next scan in {self.scan_interval} seconds...]")
                time.sleep(self.scan_interval)
                continue

            # Scan for opportunities
            opportunities = self.scan_opportunities()

            print(f"\n[SCAN COMPLETE] Found {len(opportunities)} opportunities")

            if not opportunities:
                print("  No high-conviction setups found")
                print(f"\n[Next scan in {self.scan_interval} seconds...]")
                time.sleep(self.scan_interval)
                continue

            # Execute trades
            trades_needed = self.max_positions - current_positions
            trades_to_execute = opportunities[:trades_needed]

            if trades_to_execute:
                print(f"\n[EXECUTING] {len(trades_to_execute)} OPTIONS TRADES")
                trades_executed = self.executor.execute_from_scanner(trades_to_execute)
                print(f"\n[TRADES EXECUTED: {trades_executed}]")

            print(f"\n[Next scan in {self.scan_interval} seconds...]")
            time.sleep(self.scan_interval)

if __name__ == "__main__":
    trader = RealOptionsTrader()
    trader.run()
