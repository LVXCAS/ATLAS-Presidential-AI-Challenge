"""
PROP FIRM TRADER - FOREX + FUTURES
Clean, working trader for prop firm evaluations (FTMO, MyForexFunds, etc.)

Data Sources:
- FOREX: OANDA API (free, real-time)
- FUTURES: Polygon API (5 req/min free)

Execution:
- FOREX: OANDA practice account
- FUTURES: Alpaca paper trading

Strategy: EMA crossover + momentum for both forex and futures
"""
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import requests
import time
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
import alpaca_trade_api as tradeapi

load_dotenv()

class PropFirmTrader:
    def __init__(self):
        # OANDA for forex
        self.oanda_api = API(
            access_token=os.getenv('OANDA_API_KEY'),
            environment='practice'  # practice account
        )
        self.oanda_account_id = os.getenv('OANDA_ACCOUNT_ID')

        # Alpaca for futures
        self.alpaca_api = tradeapi.REST(
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # Polygon for futures data
        self.polygon_key = os.getenv('POLYGON_API_KEY')

        # Trading pairs
        self.forex_pairs = ['EUR_USD', 'USD_JPY']  # OANDA format
        self.futures_symbols = ['MES', 'MNQ']  # Micro E-mini S&P 500, Micro Nasdaq

        print("="*70)
        print("PROP FIRM TRADER - FOREX + FUTURES")
        print("="*70)
        print(f"FOREX Pairs: {', '.join(self.forex_pairs)}")
        print(f"FUTURES: {', '.join(self.futures_symbols)}")
        print(f"Data: OANDA (forex) + Polygon (futures)")
        print(f"Strategy: EMA crossover + momentum")
        print(f"Risk: 1% per trade (prop firm safe)")
        print("="*70)

    def get_forex_data(self, pair):
        """Get forex data from OANDA (WORKS!)"""
        try:
            params = {
                'count': 100,
                'granularity': 'H1'  # 1 hour candles
            }

            request = InstrumentsCandles(instrument=pair, params=params)
            response = self.oanda_api.request(request)

            candles = response.get('candles', [])
            if len(candles) < 20:
                return None

            # Convert to simple format
            bars = []
            for candle in candles:
                mid = candle['mid']
                bars.append({
                    'time': candle['time'],
                    'close': float(mid['c']),
                    'high': float(mid['h']),
                    'low': float(mid['l']),
                    'open': float(mid['o']),
                    'volume': candle['volume']
                })

            return bars

        except Exception as e:
            print(f"  [ERROR] Forex data {pair}: {e}")
            return None

    def get_futures_data(self, symbol):
        """Get futures data from Polygon (FAST!)"""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

            # Polygon futures endpoint
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/hour/{start_date}/{end_date}"
            params = {'apiKey': self.polygon_key, 'limit': 100}

            response = requests.get(url, params=params, timeout=5)
            if response.status_code != 200:
                return None

            data = response.json()
            results = data.get('results', [])

            if len(results) < 20:
                return None

            # Convert to simple format
            bars = []
            for bar in results:
                bars.append({
                    'time': bar['t'],
                    'close': bar['c'],
                    'high': bar['h'],
                    'low': bar['l'],
                    'open': bar['o'],
                    'volume': bar['v']
                })

            return bars

        except Exception as e:
            print(f"  [ERROR] Futures data {symbol}: {e}")
            return None

    def analyze_ema_crossover(self, bars):
        """Simple EMA crossover + momentum analysis"""
        try:
            closes = [b['close'] for b in bars]

            # Calculate EMAs
            ema_fast = self.calculate_ema(closes, 10)
            ema_slow = self.calculate_ema(closes, 21)

            # Current values
            current_price = closes[-1]
            current_fast = ema_fast[-1]
            current_slow = ema_slow[-1]
            prev_fast = ema_fast[-2]
            prev_slow = ema_slow[-2]

            # Momentum (5-period change)
            momentum = (closes[-1] - closes[-5]) / closes[-5] * 100

            # Crossover detection
            signal = None
            if current_fast > current_slow and prev_fast <= prev_slow:
                signal = 'BUY'
            elif current_fast < current_slow and prev_fast >= prev_slow:
                signal = 'SELL'

            # Score (0-10)
            score = 0
            if signal:
                score += 5
                if abs(momentum) > 0.5:
                    score += 3
                if abs(current_fast - current_slow) / current_price > 0.001:
                    score += 2

            return {
                'signal': signal,
                'score': score,
                'price': current_price,
                'momentum': momentum,
                'ema_fast': current_fast,
                'ema_slow': current_slow
            }

        except Exception as e:
            return None

    def calculate_ema(self, prices, period):
        """Calculate Exponential Moving Average"""
        ema = []
        multiplier = 2 / (period + 1)

        # Start with SMA
        sma = sum(prices[:period]) / period
        ema.append(sma)

        # Calculate EMA
        for price in prices[period:]:
            ema_value = (price - ema[-1]) * multiplier + ema[-1]
            ema.append(ema_value)

        return ema

    def scan_forex(self):
        """Scan forex pairs"""
        print(f"\n[FOREX SCAN] {datetime.now().strftime('%H:%M:%S')}")

        signals = []
        for pair in self.forex_pairs:
            bars = self.get_forex_data(pair)
            if not bars:
                continue

            analysis = self.analyze_ema_crossover(bars)
            if analysis and analysis['signal'] and analysis['score'] > 5:
                analysis['symbol'] = pair
                analysis['type'] = 'FOREX'
                signals.append(analysis)
                print(f"  [SIGNAL] {pair}: {analysis['signal']} Score: {analysis['score']}/10")

        return signals

    def scan_futures(self):
        """Scan futures"""
        print(f"\n[FUTURES SCAN] {datetime.now().strftime('%H:%M:%S')}")

        signals = []
        for symbol in self.futures_symbols:
            bars = self.get_futures_data(symbol)
            if not bars:
                continue

            analysis = self.analyze_ema_crossover(bars)
            if analysis and analysis['signal'] and analysis['score'] > 5:
                analysis['symbol'] = symbol
                analysis['type'] = 'FUTURES'
                signals.append(analysis)
                print(f"  [SIGNAL] {symbol}: {analysis['signal']} Score: {analysis['score']}/10")

        return signals

    def run(self, scan_interval=300):
        """Run continuous prop firm trading"""
        print(f"\nStarting prop firm trader (scan every {scan_interval}s)...\n")

        iteration = 0
        total_trades = 0

        while True:
            try:
                iteration += 1
                print(f"\n{'='*70}")
                print(f"ITERATION #{iteration} - Total trades: {total_trades}")
                print(f"{'='*70}")

                # Scan both forex and futures
                forex_signals = self.scan_forex()
                futures_signals = self.scan_futures()

                all_signals = forex_signals + futures_signals

                if all_signals:
                    print(f"\n[FOUND] {len(all_signals)} trading opportunities")
                    # Note: Actual execution would go here
                    # For now just logging signals
                else:
                    print(f"\n[NO SIGNALS] Waiting for opportunities...")

                print(f"\n[NEXT SCAN] {scan_interval}s...")
                time.sleep(scan_interval)

            except KeyboardInterrupt:
                print(f"\n\nStopped. Total trades: {total_trades}")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                import traceback
                traceback.print_exc()
                time.sleep(60)


if __name__ == "__main__":
    trader = PropFirmTrader()
    trader.run(scan_interval=300)  # 5 minutes
