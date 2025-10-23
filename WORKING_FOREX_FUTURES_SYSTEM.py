"""
WORKING FOREX + FUTURES SYSTEM
Fixed thresholds that actually find trades
Target: 50% monthly ROI across both markets
"""
import os
import time
import requests
from datetime import datetime
import alpaca_trade_api as tradeapi

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# TA-Lib for professional indicators
try:
    import talib
    import numpy as np
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

class WorkingForexFuturesSystem:
    def __init__(self):
        # Alpaca API (for FOREX data + account)
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_SECRET_KEY')

        if not self.api_key or not self.api_secret:
            raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in .env")

        self.api = tradeapi.REST(
            self.api_key,
            self.api_secret,
            'https://paper-api.alpaca.markets',
            api_version='v2'
        )

        # FOREX Pairs (24/5 trading)
        self.forex_pairs = ['EUR/USD', 'USD/JPY', 'GBP/USD']

        # FUTURES Proxies (use ETFs during market hours)
        self.futures_proxies = {
            'MES': 'SPY',   # S&P 500 micro futures → SPY ETF
            'MNQ': 'QQQ'    # NASDAQ micro futures → QQQ ETF
        }

        # REALISTIC PARAMETERS (not the 8.0 bullshit that finds nothing)
        self.min_score = 3.0  # LOWERED from 8.0
        self.position_size = 1000  # $1k per position
        self.max_positions = 5  # Can hold 5 total positions
        self.profit_target = 0.10  # 10% gain (realistic for FX/futures)
        self.stop_loss = 0.03  # 3% stop (tighter risk)

        # Scan intervals
        self.forex_scan_interval = 3600  # 1 hour (FX moves slower)
        self.futures_scan_interval = 900  # 15 minutes (futures move faster)

        print("=" * 70)
        print("WORKING FOREX + FUTURES SYSTEM")
        print("=" * 70)
        print(f"FOREX Pairs: {', '.join(self.forex_pairs)}")
        print(f"FUTURES: {', '.join(self.futures_proxies.keys())} (via ETF proxies)")
        print(f"Min Score: {self.min_score} (REALISTIC)")
        print(f"Position Size: ${self.position_size}")
        print(f"Max Positions: {self.max_positions}")
        print(f"Profit Target: {self.profit_target*100}%")
        print(f"Stop Loss: {self.stop_loss*100}%")
        if TALIB_AVAILABLE:
            print("Quant Libraries: TA-Lib ENABLED")
        print("=" * 70)

    def get_forex_data(self, pair):
        """Get FOREX data from Alpaca"""
        try:
            symbol = pair.replace('/', '')  # EUR/USD → EURUSD
            bars = self.api.get_crypto_bars(
                symbol,
                '1Hour',
                limit=100
            ).df

            if len(bars) < 50:
                return None

            return bars
        except Exception as e:
            print(f"  [ERROR] FOREX {pair}: {e}")
            return None

    def get_futures_data(self, symbol):
        """Get FUTURES data via ETF proxy"""
        try:
            proxy = self.futures_proxies[symbol]
            bars = self.api.get_bars(
                proxy,
                '15Min',
                limit=100
            ).df

            if len(bars) < 50:
                return None

            return bars
        except Exception as e:
            print(f"  [ERROR] FUTURES {symbol}: {e}")
            return None

    def calculate_score(self, symbol, bars, market_type='FOREX'):
        """
        Calculate score using TA-Lib
        LOWERED thresholds so we actually find trades
        """
        closes = bars['close'].values
        highs = bars['high'].values
        lows = bars['low'].values
        current_price = closes[-1]

        score = 0
        signals = []

        if TALIB_AVAILABLE and len(closes) >= 50:
            # RSI
            rsi = talib.RSI(closes, timeperiod=14)[-1]

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(closes)

            # ATR (volatility)
            atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
            volatility = (atr / current_price) * 100

            # ADX (trend strength)
            adx = talib.ADX(highs, lows, closes, timeperiod=14)[-1]

            # EMA crossover
            ema_fast = talib.EMA(closes, timeperiod=10)
            ema_slow = talib.EMA(closes, timeperiod=21)

            # SCORING (REALISTIC THRESHOLDS)

            # 1. RSI oversold/overbought (easier to trigger)
            if rsi < 40:  # Was 30, now 40 = more signals
                score += 2
                signals.append("RSI_OVERSOLD")
            elif rsi > 60:  # Was 70, now 60 = more signals
                score += 1.5
                signals.append("RSI_OVERBOUGHT")

            # 2. MACD bullish cross
            if macd_hist[-1] > 0 and macd_hist[-2] <= 0:
                score += 2.5
                signals.append("MACD_BULLISH")

            # 3. EMA crossover
            if ema_fast[-1] > ema_slow[-1] and ema_fast[-2] <= ema_slow[-2]:
                score += 2
                signals.append("EMA_CROSS")

            # 4. Trend strength (lowered from 25 to 20)
            if adx > 20:  # Was 25, now 20 = more signals
                score += 1.5
                signals.append("TREND")

            # 5. Volatility (for FX/futures any movement is good)
            if market_type == 'FOREX' and volatility > 0.5:  # Even small moves count
                score += 1
                signals.append("FX_VOLATILITY")
            elif market_type == 'FUTURES' and volatility > 1.0:
                score += 1
                signals.append("FUTURES_VOLATILITY")

            return {
                'symbol': symbol,
                'score': score,
                'price': current_price,
                'rsi': rsi,
                'volatility': volatility,
                'adx': adx,
                'signals': signals,
                'market_type': market_type
            }
        else:
            # Fallback without TA-Lib
            return {
                'symbol': symbol,
                'score': 0,
                'price': current_price,
                'signals': [],
                'market_type': market_type
            }

    def scan_forex(self):
        """Scan FOREX pairs"""
        print(f"\n[FOREX SCAN] {datetime.now().strftime('%H:%M:%S')}")

        opportunities = []

        for pair in self.forex_pairs:
            bars = self.get_forex_data(pair)
            if bars is None:
                continue

            result = self.calculate_score(pair, bars, 'FOREX')

            if result['score'] >= self.min_score:
                opportunities.append(result)
                print(f"  [FOREX] {pair}: Score {result['score']:.1f}/10")
                if TALIB_AVAILABLE:
                    print(f"          RSI: {result.get('rsi', 0):.1f} | Signals: {', '.join(result['signals'])}")

        return opportunities

    def scan_futures(self):
        """Scan FUTURES via ETF proxies"""
        print(f"\n[FUTURES SCAN] {datetime.now().strftime('%H:%M:%S')}")

        opportunities = []

        for symbol in self.futures_proxies.keys():
            bars = self.get_futures_data(symbol)
            if bars is None:
                continue

            result = self.calculate_score(symbol, bars, 'FUTURES')

            if result['score'] >= self.min_score:
                opportunities.append(result)
                print(f"  [FUTURES] {symbol}: Score {result['score']:.1f}/10")
                if TALIB_AVAILABLE:
                    print(f"            RSI: {result.get('rsi', 0):.1f} | Signals: {', '.join(result['signals'])}")

        return opportunities

    def execute_trades(self, opportunities):
        """Execute trades (PAPER TRADING for now)"""
        if not opportunities:
            return 0

        positions = self.api.list_positions()
        current_symbols = [p.symbol for p in positions]

        print(f"\n[POSITIONS] Currently holding: {len(current_symbols)}/{self.max_positions}")

        if len(current_symbols) >= self.max_positions:
            print(f"  [INFO] Max positions reached")
            return 0

        trades_executed = 0

        for opp in opportunities[:self.max_positions - len(current_symbols)]:
            symbol = opp['symbol']

            # For FOREX/FUTURES we're just tracking in paper mode for now
            print(f"\n  [PAPER TRADE] {opp['market_type']}")
            print(f"    Symbol: {symbol}")
            print(f"    Score: {opp['score']:.1f}/10")
            print(f"    Price: ${opp['price']:.4f}")
            print(f"    Signals: {', '.join(opp['signals'])}")

            trades_executed += 1

            # Send Telegram notification
            try:
                message = f"📊 {opp['market_type']} SIGNAL!\n{symbol}\nScore: {opp['score']:.1f}/10"
                if TALIB_AVAILABLE:
                    message += f"\nRSI: {opp.get('rsi', 0):.1f} | Signals: {', '.join(opp['signals'])}"

                bot_url = f"https://api.telegram.org/bot8203125300:AAE1FTiXQALCFh8cX9lKWhq8arEB2yvUGfQ/sendMessage"
                requests.post(bot_url, data={'chat_id': '7606409012', 'text': message}, timeout=3)
            except:
                pass

        return trades_executed

    def run(self):
        """Run continuous FOREX + FUTURES scanning"""
        print("\n[STARTING FOREX + FUTURES SYSTEM]")
        print("FOREX: Scanning every 1 hour (24/5)")
        print("FUTURES: Scanning every 15 minutes (market hours)")
        print("Press Ctrl+C to stop\n")

        iteration = 0
        last_forex_scan = 0
        last_futures_scan = 0

        while True:
            iteration += 1
            current_time = time.time()

            print(f"\n{'='*70}")
            print(f"ITERATION #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}")

            all_opportunities = []

            # FOREX scan (every hour)
            if current_time - last_forex_scan >= self.forex_scan_interval:
                forex_opps = self.scan_forex()
                all_opportunities.extend(forex_opps)
                last_forex_scan = current_time

            # FUTURES scan (every 15 min, only during market hours)
            clock = self.api.get_clock()
            if clock.is_open and (current_time - last_futures_scan >= self.futures_scan_interval):
                futures_opps = self.scan_futures()
                all_opportunities.extend(futures_opps)
                last_futures_scan = current_time
            elif not clock.is_open:
                print("[FUTURES] Market closed, skipping")

            # Execute trades
            if all_opportunities:
                trades = self.execute_trades(all_opportunities)
                print(f"\n[TRADES: {trades}]")
            else:
                print("\n[No opportunities found this scan]")

            # Wait for next scan (use shorter of the two intervals)
            wait_time = min(
                self.forex_scan_interval - (current_time - last_forex_scan),
                self.futures_scan_interval - (current_time - last_futures_scan)
            )
            wait_time = max(wait_time, 60)  # At least 1 minute

            print(f"\n[Next scan in {wait_time/60:.1f} minutes...]")
            time.sleep(wait_time)

if __name__ == "__main__":
    system = WorkingForexFuturesSystem()
    system.run()
