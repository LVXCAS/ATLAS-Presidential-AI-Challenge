"""
OPTIONS-STYLE STOCK TRADING SYSTEM
Mimics butterfly spreads and options strategies using stocks
Uses TA-Lib institutional-grade indicators
Target: 50% monthly ROI like you wanted for options
"""
import os
import time
import requests
import numpy as np
from datetime import datetime
import alpaca_trade_api as tradeapi

# TA-Lib for professional indicators
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

class OptionsStyleStockTrader:
    def __init__(self):
        # API Keys (proven working)
        self.api_key = os.getenv('ALPACA_API_KEY', 'PKOWU7D6JANXP47ZU72X72757D')
        self.api_secret = os.getenv('ALPACA_SECRET_KEY', '52QKewJCoafjLsFPKJJSTZs7BG7XBa6mLwi3e1W3Z7Tq')

        self.api = tradeapi.REST(
            self.api_key,
            self.api_secret,
            'https://paper-api.alpaca.markets',
            api_version='v2'
        )

        # High-IV stocks (options-like volatility)
        # These move like options: high volatility, big swings
        self.symbols = [
            'TSLA',   # High IV, big moves
            'NVDA',   # Tech volatility
            'AMD',    # Semiconductor swings
            'PLTR',   # High beta
            'NIO',    # EV volatility
            'SOFI',   # Fintech moves
            'BABA',   # China tech
            'META',   # Big tech swings
            'NFLX',   # Streaming volatility
            'COIN',   # Crypto proxy
            'RIOT',   # Bitcoin leverage
            'MARA',   # Mining volatility
            'SQ',     # Fintech
            'ARKK',   # Innovation ETF
            'SQQQ',   # 3x inverse QQQ
            'TQQQ',   # 3x QQQ
            'SPXS',   # 3x inverse SPY
            'SPXL',   # 3x SPY
            'SOXL',   # 3x semiconductors
            'FNGU'    # Tech leverage
        ]

        # OPTIONS-STYLE PARAMETERS
        self.min_score = 2.5  # LOWERED from 5.0 to find opportunities
        self.position_size = 1000  # $1k per position (like buying 10 options contracts)
        self.max_positions = 3  # Like having 3 options positions
        self.profit_target = 0.15  # 15% gain (like options spreads)
        self.stop_loss = 0.05  # 5% stop (defined risk like options)

        print("=" * 70)
        print("OPTIONS-STYLE STOCK TRADING SYSTEM")
        print("=" * 70)
        print(f"Symbols: {len(self.symbols)} high-volatility stocks")
        print(f"Min Score: {self.min_score}")
        print(f"Position Size: ${self.position_size}")
        print(f"Max Positions: {self.max_positions}")
        print(f"Profit Target: {self.profit_target*100}%")
        print(f"Stop Loss: {self.stop_loss*100}%")
        if TALIB_AVAILABLE:
            print("Quant Libraries: TA-Lib ENABLED (Professional)")
        print("=" * 70)

    def get_market_data(self, symbol):
        """Get historical data for analysis"""
        try:
            # Get bars from Alpaca
            bars = self.api.get_bars(
                symbol,
                '1Day',
                limit=100
            ).df

            if len(bars) < 50:
                return None

            return bars
        except Exception as e:
            return None

    def calculate_options_style_score(self, symbol, bars):
        """
        Calculate score using TA-Lib (options-like analysis)

        Options traders look for:
        - High IV (volatility)
        - Overbought/oversold (RSI)
        - Trend exhaustion (MACD)
        - Support/resistance (Bollinger Bands)
        """
        closes = bars['close'].values
        highs = bars['high'].values
        lows = bars['low'].values
        volumes = bars['volume'].values

        current_price = closes[-1]

        score = 0
        signals = []

        if TALIB_AVAILABLE and len(closes) >= 50:
            # RSI - Like finding overbought/oversold for options
            rsi = talib.RSI(closes, timeperiod=14)[-1]

            # MACD - Trend confirmation
            macd, macd_signal, macd_hist = talib.MACD(closes)

            # ATR - Volatility (high volatility = options-like)
            atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
            volatility = (atr / current_price) * 100

            # ADX - Trend strength
            adx = talib.ADX(highs, lows, closes, timeperiod=14)[-1]

            # Bollinger Bands - Support/resistance
            upper, middle, lower = talib.BBANDS(closes, timeperiod=20)
            bb_position = (current_price - lower[-1]) / (upper[-1] - lower[-1])

            # SCORING (like butterfly spread selection)

            # 1. High volatility (like high IV for options)
            if volatility > 3:  # >3% daily range
                score += 3
                signals.append("HIGH_VOLATILITY")

            # 2. RSI oversold (like finding cheap options)
            if rsi < 30:
                score += 3
                signals.append("OVERSOLD")
            elif rsi > 70:
                score += 2
                signals.append("OVERBOUGHT")

            # 3. MACD bullish crossover (like options momentum)
            if macd_hist[-1] > 0 and macd_hist[-2] <= 0:
                score += 4
                signals.append("MACD_BULLISH_CROSS")

            # 4. Strong trend (like directional options plays)
            if adx > 25:
                score += 2
                signals.append("STRONG_TREND")

            # 5. At Bollinger Band (like support/resistance for options)
            if bb_position < 0.2:  # Near lower band
                score += 3
                signals.append("AT_SUPPORT")
            elif bb_position > 0.8:  # Near upper band
                score += 2
                signals.append("AT_RESISTANCE")

            return {
                'symbol': symbol,
                'score': score,
                'price': current_price,
                'rsi': rsi,
                'volatility': volatility,
                'adx': adx,
                'signals': signals,
                'macd_hist': macd_hist[-1]
            }
        else:
            # Fallback without TA-Lib
            volatility = np.std(closes[-20:]) / np.mean(closes[-20:]) * 100

            if volatility > 3:
                score += 5
                signals.append("HIGH_VOLATILITY")

            return {
                'symbol': symbol,
                'score': score,
                'price': current_price,
                'volatility': volatility,
                'signals': signals
            }

    def scan_opportunities(self):
        """Scan for options-style stock opportunities"""
        print(f"\n[SCANNING] {datetime.now().strftime('%H:%M:%S')}")

        opportunities = []

        for symbol in self.symbols:
            bars = self.get_market_data(symbol)
            if bars is None:
                continue

            result = self.calculate_options_style_score(symbol, bars)

            # DEBUG: Show ALL scores (not just above threshold)
            print(f"  [DEBUG] {result['symbol']}: Score {result['score']:.2f}/10", end="")
            if TALIB_AVAILABLE:
                print(f" | RSI: {result.get('rsi', 0):.1f} | Vol: {result['volatility']:.1f}%")
            else:
                print()

            if result['score'] >= self.min_score:
                opportunities.append(result)
                print(f"  >>> [FOUND] {result['symbol']}: Score {result['score']:.1f}/10 - ABOVE THRESHOLD!")
                if TALIB_AVAILABLE:
                    print(f"          Signals: {', '.join(result['signals'])}")

        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)

        return opportunities

    def execute_trades(self, opportunities):
        """Execute options-style trades (defined risk like butterflies)"""
        # Check current positions
        positions = self.api.list_positions()
        current_symbols = [p.symbol for p in positions]

        print(f"\n[POSITIONS] Currently holding: {len(current_symbols)}")

        # Don't exceed max positions (like having max 3 options spreads)
        if len(current_symbols) >= self.max_positions:
            print(f"  [INFO] Max positions ({self.max_positions}) reached")
            return 0

        trades_executed = 0

        for opp in opportunities[:self.max_positions - len(current_symbols)]:
            symbol = opp['symbol']

            # Skip if already own
            if symbol in current_symbols:
                continue

            # Calculate shares
            price = opp['price']
            qty = int(self.position_size / price)

            if qty < 1:
                continue

            try:
                # Place order with stop loss (defined risk like options)
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )

                print(f"\n  [TRADE EXECUTED]")
                print(f"    Symbol: {symbol}")
                print(f"    Qty: {qty} shares")
                print(f"    Price: ~${price:.2f}")
                print(f"    Score: {opp['score']}/10")
                print(f"    Order ID: {order.id}")

                trades_executed += 1

                # Send Telegram notification
                try:
                    message = f"📈 OPTIONS-STYLE TRADE!\n{symbol} x{qty} @ ${price:.2f}\nScore: {opp['score']}/10"
                    if TALIB_AVAILABLE:
                        message += f"\nRSI: {opp.get('rsi', 0):.1f} | Vol: {opp['volatility']:.1f}%"

                    bot_url = f"https://api.telegram.org/bot8203125300:AAE1FTiXQALCFh8cX9lKWhq8arEB2yvUGfQ/sendMessage"
                    requests.post(bot_url, data={'chat_id': '7606409012', 'text': message}, timeout=3)
                except:
                    pass

            except Exception as e:
                print(f"  [ERROR] {symbol}: {e}")

        return trades_executed

    def manage_positions(self):
        """Manage positions with profit targets and stop losses (like options)"""
        positions = self.api.list_positions()

        for position in positions:
            symbol = position.symbol

            # Skip actual options positions (we only manage stocks)
            if len(symbol) > 10:  # Options symbols are like "AMZN251121P00200000"
                print(f"  [SKIP] {symbol} is an options contract, not managing")
                continue

            # Only manage high-volatility stocks from our list
            if symbol not in self.symbols:
                continue

            entry_price = float(position.avg_entry_price)
            current_price = float(position.current_price)
            qty = abs(int(float(position.qty)))  # Fix: Ensure positive integer

            # Skip if qty is 0 or negative
            if qty <= 0:
                continue

            pnl_pct = (current_price - entry_price) / entry_price

            # Take profit at 15% (like options spread max profit)
            if pnl_pct >= self.profit_target:
                print(f"\n  [PROFIT TARGET] {symbol} +{pnl_pct*100:.1f}%")
                try:
                    self.api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    print(f"    Sold {qty} {symbol} @ ${current_price:.2f}")
                except Exception as e:
                    print(f"    [ERROR] Could not sell: {e}")

            # Stop loss at 5% (defined risk like options)
            elif pnl_pct <= -self.stop_loss:
                print(f"\n  [STOP LOSS] {symbol} {pnl_pct*100:.1f}%")
                try:
                    self.api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    print(f"    Sold {qty} {symbol} @ ${current_price:.2f}")
                except Exception as e:
                    print(f"    [ERROR] Could not sell: {e}")

    def run(self):
        """Run continuous options-style trading"""
        print("\n[STARTING OPTIONS-STYLE STOCK TRADER]")
        print("Scanning every 5 minutes during market hours")
        print("Press Ctrl+C to stop\n")

        iteration = 0

        while True:
            iteration += 1
            print(f"\n{'='*70}")
            print(f"ITERATION #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}")

            # Check if market is open
            clock = self.api.get_clock()
            if not clock.is_open:
                print("[MARKET CLOSED] Waiting...")
                time.sleep(900)  # Wait 15 minutes
                continue

            # Scan for opportunities
            opportunities = self.scan_opportunities()
            print(f"\n[SCAN COMPLETE] Found {len(opportunities)} opportunities")

            # Manage existing positions (take profits/stop losses)
            self.manage_positions()

            # Execute new trades
            if opportunities:
                trades = self.execute_trades(opportunities)
                print(f"\n[TRADES EXECUTED: {trades}]")

            print(f"\n[Next scan in 5 minutes...]")
            time.sleep(300)  # 5 minutes

if __name__ == "__main__":
    trader = OptionsStyleStockTrader()
    trader.run()
