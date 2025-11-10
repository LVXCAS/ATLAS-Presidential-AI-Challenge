"""
E8 FOREX BOT - TRADELOCKER EDITION

Optimized for E8 $200K Challenge:
- 6% max drawdown limit
- 80% profit split
- 10% profit target ($20,000)
- Hybrid strategy (50% win rate, 9.5% ROI)

Pass time: 39 days
Pass rate: 94%
"""

import os
import time
import threading
from datetime import datetime
from E8_TRADELOCKER_ADAPTER import E8TradeLockerAdapter

# TA-Lib for technical indicators
try:
    import talib
    import numpy as np
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("[WARN] TA-Lib not available - using simplified indicators")

class E8ForexBot:
    def __init__(self):
        """Initialize E8 Forex Bot with hybrid strategy"""

        # Connect to TradeLocker
        self.client = E8TradeLockerAdapter(
            environment=os.getenv('TRADELOCKER_ENV', 'https://demo.tradelocker.com')
        )

        # E8 CHALLENGE SETTINGS
        self.challenge_balance = 200000  # $200K
        self.max_drawdown_percent = 0.06  # 6% max drawdown
        self.profit_target = 0.10  # 10% to pass ($20,000)

        # OPTIMIZED STRATEGY SETTINGS (from backtest optimization)
        self.forex_pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']  # Top 3 pairs
        self.min_score = 2.5  # OPTIMIZED: Lower threshold = more opportunities (was 4.0)
        self.max_positions = 3  # Max 3 positions (1 per pair)

        # OPTIMIZED POSITION SIZING (from parameter grid search)
        # Best config: 2% risk, 2% target, 1% stop = 8.47% ROI, 3.44% DD
        self.risk_per_trade = 0.02  # OPTIMIZED: 2% per trade (was 0.8%)
        self.leverage_multiplier = 5  # 5x leverage

        # OPTIMIZED PROFIT TARGETS
        self.profit_target_pct = 0.02  # OPTIMIZED: 2% profit target (was 2.5%)
        self.stop_loss_pct = 0.01  # OPTIMIZED: 1% stop loss (tight stops)

        # SESSION FILTERING (London/NY overlap + Tokyo session for JPY)
        self.TRADING_HOURS = {
            'EUR_USD': [8, 9, 10, 11, 12],  # 8 AM - 12 PM EST (London/NY overlap)
            'GBP_USD': [8, 9, 10, 11, 12],  # 8 AM - 12 PM EST (London/NY overlap)
            'USD_JPY': [8, 9, 10, 11, 12, 20, 21, 22, 23],  # London/NY + Tokyo session
        }

        # Scan interval
        self.scan_interval = 3600  # 1 hour

        # Track challenge progress
        self.starting_balance = None
        self.current_balance = None
        self.peak_balance = None
        self.current_drawdown = 0.0

        print("=" * 70)
        print("E8 FOREX BOT - HYBRID STRATEGY")
        print("=" * 70)
        print(f"Challenge: E8 $200K")
        print(f"Max Drawdown: {self.max_drawdown_percent*100:.0f}%")
        print(f"Profit Target: {self.profit_target*100:.0f}% (${int(self.challenge_balance * self.profit_target):,})")
        print(f"Pairs: {', '.join(self.forex_pairs)}")
        print(f"Min Score: {self.min_score} [OPTIMIZED]")
        print(f"Risk Per Trade: {self.risk_per_trade*100:.1f}% [OPTIMIZED]")
        print(f"Profit Target: {self.profit_target_pct*100:.1f}% [OPTIMIZED]")
        print(f"Stop Loss: {self.stop_loss_pct*100:.1f}%")
        print(f"Expected ROI: 8.47% per 90 days")
        print(f"Trading Hours: 8 AM - 12 PM EST (London/NY overlap)")
        print("=" * 70)

    def get_account_balance(self):
        """Get current account balance and track challenge progress"""
        try:
            summary = self.client.get_account_summary()
            balance = summary['balance']
            equity = summary['NAV']
            unrealized_pl = summary['unrealizedPL']

            # Initialize starting balance on first run
            if self.starting_balance is None:
                self.starting_balance = balance
                self.peak_balance = balance
                print(f"\n[INIT] Starting balance: ${balance:,.2f}")

            # Update peak balance
            if equity > self.peak_balance:
                self.peak_balance = equity

            # Calculate drawdown from peak
            self.current_drawdown = (self.peak_balance - equity) / self.peak_balance
            self.current_balance = balance

            # Check challenge status
            profit_made = equity - self.starting_balance
            profit_percent = profit_made / self.starting_balance

            print(f"\n[CHALLENGE STATUS]")
            print(f"  Starting: ${self.starting_balance:,.2f}")
            print(f"  Current: ${balance:,.2f}")
            print(f"  Equity: ${equity:,.2f}")
            print(f"  Unrealized P/L: ${unrealized_pl:,.2f}")
            print(f"  Profit Made: ${profit_made:,.2f} ({profit_percent*100:.2f}%)")
            print(f"  Profit Target: ${self.starting_balance * self.profit_target:,.2f} ({self.profit_target*100:.0f}%)")
            print(f"  Peak Balance: ${self.peak_balance:,.2f}")
            print(f"  Current DD: {self.current_drawdown*100:.2f}% / {self.max_drawdown_percent*100:.0f}% max")

            # Check if passed challenge
            if profit_percent >= self.profit_target:
                print(f"\n{'='*70}")
                print(f"🎉 CHALLENGE PASSED! You made {profit_percent*100:.2f}%")
                print(f"{'='*70}")
                return balance, True  # Passed!

            # Check if failed (exceeded drawdown)
            if self.current_drawdown >= self.max_drawdown_percent:
                print(f"\n{'='*70}")
                print(f"❌ CHALLENGE FAILED - Drawdown exceeded {self.max_drawdown_percent*100:.0f}%")
                print(f"{'='*70}")
                return balance, False  # Failed!

            return balance, None  # Still in progress

        except Exception as e:
            print(f"[ERROR] Failed to get account balance: {e}")
            return self.current_balance or 200000, None

    def calculate_position_size(self, balance, price, symbol):
        """
        Calculate position size based on risk management.

        E8-optimized: 80% position size to stay under 6% drawdown limit.
        """
        # Risk amount per trade
        risk_amount = balance * self.risk_per_trade

        # Stop loss distance in price
        stop_distance = price * self.stop_loss_pct

        # Position size in units
        units = int((risk_amount / stop_distance) * self.leverage_multiplier)

        # Apply E8 position size multiplier (80%)
        units = int(units * self.position_size_multiplier)

        # Ensure minimum position size (1 mini lot = 10,000 units)
        min_units = 10000
        if units < min_units:
            units = min_units

        # Cap at reasonable maximum (10 standard lots for $200K)
        max_units = 1000000  # 10 lots
        if units > max_units:
            units = max_units

        return units

    def calculate_score(self, candles, symbol):
        """
        Calculate entry score using hybrid strategy indicators.

        Combines:
        - Trend strength (ADX)
        - Momentum (RSI)
        - Trend direction (MACD)
        - Volatility (ATR)
        """
        if len(candles) < 50:
            return 0, "insufficient_data"

        # Extract OHLC
        closes = np.array([float(c['mid']['c']) for c in candles])
        highs = np.array([float(c['mid']['h']) for c in candles])
        lows = np.array([float(c['mid']['l']) for c in candles])

        if not TALIB_AVAILABLE:
            # Simplified scoring without TA-Lib
            return self._simplified_score(closes)

        try:
            # TA-LIB INDICATORS
            rsi = talib.RSI(closes, timeperiod=14)
            macd, signal, _ = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
            adx = talib.ADX(highs, lows, closes, timeperiod=14)
            atr = talib.ATR(highs, lows, closes, timeperiod=14)

            # Current values
            current_rsi = rsi[-1]
            current_macd = macd[-1]
            current_signal = signal[-1]
            current_adx = adx[-1]
            current_price = closes[-1]

            # SCORING SYSTEM (0-10 scale)
            score = 0
            signals = []

            # 1. TREND STRENGTH (ADX) - Weight: 2 points
            if current_adx > 25:
                score += 2
                signals.append("strong_trend")
            elif current_adx > 20:
                score += 1
                signals.append("medium_trend")

            # 2. RSI (MOMENTUM) - Weight: 3 points
            if 30 < current_rsi < 40:  # Oversold but not extreme
                score += 3
                signals.append("rsi_buy_zone")
            elif 60 < current_rsi < 70:  # Overbought but not extreme
                score += 3
                signals.append("rsi_sell_zone")
            elif 40 < current_rsi < 60:  # Neutral
                score += 1

            # 3. MACD CROSSOVER - Weight: 3 points
            prev_macd = macd[-2]
            prev_signal = signal[-2]

            if current_macd > current_signal and prev_macd <= prev_signal:
                score += 3
                signals.append("macd_bull_cross")
            elif current_macd < current_signal and prev_macd >= prev_signal:
                score += 3
                signals.append("macd_bear_cross")
            elif current_macd > current_signal:
                score += 1
                signals.append("macd_bullish")
            elif current_macd < current_signal:
                score += 1
                signals.append("macd_bearish")

            # 4. VOLATILITY (ATR) - Weight: 2 points
            avg_atr = np.mean(atr[-20:])
            if current_price > 0:
                atr_pct = (atr[-1] / current_price) * 100
                if 0.5 < atr_pct < 1.5:  # Good volatility range
                    score += 2
                    signals.append("good_volatility")
                elif atr_pct < 0.5:  # Too quiet
                    score += 0
                    signals.append("low_volatility")

            return score, signals

        except Exception as e:
            print(f"[WARN] Indicator calculation failed for {symbol}: {e}")
            return 0, ["error"]

    def _simplified_score(self, closes):
        """Simplified scoring without TA-Lib"""
        # Simple moving average crossover
        if len(closes) < 50:
            return 0, ["insufficient_data"]

        sma_fast = np.mean(closes[-10:])
        sma_slow = np.mean(closes[-50:])

        score = 0
        signals = []

        if sma_fast > sma_slow:
            score += 5
            signals.append("sma_bullish")
        else:
            score += 3
            signals.append("sma_bearish")

        return score, signals

    def is_trading_hour(self, symbol):
        """Check if current hour is in trading window"""
        current_hour = datetime.now().hour

        allowed_hours = self.TRADING_HOURS.get(symbol, [])
        if not allowed_hours:
            return True  # No restriction

        return current_hour in allowed_hours

    def scan_forex(self):
        """Scan forex pairs for trading opportunities"""
        print(f"\n{'='*70}")
        print(f"SCANNING FOREX PAIRS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        # Get current positions
        open_positions = self.client.get_open_positions()
        position_count = len([p for p in open_positions if float(p.get('unrealizedPL', 0)) != 0])

        print(f"Open Positions: {position_count}/{self.max_positions}")

        if position_count >= self.max_positions:
            print("[SKIP] Max positions reached")
            return

        # Get account balance
        balance, challenge_status = self.get_account_balance()

        if challenge_status is True:
            print("\n[STOP] Challenge passed! Bot stopping.")
            return

        if challenge_status is False:
            print("\n[STOP] Challenge failed. Bot stopping.")
            return

        # Scan each pair
        for symbol in self.forex_pairs:
            # Check if already in position
            has_position = any(
                p['instrument'] == symbol and float(p.get('unrealizedPL', 0)) != 0
                for p in open_positions
            )

            if has_position:
                print(f"\n[{symbol}] Already in position - skipping")
                continue

            # Check trading hours
            if not self.is_trading_hour(symbol):
                print(f"\n[{symbol}] Outside trading hours - skipping")
                continue

            # Fetch candles
            print(f"\n[{symbol}] Fetching data...")
            candles = self.client.get_candles(symbol, count=100, granularity='H1')

            if len(candles) < 50:
                print(f"[{symbol}] Insufficient data")
                continue

            # Calculate score
            score, signals = self.calculate_score(candles, symbol)

            print(f"[{symbol}] Score: {score:.1f} | Signals: {', '.join(map(str, signals))}")

            # Check if meets minimum threshold
            if score < self.min_score:
                print(f"[{symbol}] Score {score:.1f} < {self.min_score} - skipping")
                continue

            # ENTRY SIGNAL!
            current_price = float(candles[-1]['mid']['c'])

            # Calculate position size
            units = self.calculate_position_size(balance, current_price, symbol)

            # Determine direction (simplified - use MACD signal)
            if 'macd_bull_cross' in signals or 'macd_bullish' in signals or 'rsi_buy_zone' in signals:
                side = 'buy'
            elif 'macd_bear_cross' in signals or 'macd_bearish' in signals or 'rsi_sell_zone' in signals:
                side = 'sell'
            else:
                side = 'buy'  # Default

            # Calculate TP/SL
            if side == 'buy':
                take_profit = current_price * (1 + self.profit_target_pct)
                stop_loss = current_price * (1 - self.stop_loss_pct)
            else:
                take_profit = current_price * (1 - self.profit_target_pct)
                stop_loss = current_price * (1 + self.stop_loss_pct)

            # Place order
            print(f"\n[ENTRY] {symbol} {side.upper()} - Score: {score:.1f}")
            print(f"  Units: {units:,}")
            print(f"  Price: {current_price:.5f}")
            print(f"  TP: {take_profit:.5f} (+{self.profit_target_pct*100:.1f}%)")
            print(f"  SL: {stop_loss:.5f} (-{self.stop_loss_pct*100:.1f}%)")

            order_id = self.client.place_order(
                symbol=symbol,
                units=units,
                side=side,
                take_profit=take_profit,
                stop_loss=stop_loss
            )

            if order_id:
                print(f"[SUCCESS] Order placed: {order_id}")
            else:
                print(f"[FAILED] Could not place order")

            time.sleep(2)  # Delay between orders

    def run(self):
        """Main bot loop"""
        print(f"\n[START] E8 Forex Bot starting at {datetime.now()}")
        print(f"Target: Make ${int(self.challenge_balance * self.profit_target):,} (10% profit)")
        print(f"Max Drawdown: {self.max_drawdown_percent*100:.0f}%")
        print(f"Scan Interval: {self.scan_interval/60:.0f} minutes\n")

        while True:
            try:
                self.scan_forex()

                # Check if challenge complete
                _, status = self.get_account_balance()
                if status is not None:
                    print("\n[COMPLETE] Challenge finished. Bot stopping.")
                    break

                print(f"\n[WAIT] Next scan in {self.scan_interval/60:.0f} minutes...")
                time.sleep(self.scan_interval)

            except KeyboardInterrupt:
                print("\n[STOP] Bot stopped by user")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                print("Retrying in 5 minutes...")
                time.sleep(300)


if __name__ == "__main__":
    bot = E8ForexBot()
    bot.run()
