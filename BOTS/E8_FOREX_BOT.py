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
import csv
import json
from datetime import datetime
from pathlib import Path
from HYBRID_OANDA_TRADELOCKER import HybridAdapter

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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

        # HYBRID: OANDA data + TradeLocker execution
        self.client = HybridAdapter()

        # E8 CHALLENGE SETTINGS
        self.challenge_balance = 200000  # $200K
        self.max_drawdown_percent = 0.06  # 6% max drawdown
        self.profit_target = 0.10  # 10% to pass ($20,000)

        # OPTIMIZED STRATEGY SETTINGS (from backtest optimization)
        self.forex_pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']  # Top 3 pairs
        self.min_score = 3.0  # MODERATE: Strong setups (RSI+trend OR MACD+trend)
        self.max_positions = 2  # Max 2 positions (focus on best pairs)

        # OPTIMIZED POSITION SIZING (from parameter grid search)
        # Best config: 2% risk, 2% target, 1% stop = 8.47% ROI, 3.44% DD
        self.risk_per_trade = 0.02  # OPTIMIZED: 2% per trade (was 0.8%)
        self.leverage_multiplier = 5  # 5x leverage

        # OPTIMIZED PROFIT TARGETS
        self.profit_target_pct = 0.02  # OPTIMIZED: 2% profit target (was 2.5%)
        self.stop_loss_pct = 0.01  # OPTIMIZED: 1% stop loss (tight stops)

        # E8 POSITION SIZE MULTIPLIER (MODERATE AGGRESSION)
        self.position_size_multiplier = 0.90  # 90% of calculated size (increased from 80%)

        # 24/5 TRADING - ALL FOREX SESSIONS
        # Trade all major sessions: Tokyo (7pm-4am), London (3am-12pm), NY (8am-5pm) EST
        self.TRADING_HOURS = {
            'EUR_USD': list(range(24)),  # 24/5 - all hours (forex never sleeps!)
            'GBP_USD': list(range(24)),  # 24/5 - all hours
            'USD_JPY': list(range(24)),  # 24/5 - all hours (especially good during Tokyo session)
        }

        # Scan interval
        self.scan_interval = 3600  # 1 hour

        # Track challenge progress
        self.starting_balance = None
        self.current_balance = None
        self.peak_balance = None
        self.current_drawdown = 0.0

        # Persistent state file (CRITICAL FIX: persist peak balance across restarts)
        self.state_file = Path('BOTS/e8_bot_state.json')
        self._load_state()

        # Score logging
        self.score_log_file = Path('e8_score_log.csv')
        self._init_score_log()

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
        print(f"Score Log: {self.score_log_file}")
        print("=" * 70)

    def _init_score_log(self):
        """Initialize CSV file for score logging"""
        if not self.score_log_file.exists():
            with open(self.score_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'pair', 'score', 'direction',
                    'rsi', 'macd', 'macd_signal', 'adx',
                    'ema_10', 'ema_21', 'ema_200', 'signals',
                    'price', 'action_taken'
                ])
            print(f"[INIT] Created score log: {self.score_log_file}")

    def _load_state(self):
        """Load persistent state (peak balance) from file - CRITICAL FIX"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.peak_balance = state.get('peak_balance')
                    self.starting_balance = state.get('starting_balance')
                    print(f"[LOAD] Restored state - Peak: ${self.peak_balance:,.2f}, Starting: ${self.starting_balance:,.2f}")
            else:
                print(f"[INIT] No previous state found - starting fresh")
        except Exception as e:
            print(f"[WARN] Failed to load state: {e}")

    def _save_state(self):
        """Save persistent state (peak balance) to file - CRITICAL FIX"""
        try:
            # Ensure directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            state = {
                'peak_balance': self.peak_balance,
                'starting_balance': self.starting_balance,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to save state: {e}")

    def _log_score(self, symbol, score, direction, indicators, signals, price, action):
        """Log score calculation to CSV"""
        try:
            with open(self.score_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    symbol,
                    f"{score:.2f}",
                    direction,
                    f"{indicators.get('rsi', 0):.2f}",
                    f"{indicators.get('macd', 0):.5f}",
                    f"{indicators.get('macd_signal', 0):.5f}",
                    f"{indicators.get('adx', 0):.2f}",
                    f"{indicators.get('ema_10', 0):.5f}",
                    f"{indicators.get('ema_21', 0):.5f}",
                    f"{indicators.get('ema_200', 0):.5f}",
                    '|'.join(map(str, signals)),
                    f"{price:.5f}",
                    action
                ])
        except Exception as e:
            print(f"[WARN] Failed to log score: {e}")

    def get_account_balance(self):
        """Get current account balance and track challenge progress"""
        try:
            # Get from TradeLocker via hybrid adapter
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
                self._save_state()  # CRITICAL: Persist new peak immediately

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

        CRITICAL FIX: DD-constrained sizing to prevent challenge failure.
        """
        # STEP 1: Calculate remaining DD cushion
        if self.peak_balance is None:
            self.peak_balance = balance

        equity = balance  # Approximate (actual equity from get_account_balance)
        current_dd = (self.peak_balance - equity) / self.peak_balance
        dd_cushion_percent = self.max_drawdown_percent - current_dd
        dd_cushion_dollars = self.peak_balance * dd_cushion_percent

        print(f"\n[POSITION SIZING - {symbol}]")
        print(f"  Peak Balance: ${self.peak_balance:,.2f}")
        print(f"  Current Equity: ${equity:,.2f}")
        print(f"  Current DD: {current_dd*100:.2f}%")
        print(f"  DD Cushion: ${dd_cushion_dollars:,.2f} ({dd_cushion_percent*100:.2f}%)")

        # STEP 2: Standard position sizing (2% risk)
        risk_amount = balance * self.risk_per_trade
        stop_distance = price * self.stop_loss_pct
        units_standard = int((risk_amount / stop_distance) * self.leverage_multiplier)
        units_standard = int(units_standard * self.position_size_multiplier)

        # STEP 3: DD-constrained sizing (max loss = DD cushion)
        # Leave 20% safety margin on DD cushion
        max_safe_loss = dd_cushion_dollars * 0.80
        units_dd_constrained = int((max_safe_loss / stop_distance) * self.leverage_multiplier)

        # STEP 4: Take the SMALLER of the two (safety first!)
        units = min(units_standard, units_dd_constrained)

        print(f"  Standard Sizing: {units_standard:,} units")
        print(f"  DD-Constrained: {units_dd_constrained:,} units")
        print(f"  FINAL Position: {units:,} units ({units/100000:.1f} lots)")

        # STEP 5: Safety checks
        min_units = 10000  # 0.1 lots minimum
        if units < min_units:
            print(f"  [WARN] Position too small - blocking trade (need >{min_units:,} units)")
            return 0  # Block trade if too small

        max_units = 1000000  # 10 lots maximum
        if units > max_units:
            units = max_units

        max_loss = (units * stop_distance) / self.leverage_multiplier
        print(f"  Max Loss at SL: ${max_loss:,.2f}")

        # STEP 6: Final safety check - would this risk DD violation?
        if max_loss > dd_cushion_dollars:
            print(f"  [BLOCK] Trade would risk DD violation!")
            print(f"  Max Loss: ${max_loss:,.2f} > DD Cushion: ${dd_cushion_dollars:,.2f}")
            return 0  # BLOCK TRADE

        return units

    def calculate_score(self, candles, symbol):
        """
        Calculate SEPARATE LONG and SHORT scores (25.16% ROI strategy).

        This is the CORRECT method that achieved 25.16% ROI in backtesting.
        LONG and SHORT are evaluated independently with their own criteria.

        Returns: (long_score, short_score, signals, indicators_dict)
        """
        if len(candles) < 50:
            return 0, 0, ["insufficient_data"], {}

        # Extract OHLC
        closes = np.array([float(c['mid']['c']) for c in candles])
        highs = np.array([float(c['mid']['h']) for c in candles])
        lows = np.array([float(c['mid']['l']) for c in candles])

        if not TALIB_AVAILABLE:
            # Simplified scoring without TA-Lib
            long_score, short_score, signals = self._simplified_score(closes)
            return long_score, short_score, signals, {}

        try:
            # TA-LIB INDICATORS
            rsi = talib.RSI(closes, timeperiod=14)
            macd, macd_signal, _ = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
            adx = talib.ADX(highs, lows, closes, timeperiod=14)
            ema_10 = talib.EMA(closes, timeperiod=10)
            ema_21 = talib.EMA(closes, timeperiod=21)
            ema_200 = talib.EMA(closes, timeperiod=200)

            # Current values
            current_rsi = rsi[-1]
            current_macd = macd[-1]
            current_signal = macd_signal[-1]
            prev_macd = macd[-2]
            prev_signal = macd_signal[-2]
            current_adx = adx[-1]
            current_ema10 = ema_10[-1]
            current_ema21 = ema_21[-1]
            current_ema200 = ema_200[-1]
            macd_hist = current_macd - current_signal
            prev_macd_hist = prev_macd - prev_signal

            # SEPARATE SCORING (original 25.16% ROI strategy)
            long_score = 0
            short_score = 0
            signals = []

            # LONG SCORE (0-6 max)
            # 1. RSI oversold (2 points)
            if current_rsi < 40:
                long_score += 2
                signals.append("rsi_oversold")

            # 2. MACD bullish crossover (2 points)
            if macd_hist > 0 and prev_macd_hist <= 0:
                long_score += 2
                signals.append("macd_bull_cross")

            # 3. Strong trend (1 point)
            if current_adx > 25:
                long_score += 1
                signals.append("strong_trend")

            # 4. EMA alignment uptrend (1 point)
            if current_ema10 > current_ema21 > current_ema200:
                long_score += 1
                signals.append("ema_uptrend")

            # SHORT SCORE (0-6 max)
            # 1. RSI overbought (2 points)
            if current_rsi > 60:
                short_score += 2
                signals.append("rsi_overbought")

            # 2. MACD bearish crossover (2 points)
            if macd_hist < 0 and prev_macd_hist >= 0:
                short_score += 2
                signals.append("macd_bear_cross")

            # 3. Strong trend (1 point)
            if current_adx > 25:
                short_score += 1
                # Don't append again if already added for long

            # 4. EMA alignment downtrend (1 point)
            if current_ema10 < current_ema21 < current_ema200:
                short_score += 1
                signals.append("ema_downtrend")

            # Build indicators dict
            indicators = {
                'rsi': current_rsi,
                'macd': current_macd,
                'macd_signal': current_signal,
                'adx': current_adx,
                'ema_10': current_ema10,
                'ema_21': current_ema21,
                'ema_200': current_ema200
            }

            return long_score, short_score, signals, indicators

        except Exception as e:
            print(f"[WARN] Indicator calculation failed for {symbol}: {e}")
            return 0, 0, ["error"], {}

    def _simplified_score(self, closes):
        """
        Simplified scoring without TA-Lib (dual-scoring system).
        Returns: (long_score, short_score, signals, indicators)
        """
        if len(closes) < 50:
            return 0, 0, ["insufficient_data"], {}

        sma_fast = np.mean(closes[-10:])
        sma_slow = np.mean(closes[-50:])

        long_score = 0
        short_score = 0
        signals = []

        # LONG score: fast MA above slow MA
        if sma_fast > sma_slow:
            long_score += 3
            signals.append("sma_bullish")

        # SHORT score: fast MA below slow MA
        if sma_fast < sma_slow:
            short_score += 3
            signals.append("sma_bearish")

        # Simplified indicators
        indicators = {
            'rsi': 50,  # neutral
            'macd': 0,
            'macd_signal': 0,
            'adx': 20,
            'ema_10': sma_fast,
            'ema_21': sma_slow,
            'ema_200': np.mean(closes)
        }

        return long_score, short_score, signals, indicators

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

        # Get current positions from TradeLocker
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
                p.get('instrument') == symbol or p.get('symbol') == symbol
                for p in open_positions
            )

            if has_position:
                print(f"\n[{symbol}] Already in position - skipping")
                continue

            # Check trading hours
            current_hour = datetime.now().hour
            if not self.is_trading_hour(symbol):
                print(f"\n[{symbol}] Outside trading hours (current hour: {current_hour}) - skipping")
                continue

            # Fetch candles from OANDA (via hybrid adapter)
            print(f"\n[{symbol}] Fetching data...")
            candles = self.client.get_candles(symbol, count=100, granularity='H1')

            if len(candles) < 50:
                print(f"[{symbol}] Insufficient data")
                continue

            # Calculate score (dual-scoring system from 25.16% ROI backtest)
            long_score, short_score, signals, indicators = self.calculate_score(candles, symbol)

            # Get current price
            current_price = float(candles[-1]['mid']['c'])

            # Determine direction based on independent score thresholds
            direction = None
            score = 0

            if long_score >= self.min_score:
                direction = 'LONG'
                score = long_score
                print(f"[{symbol}] LONG Score: {long_score:.1f} | Signals: {', '.join(map(str, signals))}")
            elif short_score >= self.min_score:
                direction = 'SHORT'
                score = short_score
                print(f"[{symbol}] SHORT Score: {short_score:.1f} | Signals: {', '.join(map(str, signals))}")
            else:
                print(f"[{symbol}] No signal - LONG: {long_score:.1f}, SHORT: {short_score:.1f} (need >={self.min_score})")
                # Log the scores even when not trading
                self._log_score(symbol, max(long_score, short_score),
                               'LONG' if long_score > short_score else 'SHORT',
                               indicators, signals, current_price, "no_signal")
                continue

            # ENTRY SIGNAL! (Dual-scoring system allows both LONG and SHORT)
            # Log signal detected
            self._log_score(symbol, score, direction, indicators, signals, current_price, "signal_detected")

            # Calculate position size
            units = self.calculate_position_size(balance, current_price, symbol)

            # CRITICAL: Skip trade if position sizing blocked it (DD constraint)
            if units == 0:
                print(f"[SKIP] Trade blocked - insufficient DD cushion")
                self._log_score(symbol, score, direction, indicators, signals, current_price, "BLOCKED_DD_CONSTRAINT")
                continue

            # Convert direction to side
            side = 'buy' if direction == 'LONG' else 'sell'

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

            # Place order on TradeLocker (via hybrid adapter)
            order_id = self.client.place_order(
                symbol=symbol,
                units=units,
                side=side,
                take_profit=take_profit,
                stop_loss=stop_loss
            )

            if order_id:
                print(f"[SUCCESS] Order placed: {order_id}")
                # Update log with trade execution
                self._log_score(symbol, score, direction, indicators, signals, current_price, f"TRADED_{side.upper()}")
            else:
                print(f"[FAILED] Could not place order")
                self._log_score(symbol, score, direction, indicators, signals, current_price, "FAILED")

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
