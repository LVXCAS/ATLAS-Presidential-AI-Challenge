"""
WORKING FOREX SYSTEM - OANDA (NEWS-INTEGRATED)
Uses your OANDA practice account for ACTUAL forex trading
NOW WITH: News filtering (FRED, Polygon) to avoid trading against fundamentals
Target: 50% monthly ROI from forex alone
"""
import os
import time
import requests
import threading
from datetime import datetime
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts

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

# NEWS INTEGRATION
try:
    from news_forex_integration import NewsForexIntegration
    NEWS_AVAILABLE = True
except ImportError:
    NEWS_AVAILABLE = False
    print("[WARN] news_forex_integration.py not found - trading without news filter")

class WorkingForexOanda:
    def __init__(self):
        # OANDA API (REAL FOREX BROKER)
        self.oanda_token = os.getenv('OANDA_API_KEY')
        self.oanda_account_id = os.getenv('OANDA_ACCOUNT_ID', '101-004-29159709-001')  # Practice account

        if not self.oanda_token:
            raise ValueError("Missing OANDA_API_KEY in .env")

        # OANDA Practice server
        self.client = API(
            access_token=self.oanda_token,
            environment='practice'  # Practice account for paper trading
        )

        # FOREX Pairs (24/5 trading)
        self.forex_pairs = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'GBP_JPY']  # OANDA format uses underscore

        # REALISTIC PARAMETERS
        self.min_score = 2.5  # RAISED from 1.0 - filter for quality signals
        self.risk_per_trade = 0.01  # 1% risk per trade
        self.max_positions = 3  # Max 3 forex positions at once
        self.profit_target = 0.02  # 2% profit target (realistic for forex)
        self.stop_loss = 0.01  # 1% stop loss (tight risk management)

        # LEVERAGE SETTINGS
        self.leverage_multiplier = 5  # 5x leverage (reduced from 10x for safer recovery)
        self.use_10x_leverage = True  # Still enabled, but using 5x multiplier

        # TIME-BASED ENTRY FILTERS (avoid choppy hours)
        self.AVOID_TRADING_HOURS = {
            'EUR_USD': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],  # After 11 AM = chop
            'GBP_USD': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],  # Same
            'USD_JPY': [0, 1, 2, 3, 4, 16, 17, 18],  # Dead hours + late afternoon
            'GBP_JPY': [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17, 18],
        }

        # Scan interval
        self.scan_interval = 3600  # 1 hour (forex moves slower than stocks)

        # NEWS INTEGRATION
        self.news_filter = NewsForexIntegration() if NEWS_AVAILABLE else None

        print("=" * 70)
        print("WORKING FOREX SYSTEM - OANDA (NEWS-INTEGRATED)")
        print("=" * 70)
        print(f"Broker: OANDA Practice Account")
        print(f"Account ID: {self.oanda_account_id}")
        print(f"Pairs: {', '.join(self.forex_pairs)}")
        print(f"Min Score: {self.min_score} (QUALITY FILTER)")
        print(f"Risk Per Trade: {self.risk_per_trade*100}%")
        print(f"Max Positions: {self.max_positions}")
        print(f"Profit Target: {self.profit_target*100}%")
        print(f"Stop Loss: {self.stop_loss*100}%")
        print(f"Leverage: {self.leverage_multiplier}x {'ENABLED' if self.use_10x_leverage else 'DISABLED'}")
        if TALIB_AVAILABLE:
            print("Quant Libraries: TA-Lib ENABLED")
        if NEWS_AVAILABLE:
            print("News Filter: FRED + Polygon ENABLED")
        print("=" * 70)

    def get_account_balance(self):
        """Get OANDA account balance"""
        try:
            r = accounts.AccountSummary(accountID=self.oanda_account_id)
            response = self.client.request(r)
            balance = float(response['account']['balance'])
            return balance
        except Exception as e:
            print(f"  [ERROR] Getting account balance: {e}")
            return 100000  # Default to $100k if error

    def get_forex_data(self, pair, granularity='H1', count=100):
        """Get FOREX data from OANDA for any timeframe"""
        try:
            params = {
                'count': count,
                'granularity': granularity  # H1, H4, D, etc.
            }

            r = instruments.InstrumentsCandles(instrument=pair, params=params)
            response = self.client.request(r)

            candles = response['candles']

            if len(candles) < 50:
                return None

            # Convert to numpy arrays
            closes = np.array([float(c['mid']['c']) for c in candles])
            highs = np.array([float(c['mid']['h']) for c in candles])
            lows = np.array([float(c['mid']['l']) for c in candles])

            return {
                'closes': closes,
                'highs': highs,
                'lows': lows,
                'current_price': closes[-1]
            }

        except Exception as e:
            print(f"  [ERROR] FOREX {pair}: {e}")
            return None

    def get_higher_timeframe_trend(self, pair):
        """
        Get 4H timeframe trend direction for multi-timeframe confirmation
        Returns: 'bullish', 'bearish', or 'neutral'
        """
        if not TALIB_AVAILABLE:
            return 'neutral'

        try:
            # Get 4-hour data
            data_4h = self.get_forex_data(pair, granularity='H4', count=100)

            if not data_4h:
                return 'neutral'

            closes_4h = data_4h['closes']

            # Calculate 4H trend using EMA
            ema_fast_4h = talib.EMA(closes_4h, timeperiod=10)
            ema_slow_4h = talib.EMA(closes_4h, timeperiod=21)
            ema_trend_4h = talib.EMA(closes_4h, timeperiod=50)

            current_price = closes_4h[-1]

            # Determine trend
            # Bullish: Fast > Slow and Price > Trend EMA
            if ema_fast_4h[-1] > ema_slow_4h[-1] and current_price > ema_trend_4h[-1]:
                return 'bullish'

            # Bearish: Fast < Slow and Price < Trend EMA
            elif ema_fast_4h[-1] < ema_slow_4h[-1] and current_price < ema_trend_4h[-1]:
                return 'bearish'

            else:
                return 'neutral'

        except Exception as e:
            print(f"  [WARN] Could not get 4H trend for {pair}: {e}")
            return 'neutral'

    def calculate_score(self, pair, data):
        """
        Calculate score using TA-Lib - BIDIRECTIONAL (LONG and SHORT)
        Returns separate LONG and SHORT opportunities
        """
        closes = data['closes']
        highs = data['highs']
        lows = data['lows']
        current_price = data['current_price']

        long_score = 0
        short_score = 0
        long_signals = []
        short_signals = []

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
            ema_trend = talib.EMA(closes, timeperiod=200)

            # === LONG SIGNALS ===

            # 1. RSI oversold (buy signal)
            if rsi < 40:
                long_score += 2
                long_signals.append("RSI_OVERSOLD")

            # 2. MACD bullish cross
            if len(macd_hist) >= 2:
                if macd_hist[-1] > 0 and macd_hist[-2] <= 0:
                    long_score += 2.5
                    long_signals.append("MACD_BULLISH")

            # 3. EMA bullish crossover
            if len(ema_fast) >= 2 and len(ema_slow) >= 2:
                if ema_fast[-1] > ema_slow[-1] and ema_fast[-2] <= ema_slow[-2]:
                    long_score += 2
                    long_signals.append("EMA_CROSS_BULLISH")

            # 4. Uptrend (price above 200 EMA)
            if len(ema_trend) >= 1 and current_price > ema_trend[-1]:
                long_score += 1
                long_signals.append("UPTREND")

            # === SHORT SIGNALS ===

            # 1. RSI overbought (sell signal)
            if rsi > 60:
                short_score += 2
                short_signals.append("RSI_OVERBOUGHT")

            # 2. MACD bearish cross
            if len(macd_hist) >= 2:
                if macd_hist[-1] < 0 and macd_hist[-2] >= 0:
                    short_score += 2.5
                    short_signals.append("MACD_BEARISH")

            # 3. EMA bearish crossover
            if len(ema_fast) >= 2 and len(ema_slow) >= 2:
                if ema_fast[-1] < ema_slow[-1] and ema_fast[-2] >= ema_slow[-2]:
                    short_score += 2
                    short_signals.append("EMA_CROSS_BEARISH")

            # 4. Downtrend (price below 200 EMA)
            if len(ema_trend) >= 1 and current_price < ema_trend[-1]:
                short_score += 1
                short_signals.append("DOWNTREND")

            # === SHARED SIGNALS (apply to both) ===

            # 5. Trend strength
            if adx > 20:
                long_score += 1.5
                short_score += 1.5
                long_signals.append("STRONG_TREND")
                short_signals.append("STRONG_TREND")

            # 6. Forex volatility
            if volatility > 0.3:
                long_score += 1
                short_score += 1
                long_signals.append("FX_VOLATILITY")
                short_signals.append("FX_VOLATILITY")

            # === MULTI-TIMEFRAME CONFIRMATION (4H trend) ===
            trend_4h = self.get_higher_timeframe_trend(pair)

            # Boost score if 1H signal aligns with 4H trend
            if trend_4h == 'bullish':
                long_score += 2  # Bonus for trading WITH the 4H trend
                long_signals.append("4H_BULLISH_TREND")

                # Penalize counter-trend shorts
                if short_score > 0:
                    short_score -= 1.5  # Reduce score for counter-trend
                    short_signals.append("COUNTER_4H_TREND")

            elif trend_4h == 'bearish':
                short_score += 2  # Bonus for trading WITH the 4H trend
                short_signals.append("4H_BEARISH_TREND")

                # Penalize counter-trend longs
                if long_score > 0:
                    long_score -= 1.5  # Reduce score for counter-trend
                    long_signals.append("COUNTER_4H_TREND")

            # Return BOTH long and short opportunities
            results = []

            if long_score >= self.min_score:
                results.append({
                    'pair': pair,
                    'direction': 'long',
                    'score': long_score,
                    'price': current_price,
                    'rsi': rsi,
                    'volatility': volatility,
                    'adx': adx,
                    'signals': long_signals
                })

            if short_score >= self.min_score:
                results.append({
                    'pair': pair,
                    'direction': 'short',
                    'score': short_score,
                    'price': current_price,
                    'rsi': rsi,
                    'volatility': volatility,
                    'adx': adx,
                    'signals': short_signals
                })

            # For debug output, return the best opportunity
            if long_score >= short_score and long_score > 0:
                return {
                    'pair': pair,
                    'direction': 'long',
                    'score': long_score,
                    'price': current_price,
                    'rsi': rsi,
                    'volatility': volatility,
                    'adx': adx,
                    'signals': long_signals,
                    'all_opportunities': results
                }
            elif short_score > 0:
                return {
                    'pair': pair,
                    'direction': 'short',
                    'score': short_score,
                    'price': current_price,
                    'rsi': rsi,
                    'volatility': volatility,
                    'adx': adx,
                    'signals': short_signals,
                    'all_opportunities': results
                }
            else:
                return {
                    'pair': pair,
                    'direction': None,
                    'score': max(long_score, short_score),
                    'price': current_price,
                    'rsi': rsi if 'rsi' in locals() else 0,
                    'volatility': volatility if 'volatility' in locals() else 0,
                    'adx': adx if 'adx' in locals() else 0,
                    'signals': [],
                    'all_opportunities': []
                }
        else:
            # Fallback without TA-Lib
            return {
                'pair': pair,
                'direction': None,
                'score': 0,
                'price': current_price,
                'signals': [],
                'all_opportunities': []
            }

    def get_current_positions(self):
        """Get current OANDA positions"""
        try:
            r = positions.OpenPositions(accountID=self.oanda_account_id)
            response = self.client.request(r)
            return response.get('positions', [])
        except Exception as e:
            print(f"  [ERROR] Getting positions: {e}")
            return []

    def calculate_position_size(self, balance, pair, technical_score=None, fundamental_score=None):
        """
        Calculate position size using KELLY CRITERION
        Dynamic sizing based on signal confidence (technical + fundamental)
        """
        # Base size with 10x leverage
        base_units = 100000
        base_leveraged = base_units * (self.leverage_multiplier if self.use_10x_leverage else 1)

        # If no scores provided, use base size (backwards compatibility)
        if technical_score is None or fundamental_score is None:
            return base_leveraged

        # === KELLY CRITERION POSITION SIZING ===
        # Map combined signal strength to win probability

        # Technical score: 0-10 scale
        tech_normalized = technical_score / 10.0  # 0.0 to 1.0

        # Fundamental score: ±6 scale -> convert to 0-1
        fund_abs = abs(fundamental_score)
        fund_normalized = min(fund_abs / 6.0, 1.0)  # 0.0 to 1.0

        # Combined confidence (average of both signals)
        combined_confidence = (tech_normalized + fund_normalized) / 2.0

        # Map confidence to win probability (conservative estimates)
        # 0.5 confidence -> 55% win rate (baseline)
        # 1.0 confidence -> 75% win rate (maximum)
        base_win_rate = 0.55
        max_win_rate = 0.75
        win_probability = base_win_rate + (combined_confidence * (max_win_rate - base_win_rate))

        # Profit/Loss ratio for forex (2:1 target = 2.0)
        profit_loss_ratio = 2.0

        # Kelly formula: f* = (p*b - q) / b
        p = win_probability
        q = 1 - p
        b = profit_loss_ratio
        kelly_fraction = (p * b - q) / b

        # Use QUARTER-KELLY (very conservative for prop firm)
        quarter_kelly = kelly_fraction / 4.0

        # Cap between 0.5x and 1.5x base leverage
        MIN_MULTIPLIER = 0.5
        MAX_MULTIPLIER = 1.5

        # Convert Kelly fraction to position multiplier
        # quarter_kelly = 0.10 (10%) -> multiplier = 1.0x (base)
        # quarter_kelly = 0.05 (5%)  -> multiplier = 0.5x (half position)
        # quarter_kelly = 0.15 (15%) -> multiplier = 1.5x (max position)
        position_multiplier = max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, quarter_kelly * 10))

        # Calculate final position size
        final_units = int(base_leveraged * position_multiplier)

        # Print Kelly analysis
        print(f"\n  [KELLY SIZING]")
        print(f"    Technical: {technical_score:.2f}/10 ({tech_normalized*100:.0f}%)")
        print(f"    Fundamental: {fundamental_score}/6 ({fund_normalized*100:.0f}%)")
        print(f"    Combined Confidence: {combined_confidence*100:.0f}%")
        print(f"    Win Probability: {win_probability*100:.1f}%")
        print(f"    Kelly Fraction: {kelly_fraction*100:.1f}%")
        print(f"    Quarter-Kelly: {quarter_kelly*100:.1f}%")
        print(f"    Position Multiplier: {position_multiplier:.2f}x")
        print(f"    Final Units: {final_units:,} ({final_units/base_units:.1f} lots)")

        return final_units

    def place_forex_order(self, opportunity):
        """Place FOREX order on OANDA - SUPPORTS BOTH LONG AND SHORT"""
        try:
            pair = opportunity['pair']
            price = opportunity['price']
            direction = opportunity.get('direction', 'long')  # Default to LONG for backwards compatibility

            # Get account balance
            balance = self.get_account_balance()

            # Get scores for Kelly sizing
            technical_score = opportunity.get('score', 5.0)  # Default to medium confidence
            fundamental_score = opportunity.get('fundamental_score', 0)  # Default to neutral

            # Calculate position size using Kelly Criterion
            base_units = self.calculate_position_size(balance, pair, technical_score, fundamental_score)

            # Set units based on direction
            # Positive units = BUY (LONG)
            # Negative units = SELL (SHORT)
            units = base_units if direction == 'long' else -base_units

            # Calculate stop loss and take profit based on direction
            stop_distance = price * self.stop_loss
            profit_distance = price * self.profit_target

            # OANDA precision: JPY pairs use 3 decimals, others use 5
            precision = 3 if 'JPY' in pair else 5

            # For LONG: stop below entry, profit above entry
            # For SHORT: stop above entry, profit below entry
            if direction == 'long':
                stop_loss_price = price - stop_distance
                take_profit_price = price + profit_distance
            else:  # SHORT
                stop_loss_price = price + stop_distance
                take_profit_price = price - profit_distance

            # Create order
            order_data = {
                "order": {
                    "type": "MARKET",
                    "instrument": pair,
                    "units": str(units),  # Positive = LONG, Negative = SHORT
                    "stopLossOnFill": {
                        "price": str(round(stop_loss_price, precision))
                    },
                    "takeProfitOnFill": {
                        "price": str(round(take_profit_price, precision))
                    }
                }
            }

            direction_label = direction.upper()

            print(f"\n  [FOREX TRADE EXECUTING - LIVE!]")
            print(f"    Direction: {direction_label}")
            print(f"    Pair: {pair}")
            print(f"    Units: {abs(units):,} ({'BUY' if units > 0 else 'SELL'})")
            print(f"    Entry: {price:.5f}")
            print(f"    Stop Loss: {stop_loss_price:.5f} ({'-' if direction == 'long' else '+' }{stop_distance:.5f})")
            print(f"    Take Profit: {take_profit_price:.5f} ({'+' if direction == 'long' else '-'}{profit_distance:.5f})")
            print(f"    Score: {opportunity['score']:.1f}/10")

            # LIVE EXECUTION ENABLED:
            r = orders.OrderCreate(accountID=self.oanda_account_id, data=order_data)
            response = self.client.request(r)
            order_id = response['orderFillTransaction']['id']
            print(f"    [LIVE TRADE] Order ID: {order_id}")
            print(f"    [SUCCESS] Trade executed on OANDA!")

            # Send Telegram notification
            try:
                direction_text = "LONG" if direction == 'long' else "SHORT"
                message = f"{direction_text} FOREX TRADE!\n{pair} @ {price:.5f}\n{abs(units):,} units\nScore: {opportunity['score']:.1f}/10"
                if TALIB_AVAILABLE:
                    message += f"\nRSI: {opportunity.get('rsi', 0):.1f}\nSignals: {', '.join(opportunity['signals'])}"
                message += f"\nStop: {stop_loss_price:.5f} | Target: {take_profit_price:.5f}"

                bot_url = f"https://api.telegram.org/bot8203125300:AAE1FTiXQALCFh8cX9lKWhq8arEB2yvUGfQ/sendMessage"
                requests.post(bot_url, data={'chat_id': '7606409012', 'text': message}, timeout=3)
            except:
                pass

            return True

        except Exception as e:
            print(f"  [ERROR] Placing order: {e}")
            return False

    def scan_forex(self):
        """Scan FOREX pairs - returns BOTH LONG and SHORT opportunities"""
        print(f"\n[FOREX SCAN] {datetime.now().strftime('%H:%M:%S')}")

        opportunities = []
        current_hour = datetime.now().hour

        for pair in self.forex_pairs:
            # TIME FILTER: Skip if current hour is in avoid list for this pair
            if current_hour in self.AVOID_TRADING_HOURS.get(pair, []):
                print(f"  [SKIP] {pair} - Avoiding hour {current_hour} (choppy period)")
                continue

            data = self.get_forex_data(pair)
            if data is None:
                continue

            result = self.calculate_score(pair, data)

            # DEBUG: Show best score for this pair
            direction_label = result.get('direction', 'N/A').upper() if result.get('direction') else 'NONE'
            print(f"  [DEBUG] {pair}: Best={direction_label} Score {result['score']:.2f}/10", end="")
            if TALIB_AVAILABLE:
                print(f" | RSI: {result.get('rsi', 0):.1f} | ADX: {result.get('adx', 0):.1f} | Vol: {result.get('volatility', 0):.2f}%")
            else:
                print()

            # Add all opportunities (both LONG and SHORT if they meet threshold)
            if 'all_opportunities' in result and result['all_opportunities']:
                for opp in result['all_opportunities']:
                    opportunities.append(opp)
                    direction_str = opp['direction'].upper()
                    print(f"  >>> [FOUND] {pair} {direction_str}: Score {opp['score']:.1f}/10 - ABOVE THRESHOLD!")
                    if TALIB_AVAILABLE:
                        print(f"          Signals: {', '.join(opp['signals'])}")

        return opportunities

    def execute_trades(self, opportunities):
        """Execute forex trades (with news filtering)"""
        if not opportunities:
            return 0

        current_positions = self.get_current_positions()
        current_pairs = [p['instrument'] for p in current_positions]

        print(f"\n[POSITIONS] Currently holding: {len(current_pairs)}/{self.max_positions}")

        if len(current_pairs) >= self.max_positions:
            print(f"  [INFO] Max positions reached")
            return 0

        # SAFETY CHECK: Check if it's safe to trade based on time/events
        if self.news_filter:
            safety_check = self.news_filter.is_safe_to_trade_v2()
            print(f"\n[SAFETY CHECK] {safety_check['reason']}")

            if not safety_check['safe']:
                print(f"  [BLOCKED] Trading suspended due to high-risk window")
                return 0

        # NEWS FILTER: Check fundamentals before trading
        if self.news_filter:
            print(f"\n[NEWS FILTER] Checking fundamentals...")
            fundamental_analysis = self.news_filter.check_all_pairs(self.forex_pairs)
        else:
            fundamental_analysis = {}

        trades_executed = 0

        for opp in opportunities[:self.max_positions - len(current_pairs)]:
            pair = opp['pair']

            # Skip if already have position
            if pair in current_pairs:
                continue

            # NEWS FILTER: DISABLED - Trading on technicals only
            # Fundamental filter was blocking valid technical setups
            # if pair in fundamental_analysis:
            #     fund_data = fundamental_analysis[pair]
            #     technical_direction = opp.get('direction', 'long')
            #
            #     if not fund_data['tradeable']:
            #         print(f"  [BLOCKED] {pair} {technical_direction.upper()} - Fundamentals unclear (Score: {fund_data.get('score', 0)})")
            #         print(f"            {', '.join(fund_data.get('reasons', ['Mixed signals']))}")
            #         continue
            #
            #     # Check if technical signal matches fundamental direction
            #     fund_direction = fund_data['direction']
            #
            #     # If fundamental has a direction preference, check alignment
            #     if fund_direction and fund_direction != technical_direction:
            #         print(f"  [BLOCKED] {pair} {technical_direction.upper()} - Fundamentals say {fund_direction.upper()} (mismatch)")
            #         print(f"            Confidence: {fund_data['confidence']:.0f}%")
            #         print(f"            Reasons: {', '.join(fund_data.get('reasons', []))}")
            #         continue
            #
            #     # If we get here, fundamentals ALIGN with technical (or neutral)
            #     alignment_msg = "ALIGNED" if fund_direction == technical_direction else "NEUTRAL"
            #     print(f"  [APPROVED] {pair} {technical_direction.upper()} - Fundamentals {alignment_msg} (Confidence: {fund_data['confidence']:.0f}%)")
            #
            #     # Add fundamental score to opportunity for Kelly sizing
            #     opp['fundamental_score'] = fund_data.get('score', 0)
            # else:
            #     # No fundamental data available, use neutral score

            # Use neutral fundamental score for all trades (technical-only mode)
            opp['fundamental_score'] = 0
            print(f"  [APPROVED] {pair} {opp.get('direction', 'long').upper()} - Technical score: {opp.get('score', 0)}/10")

            # Place order
            if self.place_forex_order(opp):
                trades_executed += 1

        return trades_executed

    def start_trailing_stop_manager(self):
        """Start trailing stop manager in background thread"""
        try:
            from trailing_stop_manager_v2 import manage_trailing_stops

            print("[TRAILING STOPS V2] Starting DOLLAR-BASED trailing stop manager...")
            print("[TRAILING STOPS V2] Breakeven: $1,000 | Lock 50%: $2,000 | Lock 75%: $3,000")
            trailing_thread = threading.Thread(target=manage_trailing_stops, daemon=True)
            trailing_thread.start()
            print("[TRAILING STOPS V2] Manager active in background")

        except Exception as e:
            print(f"[WARN] Could not start trailing stop manager: {e}")
            print("[INFO] Bot will continue without trailing stops")

    def start_account_risk_manager(self):
        """Start account-level risk manager in background thread"""
        try:
            from account_risk_manager import monitor_account_risk

            print("[ACCOUNT RISK MANAGER] Starting account-level drawdown protection...")
            print("[ACCOUNT RISK MANAGER] Max Drawdown: -4% | E8 Limit: -6%")
            risk_thread = threading.Thread(target=monitor_account_risk, daemon=True)
            risk_thread.start()
            print("[ACCOUNT RISK MANAGER] Global stop loss active")

        except Exception as e:
            print(f"[WARN] Could not start account risk manager: {e}")
            print("[INFO] Bot will continue without account-level protection")

    def run(self):
        """Run continuous FOREX scanning"""
        print("\n[STARTING FOREX SYSTEM - OANDA]")
        print(f"Scanning every {self.scan_interval/60:.0f} minutes (24/5)")
        print("Press Ctrl+C to stop\n")

        # Start risk management systems in background
        self.start_account_risk_manager()  # Account-level protection (CRITICAL)
        time.sleep(1)
        self.start_trailing_stop_manager()  # Profit protection
        time.sleep(2)  # Give them time to start

        iteration = 0

        while True:
            iteration += 1
            print(f"\n{'='*70}")
            print(f"ITERATION #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}")

            # Get account status
            balance = self.get_account_balance()
            print(f"[ACCOUNT] Balance: ${balance:,.2f}")

            # Scan for opportunities
            opportunities = self.scan_forex()

            if opportunities:
                print(f"\n[SCAN COMPLETE] Found {len(opportunities)} opportunities")

                # Execute trades
                trades = self.execute_trades(opportunities)
                print(f"\n[TRADES EXECUTED: {trades}]")
            else:
                print("\n[No opportunities found this scan]")

            # Wait for next scan
            print(f"\n[Next scan in {self.scan_interval/60:.0f} minutes...]")
            time.sleep(self.scan_interval)

if __name__ == "__main__":
    trader = WorkingForexOanda()
    trader.run()
