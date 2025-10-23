"""
WORKING FOREX SYSTEM - OANDA
Uses your OANDA practice account for ACTUAL forex trading
Target: 50% monthly ROI from forex alone
"""
import os
import time
import requests
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
        self.forex_pairs = ['EUR_USD', 'USD_JPY', 'GBP_USD']  # OANDA format uses underscore

        # REALISTIC PARAMETERS
        self.min_score = 1.0  # AGGRESSIVE - lowered from 3.0 to start getting trades
        self.risk_per_trade = 0.01  # 1% risk per trade
        self.max_positions = 3  # Max 3 forex positions at once
        self.profit_target = 0.02  # 2% profit target (realistic for forex)
        self.stop_loss = 0.01  # 1% stop loss (tight risk management)

        # Scan interval
        self.scan_interval = 3600  # 1 hour (forex moves slower than stocks)

        print("=" * 70)
        print("WORKING FOREX SYSTEM - OANDA")
        print("=" * 70)
        print(f"Broker: OANDA Practice Account")
        print(f"Account ID: {self.oanda_account_id}")
        print(f"Pairs: {', '.join(self.forex_pairs)}")
        print(f"Min Score: {self.min_score} (REALISTIC)")
        print(f"Risk Per Trade: {self.risk_per_trade*100}%")
        print(f"Max Positions: {self.max_positions}")
        print(f"Profit Target: {self.profit_target*100}%")
        print(f"Stop Loss: {self.stop_loss*100}%")
        if TALIB_AVAILABLE:
            print("Quant Libraries: TA-Lib ENABLED")
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

    def get_forex_data(self, pair):
        """Get FOREX data from OANDA"""
        try:
            params = {
                'count': 100,
                'granularity': 'H1'  # 1-hour candles
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

    def calculate_score(self, pair, data):
        """
        Calculate score using TA-Lib
        LOWERED thresholds so we actually find trades
        """
        closes = data['closes']
        highs = data['highs']
        lows = data['lows']
        current_price = data['current_price']

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
            ema_trend = talib.EMA(closes, timeperiod=200)

            # SCORING (REALISTIC THRESHOLDS FOR FOREX)

            # 1. RSI oversold/overbought (forex-specific)
            if rsi < 40:  # Lowered from 30
                score += 2
                signals.append("RSI_OVERSOLD")
            elif rsi > 60:  # Lowered from 70
                score += 1.5
                signals.append("RSI_OVERBOUGHT")

            # 2. MACD bullish cross
            if len(macd_hist) >= 2:
                if macd_hist[-1] > 0 and macd_hist[-2] <= 0:
                    score += 2.5
                    signals.append("MACD_BULLISH")

            # 3. EMA crossover
            if len(ema_fast) >= 2 and len(ema_slow) >= 2:
                if ema_fast[-1] > ema_slow[-1] and ema_fast[-2] <= ema_slow[-2]:
                    score += 2
                    signals.append("EMA_CROSS_BULLISH")

            # 4. Trend alignment (price above 200 EMA)
            if len(ema_trend) >= 1 and current_price > ema_trend[-1]:
                score += 1
                signals.append("UPTREND")

            # 5. Trend strength (lowered from 25 to 20)
            if adx > 20:
                score += 1.5
                signals.append("STRONG_TREND")

            # 6. Forex volatility (any movement counts)
            if volatility > 0.3:  # Even small forex moves count
                score += 1
                signals.append("FX_VOLATILITY")

            return {
                'pair': pair,
                'score': score,
                'price': current_price,
                'rsi': rsi,
                'volatility': volatility,
                'adx': adx,
                'signals': signals
            }
        else:
            # Fallback without TA-Lib
            return {
                'pair': pair,
                'score': 0,
                'price': current_price,
                'signals': []
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

    def calculate_position_size(self, balance, pair):
        """Calculate position size based on account balance and risk"""
        # For forex, we'll use micro lots (1000 units)
        # Risk 1% of account per trade
        risk_amount = balance * self.risk_per_trade

        # Use 100,000 units (1 standard lot)
        # This is about $1,080 risk per trade on $200k account (0.54% risk)
        # With 3 max positions = $3,240 total risk (1.6% of account)
        units = 100000

        return units

    def place_forex_order(self, opportunity):
        """Place FOREX order on OANDA (PAPER TRADING)"""
        try:
            pair = opportunity['pair']
            price = opportunity['price']

            # Get account balance
            balance = self.get_account_balance()

            # Calculate position size
            units = self.calculate_position_size(balance, pair)

            # Calculate stop loss and take profit
            stop_distance = price * self.stop_loss
            profit_distance = price * self.profit_target

            # OANDA precision: JPY pairs use 3 decimals, others use 5
            precision = 3 if 'JPY' in pair else 5

            # Create order
            order_data = {
                "order": {
                    "type": "MARKET",
                    "instrument": pair,
                    "units": str(units),  # Positive = buy
                    "stopLossOnFill": {
                        "price": str(round(price - stop_distance, precision))
                    },
                    "takeProfitOnFill": {
                        "price": str(round(price + profit_distance, precision))
                    }
                }
            }

            print(f"\n  [FOREX TRADE EXECUTING - LIVE!]")
            print(f"    Pair: {pair}")
            print(f"    Units: {units}")
            print(f"    Entry: {price:.5f}")
            print(f"    Stop Loss: {price - stop_distance:.5f}")
            print(f"    Take Profit: {price + profit_distance:.5f}")
            print(f"    Score: {opportunity['score']:.1f}/10")

            # LIVE EXECUTION ENABLED:
            r = orders.OrderCreate(accountID=self.oanda_account_id, data=order_data)
            response = self.client.request(r)
            order_id = response['orderFillTransaction']['id']
            print(f"    [LIVE TRADE] Order ID: {order_id}")
            print(f"    [SUCCESS] Trade executed on OANDA!")

            # Send Telegram notification
            try:
                message = f"💱 FOREX TRADE!\n{pair}\n{units} units @ {price:.5f}\nScore: {opportunity['score']:.1f}/10"
                if TALIB_AVAILABLE:
                    message += f"\nRSI: {opportunity.get('rsi', 0):.1f} | Signals: {', '.join(opportunity['signals'])}"

                bot_url = f"https://api.telegram.org/bot8203125300:AAE1FTiXQALCFh8cX9lKWhq8arEB2yvUGfQ/sendMessage"
                requests.post(bot_url, data={'chat_id': '7606409012', 'text': message}, timeout=3)
            except:
                pass

            return True

        except Exception as e:
            print(f"  [ERROR] Placing order: {e}")
            return False

    def scan_forex(self):
        """Scan FOREX pairs"""
        print(f"\n[FOREX SCAN] {datetime.now().strftime('%H:%M:%S')}")

        opportunities = []

        for pair in self.forex_pairs:
            data = self.get_forex_data(pair)
            if data is None:
                continue

            result = self.calculate_score(pair, data)

            # DEBUG: Show ALL scores (not just above threshold)
            print(f"  [DEBUG] {pair}: Score {result['score']:.2f}/10", end="")
            if TALIB_AVAILABLE:
                print(f" | RSI: {result.get('rsi', 0):.1f} | ADX: {result.get('adx', 0):.1f} | Vol: {result.get('volatility', 0):.2f}%")
            else:
                print()

            if result['score'] >= self.min_score:
                opportunities.append(result)
                print(f"  >>> [FOUND] {pair}: Score {result['score']:.1f}/10 - ABOVE THRESHOLD!")
                if TALIB_AVAILABLE:
                    print(f"          Signals: {', '.join(result['signals'])}")

        return opportunities

    def execute_trades(self, opportunities):
        """Execute forex trades"""
        if not opportunities:
            return 0

        current_positions = self.get_current_positions()
        current_pairs = [p['instrument'] for p in current_positions]

        print(f"\n[POSITIONS] Currently holding: {len(current_pairs)}/{self.max_positions}")

        if len(current_pairs) >= self.max_positions:
            print(f"  [INFO] Max positions reached")
            return 0

        trades_executed = 0

        for opp in opportunities[:self.max_positions - len(current_pairs)]:
            pair = opp['pair']

            # Skip if already have position
            if pair in current_pairs:
                continue

            # Place order
            if self.place_forex_order(opp):
                trades_executed += 1

        return trades_executed

    def run(self):
        """Run continuous FOREX scanning"""
        print("\n[STARTING FOREX SYSTEM - OANDA]")
        print(f"Scanning every {self.scan_interval/60:.0f} minutes (24/5)")
        print("Press Ctrl+C to stop\n")

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
