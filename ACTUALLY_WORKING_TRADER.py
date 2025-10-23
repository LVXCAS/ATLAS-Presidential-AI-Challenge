"""
ACTUALLY WORKING TRADER
Uses FREE yfinance data + Alpaca execution
Will actually execute trades!
"""
import os
from dotenv import load_dotenv
from datetime import datetime
import yfinance as yf
import alpaca_trade_api as tradeapi
import time

load_dotenv()

class WorkingTrader:
    def __init__(self):
        # Alpaca for execution ONLY
        self.api = tradeapi.REST(
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # High-volume symbols
        self.symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA', 'META']
        self.position_size = 1  # 1 share (safe)

        account = self.api.get_account()
        print("="*70)
        print("ACTUALLY WORKING TRADER - YFINANCE DATA + ALPACA EXECUTION")
        print("="*70)
        print(f"Account: ${float(account.equity):,.2f}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Position Size: {self.position_size} shares")
        print(f"Data Source: yfinance (FREE)")
        print(f"Execution: Alpaca Paper Trading")
        print("="*70)

    def get_data(self, symbol):
        """Get data from yfinance (FREE, no subscription needed)"""
        try:
            ticker = yf.Ticker(symbol)
            # Get 5 days of data
            hist = ticker.history(period='5d')

            if len(hist) > 0:
                return hist
        except Exception as e:
            print(f"  [ERROR] {symbol}: {e}")
        return None

    def analyze(self, symbol, data):
        """Simple momentum analysis"""
        try:
            close_prices = data['Close'].values
            volumes = data['Volume'].values

            # 5-day momentum
            momentum = (close_prices[-1] - close_prices[0]) / close_prices[0] * 100

            # Volume surge
            volume_ratio = volumes[-1] / volumes.mean()

            # Simple scoring
            score = 0
            if momentum > 0:
                score += 5
            if momentum > 1:
                score += 2
            if volume_ratio > 1.2:
                score += 3

            return {
                'symbol': symbol,
                'price': close_prices[-1],
                'momentum': momentum,
                'volume_ratio': volume_ratio,
                'score': score
            }
        except Exception as e:
            print(f"  [ERROR] Analyzing {symbol}: {e}")
            return None

    def execute_trade(self, signal):
        """Place the trade via Alpaca"""
        try:
            # Check if already have position
            try:
                position = self.api.get_position(signal['symbol'])
                print(f"  [SKIP] Already own {position.qty} {signal['symbol']}")
                return False
            except:
                pass  # No position

            # Submit order
            order = self.api.submit_order(
                symbol=signal['symbol'],
                qty=self.position_size,
                side='buy',
                type='market',
                time_in_force='day'
            )

            print(f"\n  *** TRADE EXECUTED ***")
            print(f"  Symbol: {signal['symbol']}")
            print(f"  Qty: {self.position_size} shares")
            print(f"  Price: ~${signal['price']:.2f}")
            print(f"  Score: {signal['score']}/10")
            print(f"  Momentum: {signal['momentum']:.2f}%")
            print(f"  Order ID: {order.id}")
            print(f"  *********************\n")
            return True

        except Exception as e:
            print(f"  [ERROR] Executing {signal['symbol']}: {e}")
            return False

    def scan(self):
        """Scan all symbols"""
        print(f"\n[SCAN] {datetime.now().strftime('%H:%M:%S')}")

        trades = 0

        for symbol in self.symbols:
            # Get data from yfinance
            data = self.get_data(symbol)
            if data is None:
                continue

            # Analyze
            signal = self.analyze(symbol, data)
            if signal is None:
                continue

            # Execute if score > 5
            if signal['score'] > 5:
                print(f"\n  [SIGNAL] {symbol}: Score {signal['score']}/10, Momentum {signal['momentum']:.2f}%")
                if self.execute_trade(signal):
                    trades += 1

        print(f"\n[COMPLETE] Trades executed this scan: {trades}")
        return trades

    def run(self, scan_interval=180):
        """Continuous trading"""
        print(f"\nStarting (scan every {scan_interval}s)...\n")

        iteration = 0
        total_trades = 0

        while True:
            try:
                iteration += 1
                print(f"\n{'='*70}")
                print(f"ITERATION #{iteration} - Total trades today: {total_trades}")
                print(f"{'='*70}")

                trades = self.scan()
                total_trades += trades

                print(f"\n[NEXT SCAN] Waiting {scan_interval}s...")
                time.sleep(scan_interval)

            except KeyboardInterrupt:
                print(f"\n\nStopped. Total trades: {total_trades}")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                time.sleep(60)


if __name__ == "__main__":
    trader = WorkingTrader()
    trader.run(scan_interval=180)  # Every 3 minutes
