"""
SIMPLE WORKING TRADER - Actually executes trades
Fixed data fetching, simple momentum strategy
"""
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import time

load_dotenv()

class SimpleTrader:
    def __init__(self):
        self.api = tradeapi.REST(
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # High-volume symbols
        self.symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMD', 'META']
        self.position_size = 1  # 1 share per trade (safe)

        account = self.api.get_account()
        print("="*70)
        print("SIMPLE WORKING TRADER - LIVE")
        print("="*70)
        print(f"Account: ${float(account.equity):,.2f}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Position Size: {self.position_size} shares")
        print(f"Threshold: ANY positive momentum")
        print("="*70)

    def get_data(self, symbol):
        """Get market data with CORRECT date format"""
        try:
            # Fix: Use .date() to get just YYYY-MM-DD format
            end = datetime.now().date()
            start = (datetime.now() - timedelta(days=5)).date()

            bars = self.api.get_bars(
                symbol,
                '1Day',
                start=start.isoformat(),  # Now returns "2025-10-16" not "2025-10-16T07:10..."
                end=end.isoformat()
            ).df

            if len(bars) > 0:
                return bars
        except Exception as e:
            print(f"  [ERROR] {symbol}: {e}")
        return None

    def analyze(self, symbol, bars):
        """Simple momentum + volume analysis"""
        try:
            close_prices = bars['close'].values
            volumes = bars['volume'].values

            # Calculate 5-day momentum
            momentum = (close_prices[-1] - close_prices[0]) / close_prices[0] * 100

            # Volume surge
            volume_ratio = volumes[-1] / volumes.mean()

            # Simple score
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
        """Actually place the trade"""
        try:
            # Check if already have position
            try:
                position = self.api.get_position(signal['symbol'])
                print(f"  [SKIP] Already have {position.qty} {signal['symbol']}")
                return False
            except:
                pass  # No position, good to trade

            # Place market order
            order = self.api.submit_order(
                symbol=signal['symbol'],
                qty=self.position_size,
                side='buy',
                type='market',
                time_in_force='day'
            )

            print(f"  [EXECUTED] BUY {self.position_size} {signal['symbol']} @ ${signal['price']:.2f}")
            print(f"             Order ID: {order.id}")
            print(f"             Score: {signal['score']}/10, Momentum: {signal['momentum']:.2f}%")
            return True

        except Exception as e:
            print(f"  [ERROR] Executing {signal['symbol']}: {e}")
            return False

    def scan(self):
        """Scan all symbols and execute trades"""
        print(f"\n[SCAN] {datetime.now().strftime('%H:%M:%S')}")

        trades_executed = 0

        for symbol in self.symbols:
            bars = self.get_data(symbol)
            if bars is None:
                continue

            signal = self.analyze(symbol, bars)
            if signal is None:
                continue

            # AGGRESSIVE: Execute if score > 5
            if signal['score'] > 5:
                print(f"\n  [SIGNAL] {symbol}: Score {signal['score']}/10")
                if self.execute_trade(signal):
                    trades_executed += 1

        print(f"\n[COMPLETE] Trades executed: {trades_executed}")
        return trades_executed

    def run(self, scan_interval=120):
        """Run continuous trading"""
        print(f"\nStarting continuous trading (scan every {scan_interval}s)...")
        print("Press Ctrl+C to stop\n")

        iteration = 0
        while True:
            try:
                iteration += 1
                print(f"\n{'='*70}")
                print(f"ITERATION #{iteration}")
                print(f"{'='*70}")

                self.scan()

                print(f"\n[WAITING] Next scan in {scan_interval}s...")
                time.sleep(scan_interval)

            except KeyboardInterrupt:
                print("\n\nStopped by user")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                time.sleep(60)


if __name__ == "__main__":
    trader = SimpleTrader()
    trader.run(scan_interval=120)  # Scan every 2 minutes
