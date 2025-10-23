"""
OPENBB OPTIONS TRADER - FAST & FREE
Uses OpenBB for quick stock data, then executes OPTIONS on Alpaca
"""
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from openbb import obb
import alpaca_trade_api as tradeapi
import time

load_dotenv()

class OpenBBOptionsTrader:
    def __init__(self):
        self.api = tradeapi.REST(
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # High-liquidity symbols
        self.symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD']
        self.contracts_per_trade = 1

        account = self.api.get_account()
        print("="*70)
        print("OPENBB OPTIONS TRADER - LIVE")
        print("="*70)
        print(f"Account: ${float(account.equity):,.2f}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Strategy: BUY CALLS on momentum")
        print(f"Contracts: {self.contracts_per_trade} (100 shares exposure each)")
        print(f"Data: OpenBB (FAST & FREE)")
        print(f"Execution: Alpaca Options Paper Trading")
        print("="*70)

    def get_stock_data(self, symbol):
        """Get stock data from OpenBB"""
        try:
            # Use OpenBB to get historical data
            data = obb.equity.price.historical(
                symbol=symbol,
                start_date=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
                provider='yfinance'  # OpenBB wraps yfinance but optimized
            )

            if data and hasattr(data, 'results') and data.results:
                return data.results
            return None
        except Exception as e:
            print(f"  [ERROR] {symbol} data fetch: {e}")
            return None

    def analyze_momentum(self, symbol, data):
        """Calculate momentum from OpenBB data"""
        try:
            # Convert OpenBB results to list
            bars = list(data)
            if len(bars) < 2:
                return None

            # Get price and volume info
            first_close = bars[0].close
            last_close = bars[-1].close
            last_volume = bars[-1].volume
            avg_volume = sum(b.volume for b in bars) / len(bars)

            momentum = (last_close - first_close) / first_close * 100
            volume_ratio = last_volume / avg_volume if avg_volume > 0 else 1

            # Score
            score = 0
            if momentum > 0:
                score += 5
            if momentum > 1:
                score += 2
            if volume_ratio > 1.2:
                score += 3

            return {
                'symbol': symbol,
                'price': last_close,
                'momentum': momentum,
                'volume_ratio': volume_ratio,
                'score': score
            }
        except Exception as e:
            print(f"  [ERROR] Analyzing {symbol}: {e}")
            return None

    def get_option_symbol(self, stock_symbol, stock_price):
        """Build option symbol - 3 weeks out, 3% OTM call"""
        try:
            exp_date = datetime.now() + timedelta(days=21)
            exp_str = exp_date.strftime('%y%m%d')

            strike = round(stock_price * 1.03)
            strike_str = f"{int(strike * 1000):08d}"

            option_symbol = f"{stock_symbol}{exp_str}C{strike_str}"
            return option_symbol, strike
        except Exception as e:
            print(f"  [ERROR] Building option symbol: {e}")
            return None, None

    def execute_option_trade(self, signal):
        """Place OPTIONS order"""
        try:
            option_symbol, strike = self.get_option_symbol(signal['symbol'], signal['price'])
            if not option_symbol:
                return False

            # Check existing positions
            try:
                positions = self.api.list_positions()
                for pos in positions:
                    if option_symbol in pos.symbol:
                        print(f"  [SKIP] Already own {pos.qty} {option_symbol}")
                        return False
            except:
                pass

            # Place order
            order = self.api.submit_order(
                symbol=option_symbol,
                qty=self.contracts_per_trade,
                side='buy',
                type='market',
                time_in_force='day'
            )

            print(f"\n  *** OPTIONS TRADE EXECUTED ***")
            print(f"  Stock: {signal['symbol']} @ ${signal['price']:.2f}")
            print(f"  Option: {option_symbol}")
            print(f"  Strike: ${strike}")
            print(f"  Contracts: {self.contracts_per_trade} (100 shares)")
            print(f"  Score: {signal['score']}/10, Momentum: {signal['momentum']:.2f}%")
            print(f"  Order ID: {order.id}")
            print(f"  ****************************\n")
            return True

        except Exception as e:
            print(f"  [ERROR] Executing {signal['symbol']}: {e}")
            return False

    def scan(self):
        """Scan stocks, buy OPTIONS on winners"""
        print(f"\n[SCAN] {datetime.now().strftime('%H:%M:%S')}")

        trades = 0

        for symbol in self.symbols:
            data = self.get_stock_data(symbol)
            if data is None:
                continue

            signal = self.analyze_momentum(symbol, data)
            if signal is None:
                continue

            if signal['score'] > 5:
                print(f"\n  [SIGNAL] {symbol}: Score {signal['score']}/10, Momentum {signal['momentum']:.2f}%")
                if self.execute_option_trade(signal):
                    trades += 1

        print(f"\n[COMPLETE] Options trades: {trades}")
        return trades

    def run(self, scan_interval=300):
        """Continuous options trading"""
        print(f"\nStarting continuous trading (every {scan_interval}s)...\n")

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

                print(f"\n[WAITING] Next scan in {scan_interval}s...")
                time.sleep(scan_interval)

            except KeyboardInterrupt:
                print(f"\n\nStopped. Total options trades: {total_trades}")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                import traceback
                traceback.print_exc()
                time.sleep(60)


if __name__ == "__main__":
    trader = OpenBBOptionsTrader()
    trader.run(scan_interval=300)
