"""
SIMPLE OPTIONS TRADER - Actually Works!
Uses yfinance for stock momentum, then BUYS OPTIONS (not shares)
"""
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import yfinance as yf
import alpaca_trade_api as tradeapi
import time

load_dotenv()

class SimpleOptionsTrader:
    def __init__(self):
        self.api = tradeapi.REST(
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # High-volume symbols with active options
        self.symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 'META']
        self.contracts_per_trade = 1  # 1 option contract (100 shares exposure)

        account = self.api.get_account()
        print("="*70)
        print("SIMPLE OPTIONS TRADER - LIVE")
        print("="*70)
        print(f"Account: ${float(account.equity):,.2f}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Strategy: BUY CALLS on positive momentum")
        print(f"Contracts: {self.contracts_per_trade} per trade (100 shares exposure)")
        print(f"Data: yfinance (FREE)")
        print(f"Execution: Alpaca Options (FREE paper trading)")
        print("="*70)

    def get_stock_data(self, symbol):
        """Get stock data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5d')

            if len(hist) > 0:
                return hist
        except Exception as e:
            print(f"  [ERROR] {symbol}: {e}")
        return None

    def analyze_momentum(self, symbol, data):
        """Calculate simple momentum"""
        try:
            close_prices = data['Close'].values
            volumes = data['Volume'].values

            momentum = (close_prices[-1] - close_prices[0]) / close_prices[0] * 100
            volume_ratio = volumes[-1] / volumes.mean()
            price = close_prices[-1]

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
                'price': price,
                'momentum': momentum,
                'volume_ratio': volume_ratio,
                'score': score
            }
        except Exception as e:
            print(f"  [ERROR] Analyzing {symbol}: {e}")
            return None

    def get_option_symbol(self, stock_symbol, stock_price):
        """Build option symbol for slightly OTM call
        Format: AAPL250117C00270000 (AAPL, Jan 17 2025, Call, $270 strike)
        """
        try:
            # Get expiration 2-4 weeks out
            exp_date = datetime.now() + timedelta(days=21)  # ~3 weeks
            exp_str = exp_date.strftime('%y%m%d')

            # Strike: 2-5% OTM (above current price)
            strike = round(stock_price * 1.03)  # 3% OTM

            # Build symbol: SYMBOL + YYMMDD + C/P + STRIKE (8 digits, padded with zeros, multiplied by 1000)
            strike_str = f"{int(strike * 1000):08d}"
            option_symbol = f"{stock_symbol}{exp_str}C{strike_str}"

            return option_symbol, strike

        except Exception as e:
            print(f"  [ERROR] Building option symbol: {e}")
            return None, None

    def execute_option_trade(self, signal):
        """Place OPTIONS order (not stock!)"""
        try:
            stock_symbol = signal['symbol']
            stock_price = signal['price']

            # Build option symbol
            option_symbol, strike = self.get_option_symbol(stock_symbol, stock_price)
            if not option_symbol:
                return False

            # Check if already have this option
            try:
                positions = self.api.list_positions()
                for pos in positions:
                    if option_symbol in pos.symbol:
                        print(f"  [SKIP] Already own {pos.qty} contracts of {option_symbol}")
                        return False
            except:
                pass

            # Place OPTION order
            order = self.api.submit_order(
                symbol=option_symbol,  # Option symbol, not stock symbol!
                qty=self.contracts_per_trade,
                side='buy',
                type='market',
                time_in_force='day',
                order_class='simple'
            )

            print(f"\n  *** OPTIONS TRADE EXECUTED ***")
            print(f"  Stock: {stock_symbol} @ ${stock_price:.2f}")
            print(f"  Option: {option_symbol}")
            print(f"  Strike: ${strike}")
            print(f"  Contracts: {self.contracts_per_trade} (={self.contracts_per_trade * 100} shares exposure)")
            print(f"  Score: {signal['score']}/10")
            print(f"  Momentum: {signal['momentum']:.2f}%")
            print(f"  Order ID: {order.id}")
            print(f"  ****************************\n")
            return True

        except Exception as e:
            print(f"  [ERROR] Executing option for {signal['symbol']}: {e}")
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

            # Execute OPTIONS if score > 5
            if signal['score'] > 5:
                print(f"\n  [SIGNAL] {symbol}: Score {signal['score']}/10, Momentum {signal['momentum']:.2f}%")
                if self.execute_option_trade(signal):
                    trades += 1

        print(f"\n[COMPLETE] Options trades executed: {trades}")
        return trades

    def run(self, scan_interval=300):
        """Continuous options trading"""
        print(f"\nStarting (scan every {scan_interval}s)...\n")

        iteration = 0
        total_trades = 0

        while True:
            try:
                iteration += 1
                print(f"\n{'='*70}")
                print(f"ITERATION #{iteration} - Total options trades today: {total_trades}")
                print(f"{'='*70}")

                trades = self.scan()
                total_trades += trades

                print(f"\n[NEXT SCAN] Waiting {scan_interval}s...")
                time.sleep(scan_interval)

            except KeyboardInterrupt:
                print(f"\n\nStopped. Total options trades: {total_trades}")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                time.sleep(60)


if __name__ == "__main__":
    trader = SimpleOptionsTrader()
    trader.run(scan_interval=300)  # Every 5 minutes
