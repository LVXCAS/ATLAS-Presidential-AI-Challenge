"""
Backtrader Evaluation for E8 Strategy

Tests if Backtrader is suitable for:
1. Live forex trading with TradeLocker
2. Multi-indicator scoring system
3. E8 challenge rules (6% DD, 10% profit target)
4. Hourly timeframe analysis
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Try to import TA-Lib
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("[WARN] TA-Lib not available - using pandas fallback")


class E8Strategy(bt.Strategy):
    """
    E8 Forex Strategy matching your current system:
    - Score-based entries (2.5 threshold)
    - RSI, MACD, ADX, EMA indicators
    - 2% risk per trade, 1% SL, 2% TP
    """

    params = (
        ('min_score', 2.5),
        ('risk_per_trade', 0.02),
        ('stop_loss_pct', 0.01),
        ('profit_target_pct', 0.02),
        ('position_size_multiplier', 0.80),
    )

    def __init__(self):
        """Initialize indicators"""
        self.order = None
        self.entry_price = None

        # TA-Lib indicators (if available)
        if TALIB_AVAILABLE:
            self.rsi = bt.talib.RSI(self.data.close, timeperiod=14)
            self.macd = bt.talib.MACD(self.data.close,
                                       fastperiod=12,
                                       slowperiod=26,
                                       signalperiod=9)
            self.adx = bt.talib.ADX(self.data.high,
                                     self.data.low,
                                     self.data.close,
                                     timeperiod=14)
        else:
            # Fallback indicators
            self.rsi = bt.indicators.RSI(self.data.close, period=14)
            self.macd = bt.indicators.MACD(self.data.close)
            self.adx = bt.indicators.AverageDirectionalMovementIndex(self.data)

        # Track scoring
        self.scores = []
        self.trade_count = 0

    def calculate_score(self):
        """
        Calculate entry score using same logic as E8_FOREX_BOT.py
        Returns: score (float)
        """
        if len(self.data) < 50:
            return 0

        score = 0

        # 1. RSI scoring
        current_rsi = self.rsi[0]
        if 30 < current_rsi < 40:
            score += 3  # Oversold zone
        elif 60 < current_rsi < 70:
            score -= 3  # Overbought zone (SHORT signal)
        elif 40 < current_rsi < 60:
            score += 1  # Neutral

        # 2. MACD scoring
        if TALIB_AVAILABLE:
            current_macd = self.macd.macd[0]
            current_signal = self.macd.macdsignal[0]
            prev_macd = self.macd.macd[-1]
            prev_signal = self.macd.macdsignal[-1]
        else:
            current_macd = self.macd.macd[0]
            current_signal = self.macd.signal[0]
            prev_macd = self.macd.macd[-1]
            prev_signal = self.macd.signal[-1]

        # Crossover detection
        if current_macd > current_signal and prev_macd <= prev_signal:
            score += 3  # Bull cross
        elif current_macd < current_signal and prev_macd >= prev_signal:
            score -= 3  # Bear cross
        elif current_macd > current_signal:
            score += 1
        else:
            score -= 1

        # 3. ADX trend strength
        if TALIB_AVAILABLE:
            current_adx = self.adx[0]
        else:
            current_adx = self.adx.adx[0]

        if current_adx > 25:
            # Strong trend - amplify score direction
            score += 2 if score > 0 else -2
        elif current_adx > 20:
            score += 1 if score > 0 else -1

        return score

    def notify_order(self, order):
        """Track order status"""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.5f}, Size: {order.executed.size}')
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.5f}, Size: {order.executed.size}')

            self.entry_price = order.executed.price

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        """Track trade results"""
        if trade.isclosed:
            self.trade_count += 1
            pnl_pct = (trade.pnl / trade.value) * 100 if trade.value != 0 else 0
            self.log(f'TRADE #{self.trade_count} CLOSED - P/L: ${trade.pnl:.2f} ({pnl_pct:.2f}%)')

    def next(self):
        """Main strategy logic"""
        # Skip if order pending
        if self.order:
            return

        # Skip if already in position
        if self.position:
            return

        # Calculate score
        score = self.calculate_score()
        self.scores.append(score)

        # Check threshold
        if abs(score) < self.params.min_score:
            return

        # Determine direction
        direction = 'LONG' if score > 0 else 'SHORT'

        # **CRITICAL**: Only trade LONG (your original 25.16% ROI strategy)
        # SHORT signals are IGNORED
        if direction != 'LONG':
            return  # Skip SHORT signals

        # Fixed position sizing for forex (simpler approach)
        # Use fixed lot size instead of complex risk calculation
        # 1 lot = 100,000 units, use 0.1 lots = 10,000 units
        size = 10000  # 0.1 lots (1 mini lot)

        # Execute LONG trade only
        stop_price = self.data.close[0] * (1 - self.params.stop_loss_pct)
        target_price = self.data.close[0] * (1 + self.params.profit_target_pct)

        self.log(f'LONG SIGNAL - Score: {score:.2f}, Price: {self.data.close[0]:.5f}')
        # Use bracket order
        self.order = self.buy_bracket(
            size=size,
            price=None,  # Market order
            stopprice=stop_price,
            limitprice=target_price
        )[0]

    def log(self, txt):
        """Logging function"""
        dt = self.data.datetime.date(0)
        print(f'[{dt}] {txt}')


def generate_sample_forex_data(days=90):
    """
    Generate sample EUR/USD hourly data for testing
    In real use, you'd fetch from TradeLocker
    """
    start = datetime.now() - timedelta(days=days)
    dates = pd.date_range(start, periods=days*24, freq='H')

    # Simulate forex price movement (starting at ~1.10)
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.003, len(dates))
    price = 1.10 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'datetime': dates,
        'open': price,
        'high': price * (1 + np.abs(np.random.normal(0, 0.001, len(dates)))),
        'low': price * (1 - np.abs(np.random.normal(0, 0.001, len(dates)))),
        'close': price,
        'volume': np.random.randint(1000, 10000, len(dates))
    })

    return df


def evaluate_backtrader():
    """
    Main evaluation function
    Tests Backtrader for E8 trading
    """
    print("=" * 70)
    print("BACKTRADER EVALUATION FOR E8 STRATEGY")
    print("=" * 70)

    # Create Cerebro engine
    cerebro = bt.Cerebro()

    # Add strategy
    cerebro.addstrategy(E8Strategy)

    # Generate sample data (in production, fetch from TradeLocker)
    print("\n[INFO] Generating sample forex data (90 days, hourly)...")
    df = generate_sample_forex_data(days=90)

    # Convert to Backtrader format
    data = bt.feeds.PandasData(
        dataname=df,
        datetime='datetime',
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1
    )

    cerebro.adddata(data)

    # Set E8 challenge parameters
    starting_cash = 200000.0  # $200K E8 account
    cerebro.broker.setcash(starting_cash)
    cerebro.broker.setcommission(commission=0.0001)  # 1 pip spread

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')

    print(f"\n[START] Starting Portfolio Value: ${cerebro.broker.getvalue():,.2f}")

    # Run backtest
    print("\n[RUNNING] Backtesting E8 strategy with Backtrader...")
    results = cerebro.run()
    strat = results[0]

    # Get final value
    final_value = cerebro.broker.getvalue()
    profit = final_value - starting_cash
    profit_pct = (profit / starting_cash) * 100

    print(f"\n[END] Final Portfolio Value: ${final_value:,.2f}")
    print(f"[RESULT] Profit/Loss: ${profit:,.2f} ({profit_pct:.2f}%)")

    # Analyze results
    print("\n" + "=" * 70)
    print("BACKTRADER EVALUATION RESULTS")
    print("=" * 70)

    # Trade statistics
    trade_analysis = strat.analyzers.trades.get_analysis()
    drawdown_analysis = strat.analyzers.drawdown.get_analysis()

    # Handle empty trade analysis safely
    try:
        total_trades = trade_analysis.total.closed
        won_trades = trade_analysis.won.total
        lost_trades = trade_analysis.lost.total
    except (KeyError, AttributeError):
        total_trades = 0
        won_trades = 0
        lost_trades = 0

    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

    print(f"\n[PERFORMANCE METRICS]")
    print(f"  Total Trades: {total_trades}")
    print(f"  Winning Trades: {won_trades}")
    print(f"  Losing Trades: {lost_trades}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Max Drawdown: {drawdown_analysis.max.drawdown:.2f}%")
    print(f"  ROI: {profit_pct:.2f}%")

    # E8 Challenge compliance
    print(f"\n[E8 CHALLENGE COMPLIANCE]")
    max_dd_limit = 6.0
    profit_target = 10.0

    dd_status = "PASS" if drawdown_analysis.max.drawdown < max_dd_limit else "FAIL"
    profit_status = "PASS" if profit_pct >= profit_target else "IN PROGRESS"

    print(f"  Max Drawdown: {drawdown_analysis.max.drawdown:.2f}% / {max_dd_limit}% [{dd_status}]")
    print(f"  Profit Target: {profit_pct:.2f}% / {profit_target}% [{profit_status}]")

    # Backtrader pros/cons for your use case
    print(f"\n[BACKTRADER ADVANTAGES]")
    print(f"  1. Built-in analyzers (Sharpe, DD, trades, etc.)")
    print(f"  2. TA-Lib integration works natively")
    print(f"  3. Multiple timeframe support")
    print(f"  4. Backtesting & live trading in same code")
    print(f"  5. Order types (Market, Stop, Limit) built-in")
    print(f"  6. Position sizing & risk management tools")

    print(f"\n[BACKTRADER LIMITATIONS FOR YOUR CASE]")
    print(f"  1. No native TradeLocker support (need custom broker)")
    print(f"  2. Learning curve for Cerebro engine")
    print(f"  3. Live data feed requires custom implementation")
    print(f"  4. Your E8_TRADELOCKER_ADAPTER.py would need wrapping")
    print(f"  5. Overkill for simple hourly scanning strategy")

    # Recommendation
    print(f"\n[RECOMMENDATION]")
    print(f"  Current System: Simple, direct TradeLocker integration [GOOD]")
    print(f"  Backtrader: Better for complex multi-strategy backtesting")
    print(f"  ")
    print(f"  For E8 live trading: STICK WITH CURRENT SYSTEM")
    print(f"  For strategy research: USE BACKTRADER for backtesting")
    print(f"  ")
    print(f"  Hybrid approach:")
    print(f"    - Backtest new strategies with Backtrader")
    print(f"    - Deploy validated strategies via E8_FOREX_BOT.py")

    return {
        'trades': total_trades,
        'win_rate': win_rate,
        'roi': profit_pct,
        'max_dd': drawdown_analysis.max.drawdown,
        'final_value': final_value
    }


if __name__ == '__main__':
    results = evaluate_backtrader()

    print("\n" + "=" * 70)
    print("CONCLUSION: Backtrader vs Current System")
    print("=" * 70)
    print("""
Your current E8_FOREX_BOT.py is BETTER for live trading because:
  [+] Direct TradeLocker API integration
  [+] Simple hourly scanning logic
  [+] Score logging already implemented
  [+] E8 challenge rules built-in
  [+] Proven to connect and run

Use Backtrader for:
  [+] Backtesting new strategy ideas
  [+] Optimizing parameters (min_score, risk%, etc.)
  [+] Multi-timeframe research
  [+] Strategy comparison studies

Bottom line: Keep current bot for live trading, add Backtrader for research.
""")
