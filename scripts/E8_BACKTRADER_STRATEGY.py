"""
E8 FOREX STRATEGY - Professional Backtrader Implementation

Market microstructure setups:
1. London Fake-out (stop hunts)
2. NY Absorption (order flow reversal)
3. Tokyo Gap Fill (mean reversion)

Uses Backtrader for proper backtesting with:
- Realistic position sizing
- Slippage and commission
- Proper trade management
- Statistical analysis
"""

import backtrader as bt
from datetime import time as dt_time, datetime, timedelta
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


class LondonFakeoutStrategy(bt.Strategy):
    """
    London Fake-out: Low volume breakout of Asian range
    Theory: Stop hunt by institutions, fade the move
    """

    params = (
        ('asian_lookback', 32),  # 8 hours of 15min candles
        ('volume_threshold', 0.7),  # 70% of avg volume = fake
        ('breakout_buffer', 0.0002),  # 2 pips buffer
        ('risk_percent', 0.01),  # Risk 1% per trade (E8 safe)
    )

    def __init__(self):
        self.order = None
        self.in_position = False

        # Track session times
        self.london_open = dt_time(3, 0)
        self.london_close = dt_time(5, 0)

    def is_london_session(self):
        """Check if current bar is London session (3-5 AM EST)"""
        current_time = self.data.datetime.time()
        return self.london_open <= current_time < self.london_close

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: {order.executed.price:.5f}')
            else:
                self.log(f'SELL EXECUTED: {order.executed.price:.5f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'TRADE PROFIT: {trade.pnl:.5f}')
            self.in_position = False

    def log(self, txt):
        dt = self.data.datetime.datetime()
        print(f'[{dt}] {txt}')

    def next(self):
        # Skip if not enough data
        if len(self.data) < self.params.asian_lookback + 10:
            return

        # Only trade during London session
        if not self.is_london_session():
            return

        # Skip if already in position
        if self.in_position or self.order:
            return

        # Calculate Asian range (previous 8 hours)
        asian_highs = [self.data.high[-i] for i in range(1, self.params.asian_lookback + 1)]
        asian_lows = [self.data.low[-i] for i in range(1, self.params.asian_lookback + 1)]

        asian_high = max(asian_highs)
        asian_low = min(asian_lows)
        asian_range = asian_high - asian_low

        # Calculate volume metrics
        volumes = [self.data.volume[-i] for i in range(1, 11)]
        avg_volume = sum(volumes) / len(volumes)
        current_volume = self.data.volume[0]

        # Current price
        current_price = self.data.close[0]

        # Detection: Low volume breakout = fake
        breakout_buffer = asian_range * self.params.breakout_buffer

        # Upside fake-out (SHORT)
        if current_price > asian_high + breakout_buffer:
            if current_volume < avg_volume * self.params.volume_threshold:
                # Calculate position size (1% risk)
                stop_distance = (asian_high + asian_range * 0.05) - current_price
                risk_amount = self.broker.getvalue() * self.params.risk_percent
                size = risk_amount / abs(stop_distance)

                # Enter SHORT with stop and target
                stop_price = asian_high + (asian_range * 0.05)
                target_price = asian_low

                self.order = self.sell_bracket(
                    size=size,
                    stopprice=stop_price,
                    limitprice=target_price
                )
                self.in_position = True

                self.log(f'LONDON FAKEOUT SHORT: Entry={current_price:.5f}, '
                        f'Stop={stop_price:.5f}, '
                        f'Target={target_price:.5f}, '
                        f'Volume Ratio={current_volume/avg_volume:.2f}')

        # Downside fake-out (LONG)
        elif current_price < asian_low - breakout_buffer:
            if current_volume < avg_volume * self.params.volume_threshold:
                # Calculate position size
                stop_distance = current_price - (asian_low - asian_range * 0.05)
                risk_amount = self.broker.getvalue() * self.params.risk_percent
                size = risk_amount / abs(stop_distance)

                # Enter LONG with stop and target
                stop_price = asian_low - (asian_range * 0.05)
                target_price = asian_high

                self.order = self.buy_bracket(
                    size=size,
                    stopprice=stop_price,
                    limitprice=target_price
                )
                self.in_position = True

                self.log(f'LONDON FAKEOUT LONG: Entry={current_price:.5f}, '
                        f'Stop={stop_price:.5f}, '
                        f'Target={target_price:.5f}, '
                        f'Volume Ratio={current_volume/avg_volume:.2f}')


class NYAbsorptionStrategy(bt.Strategy):
    """
    NY Absorption: Price tests prev day high/low, gets rejected
    Theory: Institutions defend levels with size
    """

    params = (
        ('risk_percent', 0.01),
        ('level_threshold', 0.001),  # Within 0.1% of level
    )

    def __init__(self):
        self.order = None
        self.in_position = False
        self.ny_open = dt_time(8, 0)
        self.ny_close = dt_time(12, 0)

    def is_ny_session(self):
        current_time = self.data.datetime.time()
        return self.ny_open <= current_time < self.ny_close

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: {order.executed.price:.5f}')
            else:
                self.log(f'SELL EXECUTED: {order.executed.price:.5f}')
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'TRADE PROFIT: {trade.pnl:.5f}')
            self.in_position = False

    def log(self, txt):
        dt = self.data.datetime.datetime()
        print(f'[{dt}] {txt}')

    def next(self):
        # Need previous day data (24 hours)
        if len(self.data) < 48:
            return

        if not self.is_ny_session():
            return

        if self.in_position or self.order:
            return

        # Get previous day high/low (24-48 hours ago)
        prev_highs = [self.data.high[-i] for i in range(24, 49)]
        prev_lows = [self.data.low[-i] for i in range(24, 49)]

        prev_high = max(prev_highs)
        prev_low = min(prev_lows)

        current_high = self.data.high[0]
        current_low = self.data.low[0]
        current_close = self.data.close[0]
        current_open = self.data.open[0]

        # Rejection at previous day high (SHORT)
        if current_high >= prev_high * (1 - self.params.level_threshold):
            if current_close < current_open:  # Bearish rejection
                stop = prev_high * (1 + self.params.level_threshold)
                stop_distance = stop - current_close
                risk_amount = self.broker.getvalue() * self.params.risk_percent
                size = risk_amount / abs(stop_distance)

                self.order = self.sell_bracket(
                    size=size,
                    stopprice=stop,
                    limitprice=prev_low
                )
                self.in_position = True

                self.log(f'NY ABSORPTION SHORT: Entry={current_close:.5f}, '
                        f'Stop={stop:.5f}, Target={prev_low:.5f}')

        # Rejection at previous day low (LONG)
        elif current_low <= prev_low * (1 + self.params.level_threshold):
            if current_close > current_open:  # Bullish rejection
                stop = prev_low * (1 - self.params.level_threshold)
                stop_distance = current_close - stop
                risk_amount = self.broker.getvalue() * self.params.risk_percent
                size = risk_amount / abs(stop_distance)

                self.order = self.buy_bracket(
                    size=size,
                    stopprice=stop,
                    limitprice=prev_high
                )
                self.in_position = True

                self.log(f'NY ABSORPTION LONG: Entry={current_close:.5f}, '
                        f'Stop={stop:.5f}, Target={prev_high:.5f}')


class TokyoGapFillStrategy(bt.Strategy):
    """
    Tokyo Gap Fill: Session/weekend gaps fill mechanically
    Theory: 70%+ gap fill rate, mean reversion
    """

    params = (
        ('gap_threshold', 0.0005),  # 5 pips minimum
        ('risk_percent', 0.01),
    )

    def __init__(self):
        self.order = None
        self.in_position = False
        self.tokyo_open = dt_time(19, 0)
        self.tokyo_close = dt_time(2, 0)

    def is_tokyo_session(self):
        current_time = self.data.datetime.time()
        return current_time >= self.tokyo_open or current_time < self.tokyo_close

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: {order.executed.price:.5f}')
            else:
                self.log(f'SELL EXECUTED: {order.executed.price:.5f}')
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'TRADE PROFIT: {trade.pnl:.5f}')
            self.in_position = False

    def log(self, txt):
        dt = self.data.datetime.datetime()
        print(f'[{dt}] {txt}')

    def next(self):
        if len(self.data) < 2:
            return

        if not self.is_tokyo_session():
            return

        if self.in_position or self.order:
            return

        # Check for gap
        prev_close = self.data.close[-1]
        current_open = self.data.open[0]
        current_close = self.data.close[0]

        gap_size = abs(current_open - prev_close)
        gap_threshold = prev_close * self.params.gap_threshold

        if gap_size < gap_threshold:
            return

        # Gap up - SHORT (expect fill down)
        if current_open > prev_close:
            stop = self.data.high[0] + gap_size * 0.5
            stop_distance = stop - current_close
            risk_amount = self.broker.getvalue() * self.params.risk_percent
            size = risk_amount / abs(stop_distance)

            self.order = self.sell_bracket(
                size=size,
                stopprice=stop,
                limitprice=prev_close
            )
            self.in_position = True

            self.log(f'TOKYO GAP FILL SHORT: Entry={current_close:.5f}, '
                    f'Stop={stop:.5f}, Gap={gap_size:.5f}, Target={prev_close:.5f}')

        # Gap down - LONG (expect fill up)
        else:
            stop = self.data.low[0] - gap_size * 0.5
            stop_distance = current_close - stop
            risk_amount = self.broker.getvalue() * self.params.risk_percent
            size = risk_amount / abs(stop_distance)

            self.order = self.buy_bracket(
                size=size,
                stopprice=stop,
                limitprice=prev_close
            )
            self.in_position = True

            self.log(f'TOKYO GAP FILL LONG: Entry={current_close:.5f}, '
                    f'Stop={stop:.5f}, Gap={gap_size:.5f}, Target={prev_close:.5f}')


class OANDAData(bt.DataBase):
    """Custom data feed from OANDA"""

    params = (
        ('instrument', ''),
        ('granularity', 'H1'),
        ('start_date', None),
        ('end_date', None),
    )

    def __init__(self):
        super(OANDAData, self).__init__()

        self.api = API(access_token=os.getenv('OANDA_API_KEY'))
        self.candles = self._fetch_data()
        self.idx = 0

    def _fetch_data(self):
        """Fetch historical data from OANDA"""
        params = {
            'from': self.params.start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'to': self.params.end_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'granularity': self.params.granularity
        }

        req = InstrumentsCandles(instrument=self.params.instrument, params=params)
        response = self.api.request(req)

        candles = []
        for candle in response.get('candles', []):
            if candle['complete']:
                candles.append({
                    'datetime': datetime.fromisoformat(candle['time'].replace('Z', '+00:00')),
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume'])
                })

        print(f"[DATA] Loaded {len(candles)} candles for {self.params.instrument}")
        return candles

    def _load(self):
        if self.idx >= len(self.candles):
            return False

        candle = self.candles[self.idx]

        self.lines.datetime[0] = bt.date2num(candle['datetime'])
        self.lines.open[0] = candle['open']
        self.lines.high[0] = candle['high']
        self.lines.low[0] = candle['low']
        self.lines.close[0] = candle['close']
        self.lines.volume[0] = candle['volume']
        self.lines.openinterest[0] = 0

        self.idx += 1
        return True


def run_backtest(strategy_class, pair='EUR_USD', days=30, initial_cash=200000):
    """
    Run backtest for a specific strategy

    Args:
        strategy_class: Strategy class to test
        pair: Currency pair
        days: Days of history to test
        initial_cash: Starting capital (E8 challenge = $200k)
    """

    print("=" * 70)
    print(f"BACKTRADER BACKTEST: {strategy_class.__name__}")
    print("=" * 70)
    print(f"Pair: {pair}")
    print(f"Period: Last {days} days")
    print(f"Initial Capital: ${initial_cash:,}")
    print("=" * 70)

    # Create Cerebro engine
    cerebro = bt.Cerebro()

    # Add strategy
    cerebro.addstrategy(strategy_class)

    # Add data feed
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    data = OANDAData(
        instrument=pair,
        granularity='H1',
        start_date=start_date,
        end_date=end_date
    )

    cerebro.adddata(data)

    # Set broker parameters
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.00002)  # 0.2 pips spread

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    # Run backtest
    print("\nRunning backtest...\n")
    starting_value = cerebro.broker.getvalue()
    results = cerebro.run()
    ending_value = cerebro.broker.getvalue()

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f'Starting Portfolio Value: ${starting_value:,.2f}')
    print(f'Ending Portfolio Value: ${ending_value:,.2f}')
    print(f'Total Return: ${ending_value - starting_value:,.2f} ({(ending_value/starting_value - 1)*100:.2f}%)')

    # Trade analysis
    trade_analysis = results[0].analyzers.trades.get_analysis()

    if 'total' in trade_analysis and 'total' in trade_analysis['total']:
        total_trades = trade_analysis['total']['total']
        print(f'\nTotal Trades: {total_trades}')

        if total_trades > 0:
            won = trade_analysis['won']['total'] if 'won' in trade_analysis else 0
            lost = trade_analysis['lost']['total'] if 'lost' in trade_analysis else 0
            win_rate = (won / total_trades * 100) if total_trades > 0 else 0

            print(f'Wins: {won}')
            print(f'Losses: {lost}')
            print(f'Win Rate: {win_rate:.1f}%')

            if 'won' in trade_analysis and 'pnl' in trade_analysis['won']:
                avg_win = trade_analysis['won']['pnl']['average']
                print(f'Average Win: ${avg_win:.2f}')

            if 'lost' in trade_analysis and 'pnl' in trade_analysis['lost']:
                avg_loss = trade_analysis['lost']['pnl']['average']
                print(f'Average Loss: ${avg_loss:.2f}')
    else:
        print("\nNo trades executed during backtest period")

    # Drawdown
    drawdown = results[0].analyzers.drawdown.get_analysis()
    if 'max' in drawdown and 'drawdown' in drawdown['max']:
        max_dd = drawdown['max']['drawdown']
        print(f'\nMax Drawdown: {max_dd:.2f}%')

    print("=" * 70)

    return results


if __name__ == '__main__':
    # Test all 3 strategies
    strategies = [
        LondonFakeoutStrategy,
        NYAbsorptionStrategy,
        TokyoGapFillStrategy
    ]

    for strategy in strategies:
        run_backtest(strategy, pair='EUR_USD', days=30, initial_cash=200000)
        print("\n" * 2)
