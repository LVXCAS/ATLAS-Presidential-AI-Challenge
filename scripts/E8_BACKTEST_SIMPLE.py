"""
E8 STRATEGY BACKTEST - Simplified Backtrader Version

Tests the 3 setups with proper risk management for E8 challenge:
- 1% risk per trade
- Automatic stop-loss and take-profit
- Max 6% drawdown limit

Uses OANDA data, applies to E8 live trading.
"""

import backtrader as bt
from datetime import datetime, timedelta, time as dt_time
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
import os
from dotenv import load_dotenv

load_dotenv()


class TokyoGapFillStrategy(bt.Strategy):
    """
    Tokyo Gap Fill - Simplest setup to test first

    Theory: Session gaps fill 70%+ of the time
    Entry: Trade into the gap
    Exit: Target = previous close, Stop = 1.5x gap size
    """

    params = (
        ('gap_threshold', 0.0005),  # 5 pips minimum
        ('risk_percent', 0.01),  # 1% risk per trade
        ('printlog', True),
    )

    def __init__(self):
        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.position_type = None

    def log(self, txt):
        if self.params.printlog:
            dt = self.data.datetime.datetime()
            print(f'[{dt.strftime("%Y-%m-%d %H:%M")}] {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED @ {order.executed.price:.5f}')
            else:
                self.log(f'SELL EXECUTED @ {order.executed.price:.5f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Failed: {order.getstatusname()}')

        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            pnl_pct = (trade.pnl / self.broker.getvalue()) * 100
            self.log(f'TRADE CLOSED | P&L: ${trade.pnl:,.2f} ({pnl_pct:+.2f}%)')

    def is_tokyo_session(self):
        """Tokyo session: 7 PM - 2 AM EST"""
        t = self.data.datetime.time()
        return t >= dt_time(19, 0) or t < dt_time(2, 0)

    def next(self):
        # Wait for pending orders
        if self.order:
            return

        # Exit management - check if we should close position
        if self.position:
            current_price = self.data.close[0]

            if self.position_type == 'LONG':
                # Check stop
                if self.data.low[0] <= self.stop_price:
                    self.log(f'STOP HIT @ {self.stop_price:.5f}')
                    self.order = self.close()
                    return
                # Check target
                if self.data.high[0] >= self.target_price:
                    self.log(f'TARGET HIT @ {self.target_price:.5f}')
                    self.order = self.close()
                    return

            elif self.position_type == 'SHORT':
                # Check stop
                if self.data.high[0] >= self.stop_price:
                    self.log(f'STOP HIT @ {self.stop_price:.5f}')
                    self.order = self.close()
                    return
                # Check target
                if self.data.low[0] <= self.target_price:
                    self.log(f'TARGET HIT @ {self.target_price:.5f}')
                    self.order = self.close()
                    return

            return  # Don't enter new trades while in position

        # Entry logic
        if len(self.data) < 2:
            return

        if not self.is_tokyo_session():
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
            self.entry_price = current_close
            self.stop_price = self.data.high[0] + gap_size * 0.5
            self.target_price = prev_close
            self.position_type = 'SHORT'

            # Calculate position size (1% risk)
            stop_distance = abs(self.stop_price - self.entry_price)
            risk_amount = self.broker.getvalue() * self.params.risk_percent
            size = risk_amount / stop_distance

            self.log(f'=== TOKYO GAP FILL SHORT ===')
            self.log(f'Gap: {gap_size * 10000:.1f} pips')
            self.log(f'Entry: {self.entry_price:.5f}')
            self.log(f'Stop: {self.stop_price:.5f} (risk: ${stop_distance * size:.2f})')
            self.log(f'Target: {self.target_price:.5f}')

            self.order = self.sell(size=size)

        # Gap down - LONG (expect fill up)
        elif current_open < prev_close:
            self.entry_price = current_close
            self.stop_price = self.data.low[0] - gap_size * 0.5
            self.target_price = prev_close
            self.position_type = 'LONG'

            # Calculate position size
            stop_distance = abs(self.entry_price - self.stop_price)
            risk_amount = self.broker.getvalue() * self.params.risk_percent
            size = risk_amount / stop_distance

            self.log(f'=== TOKYO GAP FILL LONG ===')
            self.log(f'Gap: {gap_size * 10000:.1f} pips')
            self.log(f'Entry: {self.entry_price:.5f}')
            self.log(f'Stop: {self.stop_price:.5f} (risk: ${stop_distance * size:.2f})')
            self.log(f'Target: {self.target_price:.5f}')

            self.order = self.buy(size=size)


class OANDAData(bt.DataBase):
    """Load OANDA historical data"""

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


def run_backtest(days=60):
    """Run Tokyo Gap Fill backtest"""

    print("=" * 70)
    print("E8 BACKTEST - TOKYO GAP FILL STRATEGY")
    print("=" * 70)
    print(f"Period: Last {days} days")
    print(f"Initial Capital: $200,000 (E8 Challenge)")
    print(f"Risk Per Trade: 1% ($2,000)")
    print(f"Max Drawdown Limit: 6% ($12,000)")
    print("=" * 70)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(TokyoGapFillStrategy)

    # Load data
    end_date = datetime(2024, 11, 10)  # Use correct date
    start_date = end_date - timedelta(days=days)

    data = OANDAData(
        instrument='EUR_USD',
        granularity='H1',
        start_date=start_date,
        end_date=end_date
    )

    cerebro.adddata(data)

    # Broker settings
    cerebro.broker.setcash(200000)
    cerebro.broker.setcommission(commission=0.00002)  # 0.2 pips

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')

    # Run
    print("\nRunning backtest...\n")
    start_value = cerebro.broker.getvalue()
    results = cerebro.run()
    end_value = cerebro.broker.getvalue()

    # Results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f'Starting Value: ${start_value:,.2f}')
    print(f'Ending Value: ${end_value:,.2f}')
    print(f'Total Return: ${end_value - start_value:,.2f} ({(end_value/start_value - 1)*100:+.2f}%)')

    # Trade stats
    trade_analysis = results[0].analyzers.trades.get_analysis()

    if 'total' in trade_analysis and 'total' in trade_analysis['total']:
        total = trade_analysis['total']['total']
        print(f'\nTotal Trades: {total}')

        if total > 0:
            won = trade_analysis.get('won', {}).get('total', 0)
            lost = trade_analysis.get('lost', {}).get('total', 0)
            win_rate = (won / total * 100) if total > 0 else 0

            print(f'Wins: {won}')
            print(f'Losses: {lost}')
            print(f'Win Rate: {win_rate:.1f}%')

            if 'won' in trade_analysis and 'pnl' in trade_analysis['won']:
                print(f'Average Win: ${trade_analysis["won"]["pnl"]["average"]:,.2f}')

            if 'lost' in trade_analysis and 'pnl' in trade_analysis['lost']:
                print(f'Average Loss: ${trade_analysis["lost"]["pnl"]["average"]:,.2f}')
    else:
        print('\nNo trades executed')

    # Drawdown
    dd = results[0].analyzers.drawdown.get_analysis()
    if 'max' in dd and 'drawdown' in dd['max']:
        max_dd = dd['max']['drawdown']
        print(f'\nMax Drawdown: {max_dd:.2f}%')

        if max_dd > 6.0:
            print('⚠️  WARNING: Exceeded E8 6% drawdown limit!')
        else:
            print('✓ Within E8 drawdown limits')

    print("=" * 70)


if __name__ == '__main__':
    run_backtest(days=60)
