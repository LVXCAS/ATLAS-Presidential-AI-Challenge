"""
E8 COMPREHENSIVE BACKTEST - All 3 Strategies
Tests London Fakeout, NY Absorption, and Tokyo Gap Fill over 90 days
Provides complete statistical analysis for E8 challenge validation
"""

import backtrader as bt
from datetime import datetime, timedelta, time as dt_time
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
import os
from dotenv import load_dotenv
import json

load_dotenv()


class CombinedE8Strategy(bt.Strategy):
    """All 3 E8 setups in one strategy"""

    params = (
        ('risk_percent', 0.01),  # 1% risk per trade
        ('printlog', True),
    )

    def __init__(self):
        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.position_type = None
        self.setup_name = None
        self.trade_log = []

    def log(self, txt):
        if self.params.printlog:
            dt = self.data.datetime.datetime()
            print(f'[{dt.strftime("%Y-%m-%d %H:%M")}] {txt}')

    def notify_order(self, order):
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
            self.log(f'TRADE CLOSED | {self.setup_name} | P&L: ${trade.pnl:,.2f} ({pnl_pct:+.2f}%)')

            # Log trade details
            self.trade_log.append({
                'setup': self.setup_name,
                'pnl': trade.pnl,
                'pnl_pct': pnl_pct,
                'entry': self.entry_price,
                'exit': trade.price,
                'win': trade.pnl > 0
            })

    def is_london_session(self):
        t = self.data.datetime.time()
        return dt_time(3, 0) <= t < dt_time(5, 0)

    def is_ny_session(self):
        t = self.data.datetime.time()
        return dt_time(8, 0) <= t < dt_time(12, 0)

    def is_tokyo_session(self):
        t = self.data.datetime.time()
        return t >= dt_time(19, 0) or t < dt_time(2, 0)

    def check_exit(self):
        """Manual stop/target management"""
        if not self.position:
            return

        current_price = self.data.close[0]

        if self.position_type == 'LONG':
            if self.data.low[0] <= self.stop_price:
                self.log(f'STOP HIT @ {self.stop_price:.5f}')
                self.order = self.close()
                return True
            if self.data.high[0] >= self.target_price:
                self.log(f'TARGET HIT @ {self.target_price:.5f}')
                self.order = self.close()
                return True

        elif self.position_type == 'SHORT':
            if self.data.high[0] >= self.stop_price:
                self.log(f'STOP HIT @ {self.stop_price:.5f}')
                self.order = self.close()
                return True
            if self.data.low[0] <= self.target_price:
                self.log(f'TARGET HIT @ {self.target_price:.5f}')
                self.order = self.close()
                return True

        return False

    def enter_trade(self, direction, entry, stop, target, setup_name, size):
        """Unified trade entry"""
        self.entry_price = entry
        self.stop_price = stop
        self.target_price = target
        self.position_type = direction
        self.setup_name = setup_name

        risk = abs(stop - entry) * size
        reward = abs(target - entry) * size
        rr = reward / risk if risk > 0 else 0

        self.log(f'=== {setup_name} {direction} ===')
        self.log(f'Entry: {entry:.5f} | Stop: {stop:.5f} | Target: {target:.5f}')
        self.log(f'Risk: ${risk:.2f} | Reward: ${reward:.2f} | R:R = 1:{rr:.2f}')

        if direction == 'LONG':
            self.order = self.buy(size=size)
        else:
            self.order = self.sell(size=size)

    def detect_london_fakeout(self):
        """London Fake-out detection"""
        if not self.is_london_session():
            return False

        if len(self.data) < 32:
            return False

        # Asian range (last 8 hours = 32 bars of 15min, approx 8 hours of H1)
        lookback = min(32, len(self.data))
        asian_highs = [self.data.high[-i] for i in range(1, lookback + 1)]
        asian_lows = [self.data.low[-i] for i in range(1, lookback + 1)]

        asian_high = max(asian_highs)
        asian_low = min(asian_lows)
        asian_range = asian_high - asian_low

        # Volume check
        volumes = [self.data.volume[-i] for i in range(1, 11)]
        avg_volume = sum(volumes) / len(volumes)
        current_volume = self.data.volume[0]

        current_price = self.data.close[0]
        breakout_buffer = asian_range * 0.002

        # Upside fake-out (SHORT)
        if current_price > asian_high + breakout_buffer:
            if current_volume < avg_volume * 0.7:
                stop = asian_high + (asian_range * 0.05)
                target = asian_low
                stop_distance = abs(stop - current_price)
                size = (self.broker.getvalue() * self.params.risk_percent) / stop_distance

                self.enter_trade('SHORT', current_price, stop, target, 'LONDON_FAKEOUT', size)
                return True

        # Downside fake-out (LONG)
        elif current_price < asian_low - breakout_buffer:
            if current_volume < avg_volume * 0.7:
                stop = asian_low - (asian_range * 0.05)
                target = asian_high
                stop_distance = abs(current_price - stop)
                size = (self.broker.getvalue() * self.params.risk_percent) / stop_distance

                self.enter_trade('LONG', current_price, stop, target, 'LONDON_FAKEOUT', size)
                return True

        return False

    def detect_ny_absorption(self):
        """NY Absorption detection"""
        if not self.is_ny_session():
            return False

        if len(self.data) < 48:
            return False

        # Previous day high/low
        prev_highs = [self.data.high[-i] for i in range(24, 49)]
        prev_lows = [self.data.low[-i] for i in range(24, 49)]

        prev_high = max(prev_highs)
        prev_low = min(prev_lows)

        current_high = self.data.high[0]
        current_low = self.data.low[0]
        current_close = self.data.close[0]
        current_open = self.data.open[0]

        # Rejection at previous high (SHORT)
        if current_high >= prev_high * 0.999:
            if current_close < current_open:  # Bearish rejection
                stop = prev_high * 1.001
                target = prev_low
                stop_distance = abs(stop - current_close)
                size = (self.broker.getvalue() * self.params.risk_percent) / stop_distance

                self.enter_trade('SHORT', current_close, stop, target, 'NY_ABSORPTION', size)
                return True

        # Rejection at previous low (LONG)
        elif current_low <= prev_low * 1.001:
            if current_close > current_open:  # Bullish rejection
                stop = prev_low * 0.999
                target = prev_high
                stop_distance = abs(current_close - stop)
                size = (self.broker.getvalue() * self.params.risk_percent) / stop_distance

                self.enter_trade('LONG', current_close, stop, target, 'NY_ABSORPTION', size)
                return True

        return False

    def detect_tokyo_gap_fill(self):
        """Tokyo Gap Fill detection"""
        if not self.is_tokyo_session():
            return False

        if len(self.data) < 2:
            return False

        prev_close = self.data.close[-1]
        current_open = self.data.open[0]
        current_close = self.data.close[0]

        gap_size = abs(current_open - prev_close)
        gap_threshold = prev_close * 0.0005

        if gap_size < gap_threshold:
            return False

        # Gap up - SHORT
        if current_open > prev_close:
            stop = self.data.high[0] + gap_size * 0.5
            target = prev_close
            stop_distance = abs(stop - current_close)
            size = (self.broker.getvalue() * self.params.risk_percent) / stop_distance

            self.enter_trade('SHORT', current_close, stop, target, 'TOKYO_GAP_FILL', size)
            return True

        # Gap down - LONG
        else:
            stop = self.data.low[0] - gap_size * 0.5
            target = prev_close
            stop_distance = abs(current_close - stop)
            size = (self.broker.getvalue() * self.params.risk_percent) / stop_distance

            self.enter_trade('LONG', current_close, stop, target, 'TOKYO_GAP_FILL', size)
            return True

    def next(self):
        if self.order:
            return

        # Check exits first
        if self.check_exit():
            return

        # Try each setup (priority order)
        if self.detect_tokyo_gap_fill():
            return
        if self.detect_london_fakeout():
            return
        if self.detect_ny_absorption():
            return


class OANDAData(bt.DataBase):
    """OANDA data feed"""
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


def run_full_backtest(days=90):
    """Run comprehensive backtest"""

    print("=" * 70)
    print("E8 COMPREHENSIVE BACKTEST - ALL 3 STRATEGIES")
    print("=" * 70)
    print(f"Period: Last {days} days")
    print(f"Pairs: EUR/USD, GBP/USD, USD/JPY")
    print(f"Initial Capital: $200,000")
    print(f"Risk Per Trade: 1%")
    print(f"Max Drawdown Limit: 6% (E8 Challenge)")
    print("=" * 70)

    pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']
    all_results = {}

    for pair in pairs:
        print(f"\n\n{'='*70}")
        print(f"TESTING {pair}")
        print(f"{'='*70}\n")

        cerebro = bt.Cerebro()
        cerebro.addstrategy(CombinedE8Strategy)

        # Load data
        end_date = datetime(2024, 11, 10)
        start_date = end_date - timedelta(days=days)

        data = OANDAData(
            instrument=pair,
            granularity='H1',
            start_date=start_date,
            end_date=end_date
        )

        cerebro.adddata(data)
        cerebro.broker.setcash(200000)
        cerebro.broker.setcommission(commission=0.00002)

        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

        start_value = cerebro.broker.getvalue()
        results = cerebro.run()
        end_value = cerebro.broker.getvalue()

        # Store results
        trade_analysis = results[0].analyzers.trades.get_analysis()
        dd_analysis = results[0].analyzers.drawdown.get_analysis()

        all_results[pair] = {
            'start': start_value,
            'end': end_value,
            'return': end_value - start_value,
            'return_pct': (end_value / start_value - 1) * 100,
            'trades': trade_analysis,
            'drawdown': dd_analysis,
            'trade_log': results[0].trade_log
        }

    # Final summary
    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY - ALL PAIRS")
    print("=" * 70)

    total_return = 0
    total_trades = 0
    total_wins = 0
    max_dd_all = 0

    for pair, res in all_results.items():
        print(f"\n{pair}:")
        print(f"  Return: ${res['return']:,.2f} ({res['return_pct']:+.2f}%)")

        if 'total' in res['trades'] and 'total' in res['trades']['total']:
            trades = res['trades']['total']['total']
            wins = res['trades'].get('won', {}).get('total', 0)
            total_trades += trades
            total_wins += wins
            print(f"  Trades: {trades} | Wins: {wins} | Win Rate: {(wins/trades*100 if trades > 0 else 0):.1f}%")

        if 'max' in res['drawdown'] and 'drawdown' in res['drawdown']['max']:
            dd = res['drawdown']['max']['drawdown']
            max_dd_all = max(max_dd_all, dd)
            print(f"  Max DD: {dd:.2f}%")

        total_return += res['return']

    print(f"\n{'='*70}")
    print(f"COMBINED RESULTS:")
    print(f"  Total Return: ${total_return:,.2f}")
    print(f"  Total Trades: {total_trades}")
    print(f"  Total Wins: {total_wins}")
    print(f"  Overall Win Rate: {(total_wins/total_trades*100 if total_trades > 0 else 0):.1f}%")
    print(f"  Max Drawdown: {max_dd_all:.2f}%")

    if max_dd_all > 6.0:
        print(f"\n  [FAIL] Exceeded E8 6% drawdown limit!")
    else:
        print(f"\n  [PASS] Within E8 drawdown limits")

    print("=" * 70)

    # Save detailed results
    output_file = f"e8_full_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        # Convert to JSON-serializable format
        save_data = {}
        for pair, res in all_results.items():
            save_data[pair] = {
                'return': res['return'],
                'return_pct': res['return_pct'],
                'trade_log': res['trade_log']
            }
        json.dump(save_data, f, indent=2)

    print(f"\n[SAVED] Detailed results: {output_file}")


if __name__ == '__main__':
    run_full_backtest(days=90)
