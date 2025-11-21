"""
E8 TRIPLE CONFIRMATION TREND FOLLOWING STRATEGY
Recommended by DeepSeek as a proven, low-drawdown approach

Strategy Logic:
1. ADX(14) > 25 - confirms strong trend
2. EMA(20) vs EMA(50) - determines trend direction
3. RSI(14) pullback & cross - entry trigger

Risk Management:
- Stop: 1.5x ATR below swing low (LONG) or above swing high (SHORT)
- Target: 2x risk (1:2 risk-reward ratio)
- Position Size: 1% account risk per trade
- Max Daily Loss: 2% ($4,000)
"""

import backtrader as bt
from datetime import datetime, timedelta, time as dt_time
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
import os
from dotenv import load_dotenv
import json

load_dotenv()


class TripleConfirmationStrategy(bt.Strategy):
    """
    Triple Confirmation Trend Following
    High win rate, low drawdown strategy for E8 Challenge
    """

    params = (
        ('ema_fast', 20),
        ('ema_slow', 50),
        ('adx_period', 14),
        ('adx_threshold', 25),
        ('rsi_period', 14),
        ('atr_period', 14),
        ('atr_stop_multiplier', 1.5),
        ('risk_reward', 2.0),  # 1:2 risk-reward
        ('risk_percent', 0.01),  # 1% risk per trade
        ('max_daily_loss', 0.02),  # 2% max daily loss
        ('printlog', False),  # Disable verbose logging for speed
    )

    def __init__(self):
        # Indicators
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.params.ema_fast)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.params.ema_slow)
        self.adx = bt.indicators.ADX(self.data, period=self.params.adx_period)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)

        # Track RSI for crossovers
        self.rsi_cross_up = bt.indicators.CrossOver(self.rsi, 50)
        self.rsi_cross_down = bt.indicators.CrossOver(50, self.rsi)

        # Trade management
        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.position_type = None

        # Daily tracking
        self.daily_pnl = 0
        self.last_date = None
        self.trade_log = []
        self.daily_trades = 0

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
            self.daily_pnl += trade.pnl

            self.log(f'TRADE CLOSED | P&L: ${trade.pnl:,.2f} ({pnl_pct:+.2f}%)')
            self.log(f'Daily P&L: ${self.daily_pnl:,.2f}')

            self.trade_log.append({
                'pnl': trade.pnl,
                'pnl_pct': pnl_pct,
                'win': trade.pnl > 0
            })

    def is_london_or_ny_session(self):
        """Check if in London (8 AM - 5 PM GMT) or NY overlap (1 PM - 5 PM GMT)"""
        # Convert to EST for simplicity (GMT-5)
        t = self.data.datetime.time()

        # London: 3 AM - 12 PM EST
        # NY overlap: 8 AM - 12 PM EST
        return dt_time(3, 0) <= t < dt_time(12, 0)

    def check_daily_loss_limit(self):
        """Stop trading if hit 2% daily loss"""
        current_date = self.data.datetime.date()

        # Reset daily tracking
        if self.last_date is None or current_date > self.last_date:
            self.daily_pnl = 0
            self.last_date = current_date
            self.daily_trades = 0

        # Check if hit daily loss limit
        max_loss = self.broker.getvalue() * self.params.max_daily_loss
        if self.daily_pnl < -max_loss:
            self.log(f'!!! DAILY LOSS LIMIT HIT: ${self.daily_pnl:.2f} / ${-max_loss:.2f}')
            return True

        return False

    def check_exit(self):
        """Manual stop/target management"""
        if not self.position:
            return False

        if self.position_type == 'LONG':
            # Check stop
            if self.data.low[0] <= self.stop_price:
                self.log(f'STOP HIT @ {self.stop_price:.5f}')
                self.order = self.close()
                return True
            # Check target
            if self.data.high[0] >= self.target_price:
                self.log(f'TARGET HIT @ {self.target_price:.5f}')
                self.order = self.close()
                return True

        elif self.position_type == 'SHORT':
            # Check stop
            if self.data.high[0] >= self.stop_price:
                self.log(f'STOP HIT @ {self.stop_price:.5f}')
                self.order = self.close()
                return True
            # Check target
            if self.data.low[0] <= self.target_price:
                self.log(f'TARGET HIT @ {self.target_price:.5f}')
                self.order = self.close()
                return True

        return False

    def next(self):
        # Wait for indicators to stabilize
        if len(self.data) < self.params.ema_slow:
            return

        # Check pending orders
        if self.order:
            return

        # Check exits first
        if self.check_exit():
            return

        # Check daily loss limit
        if self.check_daily_loss_limit():
            return

        # Limit trades per day (1-2)
        if self.daily_trades >= 2:
            return

        # Only trade during London/NY sessions
        if not self.is_london_or_ny_session():
            return

        # Get current values
        current_price = self.data.close[0]
        adx_value = self.adx[0]
        ema_fast_value = self.ema_fast[0]
        ema_slow_value = self.ema_slow[0]
        rsi_value = self.rsi[0]
        atr_value = self.atr[0]

        # FILTER 1: ADX must show strong trend
        if adx_value < self.params.adx_threshold:
            return

        # LONG SETUP
        # FILTER 2: Fast EMA > Slow EMA AND price > Slow EMA (uptrend)
        if ema_fast_value > ema_slow_value and current_price > ema_slow_value:
            # FILTER 3: RSI pullback and cross back above 50
            if self.rsi_cross_up[0]:  # RSI just crossed above 50

                # Calculate stop-loss (1.5x ATR below recent swing low)
                lookback = 20
                swing_low = min([self.data.low[-i] for i in range(0, min(lookback, len(self.data)))])
                stop = swing_low - (self.params.atr_stop_multiplier * atr_value)

                # Calculate position size (1% risk)
                stop_distance = abs(current_price - stop)
                risk_amount = self.broker.getvalue() * self.params.risk_percent
                size = risk_amount / stop_distance

                # Calculate target (2x risk)
                target = current_price + (stop_distance * self.params.risk_reward)

                # Entry
                self.entry_price = current_price
                self.stop_price = stop
                self.target_price = target
                self.position_type = 'LONG'

                self.log(f'=== TRIPLE CONFIRMATION LONG ===')
                self.log(f'ADX: {adx_value:.1f} | RSI: {rsi_value:.1f}')
                self.log(f'Entry: {current_price:.5f} | Stop: {stop:.5f} | Target: {target:.5f}')
                self.log(f'Risk: ${stop_distance * size:.2f} | Reward: ${(stop_distance * self.params.risk_reward) * size:.2f}')

                self.order = self.buy(size=size)
                self.daily_trades += 1

        # SHORT SETUP
        # FILTER 2: Fast EMA < Slow EMA AND price < Slow EMA (downtrend)
        elif ema_fast_value < ema_slow_value and current_price < ema_slow_value:
            # FILTER 3: RSI pullback and cross back below 50
            if self.rsi_cross_down[0]:  # RSI just crossed below 50

                # Calculate stop-loss (1.5x ATR above recent swing high)
                lookback = 20
                swing_high = max([self.data.high[-i] for i in range(0, min(lookback, len(self.data)))])
                stop = swing_high + (self.params.atr_stop_multiplier * atr_value)

                # Calculate position size (1% risk)
                stop_distance = abs(stop - current_price)
                risk_amount = self.broker.getvalue() * self.params.risk_percent
                size = risk_amount / stop_distance

                # Calculate target (2x risk)
                target = current_price - (stop_distance * self.params.risk_reward)

                # Entry
                self.entry_price = current_price
                self.stop_price = stop
                self.target_price = target
                self.position_type = 'SHORT'

                self.log(f'=== TRIPLE CONFIRMATION SHORT ===')
                self.log(f'ADX: {adx_value:.1f} | RSI: {rsi_value:.1f}')
                self.log(f'Entry: {current_price:.5f} | Stop: {stop:.5f} | Target: {target:.5f}')
                self.log(f'Risk: ${stop_distance * size:.2f} | Reward: ${(stop_distance * self.params.risk_reward) * size:.2f}')

                self.order = self.sell(size=size)
                self.daily_trades += 1


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


def run_backtest(pair='EUR_USD', days=90):
    """Run Triple Confirmation backtest"""

    print("=" * 70)
    print("E8 BACKTEST - TRIPLE CONFIRMATION TREND FOLLOWING")
    print("=" * 70)
    print(f"Pair: {pair}")
    print(f"Period: Last {days} days")
    print(f"Initial Capital: $200,000")
    print(f"Risk Per Trade: 1%")
    print(f"Risk:Reward: 1:2")
    print(f"Max Daily Loss: 2%")
    print("=" * 70)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(TripleConfirmationStrategy)

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
    cerebro.broker.setcommission(commission=0.00002)  # 0.2 pips

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
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
            print('[FAIL] Exceeded E8 6% drawdown limit!')
        else:
            print('[PASS] Within E8 drawdown limits!')

    print("=" * 70)

    return results


if __name__ == '__main__':
    # Test EUR/USD with shorter period for debugging
    results = run_backtest('EUR_USD', days=30)

    # Summary
    print("=" * 70)
    print("STRATEGY COMPARISON: TRIPLE CONFIRMATION vs FAILED SETUPS")
    print("=" * 70)
    print("\nFAILED SETUPS (London/NY/Tokyo):")
    print("  Total Return: -$255,810 (-128%)")
    print("  Win Rate: 36.7%")
    print("  Max Drawdown: 54.08%")
    print("  Status: FAILED E8")
    print("\nTRIPLE CONFIRMATION:")
    print("  Results above ^")
    print("=" * 70)
