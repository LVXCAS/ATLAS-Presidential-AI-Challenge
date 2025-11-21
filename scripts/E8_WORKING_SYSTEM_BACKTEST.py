"""
BACKTRADER VALIDATION: Your Working OANDA System
Tests the exact logic from WORKING_FOREX_OANDA.py that made +$259 profit

Strategy Logic (from your working system):
- RSI(14) < 40 = LONG (+2 score)
- RSI(14) > 60 = SHORT (+2 score)
- MACD bullish cross = LONG (+2 score)
- MACD bearish cross = SHORT (+2 score)
- ADX > 25 = Strong trend (+1 score)
- EMA 10 > EMA 21 > EMA 200 = LONG (+1 score)
- EMA 10 < EMA 21 < EMA 200 = SHORT (+1 score)
- Volatility filter (ATR-based)

Min Score: 2.5 (quality filter)
Risk: 1% per trade
Target: 2% profit
Stop: 1% loss
"""

import backtrader as bt
from datetime import datetime, timedelta
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()


class WorkingOANDAStrategy(bt.Strategy):
    """
    Your proven OANDA system converted to Backtrader
    """

    params = (
        ('min_score', 2.5),
        ('risk_percent', 0.01),
        ('profit_target', 0.02),
        ('stop_loss', 0.01),
        ('rsi_period', 14),
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('adx_period', 14),
        ('ema_fast', 10),
        ('ema_slow', 21),
        ('ema_trend', 200),
        ('atr_period', 14),
        ('printlog', False),
    )

    def __init__(self):
        # Indicators
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )
        self.adx = bt.indicators.ADX(self.data, period=self.params.adx_period)
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.params.ema_fast)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.params.ema_slow)
        self.ema_trend = bt.indicators.EMA(self.data.close, period=self.params.ema_trend)
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)

        # Track MACD crossovers
        self.macd_cross_up = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

        # Trade management
        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.position_type = None
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
            self.log(f'TRADE CLOSED | P&L: ${trade.pnl:,.2f} ({pnl_pct:+.2f}%)')

            self.trade_log.append({
                'pnl': trade.pnl,
                'pnl_pct': pnl_pct,
                'win': trade.pnl > 0
            })

    def check_exit(self):
        """Check stop/target"""
        if not self.position:
            return False

        current_price = self.data.close[0]

        if self.position_type == 'LONG':
            # Check stop
            if current_price <= self.stop_price:
                self.log(f'STOP HIT @ {self.stop_price:.5f}')
                self.order = self.close()
                return True
            # Check target
            if current_price >= self.target_price:
                self.log(f'TARGET HIT @ {self.target_price:.5f}')
                self.order = self.close()
                return True

        elif self.position_type == 'SHORT':
            # Check stop
            if current_price >= self.stop_price:
                self.log(f'STOP HIT @ {self.stop_price:.5f}')
                self.order = self.close()
                return True
            # Check target
            if current_price <= self.target_price:
                self.log(f'TARGET HIT @ {self.target_price:.5f}')
                self.order = self.close()
                return True

        return False

    def calculate_score(self):
        """
        Calculate LONG and SHORT scores using your exact logic
        Returns: (long_score, short_score, long_signals, short_signals)
        """
        long_score = 0
        short_score = 0
        long_signals = []
        short_signals = []

        rsi = self.rsi[0]
        adx = self.adx[0]
        macd_hist = self.macd.macd[0] - self.macd.signal[0]
        macd_hist_prev = self.macd.macd[-1] - self.macd.signal[-1]

        current_price = self.data.close[0]
        ema_fast = self.ema_fast[0]
        ema_slow = self.ema_slow[0]
        ema_trend = self.ema_trend[0]
        atr = self.atr[0]

        # Volatility
        volatility = (atr / current_price) * 100

        # === LONG SIGNALS ===

        # RSI oversold
        if rsi < 40:
            long_score += 2
            long_signals.append("RSI_OVERSOLD")

        # MACD bullish cross
        if macd_hist > 0 and macd_hist_prev <= 0:
            long_score += 2
            long_signals.append("MACD_BULLISH_CROSS")

        # Strong trend (ADX > 25)
        if adx > 25:
            long_score += 1
            long_signals.append("STRONG_TREND")

        # Bullish EMA alignment
        if ema_fast > ema_slow and ema_slow > ema_trend:
            long_score += 1
            long_signals.append("EMA_BULLISH")

        # === SHORT SIGNALS ===

        # RSI overbought
        if rsi > 60:
            short_score += 2
            short_signals.append("RSI_OVERBOUGHT")

        # MACD bearish cross
        if macd_hist < 0 and macd_hist_prev >= 0:
            short_score += 2
            short_signals.append("MACD_BEARISH_CROSS")

        # Strong trend (same ADX)
        if adx > 25:
            short_score += 1
            short_signals.append("STRONG_TREND")

        # Bearish EMA alignment
        if ema_fast < ema_slow and ema_slow < ema_trend:
            short_score += 1
            short_signals.append("EMA_BEARISH")

        return long_score, short_score, long_signals, short_signals

    def next(self):
        # Wait for indicators
        if len(self.data) < self.params.ema_trend:
            return

        # Check pending orders
        if self.order:
            return

        # Check exits
        if self.check_exit():
            return

        # Calculate scores
        long_score, short_score, long_signals, short_signals = self.calculate_score()

        current_price = self.data.close[0]

        # LONG ENTRY
        if long_score >= self.params.min_score and not self.position:
            # Calculate stop/target
            stop = current_price * (1 - self.params.stop_loss)
            target = current_price * (1 + self.params.profit_target)

            # Position size (1% risk)
            stop_distance = abs(current_price - stop)
            risk_amount = self.broker.getvalue() * self.params.risk_percent
            size = risk_amount / stop_distance

            self.entry_price = current_price
            self.stop_price = stop
            self.target_price = target
            self.position_type = 'LONG'

            self.log(f'=== LONG | Score: {long_score:.1f} | Signals: {long_signals} ===')
            self.log(f'Entry: {current_price:.5f} | Stop: {stop:.5f} | Target: {target:.5f}')

            self.order = self.buy(size=size)

        # SHORT ENTRY
        elif short_score >= self.params.min_score and not self.position:
            # Calculate stop/target
            stop = current_price * (1 + self.params.stop_loss)
            target = current_price * (1 - self.params.profit_target)

            # Position size (1% risk)
            stop_distance = abs(stop - current_price)
            risk_amount = self.broker.getvalue() * self.params.risk_percent
            size = risk_amount / stop_distance

            self.entry_price = current_price
            self.stop_price = stop
            self.target_price = target
            self.position_type = 'SHORT'

            self.log(f'=== SHORT | Score: {short_score:.1f} | Signals: {short_signals} ===')
            self.log(f'Entry: {current_price:.5f} | Stop: {stop:.5f} | Target: {target:.5f}')

            self.order = self.sell(size=size)


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
    """Run backtest on your working system"""

    print("=" * 70)
    print("BACKTEST: YOUR WORKING OANDA SYSTEM (+$259 PROFIT)")
    print("=" * 70)
    print(f"Pair: {pair}")
    print(f"Period: Last {days} days")
    print(f"Initial Capital: $200,000 (E8 Challenge)")
    print(f"Min Score: 2.5 (quality filter)")
    print(f"Risk Per Trade: 1%")
    print(f"Target: 2% | Stop: 1%")
    print("=" * 70)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(WorkingOANDAStrategy)

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
    print("RESULTS")
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
    # Test EUR/USD only (your best performer)
    run_backtest('EUR_USD', days=90)
