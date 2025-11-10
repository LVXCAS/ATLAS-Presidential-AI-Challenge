"""
E8 ROI OPTIMIZER - Find Best Parameters for Your Working System

Tests different parameter combinations to maximize ROI while staying under 6% drawdown.

Parameters to optimize:
1. Min Score (2.0, 2.5, 3.0)
2. Profit Target (2%, 3%, 4%)
3. Risk Per Trade (1%, 1.5%, 2%)
4. Stop Loss (1%, 1.5%)

Goal: Find combination that maximizes return while keeping drawdown < 6%
"""

import backtrader as bt
from datetime import datetime, timedelta
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
import os
from dotenv import load_dotenv
import itertools

load_dotenv()


class OptimizableStrategy(bt.Strategy):
    """Your working system with optimizable parameters"""

    params = (
        ('min_score', 2.5),
        ('risk_percent', 0.01),
        ('profit_target', 0.02),
        ('stop_loss', 0.01),
        ('rsi_period', 14),
        ('adx_period', 14),
        ('ema_fast', 10),
        ('ema_slow', 21),
        ('ema_trend', 200),
    )

    def __init__(self):
        # Indicators
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.macd = bt.indicators.MACD(self.data.close)
        self.adx = bt.indicators.ADX(self.data, period=self.params.adx_period)
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.params.ema_fast)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.params.ema_slow)
        self.ema_trend = bt.indicators.EMA(self.data.close, period=self.params.ema_trend)
        self.atr = bt.indicators.ATR(self.data)

        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.position_type = None

        # Track cumulative P&L from closed trades
        self.total_pnl = 0
        self.total_commission = 0

    def notify_order(self, order):
        if order.status in [order.Completed]:
            pass  # Silent
        self.order = None

    def notify_trade(self, trade):
        """Track closed trades for accurate P&L calculation"""
        if trade.isclosed:
            self.total_pnl += trade.pnl
            self.total_commission += trade.commission

    def check_exit(self):
        if not self.position:
            return False

        current_price = self.data.close[0]

        if self.position_type == 'LONG':
            if current_price <= self.stop_price or current_price >= self.target_price:
                self.order = self.close()
                return True
        elif self.position_type == 'SHORT':
            if current_price >= self.stop_price or current_price <= self.target_price:
                self.order = self.close()
                return True
        return False

    def calculate_score(self):
        long_score = 0
        short_score = 0

        rsi = self.rsi[0]
        adx = self.adx[0]
        macd_hist = self.macd.macd[0] - self.macd.signal[0]
        macd_hist_prev = self.macd.macd[-1] - self.macd.signal[-1]

        ema_fast = self.ema_fast[0]
        ema_slow = self.ema_slow[0]
        ema_trend = self.ema_trend[0]

        # LONG signals
        if rsi < 40:
            long_score += 2
        if macd_hist > 0 and macd_hist_prev <= 0:
            long_score += 2
        if adx > 25:
            long_score += 1
        if ema_fast > ema_slow and ema_slow > ema_trend:
            long_score += 1

        # SHORT signals
        if rsi > 60:
            short_score += 2
        if macd_hist < 0 and macd_hist_prev >= 0:
            short_score += 2
        if adx > 25:
            short_score += 1
        if ema_fast < ema_slow and ema_slow < ema_trend:
            short_score += 1

        return long_score, short_score

    def next(self):
        if len(self.data) < self.params.ema_trend:
            return

        if self.order:
            return

        if self.check_exit():
            return

        long_score, short_score = self.calculate_score()
        current_price = self.data.close[0]

        # LONG
        if long_score >= self.params.min_score and not self.position:
            stop = current_price * (1 - self.params.stop_loss)
            target = current_price * (1 + self.params.profit_target)
            stop_distance = abs(current_price - stop)
            risk_amount = self.broker.getvalue() * self.params.risk_percent
            size = risk_amount / stop_distance

            self.entry_price = current_price
            self.stop_price = stop
            self.target_price = target
            self.position_type = 'LONG'
            self.order = self.buy(size=size)

        # SHORT
        elif short_score >= self.params.min_score and not self.position:
            stop = current_price * (1 + self.params.stop_loss)
            target = current_price * (1 - self.params.profit_target)
            stop_distance = abs(stop - current_price)
            risk_amount = self.broker.getvalue() * self.params.risk_percent
            size = risk_amount / stop_distance

            self.entry_price = current_price
            self.stop_price = stop
            self.target_price = target
            self.position_type = 'SHORT'
            self.order = self.sell(size=size)


class OANDAData(bt.DataBase):
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


def test_parameters(min_score, risk_percent, profit_target, stop_loss, pair='EUR_USD', days=90):
    """Test a specific parameter combination"""

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        OptimizableStrategy,
        min_score=min_score,
        risk_percent=risk_percent,
        profit_target=profit_target,
        stop_loss=stop_loss
    )

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

    # Get the strategy instance
    strat = results[0]

    # Calculate accurate final value using tracked P&L
    end_value = start_value + strat.total_pnl - strat.total_commission

    # Extract stats
    trade_analysis = strat.analyzers.trades.get_analysis()
    dd_analysis = strat.analyzers.drawdown.get_analysis()

    total_trades = 0
    win_rate = 0
    if 'total' in trade_analysis and 'total' in trade_analysis['total']:
        total_trades = trade_analysis['total']['total']
        won = trade_analysis.get('won', {}).get('total', 0)
        win_rate = (won / total_trades * 100) if total_trades > 0 else 0

    max_dd = 0
    if 'max' in dd_analysis and 'drawdown' in dd_analysis['max']:
        max_dd = dd_analysis['max']['drawdown']

    roi = ((end_value / start_value) - 1) * 100

    return {
        'min_score': min_score,
        'risk_percent': risk_percent,
        'profit_target': profit_target,
        'stop_loss': stop_loss,
        'roi': roi,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'final_value': end_value,
        'e8_compliant': max_dd <= 6.0
    }


def optimize():
    """Run optimization across parameter space"""

    print("=" * 70)
    print("E8 ROI OPTIMIZER - Testing Parameter Combinations")
    print("=" * 70)
    print("Goal: Maximize ROI while keeping drawdown < 6%")
    print("This may take 5-10 minutes...\n")

    # Parameter grid
    min_scores = [2.0, 2.5, 3.0]
    risk_percents = [0.01, 0.015, 0.02]
    profit_targets = [0.02, 0.03, 0.04]
    stop_losses = [0.01, 0.015]

    # Load data once (cache it)
    print("[DATA] Loading EUR_USD historical data...")

    results = []
    total_combinations = len(min_scores) * len(risk_percents) * len(profit_targets) * len(stop_losses)
    tested = 0

    for min_score in min_scores:
        for risk in risk_percents:
            for target in profit_targets:
                for stop in stop_losses:
                    tested += 1
                    print(f"\n[{tested}/{total_combinations}] Testing: MinScore={min_score}, Risk={risk*100}%, Target={target*100}%, Stop={stop*100}%")

                    result = test_parameters(min_score, risk, target, stop)
                    results.append(result)

                    print(f"  > ROI: {result['roi']:+.2f}% | Trades: {result['total_trades']} | Win Rate: {result['win_rate']:.1f}% | Max DD: {result['max_dd']:.2f}%")
                    if result['e8_compliant']:
                        print(f"  [OK] E8 Compliant (DD < 6%)")
                    else:
                        print(f"  [FAIL] Exceeded E8 DD limit (>6%)")

    # Find best E8-compliant result
    e8_compliant = [r for r in results if r['e8_compliant']]

    print("\n\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    if e8_compliant:
        # Sort by ROI
        best = max(e8_compliant, key=lambda x: x['roi'])

        print("\n*** BEST E8-COMPLIANT CONFIGURATION ***")
        print(f"  Min Score: {best['min_score']}")
        print(f"  Risk Per Trade: {best['risk_percent']*100}%")
        print(f"  Profit Target: {best['profit_target']*100}%")
        print(f"  Stop Loss: {best['stop_loss']*100}%")
        print(f"\n  ROI: {best['roi']:+.2f}%")
        print(f"  Total Trades: {best['total_trades']}")
        print(f"  Win Rate: {best['win_rate']:.1f}%")
        print(f"  Max Drawdown: {best['max_dd']:.2f}%")
        print(f"  Final Value: ${best['final_value']:,.2f}")

        # Show top 5
        print("\n*** TOP 5 E8-COMPLIANT CONFIGURATIONS ***")
        top_5 = sorted(e8_compliant, key=lambda x: x['roi'], reverse=True)[:5]
        for i, r in enumerate(top_5, 1):
            print(f"\n{i}. ROI: {r['roi']:+.2f}% | DD: {r['max_dd']:.2f}%")
            print(f"   Score={r['min_score']}, Risk={r['risk_percent']*100}%, Target={r['profit_target']*100}%, Stop={r['stop_loss']*100}%")
            print(f"   Trades: {r['total_trades']} | Win Rate: {r['win_rate']:.1f}%")

    else:
        print("\n[WARNING] NO E8-COMPLIANT CONFIGURATIONS FOUND")
        print("All parameter combinations exceeded 6% drawdown limit")

    print("=" * 70)


if __name__ == '__main__':
    optimize()
