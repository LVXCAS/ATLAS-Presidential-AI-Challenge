"""
E8 PER-PAIR OPTIMIZER
Optimize parameters for each forex pair individually, then find best combination
"""

import backtrader as bt
import oandapyV20
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

# ===== BACKTRADER DATA FEED FOR OANDA =====
class OANDAData(bt.DataBase):
    """Custom Backtrader data feed from OANDA"""

    params = (
        ('instrument', ''),
        ('api', None),
        ('start_date', None),
        ('end_date', None),
        ('granularity', 'H1'),
    )

    def __init__(self):
        super(OANDAData, self).__init__()
        self.candles = []
        self.idx = 0
        self._fetch_data()

    def _fetch_data(self):
        """Fetch historical data from OANDA"""
        params = {
            'from': self.params.start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'to': self.params.end_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'granularity': self.params.granularity
        }

        req = InstrumentsCandles(instrument=self.params.instrument, params=params)
        response = self.params.api.request(req)

        for candle in response['candles']:
            if candle['complete']:
                self.candles.append({
                    'datetime': datetime.strptime(candle['time'][:19], '%Y-%m-%dT%H:%M:%S'),
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': float(candle['volume'])
                })

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

        self.idx += 1
        return True


# ===== STRATEGY =====
class OptimizableStrategy(bt.Strategy):
    """Single-pair strategy with P&L tracking"""

    params = (
        ('min_score', 2.5),
        ('risk_percent', 0.02),
        ('profit_target', 0.02),
        ('stop_loss', 0.01),
        ('rsi_period', 14),
        ('adx_period', 14),
        ('ema_fast', 10),
        ('ema_slow', 21),
        ('ema_trend', 200),
    )

    def __init__(self):
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

        # Track cumulative P&L
        self.total_pnl = 0
        self.total_commission = 0

    def notify_order(self, order):
        if order.status in [order.Completed]:
            pass
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.total_pnl += trade.pnl
            self.total_commission += trade.commission

    def check_exit(self):
        if not self.position:
            return False

        current_price = self.data.close[0]

        if self.position_type == 'LONG':
            if current_price <= self.stop_price:
                self.close()
                return True
            elif current_price >= self.target_price:
                self.close()
                return True
        elif self.position_type == 'SHORT':
            if current_price >= self.stop_price:
                self.close()
                return True
            elif current_price <= self.target_price:
                self.close()
                return True

        return False

    def calculate_score(self):
        if len(self.data) < self.params.ema_trend:
            return None, 0

        rsi = self.rsi[0]
        macd_hist = self.macd.macd[0] - self.macd.signal[0]
        macd_hist_prev = self.macd.macd[-1] - self.macd.signal[-1]
        adx = self.adx[0]
        ema_fast = self.ema_fast[0]
        ema_slow = self.ema_slow[0]
        ema_trend = self.ema_trend[0]

        long_score = 0
        short_score = 0

        if rsi < 40:
            long_score += 2
        if macd_hist > 0 and macd_hist_prev <= 0:
            long_score += 2
        if adx > 25:
            long_score += 1
        if ema_fast > ema_slow > ema_trend:
            long_score += 1

        if rsi > 60:
            short_score += 2
        if macd_hist < 0 and macd_hist_prev >= 0:
            short_score += 2
        if adx > 25:
            short_score += 1
        if ema_fast < ema_slow < ema_trend:
            short_score += 1

        if long_score >= self.params.min_score:
            return 'LONG', long_score
        elif short_score >= self.params.min_score:
            return 'SHORT', short_score

        return None, 0

    def next(self):
        if self.check_exit():
            return

        if self.position:
            return

        signal, score = self.calculate_score()
        if signal is None:
            return

        account_value = self.broker.getvalue()
        risk_amount = account_value * self.params.risk_percent
        stop_distance = self.data.close[0] * self.params.stop_loss
        position_size = int(risk_amount / stop_distance) if stop_distance > 0 else 0

        if position_size <= 0:
            return

        if signal == 'LONG':
            self.order = self.buy(size=position_size)
            self.entry_price = self.data.close[0]
            self.stop_price = self.data.close[0] * (1 - self.params.stop_loss)
            self.target_price = self.data.close[0] * (1 + self.params.profit_target)
            self.position_type = 'LONG'
        elif signal == 'SHORT':
            self.order = self.sell(size=position_size)
            self.entry_price = self.data.close[0]
            self.stop_price = self.data.close[0] * (1 + self.params.stop_loss)
            self.target_price = self.data.close[0] * (1 - self.params.profit_target)
            self.position_type = 'SHORT'


def test_parameters(pair, min_score, risk_percent, profit_target, stop_loss, days=90):
    """Test parameters for a single pair"""

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        OptimizableStrategy,
        min_score=min_score,
        risk_percent=risk_percent,
        profit_target=profit_target,
        stop_loss=stop_loss
    )

    api = API(access_token=os.getenv('OANDA_API_KEY'))
    end_date = datetime(2024, 11, 10)
    start_date = end_date - timedelta(days=days)

    data = OANDAData(
        instrument=pair,
        api=api,
        start_date=start_date,
        end_date=end_date,
        granularity='H1'
    )
    cerebro.adddata(data)

    cerebro.broker.setcash(200000.0)
    cerebro.broker.setcommission(commission=0.0001)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    start_value = cerebro.broker.getvalue()
    results = cerebro.run()

    strat = results[0]
    end_value = start_value + strat.total_pnl - strat.total_commission

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
        'roi': roi,
        'trades': total_trades,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'final_value': end_value,
        'e8_compliant': max_dd < 6.0
    }


def optimize_per_pair():
    """Optimize each pair individually"""

    print("=" * 70)
    print("E8 PER-PAIR OPTIMIZER")
    print("=" * 70)
    print("Testing EUR/USD, GBP/USD, USD/JPY with optimized parameters")
    print("Finding best config for each pair individually")
    print("=" * 70)

    pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']

    # Test grid for each pair
    param_grid = {
        'min_score': [2.0, 2.5, 3.0],
        'risk': [0.015, 0.02, 0.025],  # 1.5%, 2%, 2.5%
        'target': [0.02, 0.025, 0.03],  # 2%, 2.5%, 3%
        'stop': [0.01, 0.015]  # 1%, 1.5%
    }

    best_results = {}

    for pair in pairs:
        print(f"\n[{pair}] Loading data...")

        best_config = None
        best_roi = -999

        configs_tested = 0
        total_configs = len(param_grid['min_score']) * len(param_grid['risk']) * len(param_grid['target']) * len(param_grid['stop'])

        for min_score in param_grid['min_score']:
            for risk in param_grid['risk']:
                for target in param_grid['target']:
                    for stop in param_grid['stop']:
                        configs_tested += 1

                        result = test_parameters(pair, min_score, risk, target, stop)

                        # Only consider E8-compliant configs
                        if result['e8_compliant'] and result['roi'] > best_roi:
                            best_roi = result['roi']
                            best_config = {
                                'min_score': min_score,
                                'risk': risk,
                                'target': target,
                                'stop': stop,
                                'result': result
                            }

                        if configs_tested % 6 == 0:
                            print(f"  [{configs_tested}/{total_configs}] Best so far: {best_roi:+.2f}% ROI")

        best_results[pair] = best_config

        if best_config:
            r = best_config['result']
            print(f"\n[{pair}] BEST CONFIG:")
            print(f"  Score={best_config['min_score']}, Risk={best_config['risk']*100:.1f}%, Target={best_config['target']*100:.1f}%, Stop={best_config['stop']*100:.1f}%")
            print(f"  ROI: {r['roi']:+.2f}% | Trades: {r['trades']} | Win Rate: {r['win_rate']:.1f}% | DD: {r['max_dd']:.2f}%")
        else:
            print(f"\n[{pair}] NO E8-COMPLIANT CONFIG FOUND")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - BEST CONFIG PER PAIR")
    print("=" * 70)

    total_roi = 0
    total_trades = 0
    compliant_pairs = 0

    for pair, config in best_results.items():
        if config:
            r = config['result']
            print(f"\n{pair}:")
            print(f"  Parameters: Score={config['min_score']}, Risk={config['risk']*100:.1f}%, Target={config['target']*100:.1f}%, Stop={config['stop']*100:.1f}%")
            print(f"  Performance: ROI={r['roi']:+.2f}%, Trades={r['trades']}, Win={r['win_rate']:.1f}%, DD={r['max_dd']:.2f}%")
            total_roi += r['roi']
            total_trades += r['trades']
            compliant_pairs += 1

    print(f"\n" + "=" * 70)
    print(f"COMBINED PROJECTION (if run simultaneously):")
    print(f"  Total ROI: ~{total_roi:.2f}% (sum of individual pairs)")
    print(f"  Total Trades: ~{total_trades}")
    print(f"  E8-Compliant Pairs: {compliant_pairs}/3")
    print("=" * 70)


if __name__ == '__main__':
    optimize_per_pair()
