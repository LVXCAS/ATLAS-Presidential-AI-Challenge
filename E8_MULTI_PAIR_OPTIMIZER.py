"""
E8 MULTI-PAIR OPTIMIZER
Test EUR/USD, GBP/USD, and USD/JPY simultaneously to maximize combined ROI
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


# ===== OPTIMIZABLE STRATEGY =====
class MultiPairStrategy(bt.Strategy):
    """Strategy that tracks P&L across multiple pairs"""

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
        # Track indicators per data feed (per pair)
        self.indicators = {}

        for i, d in enumerate(self.datas):
            self.indicators[d] = {
                'rsi': bt.indicators.RSI(d.close, period=self.params.rsi_period),
                'macd': bt.indicators.MACD(d.close),
                'adx': bt.indicators.ADX(d, period=self.params.adx_period),
                'ema_fast': bt.indicators.EMA(d.close, period=self.params.ema_fast),
                'ema_slow': bt.indicators.EMA(d.close, period=self.params.ema_slow),
                'ema_trend': bt.indicators.EMA(d.close, period=self.params.ema_trend),
                'atr': bt.indicators.ATR(d)
            }

        self.orders = {}  # Track orders per data feed
        self.entry_prices = {}
        self.stop_prices = {}
        self.target_prices = {}

        # Track cumulative P&L
        self.total_pnl = 0
        self.total_commission = 0

    def notify_order(self, order):
        if order.status in [order.Completed]:
            pass  # Silent
        if order.data not in self.orders:
            return
        if self.orders[order.data] == order:
            self.orders[order.data] = None

    def notify_trade(self, trade):
        """Track closed trades for accurate P&L calculation"""
        if trade.isclosed:
            self.total_pnl += trade.pnl
            self.total_commission += trade.commission

    def check_exit(self, data):
        """Check if we should exit position"""
        if not self.getposition(data):
            return False

        current_price = data.close[0]

        # Check stop loss or profit target
        if data in self.stop_prices and current_price <= self.stop_prices[data]:
            self.close(data=data)
            return True
        elif data in self.target_prices and current_price >= self.target_prices[data]:
            self.close(data=data)
            return True

        return False

    def calculate_score(self, data):
        """Calculate entry signal score"""
        ind = self.indicators[data]

        rsi = ind['rsi'][0]
        macd_hist = ind['macd'].macd[0] - ind['macd'].signal[0]
        macd_hist_prev = ind['macd'].macd[-1] - ind['macd'].signal[-1]
        adx = ind['adx'][0]
        ema_fast = ind['ema_fast'][0]
        ema_slow = ind['ema_slow'][0]
        ema_trend = ind['ema_trend'][0]

        long_score = 0
        short_score = 0

        # LONG signals
        if rsi < 40:
            long_score += 2
        if macd_hist > 0 and macd_hist_prev <= 0:
            long_score += 2
        if adx > 25:
            long_score += 1
        if ema_fast > ema_slow > ema_trend:
            long_score += 1

        # SHORT signals
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
        # Process each data feed (each pair)
        for data in self.datas:
            # Check exit first
            if self.check_exit(data):
                continue

            # Skip if already in position
            if self.getposition(data):
                continue

            # Check for entry signal
            signal, score = self.calculate_score(data)
            if signal is None:
                continue

            # Calculate position size
            account_value = self.broker.getvalue()
            risk_amount = account_value * self.params.risk_percent

            atr = self.indicators[data]['atr'][0]
            stop_distance = data.close[0] * self.params.stop_loss

            # Position sizing based on risk
            position_size = int(risk_amount / stop_distance) if stop_distance > 0 else 0

            if position_size <= 0:
                continue

            # Enter position
            if signal == 'LONG':
                self.orders[data] = self.buy(data=data, size=position_size)
                self.entry_prices[data] = data.close[0]
                self.stop_prices[data] = data.close[0] * (1 - self.params.stop_loss)
                self.target_prices[data] = data.close[0] * (1 + self.params.profit_target)
            elif signal == 'SHORT':
                self.orders[data] = self.sell(data=data, size=position_size)
                self.entry_prices[data] = data.close[0]
                self.stop_prices[data] = data.close[0] * (1 + self.params.stop_loss)
                self.target_prices[data] = data.close[0] * (1 - self.params.profit_target)


def test_multi_pair(min_score=2.5, risk_percent=0.02, profit_target=0.02, stop_loss=0.01, days=90):
    """Test strategy across all 3 pairs simultaneously"""

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        MultiPairStrategy,
        min_score=min_score,
        risk_percent=risk_percent,
        profit_target=profit_target,
        stop_loss=stop_loss
    )

    # Load data for all 3 pairs
    api = API(access_token=os.getenv('OANDA_API_KEY'))
    end_date = datetime(2024, 11, 10)
    start_date = end_date - timedelta(days=days)

    pairs = {
        'EUR_USD': 'EUR_USD',
        'GBP_USD': 'GBP_USD',
        'USD_JPY': 'USD_JPY'
    }

    print(f"\n[DATA] Loading historical data for all 3 pairs...")
    for name, instrument in pairs.items():
        data = OANDAData(
            instrument=instrument,
            api=api,
            start_date=start_date,
            end_date=end_date,
            granularity='H1'
        )
        cerebro.adddata(data, name=name)

    cerebro.broker.setcash(200000.0)
    cerebro.broker.setcommission(commission=0.0001)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    start_value = cerebro.broker.getvalue()
    results = cerebro.run()

    strat = results[0]
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
        'roi': roi,
        'trades': total_trades,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'final_value': end_value,
        'e8_compliant': max_dd < 6.0
    }


def optimize_multi_pair():
    """Test best single-pair config on all 3 pairs simultaneously"""

    print("=" * 70)
    print("E8 MULTI-PAIR OPTIMIZER")
    print("=" * 70)
    print("Testing all 3 pairs (EUR/USD, GBP/USD, USD/JPY) simultaneously")
    print("Using best single-pair config: Score=2.5, Risk=2%, Target=2%, Stop=1%")
    print("=" * 70)

    # Test best config on all pairs
    result = test_multi_pair(
        min_score=2.5,
        risk_percent=0.02,
        profit_target=0.02,
        stop_loss=0.01,
        days=90
    )

    print("\n" + "=" * 70)
    print("MULTI-PAIR RESULTS (90 Days)")
    print("=" * 70)
    print(f"Starting Value: $200,000.00")
    print(f"Ending Value: ${result['final_value']:,.2f}")
    print(f"Total Return: ${result['final_value'] - 200000:,.2f}")
    print(f"ROI: {result['roi']:+.2f}%")
    print(f"Total Trades: {result['trades']}")
    print(f"Win Rate: {result['win_rate']:.1f}%")
    print(f"Max Drawdown: {result['max_dd']:.2f}%")
    print(f"E8 Compliant: {'[OK]' if result['e8_compliant'] else '[FAIL]'}")
    print("=" * 70)

    # Compare to single-pair
    single_pair_roi = 8.47
    improvement = ((result['roi'] / single_pair_roi) - 1) * 100 if single_pair_roi > 0 else 0

    print(f"\nCOMPARISON:")
    print(f"  Single Pair (EUR/USD only): +{single_pair_roi}% ROI")
    print(f"  Multi-Pair (All 3): +{result['roi']:.2f}% ROI")
    print(f"  Improvement: {improvement:+.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    optimize_multi_pair()
