"""
E8 NEW PAIRS OPTIMIZER - Using TradeLocker Data
Test AUD/USD, USD/CAD, NZD/USD, EUR/GBP with parameter grid search
"""

import backtrader as bt
from tradelocker import TLAPI
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Test the 4 new pairs
pairs_to_test = {
    'AUDUSD+': 6120,
    'USDCAD+': 6125,
    'NZDUSD+': 6112,
    'EURGBP+': 6136
}

print('=' * 70)
print('BACKTESTING 4 NEW FOREX PAIRS - TRADELOCKER DATA')
print('Testing each pair with parameter grid search (54 configs)')
print('=' * 70)
print()

# Import TA-Lib
try:
    import talib
    print('[OK] TA-Lib available')
except ImportError:
    print('[ERROR] TA-Lib not available - cannot proceed')
    exit(1)

# TradeLocker API setup
tl = TLAPI(environment='https://demo.tradelocker.com', username='kkdo@hotmail.com', password='nVFW7@6P', server='E8-Live')
print('[OK] TradeLocker connected')
print()

class E8Strategy(bt.Strategy):
    params = (
        ('min_score', 2.5),
        ('risk_pct', 0.02),
        ('profit_target_pct', 0.02),
        ('stop_loss_pct', 0.01),
    )

    def __init__(self):
        self.total_pnl = 0
        self.total_commission = 0
        self.trades_log = []

    def notify_trade(self, trade):
        if trade.isclosed:
            self.total_pnl += trade.pnl
            self.total_commission += trade.commission
            self.trades_log.append({
                'pnl': trade.pnl,
                'size': trade.size,
                'price': trade.price
            })

    def next(self):
        if len(self.data) < 200:
            return

        if self.position:
            return

        # Calculate indicators
        closes = np.array([self.data.close[-i] for i in range(200, 0, -1)])
        highs = np.array([self.data.high[-i] for i in range(200, 0, -1)])
        lows = np.array([self.data.low[-i] for i in range(200, 0, -1)])

        try:
            rsi = talib.RSI(closes, timeperiod=14)
            macd, signal, _ = talib.MACD(closes)
            adx = talib.ADX(highs, lows, closes, timeperiod=14)
            ema_fast = talib.EMA(closes, timeperiod=10)
            ema_slow = talib.EMA(closes, timeperiod=21)
            ema_trend = talib.EMA(closes, timeperiod=200)

            score = 0
            direction = 'long'

            # LONG signals
            if rsi[-1] < 40:
                score += 2
            if len(macd) > 1 and macd[-1] > signal[-1] and macd[-2] <= signal[-2]:
                score += 2
                direction = 'long'
            if adx[-1] > 25:
                score += 1
            if ema_fast[-1] > ema_slow[-1] > ema_trend[-1]:
                score += 1
                direction = 'long'

            # SHORT signals
            if rsi[-1] > 60:
                score += 2
            if len(macd) > 1 and macd[-1] < signal[-1] and macd[-2] >= signal[-2]:
                score += 2
                direction = 'short'
            if ema_fast[-1] < ema_slow[-1] < ema_trend[-1]:
                score += 1
                direction = 'short'

            if score >= self.params.min_score:
                # Calculate position size based on risk
                account_value = self.broker.getvalue()
                risk_amount = account_value * self.params.risk_pct
                stop_distance = self.data.close[0] * self.params.stop_loss_pct
                position_size = risk_amount / stop_distance

                if direction == 'long':
                    self.buy(size=position_size)
                else:
                    self.sell(size=position_size)

        except Exception as e:
            pass

def fetch_tradelocker_data(instrument_id, days=90):
    """Fetch historical data from TradeLocker"""
    try:
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        # Get historical bars - H1 (60 minutes)
        bars = tl.get_price_history(
            tradable_instrument_id=instrument_id,
            resolution='60',
            start_timestamp=start_time,
            end_timestamp=end_time,
            lookback=days * 24  # 24 hours per day
        )

        if bars.empty:
            return None

        return bars

    except Exception as e:
        print(f'[ERROR] Fetching data: {e}')
        return None

def run_backtest(pair_name, instrument_id, min_score, risk_pct, target_pct, stop_pct):
    """Run single backtest with given parameters"""
    bars = fetch_tradelocker_data(instrument_id, days=90)

    if bars is None or len(bars) < 200:
        return None

    try:
        # Prepare data for Backtrader
        df = bars[['t', 'o', 'h', 'l', 'c']].copy()
        df.columns = ['datetime', 'open', 'high', 'low', 'close']
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df['volume'] = 1000  # Placeholder volume
        df['openinterest'] = 0
        df = df.set_index('datetime')
        df = df.sort_index()

        # Create cerebro
        cerebro = bt.Cerebro()

        # Add data
        data_feed = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data_feed)

        # Add strategy
        cerebro.addstrategy(
            E8Strategy,
            min_score=min_score,
            risk_pct=risk_pct,
            profit_target_pct=target_pct,
            stop_loss_pct=stop_pct
        )

        # Set broker
        cerebro.broker.setcash(200000.0)
        cerebro.broker.setcommission(commission=0.0001)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        # Run
        start_value = cerebro.broker.getvalue()
        results = cerebro.run()
        strat = results[0]

        # Calculate accurate ROI using tracked P&L
        end_value = start_value + strat.total_pnl - strat.total_commission
        roi = ((end_value - start_value) / start_value) * 100

        # Get drawdown
        dd_analyzer = strat.analyzers.drawdown.get_analysis()
        max_dd = dd_analyzer.get('max', {}).get('drawdown', 0)

        # Get trade stats
        trade_analyzer = strat.analyzers.trades.get_analysis()
        total_trades = trade_analyzer.get('total', {}).get('closed', 0)
        won_trades = trade_analyzer.get('won', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

        return {
            'roi': roi,
            'max_dd': max_dd,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'e8_compliant': max_dd < 6.0
        }

    except Exception as e:
        print(f'[ERROR] Backtest: {e}')
        return None

# Test each pair with parameter grid
score_thresholds = [2.0, 2.5, 3.0]
risk_pcts = [0.015, 0.02, 0.025]
target_pcts = [0.02, 0.03]
stop_pcts = [0.01, 0.015]

results_summary = {}

for pair_name, instrument_id in pairs_to_test.items():
    print(f'\nTesting {pair_name}...')
    best_result = None
    best_params = None

    configs_tested = 0
    for score in score_thresholds:
        for risk in risk_pcts:
            for target in target_pcts:
                for stop in stop_pcts:
                    configs_tested += 1
                    print(f'  [{configs_tested}/54] Score={score}, Risk={risk*100}%, Target={target*100}%, Stop={stop*100}%', end=' ')

                    result = run_backtest(pair_name, instrument_id, score, risk, target, stop)

                    if result and result['e8_compliant'] and result['total_trades'] > 0:
                        print(f'> ROI={result["roi"]:.2f}%, DD={result["max_dd"]:.2f}%, Trades={result["total_trades"]}')

                        if best_result is None or result['roi'] > best_result['roi']:
                            best_result = result
                            best_params = {
                                'score': score,
                                'risk': risk,
                                'target': target,
                                'stop': stop
                            }
                    else:
                        print('> FAIL')

    if best_result:
        results_summary[pair_name] = {
            'params': best_params,
            'performance': best_result
        }
        print(f'\n*** BEST {pair_name} ***')
        print(f'  Score={best_params["score"]}, Risk={best_params["risk"]*100}%, Target={best_params["target"]*100}%, Stop={best_params["stop"]*100}%')
        print(f'  ROI: +{best_result["roi"]:.2f}%')
        print(f'  Trades: {best_result["total_trades"]}')
        print(f'  Win Rate: {best_result["win_rate"]:.1f}%')
        print(f'  Max DD: {best_result["max_dd"]:.2f}%')
    else:
        print(f'\n*** FAIL {pair_name}: No E8-compliant configuration found ***')

# Calculate combined results
print('\n' + '=' * 70)
print('COMBINED RESULTS')
print('=' * 70)

if results_summary:
    total_new_roi = sum(r['performance']['roi'] for r in results_summary.values())
    current_roi = 25.16  # From existing 3 pairs
    projected_total = current_roi + total_new_roi

    print(f'Current 3 pairs ROI: {current_roi:.2f}% per 90 days')
    print(f'New {len(results_summary)} pairs ROI: {total_new_roi:.2f}% per 90 days')
    print(f'TOTAL PROJECTED: {projected_total:.2f}% per 90 days')
    print(f'Annualized: {(projected_total/90)*365:.2f}% per year')
    print(f'Days to E8 pass: {(10/(projected_total/100)):.0f} days')
    print()

    if projected_total > 30:  # Threshold for adding pairs
        print('*** RECOMMENDATION: DEPLOY - New pairs significantly increase ROI! ***')
        print('\nDetailed results:')
        for pair_name, data in results_summary.items():
            print(f'  {pair_name}: +{data["performance"]["roi"]:.2f}% ROI')
    else:
        print('*** RECOMMENDATION: SKIP - New pairs provide minimal improvement ***')
else:
    print('*** FAIL: No new pairs passed E8 compliance tests ***')

print('=' * 70)
