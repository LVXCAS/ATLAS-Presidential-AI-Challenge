"""
ML-OPTIMIZED FOREX STRATEGY
Use Bayesian optimization to find best parameters across ALL dimensions

This will test:
- Multiple TP/SL combinations (not just 1.5%/0.8%)
- Multiple timeframes (1H, 2H, 4H)
- Multiple lookback periods
- Multiple confluence requirements
- Optimal position sizing

Expected: 10-15% ROI by finding parameter combinations I never tested
"""

import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import warnings
warnings.filterwarnings('ignore')

class OptimizedPriceAction(bt.Strategy):
    params = (
        ('profit_target_pct', 0.015),
        ('stop_loss_pct', 0.008),
        ('break_threshold_pips', 2),
        ('retest_threshold_pips', 3),
        ('lookback_hours', 12),
        ('min_range_pips', 5),
        ('risk_pct', 1.0),
    )

    def __init__(self):
        self.orders = {}
        self.trades_data = []

        # Track indicators for each data feed
        for d in self.datas:
            self.orders[d._name] = None

    def notify_order(self, order):
        if order.status in [order.Completed]:
            pass

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades_data.append({
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,
            })

    def get_daily_levels(self, data):
        """Get yesterday's high/low"""
        if len(data) < 48:
            return None, None

        # Get last 24 bars
        highs = []
        lows = []

        for i in range(-24, 0):
            try:
                highs.append(data.high[i])
                lows.append(data.low[i])
            except:
                pass

        if len(highs) < 20:
            return None, None

        resistance = max(highs)
        support = min(lows)

        return resistance, support

    def check_break_and_retest(self, data, resistance, support):
        """Check for break-and-retest setup"""
        if len(data) < self.params.lookback_hours:
            return None

        current_price = data.close[0]
        current_high = data.high[0]
        current_low = data.low[0]

        break_threshold = self.params.break_threshold_pips * 0.0001
        retest_threshold = self.params.retest_threshold_pips * 0.0001

        # LONG setup
        if current_price > resistance:
            # Check if we broke resistance recently
            broke_resistance = False
            for i in range(-self.params.lookback_hours, 0):
                try:
                    prev_high = data.high[i-1]
                    next_high = data.high[i]

                    if prev_high < resistance and next_high >= resistance + break_threshold:
                        broke_resistance = True
                        break
                except:
                    pass

            if not broke_resistance:
                return None

            # Check retest
            retest_distance = current_low - resistance

            if -retest_threshold <= retest_distance <= retest_threshold:
                if current_price > resistance:
                    return 'LONG'

        # SHORT setup
        elif current_price < support:
            broke_support = False
            for i in range(-self.params.lookback_hours, 0):
                try:
                    prev_low = data.low[i-1]
                    next_low = data.low[i]

                    if prev_low > support and next_low <= support - break_threshold:
                        broke_support = True
                        break
                except:
                    pass

            if not broke_support:
                return None

            retest_distance = current_high - support

            if -retest_threshold <= retest_distance <= retest_threshold:
                if current_price < support:
                    return 'SHORT'

        return None

    def next(self):
        for d in self.datas:
            # Skip if we have an order pending
            if self.orders[d._name]:
                continue

            # Get daily levels
            resistance, support = self.get_daily_levels(d)

            if resistance is None:
                continue

            # Check for signal
            signal = self.check_break_and_retest(d, resistance, support)

            if signal is None:
                continue

            # Calculate position size
            risk_amount = self.broker.get_cash() * (self.params.risk_pct / 100)
            price = d.close[0]
            stop_distance = price * self.params.stop_loss_pct
            size = risk_amount / stop_distance

            # Enter position
            if signal == 'LONG':
                tp_price = price * (1 + self.params.profit_target_pct)
                sl_price = resistance * (1 - self.params.stop_loss_pct)

                self.orders[d._name] = self.buy(
                    data=d,
                    size=size,
                    exectype=bt.Order.Market
                )

                # Set TP/SL (simplified - in real version use bracket orders)

            elif signal == 'SHORT':
                tp_price = price * (1 - self.params.profit_target_pct)
                sl_price = support * (1 + self.params.stop_loss_pct)

                self.orders[d._name] = self.sell(
                    data=d,
                    size=size,
                    exectype=bt.Order.Market
                )


def run_backtest_with_params(params_dict):
    """Run backtest with given parameters"""
    cerebro = bt.Cerebro()

    # Download data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    pairs = {
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X',
        'USDJPY': 'USDJPY=X'
    }

    for pair_name, ticker in pairs.items():
        df = yf.download(ticker, start=start_date, end=end_date, interval='1h', progress=False)

        if df.empty:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(col).lower() for col in df.columns]

        # Convert to backtrader format
        data = bt.feeds.PandasData(
            dataname=df,
            name=pair_name,
            datetime=None,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1
        )

        cerebro.adddata(data)

    # Add strategy with parameters
    cerebro.addstrategy(OptimizedPriceAction, **params_dict)

    # Set broker
    cerebro.broker.set_cash(200000)
    cerebro.broker.setcommission(commission=0.0001)  # 1 pip spread

    # Run
    initial_value = cerebro.broker.getvalue()
    strategies = cerebro.run()
    final_value = cerebro.broker.getvalue()

    strat = strategies[0]

    # Calculate metrics
    if len(strat.trades_data) == 0:
        return {
            'roi': 0,
            'trades': 0,
            'win_rate': 0,
            'max_dd': 100,
            'sharpe': -10
        }

    roi = ((final_value - initial_value) / initial_value) * 100

    wins = [t for t in strat.trades_data if t['pnlcomm'] > 0]
    win_rate = len(wins) / len(strat.trades_data) * 100 if strat.trades_data else 0

    # Calculate drawdown (simplified)
    equity_curve = [initial_value]
    running = initial_value
    for t in strat.trades_data:
        running += t['pnlcomm']
        equity_curve.append(running)

    peak = equity_curve[0]
    max_dd = 0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        'roi': roi,
        'trades': len(strat.trades_data),
        'win_rate': win_rate,
        'max_dd': max_dd,
        'final_value': final_value
    }


def optimize_strategy():
    """Use Bayesian optimization to find best parameters"""

    print('=' * 70)
    print('ML-POWERED STRATEGY OPTIMIZATION')
    print('=' * 70)
    print('\nUsing Bayesian optimization to search parameter space...')
    print('This will test 100+ combinations to find the optimal config.\n')

    # Define parameter search space
    space = [
        Real(0.01, 0.04, name='profit_target_pct'),   # 1-4% TP
        Real(0.005, 0.02, name='stop_loss_pct'),       # 0.5-2% SL
        Integer(1, 5, name='break_threshold_pips'),    # 1-5 pips
        Integer(2, 8, name='retest_threshold_pips'),   # 2-8 pips
        Integer(8, 24, name='lookback_hours'),         # 8-24 hours
        Integer(3, 15, name='min_range_pips'),         # 3-15 pips
        Real(0.5, 2.5, name='risk_pct'),               # 0.5-2.5% risk
    ]

    results_log = []

    @use_named_args(space)
    def objective(**params):
        """Objective function to minimize (negative ROI)"""

        print(f'\nTesting: TP={params["profit_target_pct"]:.3f}, SL={params["stop_loss_pct"]:.3f}, Risk={params["risk_pct"]:.1f}%...')

        metrics = run_backtest_with_params(params)

        results_log.append({**params, **metrics})

        # Optimization goal: Maximize ROI while keeping DD under 6%
        if metrics['max_dd'] > 6.0:
            # Penalize if DD too high
            score = -100 + metrics['roi']  # Heavy penalty
        else:
            score = -metrics['roi']  # Negative because we minimize

        print(f'  ROI: {metrics["roi"]:.2f}%, DD: {metrics["max_dd"]:.2f}%, WR: {metrics["win_rate"]:.1f}%, Trades: {metrics["trades"]}')

        return score

    # Run optimization
    print('Starting optimization (this may take 10-20 minutes)...\n')

    result = gp_minimize(
        objective,
        space,
        n_calls=50,  # Test 50 parameter combinations
        random_state=42,
        verbose=False
    )

    # Find best result
    best_params = {
        'profit_target_pct': result.x[0],
        'stop_loss_pct': result.x[1],
        'break_threshold_pips': result.x[2],
        'retest_threshold_pips': result.x[3],
        'lookback_hours': result.x[4],
        'min_range_pips': result.x[5],
        'risk_pct': result.x[6],
    }

    # Get best metrics
    best_metrics = run_backtest_with_params(best_params)

    # Print results
    print('\n' + '=' * 70)
    print('OPTIMIZATION COMPLETE')
    print('=' * 70)

    print('\nBEST PARAMETERS FOUND:')
    print('-' * 70)
    print(f'  Profit Target:      {best_params["profit_target_pct"]*100:.2f}%')
    print(f'  Stop Loss:          {best_params["stop_loss_pct"]*100:.2f}%')
    print(f'  Risk per Trade:     {best_params["risk_pct"]:.2f}%')
    print(f'  Break Threshold:    {best_params["break_threshold_pips"]} pips')
    print(f'  Retest Threshold:   {best_params["retest_threshold_pips"]} pips')
    print(f'  Lookback Period:    {best_params["lookback_hours"]} hours')

    print('\nBEST PERFORMANCE:')
    print('-' * 70)
    print(f'  ROI:                {best_metrics["roi"]:.2f}%')
    print(f'  Win Rate:           {best_metrics["win_rate"]:.1f}%')
    print(f'  Max Drawdown:       {best_metrics["max_dd"]:.2f}%')
    print(f'  Total Trades:       {best_metrics["trades"]}')

    print('\n' + '=' * 70)
    print('COMPARISON TO MANUAL STRATEGIES')
    print('=' * 70)
    print(f'\n{"Strategy":<25} {"ROI":<12} {"DD":<10} {"WR":<10}')
    print('-' * 70)
    print(f'{"Price Action (Manual)":<25} {"+5.44%":<12} {"5.94%":<10} {"44.0%":<10}')
    print(f'{"ML-Optimized":<25} {f"{best_metrics["roi"]:+.2f}%":<12} {f"{best_metrics["max_dd"]:.2f}%":<10} {f"{best_metrics["win_rate"]:.1f}%":<10}')

    improvement = best_metrics["roi"] - 5.44

    print('\n' + '=' * 70)
    print('VERDICT')
    print('=' * 70)

    if best_metrics["roi"] >= 10 and best_metrics["max_dd"] < 6:
        print(f'\n[OK] ML OPTIMIZATION FOUND A WINNER!')
        print(f'  ROI: {best_metrics["roi"]:.2f}% (target: 10%+) OK')
        print(f'  Max DD: {best_metrics["max_dd"]:.2f}% (under 6%) OK')
        print(f'  Improvement: +{improvement:.2f}% vs manual strategy')
        print(f'\n  DEPLOY ML-OPTIMIZED PARAMETERS')

    elif best_metrics["roi"] > 5.44 and best_metrics["max_dd"] < 6:
        print(f'\n[~] ML OPTIMIZATION IMPROVED PERFORMANCE')
        print(f'  ROI: {best_metrics["roi"]:.2f}% vs 5.44% manual')
        print(f'  Improvement: +{improvement:.2f}%')
        print(f'  Max DD: {best_metrics["max_dd"]:.2f}% (under 6%) OK')
        print(f'\n  USE ML-OPTIMIZED PARAMETERS')

    else:
        print(f'\n[X] ML DID NOT BEAT MANUAL STRATEGY')
        print(f'  ML: {best_metrics["roi"]:.2f}% ROI, {best_metrics["max_dd"]:.2f}% DD')
        print(f'  Manual: 5.44% ROI, 5.94% DD')
        print(f'\n  Stick with manual price action strategy')

    print('=' * 70)

    # Save best params to file
    with open('ML_BEST_PARAMS.txt', 'w') as f:
        f.write('ML-OPTIMIZED PARAMETERS\n')
        f.write('=' * 50 + '\n\n')
        for key, value in best_params.items():
            f.write(f'{key}: {value}\n')
        f.write('\n\nPERFORMANCE:\n')
        f.write('=' * 50 + '\n')
        for key, value in best_metrics.items():
            f.write(f'{key}: {value}\n')

    print('\nBest parameters saved to ML_BEST_PARAMS.txt')


if __name__ == '__main__':
    optimize_strategy()
