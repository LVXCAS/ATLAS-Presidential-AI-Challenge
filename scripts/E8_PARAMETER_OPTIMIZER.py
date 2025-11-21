"""
E8 FOREX BOT - PARAMETER OPTIMIZER
Tests different parameter combinations to find optimal settings for E8 challenge
"""

import backtrader as bt
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import itertools

class E8OptimizableStrategy(bt.Strategy):
    """Simplified strategy for parameter optimization"""

    params = (
        ('min_score', 3.0),  # Will be optimized
        ('adx_threshold', 20),  # Will be optimized
        ('profit_target_pct', 0.025),
        ('stop_loss_pct', 0.01),
        ('risk_per_trade', 0.008),
        ('max_positions', 3),
    )

    def __init__(self):
        self.orders = {}

        # Simple indicators for each data feed
        self.indicators = {}
        for d in self.datas:
            self.indicators[d._name] = {
                'adx': bt.indicators.AverageDirectionalMovementIndex(d, period=14),
                'rsi': bt.indicators.RSI(d, period=14),
                'macd': bt.indicators.MACD(d),
            }

    def next(self):
        # Count open positions
        open_positions = sum(1 for d in self.datas if self.getposition(d).size != 0)

        if open_positions >= self.params.max_positions:
            return

        # Scan each pair
        for d in self.datas:
            if self.getposition(d).size != 0:
                continue

            ind = self.indicators[d._name]

            # Calculate simple score
            score = 0.0

            # ADX
            if ind['adx'][0] > self.params.adx_threshold:
                score += 2.0

            # RSI
            rsi = ind['rsi'][0]
            if 30 <= rsi <= 45 or 55 <= rsi <= 70:
                score += 2.0

            # MACD
            if ind['macd'].macd[0] > ind['macd'].signal[0]:
                score += 1.0

            if score >= self.params.min_score:
                # Simple buy
                size = (self.broker.getvalue() * self.params.risk_per_trade) / (d.close[0] * self.params.stop_loss_pct)
                self.buy(data=d, size=size)

                # Set TP/SL
                tp_price = d.close[0] * (1 + self.params.profit_target_pct)
                sl_price = d.close[0] * (1 - self.params.stop_loss_pct)
                self.sell(data=d, size=size, exectype=bt.Order.Limit, price=tp_price)
                self.sell(data=d, size=size, exectype=bt.Order.Stop, price=sl_price)


def download_forex_data(pair, start_date, end_date):
    """Download forex data from Yahoo Finance"""
    ticker = pair.replace('/', '') + '=X'
    data = yf.download(ticker, start=start_date, end=end_date, interval='1h', progress=False)

    if data.empty:
        return None

    # Flatten multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [str(col).lower() for col in data.columns]
    return data


def run_backtest_with_params(min_score, adx_threshold, pairs_data):
    """Run backtest with given parameters"""
    cerebro = bt.Cerebro()

    # Add data
    for pair_name, df in pairs_data.items():
        data = bt.feeds.PandasData(
            dataname=df,
            name=pair_name,
            timeframe=bt.TimeFrame.Minutes,
            compression=60,
            datetime=None,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1,
        )
        cerebro.adddata(data)

    # Add strategy with parameters
    cerebro.addstrategy(E8OptimizableStrategy,
                       min_score=min_score,
                       adx_threshold=adx_threshold)

    cerebro.broker.setcash(200000)
    cerebro.broker.setcommission(commission=0.0001)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    # Run
    results = cerebro.run()
    strat = results[0]

    ending_value = cerebro.broker.getvalue()
    profit = ending_value - 200000

    drawdown = strat.analyzers.drawdown.get_analysis()
    trades_analysis = strat.analyzers.trades.get_analysis()

    # Calculate metrics
    total_trades = trades_analysis.total.total if trades_analysis.total.total > 0 else 0
    win_rate = (trades_analysis.won.total / total_trades * 100) if total_trades > 0 else 0
    max_dd = drawdown.max.drawdown if drawdown.max.drawdown else 0

    # Check if passed E8 challenge
    passed = profit >= 20000 and max_dd < 6.0

    return {
        'min_score': min_score,
        'adx_threshold': adx_threshold,
        'profit': profit,
        'max_dd': max_dd,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'passed': passed,
        'score': profit if passed else -100000  # Penalty for failing
    }


if __name__ == '__main__':
    print('=' * 70)
    print('E8 FOREX BOT - PARAMETER OPTIMIZATION')
    print('=' * 70)

    # Download historical data (6 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    print(f'\nDownloading 6 months of data ({start_date.date()} to {end_date.date()})...')

    pairs_data = {}
    for pair_name, ticker in [('EURUSD', 'EUR/USD'), ('GBPUSD', 'GBP/USD'), ('USDJPY', 'USD/JPY')]:
        df = download_forex_data(ticker, start_date, end_date)
        if df is not None and not df.empty:
            pairs_data[pair_name] = df
            print(f'  {pair_name}: {len(df)} bars')

    print(f'\n{len(pairs_data)} pairs loaded')

    # Parameter grid
    min_score_values = [2.0, 2.5, 3.0, 3.5]
    adx_threshold_values = [15, 20, 25]

    print(f'\nTesting {len(min_score_values) * len(adx_threshold_values)} parameter combinations...')
    print()

    results = []
    total_tests = len(min_score_values) * len(adx_threshold_values)
    test_num = 0

    for min_score, adx_threshold in itertools.product(min_score_values, adx_threshold_values):
        test_num += 1
        print(f'[{test_num}/{total_tests}] Testing min_score={min_score}, adx_threshold={adx_threshold}... ', end='', flush=True)

        result = run_backtest_with_params(min_score, adx_threshold, pairs_data)
        results.append(result)

        status = 'PASS' if result['passed'] else 'FAIL'
        print(f'{status} (Profit: ${result["profit"]:,.0f}, DD: {result["max_dd"]:.1f}%, Trades: {result["total_trades"]}, WR: {result["win_rate"]:.0f}%)')

    # Sort by best score
    results.sort(key=lambda x: x['score'], reverse=True)

    print('\n' + '=' * 70)
    print('OPTIMIZATION RESULTS - TOP 5 CONFIGURATIONS')
    print('=' * 70)

    for i, r in enumerate(results[:5], 1):
        print(f'\n#{i}:')
        print(f'  min_score: {r["min_score"]}')
        print(f'  adx_threshold: {r["adx_threshold"]}')
        print(f'  Profit: ${r["profit"]:,.2f}')
        print(f'  Max Drawdown: {r["max_dd"]:.2f}%')
        print(f'  Total Trades: {r["total_trades"]}')
        print(f'  Win Rate: {r["win_rate"]:.1f}%')
        print(f'  Passed E8: {"YES" if r["passed"] else "NO"}')

    # Best configuration
    best = results[0]
    print('\n' + '=' * 70)
    print('RECOMMENDED CONFIGURATION')
    print('=' * 70)

    if best['passed']:
        print(f'\nmin_score = {best["min_score"]}')
        print(f'adx_threshold = {best["adx_threshold"]}')
        print(f'\nExpected Performance:')
        print(f'  - Profit: ${best["profit"]:,.2f}')
        print(f'  - Max Drawdown: {best["max_dd"]:.2f}%')
        print(f'  - Total Trades: {best["total_trades"]}')
        print(f'  - Win Rate: {best["win_rate"]:.1f}%')
        print(f'\nREADY TO DEPLOY')
    else:
        print('\nNO CONFIGURATION PASSED E8 CHALLENGE')
        print('Strategy needs fundamental redesign')

    print('=' * 70)
