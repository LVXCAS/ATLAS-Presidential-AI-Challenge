"""
Backtest the FIXED strategy on real historical data
Same data that produced 16.7% win rate with old strategy
"""

import backtrader as bt
import yfinance as yf
import pandas as pd
import talib
from datetime import datetime, timedelta

class FixedForexStrategy(bt.Strategy):
    """
    CORRECTED strategy:
    - Trade WITH RSI momentum (not oversold/overbought)
    - Enter on MACD momentum (not lagging crossover)
    - ONLY trade WITH the trend
    """

    params = (
        ('min_score', 6.0),  # Require strong confluence
        ('profit_target_pct', 0.02),  # 2% TP
        ('stop_loss_pct', 0.01),  # 1% SL
        ('risk_per_trade', 0.01),  # 1% risk
        ('max_positions', 3),
    )

    def __init__(self):
        self.indicators = {}

        for d in self.datas:
            self.indicators[d._name] = {
                'rsi': bt.indicators.RSI(d, period=14),
                'macd': bt.indicators.MACD(d),
                'adx': bt.indicators.AverageDirectionalMovementIndex(d, period=14),
                'ema_20': bt.indicators.EMA(d, period=20),
                'ema_50': bt.indicators.EMA(d, period=50),
                'ema_200': bt.indicators.EMA(d, period=200),
            }

        self.order_placed = {}

    def next(self):
        # Count positions
        open_pos = sum(1 for d in self.datas if self.getposition(d).size != 0)
        if open_pos >= self.params.max_positions:
            return

        for d in self.datas:
            if self.getposition(d).size != 0:
                continue

            signal = self.calculate_signal(d)

            if signal['direction'] == 'LONG' and signal['score'] >= self.params.min_score:
                size = (self.broker.getvalue() * self.params.risk_per_trade) / (d.close[0] * self.params.stop_loss_pct)
                self.buy(data=d, size=size)

                # Set TP/SL
                tp = d.close[0] * (1 + self.params.profit_target_pct)
                sl = d.close[0] * (1 - self.params.stop_loss_pct)
                self.sell(data=d, size=size, exectype=bt.Order.Limit, price=tp)
                self.sell(data=d, size=size, exectype=bt.Order.Stop, price=sl)

            elif signal['direction'] == 'SHORT' and signal['score'] >= self.params.min_score:
                size = (self.broker.getvalue() * self.params.risk_per_trade) / (d.close[0] * self.params.stop_loss_pct)
                self.sell(data=d, size=size)

                # Set TP/SL
                tp = d.close[0] * (1 - self.params.profit_target_pct)
                sl = d.close[0] * (1 + self.params.stop_loss_pct)
                self.buy(data=d, size=size, exectype=bt.Order.Limit, price=tp)
                self.buy(data=d, size=size, exectype=bt.Order.Stop, price=sl)

    def calculate_signal(self, data):
        """FIXED entry logic"""
        ind = self.indicators[data._name]

        # Get values
        rsi = ind['rsi'][0]
        rsi_prev = ind['rsi'][-1]

        macd_hist = ind['macd'].macd[0] - ind['macd'].signal[0]
        macd_hist_prev = ind['macd'].macd[-1] - ind['macd'].signal[-1]
        macd_hist_2 = ind['macd'].macd[-2] - ind['macd'].signal[-2]

        adx = ind['adx'][0]

        ema_20 = ind['ema_20'][0]
        ema_50 = ind['ema_50'][0]
        ema_200 = ind['ema_200'][0]

        price = data.close[0]

        # Determine trend
        if ema_20 > ema_50 > ema_200:
            trend = 'BULLISH'
        elif ema_20 < ema_50 < ema_200:
            trend = 'BEARISH'
        else:
            trend = 'NEUTRAL'

        # Weak trend = skip
        if adx < 20:
            return {'direction': None, 'score': 0}

        # LONG signal (CORRECTED)
        long_score = 0
        if trend == 'BULLISH':
            # RSI WITH momentum
            if rsi > 50 and rsi > rsi_prev:
                long_score += 3
            elif 40 <= rsi <= 50 and rsi > rsi_prev:
                long_score += 2

            # MACD momentum expanding
            if macd_hist > 0 and macd_hist > macd_hist_prev > macd_hist_2:
                long_score += 3

            # MACD fresh cross
            if macd_hist > 0 and macd_hist_prev <= 0:
                long_score += 2

            # Price above EMAs
            if price > ema_20 > ema_50:
                long_score += 2

            # Strong trend
            if adx > 25:
                long_score += 2

        # SHORT signal (CORRECTED)
        short_score = 0
        if trend == 'BEARISH':
            # RSI WITH momentum
            if rsi < 50 and rsi < rsi_prev:
                short_score += 3
            elif 50 <= rsi <= 60 and rsi < rsi_prev:
                short_score += 2

            # MACD momentum expanding
            if macd_hist < 0 and macd_hist < macd_hist_prev < macd_hist_2:
                short_score += 3

            # MACD fresh cross
            if macd_hist < 0 and macd_hist_prev >= 0:
                short_score += 2

            # Price below EMAs
            if price < ema_20 < ema_50:
                short_score += 2

            # Strong trend
            if adx > 25:
                short_score += 2

        # Return best signal
        if long_score > short_score and long_score >= self.params.min_score:
            return {'direction': 'LONG', 'score': long_score}
        elif short_score > long_score and short_score >= self.params.min_score:
            return {'direction': 'SHORT', 'score': short_score}
        else:
            return {'direction': None, 'score': max(long_score, short_score)}


def download_forex_data(pair, start_date, end_date):
    ticker = pair.replace('/', '') + '=X'
    data = yf.download(ticker, start=start_date, end=end_date, interval='1h', progress=False)
    if data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [str(col).lower() for col in data.columns]
    return data


if __name__ == '__main__':
    print('=' * 70)
    print('BACKTEST: FIXED STRATEGY vs OLD STRATEGY')
    print('=' * 70)

    # Same 6 months as before
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    print(f'\nPeriod: {start_date.date()} to {end_date.date()}')
    print('(Same data that produced 16.7% win rate with old strategy)')
    print()

    # Download data
    pairs_data = {}
    for pair_name, ticker in [('EURUSD', 'EUR/USD'), ('GBPUSD', 'GBP/USD'), ('USDJPY', 'USD/JPY')]:
        df = download_forex_data(ticker, start_date, end_date)
        if df is not None and not df.empty:
            pairs_data[pair_name] = df
            print(f'{pair_name}: {len(df)} bars')

    # Run backtest
    cerebro = bt.Cerebro()

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

    cerebro.addstrategy(FixedForexStrategy)
    cerebro.broker.setcash(200000)
    cerebro.broker.setcommission(commission=0.0001)

    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    print(f'\nStarting Balance: $200,000')
    print('\nRunning backtest...\n')

    results = cerebro.run()
    strat = results[0]

    ending = cerebro.broker.getvalue()
    profit = ending - 200000

    drawdown = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    print('=' * 70)
    print('RESULTS COMPARISON')
    print('=' * 70)

    print(f'\n{"Metric":<25} {"OLD Strategy":<20} {"FIXED Strategy":<20}')
    print('-' * 70)

    # OLD strategy results (from forensic analysis)
    print(f'{"Win Rate":<25} {"16.7%":<20} ', end='')
    if trades.total.total > 0:
        wr = (trades.won.total / trades.total.total * 100)
        print(f'{wr:.1f}%')
    else:
        print('No trades')

    print(f'{"Total Trades":<25} {"24":<20} {trades.total.total if trades.total.total > 0 else 0}')

    print(f'{"Profit/Loss":<25} {"-$13,290":<20} ${profit:,.2f}')

    print(f'{"Max Drawdown":<25} {"N/A":<20} {drawdown.max.drawdown:.2f}%')

    if trades.total.total > 0 and trades.won.total > 0:
        print(f'{"Avg Win":<25} {"$1,454":<20} ${trades.won.pnl.average:,.2f}')
    if trades.total.total > 0 and trades.lost.total > 0:
        print(f'{"Avg Loss":<25} {"-$955":<20} ${trades.lost.pnl.average:,.2f}')

    print(f'{"Expectancy":<25} {"-$554/trade":<20} ', end='')
    if trades.total.total > 0:
        wr_decimal = trades.won.total / trades.total.total
        avg_win = trades.won.pnl.average if trades.won.total > 0 else 0
        avg_loss = trades.lost.pnl.average if trades.lost.total > 0 else 0
        expectancy = (wr_decimal * avg_win) + ((1-wr_decimal) * avg_loss)
        print(f'${expectancy:.2f}/trade')
    else:
        print('N/A')

    print('\n' + '=' * 70)
    print('VERDICT')
    print('=' * 70)

    if trades.total.total > 0:
        wr = (trades.won.total / trades.total.total * 100)
        if wr >= 50 and profit > 0:
            print('\n[OK] FIXED STRATEGY WORKS!')
            print(f'  Win rate improved from 16.7% to {wr:.1f}%')
            print(f'  Profit improved from -$13,290 to ${profit:,.2f}')
            print('\nREADY TO DEPLOY ON E8')
        elif wr >= 40:
            print('\n[~] FIXED STRATEGY IS BETTER, BUT NEEDS MORE WORK')
            print(f'  Win rate improved to {wr:.1f}% (target: 50%+)')
            print('  Consider additional filters or parameter tuning')
        else:
            print('\n[X] FIXED STRATEGY DID NOT IMPROVE ENOUGH')
            print(f'  Win rate: {wr:.1f}% (still below 40%)')
            print('  Recommendation: Try different indicators or timeframes')
    else:
        print('\n[!] NO TRADES EXECUTED')
        print('  min_score threshold may be too high')
        print('  Try lowering from 6.0 to 4.0-5.0')

    print('=' * 70)
