"""
E8 FOREX BOT - REAL HISTORICAL DATA BACKTEST
Uses backtrader with actual forex price data to validate strategy
"""

import backtrader as bt
import yfinance as yf
import pandas as pd
import talib
from datetime import datetime, timedelta
import numpy as np

class E8HybridStrategy(bt.Strategy):
    """
    Matches the exact logic from E8_FOREX_BOT.py
    - ADX, RSI, MACD, ATR indicators
    - 4.0 minimum score threshold
    - 2.5% TP / 1.0% SL (2.5:1 R/R)
    - 0.8% risk per trade
    - Session filtering (8 AM-12 PM EST for EUR/GBP, extended for JPY)
    """

    params = (
        ('min_score', 4.0),
        ('profit_target_pct', 0.025),  # 2.5%
        ('stop_loss_pct', 0.01),       # 1.0%
        ('risk_per_trade', 0.008),     # 0.8%
        ('max_positions', 3),
        ('challenge_balance', 200000),
        ('max_drawdown_pct', 0.06),    # 6%
        ('profit_target', 20000),       # $20K to pass
    )

    def __init__(self):
        self.orders = {}
        self.positions_count = 0
        self.starting_value = self.broker.getvalue()
        self.peak_value = self.starting_value

        # Create indicators for each data feed
        self.indicators = {}
        for i, d in enumerate(self.datas):
            self.indicators[d._name] = {
                'adx': bt.indicators.AverageDirectionalMovementIndex(d, period=14),
                'rsi': bt.indicators.RSI(d, period=14),
                'macd': bt.indicators.MACD(d),
                'atr': bt.indicators.ATR(d, period=14),
                'ema_20': bt.indicators.EMA(d, period=20),
                'ema_50': bt.indicators.EMA(d, period=50),
            }

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: {order.data._name} @ {order.executed.price:.5f}')
            elif order.issell():
                self.log(f'SELL EXECUTED: {order.data._name} @ {order.executed.price:.5f}, PnL: ${order.executed.pnl:.2f}')

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'TRADE CLOSED: {trade.data._name}, PnL: ${trade.pnl:.2f}')

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        print(f'{dt.strftime("%Y-%m-%d %H:%M")}: {txt}')

    def calculate_score(self, pair_name):
        """Calculate entry score (0-10) based on indicators"""
        ind = self.indicators[pair_name]
        score = 0.0

        # ADX (trend strength): 0-3 points
        adx_value = ind['adx'][0]
        if adx_value > 40:
            score += 3.0
        elif adx_value > 25:
            score += 2.0
        elif adx_value > 20:
            score += 1.0

        # RSI (momentum): 0-2 points
        rsi_value = ind['rsi'][0]
        if 30 <= rsi_value <= 40 or 60 <= rsi_value <= 70:
            score += 2.0
        elif 40 < rsi_value < 60:
            score += 1.0

        # MACD (trend direction): 0-2 points
        macd = ind['macd'].macd[0]
        macd_signal = ind['macd'].signal[0]
        if macd > macd_signal and macd > 0:
            score += 2.0  # Strong bullish
        elif macd > macd_signal:
            score += 1.0  # Weak bullish
        elif macd < macd_signal and macd < 0:
            score += 2.0  # Strong bearish (for shorts)

        # ATR (volatility): 0-2 points
        atr_value = ind['atr'][0]
        price = self.datas[0].close[0]
        atr_pct = (atr_value / price) * 100
        if 0.5 <= atr_pct <= 1.5:
            score += 2.0  # Ideal volatility
        elif 0.3 <= atr_pct <= 2.0:
            score += 1.0  # Acceptable

        # EMA trend: 0-1 point
        if ind['ema_20'][0] > ind['ema_50'][0]:
            score += 1.0  # Uptrend

        return score

    def check_trading_hours(self, pair_name):
        """Check if current hour is within trading window"""
        dt = self.datas[0].datetime.datetime(0)
        hour = dt.hour

        # Trading hours from E8_FOREX_BOT.py
        if 'USD' in pair_name and 'JPY' in pair_name:
            # USD/JPY: London/NY + Tokyo session
            return hour in [8, 9, 10, 11, 12, 20, 21, 22, 23]
        else:
            # EUR/USD, GBP/USD: London/NY overlap
            return hour in [8, 9, 10, 11, 12]

    def next(self):
        # Check drawdown
        current_value = self.broker.getvalue()
        if current_value > self.peak_value:
            self.peak_value = current_value

        drawdown = (self.peak_value - current_value) / self.peak_value
        if drawdown >= self.params.max_drawdown_pct:
            self.log(f'MAX DRAWDOWN HIT: {drawdown*100:.2f}% - STOPPING')
            self.env.runstop()
            return

        # Check if passed
        profit = current_value - self.starting_value
        if profit >= self.params.profit_target:
            self.log(f'PROFIT TARGET HIT: ${profit:.2f} - CHALLENGE PASSED!')
            self.env.runstop()
            return

        # Scan each pair
        for i, d in enumerate(self.datas):
            pair_name = d._name

            # Skip if already in position
            if self.getposition(d).size != 0:
                continue

            # Skip if max positions reached
            if self.positions_count >= self.params.max_positions:
                continue

            # Check trading hours
            if not self.check_trading_hours(pair_name):
                continue

            # Calculate entry score
            score = self.calculate_score(pair_name)

            if score >= self.params.min_score:
                # Determine direction based on MACD
                ind = self.indicators[pair_name]
                is_buy = ind['macd'].macd[0] > ind['macd'].signal[0]

                # Calculate position size
                risk_amount = self.broker.getvalue() * self.params.risk_per_trade
                stop_distance = d.close[0] * self.params.stop_loss_pct
                size = risk_amount / stop_distance

                # Place order
                if is_buy:
                    self.log(f'{pair_name} BUY SIGNAL - Score: {score:.1f}')
                    order = self.buy(data=d, size=size)

                    # Set TP/SL
                    tp_price = d.close[0] * (1 + self.params.profit_target_pct)
                    sl_price = d.close[0] * (1 - self.params.stop_loss_pct)
                    self.sell(data=d, size=size, exectype=bt.Order.Limit, price=tp_price)
                    self.sell(data=d, size=size, exectype=bt.Order.Stop, price=sl_price)
                else:
                    self.log(f'{pair_name} SELL SIGNAL - Score: {score:.1f}')
                    order = self.sell(data=d, size=size)

                    # Set TP/SL
                    tp_price = d.close[0] * (1 - self.params.profit_target_pct)
                    sl_price = d.close[0] * (1 + self.params.stop_loss_pct)
                    self.buy(data=d, size=size, exectype=bt.Order.Limit, price=tp_price)
                    self.buy(data=d, size=size, exectype=bt.Order.Stop, price=sl_price)

                self.positions_count += 1

            # Close position if exists
            pos = self.getposition(d)
            if pos.size != 0:
                # Check if should close (trailing stop, time-based, etc.)
                # For now, let TP/SL handle it
                pass


def download_forex_data(pair, start_date, end_date):
    """Download forex data from Yahoo Finance"""
    # Yahoo Finance forex ticker format: EURUSD=X
    ticker = pair.replace('/', '') + '=X'

    print(f'Downloading {pair} data...')
    data = yf.download(ticker, start=start_date, end=end_date, interval='1h', progress=False)

    if data.empty:
        print(f'WARNING: No data for {pair}')
        return None

    # Flatten multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Ensure columns are lowercase
    data.columns = [str(col).lower() for col in data.columns]

    print(f'  {len(data)} bars downloaded')
    return data


if __name__ == '__main__':
    print('=' * 70)
    print('E8 FOREX BOT - REAL HISTORICAL DATA BACKTEST')
    print('=' * 70)

    # Date range: Last 90 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=120)  # Extra buffer for indicators

    print(f'\nBacktest Period: {start_date.date()} to {end_date.date()}')
    print()

    # Initialize cerebro
    cerebro = bt.Cerebro()

    # Download data for each pair
    pairs = {
        'EUR/USD': 'EURUSD',
        'GBP/USD': 'GBPUSD',
        'USD/JPY': 'USDJPY',
    }

    data_loaded = 0
    for pair_name, pair_code in pairs.items():
        df = download_forex_data(pair_name, start_date, end_date)
        if df is not None and not df.empty:
            # Convert to backtrader format (columns already lowercase from download_forex_data)
            data = bt.feeds.PandasData(
                dataname=df,
                name=pair_code,
                timeframe=bt.TimeFrame.Minutes,
                compression=60,
                datetime=None,  # Use index as datetime
                open='open',
                high='high',
                low='low',
                close='close',
                volume='volume',
                openinterest=-1,  # Not used for forex
            )
            cerebro.adddata(data)
            data_loaded += 1

    if data_loaded == 0:
        print('\nERROR: No data loaded. Exiting.')
        exit(1)

    print(f'\n{data_loaded} pairs loaded successfully')

    # Add strategy
    cerebro.addstrategy(E8HybridStrategy)

    # Set broker
    cerebro.broker.setcash(200000)  # E8 $200K challenge
    cerebro.broker.setcommission(commission=0.0001)  # 1 pip spread

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    # Run backtest
    print('\n' + '=' * 70)
    print('RUNNING BACKTEST...')
    print('=' * 70)

    starting_value = cerebro.broker.getvalue()
    print(f'\nStarting Portfolio Value: ${starting_value:,.2f}')

    results = cerebro.run()
    strat = results[0]

    ending_value = cerebro.broker.getvalue()
    profit = ending_value - starting_value
    roi = (profit / starting_value) * 100

    print('\n' + '=' * 70)
    print('BACKTEST RESULTS')
    print('=' * 70)

    print(f'\nFinal Portfolio Value: ${ending_value:,.2f}')
    print(f'Total Profit/Loss: ${profit:,.2f}')
    print(f'ROI: {roi:.2f}%')

    # Analyzer results
    drawdown = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    print(f'\nMax Drawdown: {drawdown.max.drawdown:.2f}%')

    if trades.total.total > 0:
        print(f'\nTotal Trades: {trades.total.total}')
        print(f'Won: {trades.won.total}')
        print(f'Lost: {trades.lost.total}')
        win_rate = (trades.won.total / trades.total.total) * 100
        print(f'Win Rate: {win_rate:.1f}%')

        if trades.won.total > 0:
            print(f'Avg Win: ${trades.won.pnl.average:.2f}')
        if trades.lost.total > 0:
            print(f'Avg Loss: ${trades.lost.pnl.average:.2f}')
    else:
        print('\nNo trades executed during backtest period')

    # Challenge verdict
    print('\n' + '=' * 70)
    print('E8 CHALLENGE VERDICT')
    print('=' * 70)

    passed = profit >= 20000 and drawdown.max.drawdown < 6.0

    if passed:
        print('STATUS: PASSED [OK]')
        print(f'  Profit Target: ${profit:,.2f} / $20,000 required')
        print(f'  Max Drawdown: {drawdown.max.drawdown:.2f}% / 6.00% limit')
    else:
        if profit < 20000:
            print('STATUS: FAILED - Profit target not reached')
            print(f'  Profit: ${profit:,.2f} / $20,000 required')
        if drawdown.max.drawdown >= 6.0:
            print('STATUS: FAILED - Max drawdown exceeded')
            print(f'  Max Drawdown: {drawdown.max.drawdown:.2f}% / 6.00% limit')

    print('=' * 70)
