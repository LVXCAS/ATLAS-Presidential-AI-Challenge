"""
MULTI-TIMEFRAME BACKTEST WITH REAL 4H DATA
This is the CORRECT way to implement multi-timeframe analysis:
- Download ACTUAL 4H candles for trend filtering
- Download 1H candles for entry signals
- Align them properly in time
- Only take 1H entries that match 4H trend
"""

import yfinance as yf
import pandas as pd
import talib
import numpy as np
from datetime import datetime, timedelta

class MultiTimeframeBacktest:
    def __init__(self, initial_balance=200000):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}
        self.closed_trades = []

        # Parameters from fixed strategy
        self.profit_target_pct = 0.02  # 2%
        self.stop_loss_pct = 0.01  # 1%
        self.risk_per_trade = 0.01  # 1% risk
        self.max_positions = 3
        self.min_score = 5.0  # Lowered from 6.0 to get more trades

    def download_data(self, pair, timeframe, start_date, end_date):
        """Download forex data from yfinance"""
        ticker = pair.replace('/', '') + '=X'

        # Map timeframe to yfinance interval
        interval_map = {'1h': '1h', '4h': '1h'}  # yfinance doesn't have 4h, we'll resample
        interval = interval_map.get(timeframe, '1h')

        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

        if data.empty:
            return None

        # Flatten multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.columns = [str(col).lower() for col in data.columns]

        # Resample to 4H if needed
        if timeframe == '4h':
            data = data.resample('4h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

        return data

    def calculate_4h_trend(self, df_4h):
        """Calculate trend from REAL 4H candles"""
        if len(df_4h) < 200:
            return None

        closes = df_4h['close'].values

        # Calculate 4H EMAs
        ema_20_4h = talib.EMA(closes, timeperiod=20)
        ema_50_4h = talib.EMA(closes, timeperiod=50)
        ema_200_4h = talib.EMA(closes, timeperiod=200)

        # Current values
        ema20 = ema_20_4h[-1]
        ema50 = ema_50_4h[-1]
        ema200 = ema_200_4h[-1]

        # Determine trend
        if ema20 > ema50 > ema200:
            return 'BULLISH'
        elif ema20 < ema50 < ema200:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def calculate_1h_entry_signal(self, df_1h, trend_4h):
        """Calculate entry signal from 1H data, filtered by 4H trend"""
        if len(df_1h) < 200:
            return {'direction': None, 'score': 0}

        closes = df_1h['close'].values
        highs = df_1h['high'].values
        lows = df_1h['low'].values

        # Calculate 1H indicators
        rsi = talib.RSI(closes, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
        adx = talib.ADX(highs, lows, closes, timeperiod=14)

        ema_20 = talib.EMA(closes, timeperiod=20)
        ema_50 = talib.EMA(closes, timeperiod=50)

        # Current values
        rsi_now = rsi[-1]
        rsi_prev = rsi[-2]

        macd_hist_now = macd_hist[-1]
        macd_hist_prev = macd_hist[-2]
        macd_hist_2 = macd_hist[-3]

        adx_now = adx[-1]

        price_now = closes[-1]
        ema_20_now = ema_20[-1]
        ema_50_now = ema_50[-1]

        # Weak trend = skip
        if adx_now < 20:
            return {'direction': None, 'score': 0}

        # LONG signal (ONLY if 4H trend is BULLISH)
        long_score = 0
        if trend_4h == 'BULLISH':

            # RSI WITH momentum
            if rsi_now > 50 and rsi_now > rsi_prev:
                long_score += 3
            elif 40 <= rsi_now <= 50 and rsi_now > rsi_prev:
                long_score += 2

            # MACD momentum expanding
            if macd_hist_now > 0 and macd_hist_now > macd_hist_prev > macd_hist_2:
                long_score += 3

            # MACD fresh cross
            if macd_hist_now > 0 and macd_hist_prev <= 0:
                long_score += 2

            # Price above EMAs
            if price_now > ema_20_now > ema_50_now:
                long_score += 2

            # Strong trend
            if adx_now > 25:
                long_score += 2

        # SHORT signal (ONLY if 4H trend is BEARISH)
        short_score = 0
        if trend_4h == 'BEARISH':

            # RSI WITH momentum
            if rsi_now < 50 and rsi_now < rsi_prev:
                short_score += 3
            elif 50 <= rsi_now <= 60 and rsi_now < rsi_prev:
                short_score += 2

            # MACD momentum expanding
            if macd_hist_now < 0 and macd_hist_now < macd_hist_prev < macd_hist_2:
                short_score += 3

            # MACD fresh cross
            if macd_hist_now < 0 and macd_hist_prev >= 0:
                short_score += 2

            # Price below EMAs
            if price_now < ema_20_now < ema_50_now:
                short_score += 2

            # Strong trend
            if adx_now > 25:
                short_score += 2

        # Return best signal
        if long_score >= self.min_score and long_score > short_score:
            return {
                'direction': 'LONG',
                'score': long_score,
                'price': price_now,
                'rsi': rsi_now,
                'macd_hist': macd_hist_now,
                'adx': adx_now
            }
        elif short_score >= self.min_score and short_score > long_score:
            return {
                'direction': 'SHORT',
                'score': short_score,
                'price': price_now,
                'rsi': rsi_now,
                'macd_hist': macd_hist_now,
                'adx': adx_now
            }
        else:
            return {'direction': None, 'score': max(long_score, short_score)}

    def check_exit(self, position, current_price):
        """Check if position should be closed"""
        if position['direction'] == 'LONG':
            # TP hit
            if current_price >= position['tp']:
                return 'TP', current_price
            # SL hit
            if current_price <= position['sl']:
                return 'SL', current_price

        elif position['direction'] == 'SHORT':
            # TP hit
            if current_price <= position['tp']:
                return 'TP', current_price
            # SL hit
            if current_price >= position['sl']:
                return 'SL', current_price

        return None, None

    def run_backtest(self, pairs, start_date, end_date):
        """Run multi-timeframe backtest"""

        print('Downloading data...\n')

        # Download 4H and 1H data for all pairs
        data_4h = {}
        data_1h = {}

        for pair_name, ticker in pairs.items():
            df_4h = self.download_data(ticker, '4h', start_date, end_date)
            df_1h = self.download_data(ticker, '1h', start_date, end_date)

            if df_4h is not None and df_1h is not None:
                data_4h[pair_name] = df_4h
                data_1h[pair_name] = df_1h
                print(f'{pair_name}: {len(df_4h)} bars (4H), {len(df_1h)} bars (1H)')

        print(f'\nRunning backtest...\n')

        # Get all 1H timestamps (we'll iterate through these)
        all_timestamps = sorted(set().union(*[set(df.index) for df in data_1h.values()]))

        for timestamp in all_timestamps:

            # Check exits on open positions
            for pair_name in list(self.positions.keys()):
                if pair_name not in data_1h:
                    continue

                if timestamp not in data_1h[pair_name].index:
                    continue

                current_price = data_1h[pair_name].loc[timestamp, 'close']
                position = self.positions[pair_name]

                exit_reason, exit_price = self.check_exit(position, current_price)

                if exit_reason:
                    # Close position
                    if position['direction'] == 'LONG':
                        pnl = (exit_price - position['entry_price']) * position['units']
                    else:  # SHORT
                        pnl = (position['entry_price'] - exit_price) * position['units']

                    self.balance += pnl

                    self.closed_trades.append({
                        'pair': pair_name,
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl': pnl,
                        'entry_time': position['entry_time'],
                        'exit_time': timestamp,
                        'score': position['score']
                    })

                    del self.positions[pair_name]

            # Check for new entries (if we have room)
            if len(self.positions) >= self.max_positions:
                continue

            for pair_name, df_1h in data_1h.items():

                # Skip if already in position
                if pair_name in self.positions:
                    continue

                # Skip if timestamp not in data
                if timestamp not in df_1h.index:
                    continue

                # Get 4H trend
                df_4h = data_4h[pair_name]

                # Get 4H data up to this timestamp
                df_4h_slice = df_4h[df_4h.index <= timestamp]
                if len(df_4h_slice) < 200:
                    continue

                trend_4h = self.calculate_4h_trend(df_4h_slice)

                # Get 1H data up to this timestamp
                df_1h_slice = df_1h[df_1h.index <= timestamp]
                if len(df_1h_slice) < 200:
                    continue

                # Calculate entry signal
                signal = self.calculate_1h_entry_signal(df_1h_slice, trend_4h)

                if signal['direction'] is None:
                    continue

                # Enter position
                entry_price = signal['price']

                # Calculate position size (1% risk)
                risk_amount = self.balance * self.risk_per_trade
                units = risk_amount / (entry_price * self.stop_loss_pct)

                if signal['direction'] == 'LONG':
                    tp = entry_price * (1 + self.profit_target_pct)
                    sl = entry_price * (1 - self.stop_loss_pct)
                else:  # SHORT
                    tp = entry_price * (1 - self.profit_target_pct)
                    sl = entry_price * (1 + self.stop_loss_pct)

                self.positions[pair_name] = {
                    'direction': signal['direction'],
                    'entry_price': entry_price,
                    'tp': tp,
                    'sl': sl,
                    'units': units,
                    'entry_time': timestamp,
                    'score': signal['score'],
                    'trend_4h': trend_4h
                }

                # Check if we hit max positions
                if len(self.positions) >= self.max_positions:
                    break

        # Close any remaining open positions at final price
        for pair_name, position in self.positions.items():
            final_price = data_1h[pair_name]['close'].iloc[-1]

            if position['direction'] == 'LONG':
                pnl = (final_price - position['entry_price']) * position['units']
            else:
                pnl = (position['entry_price'] - final_price) * position['units']

            self.balance += pnl

            self.closed_trades.append({
                'pair': pair_name,
                'direction': position['direction'],
                'entry_price': position['entry_price'],
                'exit_price': final_price,
                'exit_reason': 'END_OF_BACKTEST',
                'pnl': pnl,
                'entry_time': position['entry_time'],
                'exit_time': data_1h[pair_name].index[-1],
                'score': position['score']
            })

        self.positions = {}

    def print_results(self):
        """Print backtest results"""

        total_trades = len(self.closed_trades)

        if total_trades == 0:
            print('[!] NO TRADES EXECUTED')
            print(f'  min_score threshold ({self.min_score}) may be too high')
            return

        wins = [t for t in self.closed_trades if t['pnl'] > 0]
        losses = [t for t in self.closed_trades if t['pnl'] <= 0]

        win_rate = len(wins) / total_trades * 100

        total_pnl = sum(t['pnl'] for t in self.closed_trades)
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0

        expectancy = (len(wins)/total_trades * avg_win) + (len(losses)/total_trades * avg_loss)

        # Drawdown calculation
        equity_curve = [self.initial_balance]
        running_balance = self.initial_balance
        for trade in self.closed_trades:
            running_balance += trade['pnl']
            equity_curve.append(running_balance)

        peak = equity_curve[0]
        max_dd = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd

        print('=' * 70)
        print('MULTI-TIMEFRAME BACKTEST RESULTS (REAL 4H DATA)')
        print('=' * 70)
        print(f'\nStarting Balance: ${self.initial_balance:,.2f}')
        print(f'Ending Balance:   ${self.balance:,.2f}')
        print(f'Total P/L:        ${total_pnl:,.2f} ({total_pnl/self.initial_balance*100:+.2f}%)')

        print(f'\nTotal Trades:     {total_trades}')
        print(f'Wins:             {len(wins)} ({win_rate:.1f}%)')
        print(f'Losses:           {len(losses)}')

        print(f'\nAvg Win:          ${avg_win:,.2f}')
        print(f'Avg Loss:         ${avg_loss:,.2f}')
        print(f'Win/Loss Ratio:   {abs(avg_win/avg_loss):.2f}:1' if avg_loss != 0 else 'N/A')

        print(f'\nExpectancy:       ${expectancy:,.2f}/trade')
        print(f'Max Drawdown:     {max_dd:.2f}%')

        print('\n' + '=' * 70)
        print('COMPARISON TO PREVIOUS STRATEGIES')
        print('=' * 70)
        print(f'\n{"Metric":<25} {"OLD":<15} {"FIXED":<15} {"MULTI-TF":<15}')
        print('-' * 70)
        print(f'{"Win Rate":<25} {"16.7%":<15} {"15.4%":<15} {f"{win_rate:.1f}%":<15}')
        print(f'{"Total Trades":<25} {"24":<15} {"13":<15} {total_trades:<15}')
        print(f'{"Profit/Loss":<25} {"-$13,290":<15} {"+$1,681":<15} {f"${total_pnl:,.0f}":<15}')
        print(f'{"Max Drawdown":<25} {"N/A":<15} {"N/A":<15} {f"{max_dd:.1f}%":<15}')
        print(f'{"Expectancy":<25} {"-$554":<15} {"-$1,101":<15} {f"${expectancy:.0f}":<15}')

        print('\n' + '=' * 70)
        print('VERDICT')
        print('=' * 70)

        if win_rate >= 50 and total_pnl > 0 and expectancy > 0:
            print('\n[OK] MULTI-TIMEFRAME STRATEGY WORKS!')
            print(f'  Win rate: {win_rate:.1f}% (target: 50%+) OK')
            print(f'  Profit: ${total_pnl:,.2f} (positive) OK')
            print(f'  Expectancy: ${expectancy:.2f}/trade (positive) OK')
            print(f'  Max DD: {max_dd:.1f}% (E8 limit: 6%) {"OK" if max_dd < 6 else "FAIL"}')

            if max_dd < 6:
                print('\n  READY TO DEPLOY ON E8 $200K CHALLENGE')
            else:
                print('\n  Need to reduce position sizing to stay under 6% DD')

        elif win_rate >= 40 and total_pnl > 0:
            print('\n[~] MULTI-TIMEFRAME IS BETTER, BUT NEEDS WORK')
            print(f'  Win rate: {win_rate:.1f}% (target: 50%+)')
            print(f'  Profit: ${total_pnl:,.2f} (positive) OK')
            print(f'  Consider lowering min_score or adjusting R/R')

        elif total_pnl > 0:
            print('\n[~] PROFITABLE BUT LOW WIN RATE')
            print(f'  Win rate: {win_rate:.1f}% (below 40%)')
            print(f'  Profit: ${total_pnl:,.2f} (positive) OK')
            print(f'  Strategy relies on high R/R ratio - risky for prop firms')

        else:
            print('\n[X] MULTI-TIMEFRAME DID NOT SOLVE THE PROBLEM')
            print(f'  Win rate: {win_rate:.1f}%')
            print(f'  Profit: ${total_pnl:,.2f} (negative)')
            print('\n  RECOMMENDATION: Pivot to different indicators or strategy')
            print('  MACD/RSI may not be suitable for forex trend trading')

        print('=' * 70)

        # Print trade breakdown by pair
        print('\nPER-PAIR BREAKDOWN')
        print('=' * 70)

        pairs_stats = {}
        for trade in self.closed_trades:
            pair = trade['pair']
            if pair not in pairs_stats:
                pairs_stats[pair] = {'trades': [], 'wins': 0, 'pnl': 0}

            pairs_stats[pair]['trades'].append(trade)
            if trade['pnl'] > 0:
                pairs_stats[pair]['wins'] += 1
            pairs_stats[pair]['pnl'] += trade['pnl']

        print(f'\n{"Pair":<12} {"Trades":<10} {"Win Rate":<12} {"Total P/L":<15}')
        print('-' * 70)

        for pair, stats in sorted(pairs_stats.items()):
            wr = stats['wins'] / len(stats['trades']) * 100 if stats['trades'] else 0
            print(f'{pair:<12} {len(stats["trades"]):<10} {wr:<11.1f}% ${stats["pnl"]:>12,.2f}')


if __name__ == '__main__':
    print('=' * 70)
    print('MULTI-TIMEFRAME BACKTEST WITH REAL 4H DATA')
    print('=' * 70)

    # Same 6-month period as previous tests
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    print(f'\nPeriod: {start_date.date()} to {end_date.date()}')
    print('(Same data that produced 16.7% WR with old strategy)')
    print()

    # Same 3 pairs
    pairs = {
        'EURUSD': 'EUR/USD',
        'GBPUSD': 'GBP/USD',
        'USDJPY': 'USD/JPY'
    }

    # Run backtest
    backtest = MultiTimeframeBacktest(initial_balance=200000)
    backtest.run_backtest(pairs, start_date, end_date)
    backtest.print_results()
