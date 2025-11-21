"""
MULTI-TIMEFRAME COMBINATION STRATEGY
Run BOTH 4H swing trades AND 1H day trades simultaneously

Philosophy:
- 4H trades: Bigger moves, wider TP (3-4%), hold 2-5 days
- 1H trades: Quick moves, normal TP (1.5%), hold 4-12 hours
- Combined: More opportunities without sacrificing quality

Expected: 4H gives 10-15 trades, 1H gives 25 trades = 35-40 total
Target: 10-15% ROI in 6 months
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MultiTimeframeCombo:
    def __init__(self, initial_balance=200000):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions_1h = {}
        self.positions_4h = {}
        self.closed_trades = []

        # 1H parameters (same as price action)
        self.tp_1h = 0.015  # 1.5%
        self.sl_1h = 0.008  # 0.8%

        # 4H parameters (BIGGER targets)
        self.tp_4h = 0.03  # 3% (double the 1H target)
        self.sl_4h = 0.015  # 1.5%

        self.risk_per_trade = 0.01  # 1% risk per trade
        self.max_positions_total = 5  # Total across both timeframes

    def download_data(self, pair, start_date, end_date, interval='1h'):
        """Download data"""
        ticker = pair.replace('/', '') + '=X'
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.columns = [str(col).lower() for col in data.columns]
        return data

    def get_daily_levels(self, df, current_time):
        """Get yesterday's high/low"""
        df_past = df[df.index <= current_time]

        if len(df_past) < 24:
            return None, None

        current_date = current_time.date()
        yesterday = current_date - timedelta(days=1)
        df_yesterday = df_past[df_past.index.date == yesterday]

        for i in range(3):
            if len(df_yesterday) > 0:
                break
            yesterday = current_date - timedelta(days=i+1)
            df_yesterday = df_past[df_past.index.date == yesterday]

        if len(df_yesterday) == 0:
            return None, None

        resistance = df_yesterday['high'].max()
        support = df_yesterday['low'].min()

        return resistance, support

    def get_weekly_levels(self, df, current_time):
        """Get last week's high/low (for 4H trades)"""
        df_past = df[df.index <= current_time]

        if len(df_past) < 50:
            return None, None

        # Get last week's data
        last_week_start = current_time - timedelta(days=7)
        last_week_end = current_time - timedelta(days=1)

        df_last_week = df_past[(df_past.index >= last_week_start) & (df_past.index <= last_week_end)]

        if len(df_last_week) == 0:
            return None, None

        resistance = df_last_week['high'].max()
        support = df_last_week['low'].min()

        return resistance, support

    def check_breakout_retest(self, df, current_idx, resistance, support, break_threshold=0.0002, retest_threshold=0.0003):
        """Check for break-and-retest setup"""
        if current_idx < 24:
            return None

        lookback_start = max(0, current_idx - 12)
        recent_bars = df.iloc[lookback_start:current_idx+1]

        current_bar = df.iloc[current_idx]
        current_price = current_bar['close']
        current_high = current_bar['high']
        current_low = current_bar['low']

        # LONG SETUP
        if current_price > resistance:
            broke_resistance = False

            for i in range(len(recent_bars) - 1):
                prev_bar = recent_bars.iloc[i]
                next_bar = recent_bars.iloc[i + 1]

                if prev_bar['high'] < resistance and next_bar['high'] >= resistance + break_threshold:
                    broke_resistance = True
                    break

            if not broke_resistance:
                return None

            retest_distance = current_low - resistance

            if -retest_threshold <= retest_distance <= retest_threshold:
                if current_price > resistance:
                    return 'LONG'

        # SHORT SETUP
        elif current_price < support:
            broke_support = False

            for i in range(len(recent_bars) - 1):
                prev_bar = recent_bars.iloc[i]
                next_bar = recent_bars.iloc[i + 1]

                if prev_bar['low'] > support and next_bar['low'] <= support - break_threshold:
                    broke_support = True
                    break

            if not broke_support:
                return None

            retest_distance = current_high - support

            if -retest_threshold <= retest_distance <= retest_threshold:
                if current_price < support:
                    return 'SHORT'

        return None

    def check_exit(self, position, current_price):
        """Check TP/SL"""
        if position['direction'] == 'LONG':
            if current_price >= position['tp']:
                return 'TP', current_price
            if current_price <= position['sl']:
                return 'SL', current_price

        elif position['direction'] == 'SHORT':
            if current_price <= position['tp']:
                return 'TP', current_price
            if current_price >= position['sl']:
                return 'SL', current_price

        return None, None

    def run_backtest(self, pairs, start_date, end_date):
        """Run multi-timeframe backtest"""

        print('Downloading data...\n')

        data_1h = {}
        data_4h = {}

        for pair_name, ticker in pairs.items():
            df_1h = self.download_data(ticker, start_date, end_date, interval='1h')
            df_4h = self.download_data(ticker, start_date, end_date, interval='1h')

            if df_1h is not None and df_4h is not None:
                # Resample 1H to 4H for 4H analysis
                df_4h = df_4h.resample('4h').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()

                data_1h[pair_name] = df_1h
                data_4h[pair_name] = df_4h
                print(f'{pair_name}: {len(df_1h)} bars (1H), {len(df_4h)} bars (4H)')

        print(f'\nRunning multi-timeframe backtest...\n')

        all_timestamps_1h = sorted(set().union(*[set(df.index) for df in data_1h.values()]))

        for timestamp in all_timestamps_1h:

            # Check exits for BOTH timeframes
            for pair_name in list(self.positions_1h.keys()):
                if pair_name not in data_1h or timestamp not in data_1h[pair_name].index:
                    continue

                current_price = data_1h[pair_name].loc[timestamp, 'close']
                position = self.positions_1h[pair_name]

                exit_reason, exit_price = self.check_exit(position, current_price)

                if exit_reason:
                    pnl = (exit_price - position['entry_price']) * position['units'] if position['direction'] == 'LONG' else (position['entry_price'] - exit_price) * position['units']

                    self.balance += pnl

                    self.closed_trades.append({
                        'pair': pair_name,
                        'timeframe': '1H',
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                    })

                    del self.positions_1h[pair_name]

            for pair_name in list(self.positions_4h.keys()):
                if pair_name not in data_1h or timestamp not in data_1h[pair_name].index:
                    continue

                current_price = data_1h[pair_name].loc[timestamp, 'close']
                position = self.positions_4h[pair_name]

                exit_reason, exit_price = self.check_exit(position, current_price)

                if exit_reason:
                    pnl = (exit_price - position['entry_price']) * position['units'] if position['direction'] == 'LONG' else (position['entry_price'] - exit_price) * position['units']

                    self.balance += pnl

                    self.closed_trades.append({
                        'pair': pair_name,
                        'timeframe': '4H',
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                    })

                    del self.positions_4h[pair_name]

            # Check for new entries
            total_positions = len(self.positions_1h) + len(self.positions_4h)

            if total_positions >= self.max_positions_total:
                continue

            # 1H entries (check every hour)
            for pair_name, df in data_1h.items():

                if pair_name in self.positions_1h or timestamp not in df.index:
                    continue

                if total_positions >= self.max_positions_total:
                    break

                current_idx = df.index.get_loc(timestamp)

                resistance, support = self.get_daily_levels(df, timestamp)

                if resistance is None:
                    continue

                signal = self.check_breakout_retest(df, current_idx, resistance, support)

                if signal:
                    entry_price = df.loc[timestamp, 'close']

                    risk_amount = self.balance * self.risk_per_trade
                    units = risk_amount / (entry_price * self.sl_1h)

                    if signal == 'LONG':
                        tp = entry_price * (1 + self.tp_1h)
                        sl = resistance * (1 - self.sl_1h)
                    else:
                        tp = entry_price * (1 - self.tp_1h)
                        sl = support * (1 + self.sl_1h)

                    self.positions_1h[pair_name] = {
                        'direction': signal,
                        'entry_price': entry_price,
                        'tp': tp,
                        'sl': sl,
                        'units': units,
                    }

                    total_positions += 1

            # 4H entries (check every 4 hours)
            if timestamp.hour % 4 == 0:
                for pair_name, df in data_4h.items():

                    if pair_name in self.positions_4h or timestamp not in df.index:
                        continue

                    if total_positions >= self.max_positions_total:
                        break

                    current_idx = df.index.get_loc(timestamp)

                    resistance, support = self.get_weekly_levels(df, timestamp)

                    if resistance is None:
                        continue

                    signal = self.check_breakout_retest(df, current_idx, resistance, support, break_threshold=0.0003, retest_threshold=0.0005)

                    if signal:
                        entry_price = data_1h[pair_name].loc[timestamp, 'close']

                        risk_amount = self.balance * self.risk_per_trade
                        units = risk_amount / (entry_price * self.sl_4h)

                        if signal == 'LONG':
                            tp = entry_price * (1 + self.tp_4h)
                            sl = resistance * (1 - self.sl_4h)
                        else:
                            tp = entry_price * (1 - self.tp_4h)
                            sl = support * (1 + self.sl_4h)

                        self.positions_4h[pair_name] = {
                            'direction': signal,
                            'entry_price': entry_price,
                            'tp': tp,
                            'sl': sl,
                            'units': units,
                        }

                        total_positions += 1

        # Close remaining positions
        for pair_name, position in {**self.positions_1h, **self.positions_4h}.items():
            final_price = data_1h[pair_name]['close'].iloc[-1]

            pnl = (final_price - position['entry_price']) * position['units'] if position['direction'] == 'LONG' else (position['entry_price'] - final_price) * position['units']

            self.balance += pnl

            tf = '1H' if pair_name in self.positions_1h else '4H'

            self.closed_trades.append({
                'pair': pair_name,
                'timeframe': tf,
                'pnl': pnl,
                'exit_reason': 'END',
            })

        self.positions_1h = {}
        self.positions_4h = {}

    def print_results(self):
        """Print results"""

        total_trades = len(self.closed_trades)

        if total_trades == 0:
            print('[!] NO TRADES')
            return

        wins = [t for t in self.closed_trades if t['pnl'] > 0]
        win_rate = len(wins) / total_trades * 100

        total_pnl = sum(t['pnl'] for t in self.closed_trades)
        roi = (total_pnl / self.initial_balance) * 100

        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        losses = [t for t in self.closed_trades if t['pnl'] <= 0]
        avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0

        expectancy = (len(wins)/total_trades * avg_win) + (len(losses)/total_trades * avg_loss)

        # Drawdown
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

        # Breakdown by timeframe
        trades_1h = [t for t in self.closed_trades if t.get('timeframe') == '1H']
        trades_4h = [t for t in self.closed_trades if t.get('timeframe') == '4H']

        pnl_1h = sum(t['pnl'] for t in trades_1h)
        pnl_4h = sum(t['pnl'] for t in trades_4h)

        print('=' * 70)
        print('MULTI-TIMEFRAME COMBINATION RESULTS')
        print('=' * 70)
        print(f'\nStarting Balance: ${self.initial_balance:,.2f}')
        print(f'Ending Balance:   ${self.balance:,.2f}')
        print(f'Total P/L:        ${total_pnl:,.2f} ({roi:+.2f}%)')

        print(f'\nTotal Trades:     {total_trades}')
        print(f'  1H Trades:      {len(trades_1h)} (P/L: ${pnl_1h:,.2f})')
        print(f'  4H Trades:      {len(trades_4h)} (P/L: ${pnl_4h:,.2f})')

        print(f'\nWin Rate:         {win_rate:.1f}%')
        print(f'Expectancy:       ${expectancy:,.2f}/trade')
        print(f'Max Drawdown:     {max_dd:.2f}%')

        print('\n' + '=' * 70)
        print('COMPARISON: MULTI-TF vs SINGLE-TF')
        print('=' * 70)
        print(f'\n{"Metric":<25} {"1H Only":<20} {"Multi-TF":<20}')
        print('-' * 70)
        print(f'{"Total Trades":<25} {"25":<20} {total_trades:<20}')
        print(f'{"Win Rate":<25} {"44.0%":<20} {f"{win_rate:.1f}%":<20}')
        print(f'{"ROI (6 months)":<25} {"+5.44%":<20} {f"{roi:+.2f}%":<20}')
        print(f'{"Max Drawdown":<25} {"5.94%":<20} {f"{max_dd:.2f}%":<20}')
        print(f'{"Expectancy":<25} {"+$435":<20} {f"${expectancy:.0f}":<20}')

        print('\n' + '=' * 70)
        print('VERDICT')
        print('=' * 70)

        if roi >= 10 and max_dd < 6:
            print(f'\n[OK] MULTI-TIMEFRAME WINS!')
            print(f'  ROI: {roi:.2f}% (target: 10%+) OK')
            print(f'  Max DD: {max_dd:.2f}% (under 6%) OK')
            print(f'  DEPLOY THIS STRATEGY')

        elif roi > 5.44 and max_dd < 6:
            print(f'\n[~] MULTI-TIMEFRAME IS BETTER')
            print(f'  ROI: {roi:.2f}% vs 5.44% (single-TF)')
            print(f'  Max DD: {max_dd:.2f}% (under 6%) OK')
            print(f'  IMPROVEMENT: {roi - 5.44:.2f}% extra ROI')

        else:
            print(f'\n[CONCLUSION] SINGLE-TF (1H) IS STILL BEST')
            print(f'  1H Only: +5.44% ROI, 5.94% DD')
            print(f'  Multi-TF: {roi:+.2f}% ROI, {max_dd:.2f}% DD')


if __name__ == '__main__':
    print('=' * 70)
    print('MULTI-TIMEFRAME COMBINATION STRATEGY')
    print('=' * 70)
    print('\n1H day trades + 4H swing trades running simultaneously\n')

    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    print(f'Period: {start_date.date()} to {end_date.date()} (6 months)\n')

    pairs = {
        'EURUSD': 'EUR/USD',
        'GBPUSD': 'GBP/USD',
        'USDJPY': 'USD/JPY'
    }

    backtest = MultiTimeframeCombo(initial_balance=200000)
    backtest.run_backtest(pairs, start_date, end_date)
    backtest.print_results()
