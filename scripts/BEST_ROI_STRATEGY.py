"""
BEST ROI STRATEGY - Final Optimization
Based on backtest results, EUR/USD has 54.5% WR with $922/trade expectancy.

Optimization: Increase position sizing to maximize ROI while staying under 6% DD
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MaxROIStrategy:
    def __init__(self, initial_balance=200000, risk_per_trade=0.01):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}
        self.closed_trades = []

        # Optimized parameters
        self.profit_target_pct = 0.015  # 1.5%
        self.stop_loss_pct = 0.008  # 0.8%
        self.risk_per_trade = risk_per_trade  # VARIABLE
        self.max_positions = 3

        self.break_threshold = 0.0002
        self.retest_threshold = 0.0003
        self.max_bars_for_retest = 12

    def download_data(self, pair, start_date, end_date):
        ticker = pair.replace('/', '') + '=X'
        data = yf.download(ticker, start=start_date, end=end_date, interval='1h', progress=False)

        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.columns = [str(col).lower() for col in data.columns]
        return data

    def get_daily_levels(self, df, current_time):
        df_past = df[df.index <= current_time]

        if len(df_past) < 24:
            return None, None

        current_date = current_time.date()
        yesterday = current_date - timedelta(days=1)
        df_yesterday = df_past[df_past.index.date == yesterday]

        if len(df_yesterday) == 0:
            yesterday = current_date - timedelta(days=2)
            df_yesterday = df_past[df_past.index.date == yesterday]

        if len(df_yesterday) == 0:
            yesterday = current_date - timedelta(days=3)
            df_yesterday = df_past[df_past.index.date == yesterday]

        if len(df_yesterday) == 0:
            return None, None

        resistance = df_yesterday['high'].max()
        support = df_yesterday['low'].min()

        return resistance, support

    def check_break_and_retest(self, df, current_idx, resistance, support):
        if current_idx < 24:
            return None

        lookback_start = max(0, current_idx - self.max_bars_for_retest)
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

                if prev_bar['high'] < resistance and next_bar['high'] >= resistance + self.break_threshold:
                    broke_resistance = True
                    break

            if not broke_resistance:
                return None

            retest_distance = current_low - resistance

            if -self.retest_threshold <= retest_distance <= self.retest_threshold:
                if current_price > resistance:
                    return 'LONG'

        # SHORT SETUP
        elif current_price < support:
            broke_support = False

            for i in range(len(recent_bars) - 1):
                prev_bar = recent_bars.iloc[i]
                next_bar = recent_bars.iloc[i + 1]

                if prev_bar['low'] > support and next_bar['low'] <= support - self.break_threshold:
                    broke_support = True
                    break

            if not broke_support:
                return None

            retest_distance = current_high - support

            if -self.retest_threshold <= retest_distance <= self.retest_threshold:
                if current_price < support:
                    return 'SHORT'

        return None

    def check_exit(self, position, current_price):
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
        data_1h = {}

        for pair_name, ticker in pairs.items():
            df = self.download_data(ticker, start_date, end_date)
            if df is not None:
                data_1h[pair_name] = df

        all_timestamps = sorted(set().union(*[set(df.index) for df in data_1h.values()]))

        for timestamp in all_timestamps:

            # Check exits
            for pair_name in list(self.positions.keys()):
                if pair_name not in data_1h or timestamp not in data_1h[pair_name].index:
                    continue

                current_price = data_1h[pair_name].loc[timestamp, 'close']
                position = self.positions[pair_name]

                exit_reason, exit_price = self.check_exit(position, current_price)

                if exit_reason:
                    if position['direction'] == 'LONG':
                        pnl = (exit_price - position['entry_price']) * position['units']
                    else:
                        pnl = (position['entry_price'] - exit_price) * position['units']

                    self.balance += pnl

                    self.closed_trades.append({
                        'pair': pair_name,
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'direction': position['direction'],
                    })

                    del self.positions[pair_name]

            # Check for new entries
            if len(self.positions) >= self.max_positions:
                continue

            for pair_name, df in data_1h.items():

                if pair_name in self.positions or timestamp not in df.index:
                    continue

                current_idx = df.index.get_loc(timestamp)

                resistance, support = self.get_daily_levels(df, timestamp)

                if resistance is None or support is None:
                    continue

                signal = self.check_break_and_retest(df, current_idx, resistance, support)

                if signal is None:
                    continue

                entry_price = df.loc[timestamp, 'close']

                risk_amount = self.balance * self.risk_per_trade
                units = risk_amount / (entry_price * self.stop_loss_pct)

                if signal == 'LONG':
                    tp = entry_price * (1 + self.profit_target_pct)
                    sl = resistance * (1 - self.stop_loss_pct)
                else:
                    tp = entry_price * (1 - self.profit_target_pct)
                    sl = support * (1 + self.stop_loss_pct)

                self.positions[pair_name] = {
                    'direction': signal,
                    'entry_price': entry_price,
                    'tp': tp,
                    'sl': sl,
                    'units': units,
                    'entry_time': timestamp,
                }

                if len(self.positions) >= self.max_positions:
                    break

        # Close remaining
        for pair_name, position in self.positions.items():
            final_price = data_1h[pair_name]['close'].iloc[-1]

            if position['direction'] == 'LONG':
                pnl = (final_price - position['entry_price']) * position['units']
            else:
                pnl = (position['entry_price'] - final_price) * position['units']

            self.balance += pnl

            self.closed_trades.append({
                'pair': pair_name,
                'pnl': pnl,
                'exit_reason': 'END',
                'direction': position['direction'],
            })

        self.positions = {}

    def get_results(self):
        total_trades = len(self.closed_trades)

        if total_trades == 0:
            return None

        wins = [t for t in self.closed_trades if t['pnl'] > 0]
        win_rate = len(wins) / total_trades * 100
        total_pnl = sum(t['pnl'] for t in self.closed_trades)
        roi = (total_pnl / self.initial_balance) * 100

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

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'roi': roi,
            'max_dd': max_dd,
        }


if __name__ == '__main__':
    print('=' * 70)
    print('MAXIMIZE ROI - POSITION SIZING TEST')
    print('=' * 70)
    print('\nEUR/USD has 54.5% WR. Can we increase position size without exceeding 6% DD?\n')

    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    pairs = {'EURUSD': 'EUR/USD'}  # Focus on best pair

    print(f'{"Risk/Trade":<15} {"Trades":<10} {"Win%":<10} {"ROI%":<10} {"Max DD%":<10} {"Passes E8?":<12}')
    print('-' * 70)

    risk_levels = [0.01, 0.015, 0.02, 0.025, 0.03]

    best_roi = 0
    best_risk = 0

    for risk in risk_levels:
        backtest = MaxROIStrategy(initial_balance=200000, risk_per_trade=risk)
        backtest.run_backtest(pairs, start_date, end_date)

        results = backtest.get_results()

        if results:
            passes = 'YES' if results['max_dd'] < 6 and results['roi'] >= 10 else 'NO'

            print(f'{risk*100:<14.1f}% {results["total_trades"]:<10} {results["win_rate"]:<9.1f}% {results["roi"]:<9.2f}% {results["max_dd"]:<9.2f}% {passes:<12}')

            if results['max_dd'] < 6 and results['roi'] > best_roi:
                best_roi = results['roi']
                best_risk = risk

    print('\n' + '=' * 70)
    print('RECOMMENDATION')
    print('=' * 70)

    if best_roi >= 10:
        print(f'\n[OK] FOUND WINNING CONFIG!')
        print(f'  Risk per trade: {best_risk * 100}%')
        print(f'  ROI: {best_roi:.2f}%')
        print(f'\n  This would PASS E8 $200K challenge in 6 months!')
        print(f'  Deploy with EUR/USD only at {best_risk * 100}% risk per trade.')
    else:
        print(f'\nBest Risk: {best_risk * 100}%')
        print(f'Best ROI: {best_roi:.2f}% (need 10%)')
        print(f'\nTo hit 10% in 6 months, need to:')
        print(f'  1. Trade all 3 pairs (not just EUR/USD)')
        print(f'  2. Use 2% risk per trade')
        print(f'  3. Run for longer period (9-12 months)')
