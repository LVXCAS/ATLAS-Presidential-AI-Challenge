"""
FINAL OPTIMAL CONFIGURATION
Test: All 3 pairs at 1.5% risk per trade
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class OptimalStrategy:
    def __init__(self, initial_balance=200000, risk_per_trade=0.015):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}
        self.closed_trades = []

        self.profit_target_pct = 0.015
        self.stop_loss_pct = 0.008
        self.risk_per_trade = risk_per_trade
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
                    })

                    del self.positions[pair_name]

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
            })

        self.positions = {}


if __name__ == '__main__':
    print('=' * 70)
    print('FINAL OPTIMAL CONFIGURATION')
    print('=' * 70)
    print('\nConfiguration: 3 pairs (EUR/USD, GBP/USD, USD/JPY) at 1.5% risk each\n')

    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    pairs = {
        'EURUSD': 'EUR/USD',
        'GBPUSD': 'GBP/USD',
        'USDJPY': 'USD/JPY'
    }

    backtest = OptimalStrategy(initial_balance=200000, risk_per_trade=0.015)
    backtest.run_backtest(pairs, start_date, end_date)

    total_trades = len(backtest.closed_trades)
    wins = [t for t in backtest.closed_trades if t['pnl'] > 0]
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    total_pnl = sum(t['pnl'] for t in backtest.closed_trades)
    roi = (total_pnl / backtest.initial_balance) * 100

    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    losses = [t for t in backtest.closed_trades if t['pnl'] <= 0]
    avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
    expectancy = (len(wins)/total_trades * avg_win) + (len(losses)/total_trades * avg_loss) if total_trades > 0 else 0

    # Drawdown
    equity_curve = [backtest.initial_balance]
    running_balance = backtest.initial_balance
    for trade in backtest.closed_trades:
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
    print('RESULTS')
    print('=' * 70)
    print(f'\nStarting Balance:  ${backtest.initial_balance:,.2f}')
    print(f'Ending Balance:    ${backtest.balance:,.2f}')
    print(f'Total P/L:         ${total_pnl:,.2f}')
    print(f'ROI:               {roi:.2f}%')

    print(f'\nTotal Trades:      {total_trades}')
    print(f'Wins:              {len(wins)} ({win_rate:.1f}%)')
    print(f'Losses:            {len(losses)}')

    print(f'\nAvg Win:           ${avg_win:,.2f}')
    print(f'Avg Loss:          ${avg_loss:,.2f}')
    print(f'Expectancy:        ${expectancy:,.2f}/trade')

    print(f'\nMax Drawdown:      {max_dd:.2f}%')

    print('\n' + '=' * 70)
    print('E8 $200K CHALLENGE')
    print('=' * 70)
    print(f'\nProfit Target:     $20,000.00 (10%)')
    print(f'Current Profit:    ${total_pnl:,.2f} ({total_pnl/20000*100:.1f}% of target)')
    print(f'\nDrawdown Limit:    $12,000.00 (6%)')
    print(f'Max Drawdown:      ${max_dd * backtest.initial_balance / 100:,.2f} ({max_dd:.2f}%)')

    print('\n' + '=' * 70)
    print('VERDICT')
    print('=' * 70)

    passes_dd = max_dd < 6
    passes_profit = total_pnl >= 20000

    if passes_dd and passes_profit:
        print('\n[OK] WOULD HAVE PASSED E8 CHALLENGE!')
        print(f'  Profit: ${total_pnl:,.2f} (target: $20,000) OK')
        print(f'  Max DD: {max_dd:.2f}% (limit: 6%) OK')
        print('\n  DEPLOY THIS CONFIGURATION IMMEDIATELY.')

    elif passes_dd:
        print(f'\n[~] PROFITABLE BUT NEED MORE TIME')
        print(f'  Profit: ${total_pnl:,.2f} / $20,000 ({total_pnl/20000*100:.1f}%)')
        print(f'  Max DD: {max_dd:.2f}% (under 6% limit) OK')

        remaining = 20000 - total_pnl
        trades_needed = remaining / expectancy if expectancy > 0 else 0
        months_needed = (trades_needed / total_trades) * 6 if total_trades > 0 else 0

        print(f'\n  Need ${remaining:,.2f} more')
        print(f'  At ${expectancy:.0f}/trade = ~{trades_needed:.0f} more trades')
        print(f'  At {total_trades/6:.1f} trades/month = ~{months_needed:.1f} more months')
        print(f'\n  Total time to pass: {6 + months_needed:.0f} months')

    else:
        print(f'\n[X] EXCEEDED DRAWDOWN LIMIT')
        print(f'  Max DD: {max_dd:.2f}% (limit: 6%) FAIL')
        print(f'\n  Need to reduce position sizing.')
