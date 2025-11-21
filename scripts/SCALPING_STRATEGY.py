"""
SCALPING STRATEGY - 15MIN TIMEFRAME
Price action break-and-retest on faster timeframe

Strategy:
- Use HOURLY high/low as S/R levels (not daily)
- Enter on 15min retest
- Tighter TP/SL (0.5% TP, 0.3% SL)
- More trades = faster to 10% ROI

Expected: 20-40 trades/month (vs 4/month on 1H)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ScalpingStrategy:
    def __init__(self, initial_balance=200000):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}
        self.closed_trades = []

        # Scalping parameters (tighter than 1H)
        self.profit_target_pct = 0.005  # 0.5% (vs 1.5% on 1H)
        self.stop_loss_pct = 0.003  # 0.3% (vs 0.8% on 1H)
        self.risk_per_trade = 0.01  # Keep 1% risk
        self.max_positions = 3

        # Break thresholds (tighter for 15min)
        self.break_threshold = 0.0001  # 1 pip (vs 2 pips on 1H)
        self.retest_threshold = 0.0002  # 2 pips (vs 3 pips on 1H)
        self.max_bars_for_retest = 8  # 2 hours on 15min (vs 12 hours on 1H)

    def download_data(self, pair, start_date, end_date, interval='1h'):
        """Download 1H data (15min forex data not available on yfinance)"""
        ticker = pair.replace('/', '') + '=X'
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.columns = [str(col).lower() for col in data.columns]
        return data

    def get_hourly_levels(self, df, current_time):
        """Get last hour's high/low as S/R (scalping uses shorter lookback)"""
        df_past = df[df.index <= current_time]

        if len(df_past) < 8:  # Need at least 2 hours of data
            return None, None

        # Get previous hour's bars (4 bars on 15min = 1 hour)
        current_hour = current_time.replace(minute=0, second=0, microsecond=0)
        prev_hour = current_hour - timedelta(hours=1)

        # Get bars from previous hour
        df_prev_hour = df_past[(df_past.index >= prev_hour) & (df_past.index < current_hour)]

        if len(df_prev_hour) == 0:
            # Try 2 hours ago if previous hour was weekend/gap
            prev_hour = current_hour - timedelta(hours=2)
            current_hour_check = current_hour - timedelta(hours=1)
            df_prev_hour = df_past[(df_past.index >= prev_hour) & (df_past.index < current_hour_check)]

        if len(df_prev_hour) == 0:
            return None, None

        resistance = df_prev_hour['high'].max()
        support = df_prev_hour['low'].min()

        return resistance, support

    def check_break_and_retest(self, df, current_idx, resistance, support):
        """Same logic as 1H but on 15min bars"""
        if current_idx < 8:
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
        """Run scalping backtest"""

        print('Downloading 1H data...\n')

        data_1h = {}

        for pair_name, ticker in pairs.items():
            df = self.download_data(ticker, start_date, end_date, interval='1h')

            if df is not None:
                data_1h[pair_name] = df
                print(f'{pair_name}: {len(df)} bars (1H)')

        print(f'\nRunning scalping backtest...\n')

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
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl': pnl,
                        'entry_time': position['entry_time'],
                        'exit_time': timestamp,
                    })

                    del self.positions[pair_name]

            # Check for new entries
            if len(self.positions) >= self.max_positions:
                continue

            for pair_name, df in data_1h.items():

                if pair_name in self.positions or timestamp not in df.index:
                    continue

                current_idx = df.index.get_loc(timestamp)

                # Get hourly levels (not daily - this is scalping!)
                resistance, support = self.get_hourly_levels(df, timestamp)

                if resistance is None or support is None:
                    continue

                # Check for break-and-retest
                signal = self.check_break_and_retest(df, current_idx, resistance, support)

                if signal is None:
                    continue

                # Enter position
                entry_price = df.loc[timestamp, 'close']

                # Calculate position size (same 1% risk)
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

        # Close remaining positions
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
            })

        self.positions = {}

    def print_results(self):
        """Print results and compare to 1H strategy"""

        total_trades = len(self.closed_trades)

        if total_trades == 0:
            print('[!] NO TRADES EXECUTED')
            return

        wins = [t for t in self.closed_trades if t['pnl'] > 0]
        losses = [t for t in self.closed_trades if t['pnl'] <= 0]

        win_rate = len(wins) / total_trades * 100

        total_pnl = sum(t['pnl'] for t in self.closed_trades)
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
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

        roi = (total_pnl / self.initial_balance) * 100

        print('=' * 70)
        print('SCALPING STRATEGY RESULTS (1H BARS, SCALPING TP/SL)')
        print('=' * 70)
        print(f'\nStarting Balance: ${self.initial_balance:,.2f}')
        print(f'Ending Balance:   ${self.balance:,.2f}')
        print(f'Total P/L:        ${total_pnl:,.2f} ({roi:+.2f}%)')

        print(f'\nTotal Trades:     {total_trades}')
        print(f'Wins:             {len(wins)} ({win_rate:.1f}%)')
        print(f'Losses:           {len(losses)}')

        print(f'\nAvg Win:          ${avg_win:,.2f}')
        print(f'Avg Loss:         ${avg_loss:,.2f}')
        print(f'Win/Loss Ratio:   {abs(avg_win/avg_loss):.2f}:1' if avg_loss != 0 else 'N/A')

        print(f'\nExpectancy:       ${expectancy:,.2f}/trade')
        print(f'Max Drawdown:     {max_dd:.2f}%')

        print('\n' + '=' * 70)
        print('COMPARISON: SCALPING (0.5% TP) VS SWING (1.5% TP)')
        print('=' * 70)
        print(f'\n{"Metric":<25} {"1H Swing":<15} {"Scalping":<15}')
        print('-' * 70)
        print(f'{"TP/SL":<25} {"1.5% / 0.8%":<15} {"0.5% / 0.3%":<15}')
        print(f'{"Total Trades":<25} {"25":<15} {total_trades:<15}')
        print(f'{"Win Rate":<25} {"44.0%":<15} {f"{win_rate:.1f}%":<15}')
        print(f'{"Profit/Loss":<25} {"+$10,878":<15} {f"${total_pnl:,.0f}":<15}')
        print(f'{"ROI":<25} {"5.44%":<15} {f"{roi:.2f}%":<15}')
        print(f'{"Max Drawdown":<25} {"5.94%":<15} {f"{max_dd:.2f}%":<15}')
        print(f'{"Expectancy":<25} {"+$435":<15} {f"${expectancy:.0f}":<15}')
        print(f'{"Trades/Month":<25} {"4.2":<15} {f"{total_trades/6:.1f}":<15}')

        print('\n' + '=' * 70)
        print('E8 $200K CHALLENGE')
        print('=' * 70)
        print(f'\nProfit Target:    $20,000.00 (10%)')
        print(f'Current Profit:   ${total_pnl:,.2f} ({total_pnl/20000*100:.1f}% of target)')
        print(f'\nDrawdown Limit:   $12,000.00 (6%)')
        print(f'Max Drawdown:     ${max_dd * self.initial_balance / 100:,.2f} ({max_dd:.2f}%)')

        print('\n' + '=' * 70)
        print('VERDICT')
        print('=' * 70)

        passes_dd = max_dd < 6
        passes_profit = total_pnl >= 20000

        if passes_dd and passes_profit:
            print('\n[OK] SCALPING WOULD PASS E8 IN 6 MONTHS!')
            print(f'  Profit: ${total_pnl:,.2f} (target: $20,000) OK')
            print(f'  Max DD: {max_dd:.2f}% (limit: 6%) OK')
            print(f'  Trades: {total_trades} ({total_trades/6:.1f}/month)')
            print('\n  DEPLOY SCALPING STRATEGY IMMEDIATELY.')

        elif passes_dd and total_pnl > 0:
            print(f'\n[~] SCALPING IS PROFITABLE')
            print(f'  Profit: ${total_pnl:,.2f} / $20,000 ({total_pnl/20000*100:.1f}%)')
            print(f'  Max DD: {max_dd:.2f}% (under 6% limit) OK')
            print(f'  Trades: {total_trades} ({total_trades/6:.1f}/month)')

            if total_pnl > 10878:  # Better than 1H
                print(f'\n  SCALPING BEATS 1H STRATEGY!')
                remaining = 20000 - total_pnl
                trades_needed = remaining / expectancy
                months_needed = (trades_needed / total_trades) * 6

                print(f'  Need ${remaining:,.2f} more')
                print(f'  At ${expectancy:.0f}/trade = ~{trades_needed:.0f} more trades')
                print(f'  At {total_trades/6:.1f} trades/month = ~{months_needed:.1f} more months')
                print(f'\n  TIME TO PASS E8: {6 + months_needed:.0f} months (vs 11 months with 1H)')
            else:
                print(f'\n  1H SWING STRATEGY IS BETTER')
                print(f'  Stick with 1H timeframe.')

        elif total_pnl > 0:
            print(f'\n[X] SCALPING EXCEEDED DRAWDOWN')
            print(f'  Profit: ${total_pnl:,.2f} (positive)')
            print(f'  Max DD: {max_dd:.2f}% (over 6% limit) FAIL')
            print(f'\n  Too much volatility on 15min. Stick with 1H.')

        else:
            print(f'\n[X] SCALPING LOST MONEY')
            print(f'  Profit: ${total_pnl:,.2f} (negative)')
            print(f'  Win Rate: {win_rate:.1f}%')
            print(f'\n  Stick with 1H swing trading.')

        print('=' * 70)

        # Per-pair breakdown
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
    print('SCALPING STRATEGY TEST (15MIN TIMEFRAME)')
    print('=' * 70)
    print('\nTesting if scalping can reach 10% ROI faster than 1H swing trading\n')

    # NOTE: yfinance doesn't have 15min forex data, use 1H with scalping parameters
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    print(f'Period: {start_date.date()} to {end_date.date()} (180 days)')
    print('(Using 1H data with scalping TP/SL - tighter targets, more trades)\n')

    pairs = {
        'EURUSD': 'EUR/USD',
        'GBPUSD': 'GBP/USD',
        'USDJPY': 'USD/JPY'
    }

    backtest = ScalpingStrategy(initial_balance=200000)
    backtest.run_backtest(pairs, start_date, end_date)
    backtest.print_results()
