"""
AGGRESSIVE BREAKOUT STRATEGY
Trade the break IMMEDIATELY - don't wait for retest

Philosophy:
- Price action gets 25 trades in 6 months (too slow)
- By trading the break directly, we get 3-5x more setups
- Accept lower win rate (35-40%) but use 2:1 R/R to compensate
- Goal: 15-20% ROI in 6 months instead of 5.44%

Changes from conservative price action:
- Entry: On the break candle (not retest)
- TP: 2% (vs 1.5%)
- SL: 1% (same)
- R/R: 2:1 (vs 1.88:1)
- Expected trades: 75-100 in 6 months (vs 25)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class AggressiveBreakout:
    def __init__(self, initial_balance=200000):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}
        self.closed_trades = []

        # AGGRESSIVE parameters
        self.profit_target_pct = 0.02  # 2% TP (bigger winners)
        self.stop_loss_pct = 0.01  # 1% SL
        self.risk_per_trade = 0.01  # Keep 1% risk
        self.max_positions = 5  # More concurrent positions

        # Breakout parameters (LOOSER than price action)
        self.break_threshold = 0.0001  # 1 pip (vs 2 pips) - easier to trigger
        self.min_candle_size = 0.0003  # Breakout candle must be at least 3 pips

    def download_data(self, pair, start_date, end_date):
        """Download 1H data"""
        ticker = pair.replace('/', '') + '=X'
        data = yf.download(ticker, start=start_date, end=end_date, interval='1h', progress=False)

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

    def check_immediate_break(self, df, current_idx, resistance, support):
        """
        AGGRESSIVE: Trade the break immediately (no retest required)

        Enter as soon as we break above resistance or below support
        """
        if current_idx < 2:
            return None

        prev_bar = df.iloc[current_idx - 1]
        current_bar = df.iloc[current_idx]

        current_close = current_bar['close']
        current_high = current_bar['high']
        current_low = current_bar['low']
        candle_size = abs(current_bar['close'] - current_bar['open'])

        # Breakout candle must have momentum (at least 3 pips)
        if candle_size < self.min_candle_size:
            return None

        # BULLISH BREAK: Previous bar below resistance, current breaks above
        if prev_bar['high'] < resistance and current_high >= resistance + self.break_threshold:
            # Current bar must CLOSE above resistance (confirming strength)
            if current_close > resistance:
                return {
                    'direction': 'LONG',
                    'entry_price': current_close,
                    'resistance': resistance,
                    'support': support,
                    'break_size': current_high - resistance
                }

        # BEARISH BREAK: Previous bar above support, current breaks below
        if prev_bar['low'] > support and current_low <= support - self.break_threshold:
            # Current bar must CLOSE below support
            if current_close < support:
                return {
                    'direction': 'SHORT',
                    'entry_price': current_close,
                    'resistance': resistance,
                    'support': support,
                    'break_size': support - current_low
                }

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
        """Run aggressive breakout backtest"""

        print('Downloading data...\n')

        data_1h = {}

        for pair_name, ticker in pairs.items():
            df = self.download_data(ticker, start_date, end_date)

            if df is not None:
                data_1h[pair_name] = df
                print(f'{pair_name}: {len(df)} bars (1H)')

        print(f'\nRunning AGGRESSIVE breakout backtest...\n')

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
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                    })

                    del self.positions[pair_name]

            # Check for new entries
            if len(self.positions) >= self.max_positions:
                continue

            for pair_name, df in data_1h.items():

                if pair_name in self.positions or timestamp not in df.index:
                    continue

                current_idx = df.index.get_loc(timestamp)

                # Get daily levels
                resistance, support = self.get_daily_levels(df, timestamp)

                if resistance is None or support is None:
                    continue

                # Check for IMMEDIATE break (no retest)
                signal = self.check_immediate_break(df, current_idx, resistance, support)

                if signal is None:
                    continue

                # Enter position
                entry_price = signal['entry_price']

                # Calculate position size
                risk_amount = self.balance * self.risk_per_trade
                units = risk_amount / (entry_price * self.stop_loss_pct)

                if signal['direction'] == 'LONG':
                    tp = entry_price * (1 + self.profit_target_pct)
                    sl = signal['resistance'] * (1 - self.stop_loss_pct)
                else:
                    tp = entry_price * (1 - self.profit_target_pct)
                    sl = signal['support'] * (1 + self.stop_loss_pct)

                self.positions[pair_name] = {
                    'direction': signal['direction'],
                    'entry_price': entry_price,
                    'tp': tp,
                    'sl': sl,
                    'units': units,
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
                'pnl': pnl,
                'exit_reason': 'END',
                'direction': position['direction'],
            })

        self.positions = {}

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

        print('=' * 70)
        print('AGGRESSIVE BREAKOUT RESULTS')
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
        print(f'Trades/Month:     {total_trades / 6:.1f}')

        print('\n' + '=' * 70)
        print('AGGRESSIVE vs CONSERVATIVE COMPARISON')
        print('=' * 70)
        print(f'\n{"Metric":<25} {"Conservative":<20} {"Aggressive":<20}')
        print('-' * 70)
        print(f'{"Strategy":<25} {"Break + Retest":<20} {"Break Only":<20}')
        print(f'{"TP / SL":<25} {"1.5% / 0.8%":<20} {"2.0% / 1.0%":<20}')
        print(f'{"Total Trades":<25} {"25":<20} {total_trades:<20}')
        print(f'{"Trades/Month":<25} {"4.2":<20} {f"{total_trades/6:.1f}":<20}')
        print(f'{"Win Rate":<25} {"44.0%":<20} {f"{win_rate:.1f}%":<20}')
        print(f'{"ROI (6 months)":<25} {"+5.44%":<20} {f"{roi:+.2f}%":<20}')
        print(f'{"Max Drawdown":<25} {"5.94%":<20} {f"{max_dd:.2f}%":<20}')
        print(f'{"Expectancy":<25} {"+$435":<20} {f"${expectancy:.0f}":<20}')

        print('\n' + '=' * 70)
        print('VERDICT')
        print('=' * 70)

        passes_dd = max_dd < 6
        better_roi = roi > 5.44

        if passes_dd and better_roi and roi >= 10:
            print(f'\n[OK] AGGRESSIVE BREAKOUT IS THE WINNER!')
            print(f'  ROI: {roi:.2f}% (vs 5.44% conservative)')
            print(f'  Max DD: {max_dd:.2f}% (under 6% limit) OK')
            print(f'  Trades: {total_trades} ({total_trades/6:.1f}/month)')
            print(f'\n  THIS IS THE STRATEGY TO DEPLOY.')

        elif passes_dd and better_roi:
            print(f'\n[~] AGGRESSIVE BREAKOUT IS BETTER')
            print(f'  ROI: {roi:.2f}% vs 5.44% (conservative)')
            print(f'  Max DD: {max_dd:.2f}% (under 6% limit) OK')
            print(f'  Win Rate: {win_rate:.1f}% vs 44.0% (lower but acceptable)')
            print(f'\n  More trades ({total_trades} vs 25) = Faster to goal')
            print(f'  DEPLOY AGGRESSIVE STRATEGY')

        elif better_roi:
            print(f'\n[X] AGGRESSIVE HAS BETTER ROI BUT TOO RISKY')
            print(f'  ROI: {roi:.2f}% (better than 5.44%)')
            print(f'  Max DD: {max_dd:.2f}% (exceeds 6% limit) FAIL')
            print(f'\n  Need to reduce position sizing')

        else:
            print(f'\n[CONCLUSION] CONSERVATIVE IS STILL BETTER')
            print(f'  Conservative: +5.44% ROI, 5.94% DD, 44% WR')
            print(f'  Aggressive: {roi:+.2f}% ROI, {max_dd:.2f}% DD, {win_rate:.1f}% WR')
            print(f'\n  Stick with conservative price action')


if __name__ == '__main__':
    print('=' * 70)
    print('AGGRESSIVE BREAKOUT STRATEGY')
    print('=' * 70)
    print('\nTrade the break immediately - no retest required')
    print('Goal: More trades = Higher ROI\n')

    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    print(f'Period: {start_date.date()} to {end_date.date()} (6 months)\n')

    pairs = {
        'EURUSD': 'EUR/USD',
        'GBPUSD': 'GBP/USD',
        'USDJPY': 'USD/JPY'
    }

    backtest = AggressiveBreakout(initial_balance=200000)
    backtest.run_backtest(pairs, start_date, end_date)
    backtest.print_results()
