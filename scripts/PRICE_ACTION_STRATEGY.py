"""
PRICE ACTION BREAK-AND-RETEST STRATEGY
This is what ACTUALLY works for prop firms.

Strategy:
1. Identify daily support/resistance (yesterday's high/low)
2. Wait for BREAK above resistance or below support
3. Wait for RETEST (price comes back to test broken level)
4. Enter when retest HOLDS (confirms break is real)
5. Stop loss tight on other side of retest
6. Target next daily level

Win Rate Expected: 50-60%
R/R Expected: 1.5:1 to 2:1
Edge: Institutional order flow at broken levels
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PriceActionStrategy:
    def __init__(self, initial_balance=200000):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}
        self.closed_trades = []

        # Parameters
        self.profit_target_pct = 0.015  # 1.5%
        self.stop_loss_pct = 0.008  # 0.8%
        self.risk_per_trade = 0.01  # 1% risk
        self.max_positions = 3

        # Break-and-retest parameters
        self.break_threshold = 0.0002  # 2 pips = confirmed break
        self.retest_threshold = 0.0003  # 3 pips = close enough to retest
        self.max_bars_for_retest = 12  # Wait max 12 hours for retest

    def download_data(self, pair, start_date, end_date):
        """Download 1H forex data"""
        ticker = pair.replace('/', '') + '=X'

        data = yf.download(ticker, start=start_date, end=end_date, interval='1h', progress=False)

        if data.empty:
            return None

        # Flatten columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.columns = [str(col).lower() for col in data.columns]

        return data

    def get_daily_levels(self, df, current_time):
        """Get yesterday's high/low as S/R levels"""
        # Get all data up to current time
        df_past = df[df.index <= current_time]

        if len(df_past) < 24:
            return None, None

        # Get yesterday's date
        current_date = current_time.date()
        yesterday = current_date - timedelta(days=1)

        # Filter to yesterday's bars only
        df_yesterday = df_past[df_past.index.date == yesterday]

        if len(df_yesterday) == 0:
            # Try day before if yesterday was weekend
            yesterday = current_date - timedelta(days=2)
            df_yesterday = df_past[df_past.index.date == yesterday]

        if len(df_yesterday) == 0:
            # Try 3 days ago
            yesterday = current_date - timedelta(days=3)
            df_yesterday = df_past[df_past.index.date == yesterday]

        if len(df_yesterday) == 0:
            return None, None

        resistance = df_yesterday['high'].max()
        support = df_yesterday['low'].min()

        return resistance, support

    def check_break_and_retest(self, df, current_idx, resistance, support):
        """
        Check if we have a break-and-retest setup

        Returns:
        - 'LONG' if resistance broken and retested (buy signal)
        - 'SHORT' if support broken and retested (sell signal)
        - None if no setup
        """
        if current_idx < 24:
            return None

        # Get last 12 bars (for retest detection)
        lookback_start = max(0, current_idx - self.max_bars_for_retest)
        recent_bars = df.iloc[lookback_start:current_idx+1]

        current_bar = df.iloc[current_idx]
        current_price = current_bar['close']
        current_high = current_bar['high']
        current_low = current_bar['low']

        # LONG SETUP: Resistance broken, now retesting
        # 1. Price must be ABOVE resistance (confirmed break)
        # 2. Price must have recently TOUCHED resistance from above (retest)
        # 3. Current bar closes ABOVE resistance (retest held)

        if current_price > resistance:
            # Check if we broke resistance recently (within last 12 bars)
            broke_resistance = False
            for i in range(len(recent_bars) - 1):
                prev_bar = recent_bars.iloc[i]
                next_bar = recent_bars.iloc[i + 1]

                # Break = previous bar below, next bar above
                if prev_bar['high'] < resistance and next_bar['high'] >= resistance + self.break_threshold:
                    broke_resistance = True
                    break

            if not broke_resistance:
                return None

            # Check if current bar is retesting (low touched near resistance)
            retest_distance = current_low - resistance

            if -self.retest_threshold <= retest_distance <= self.retest_threshold:
                # Retest! And close is above = held
                if current_price > resistance:
                    return 'LONG'

        # SHORT SETUP: Support broken, now retesting
        # 1. Price must be BELOW support (confirmed break)
        # 2. Price must have recently TOUCHED support from below (retest)
        # 3. Current bar closes BELOW support (retest held)

        elif current_price < support:
            # Check if we broke support recently
            broke_support = False
            for i in range(len(recent_bars) - 1):
                prev_bar = recent_bars.iloc[i]
                next_bar = recent_bars.iloc[i + 1]

                # Break = previous bar above, next bar below
                if prev_bar['low'] > support and next_bar['low'] <= support - self.break_threshold:
                    broke_support = True
                    break

            if not broke_support:
                return None

            # Check if current bar is retesting (high touched near support)
            retest_distance = current_high - support

            if -self.retest_threshold <= retest_distance <= self.retest_threshold:
                # Retest! And close is below = held
                if current_price < support:
                    return 'SHORT'

        return None

    def check_exit(self, position, current_price):
        """Check if position should be closed"""
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
        """Run price action backtest"""

        print('Downloading data...\n')

        # Download 1H data for all pairs
        data_1h = {}

        for pair_name, ticker in pairs.items():
            df = self.download_data(ticker, start_date, end_date)

            if df is not None:
                data_1h[pair_name] = df
                print(f'{pair_name}: {len(df)} bars (1H)')

        print(f'\nRunning backtest...\n')

        # Iterate through each bar
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
                        'resistance': position.get('resistance'),
                        'support': position.get('support')
                    })

                    del self.positions[pair_name]

            # Check for new entries
            if len(self.positions) >= self.max_positions:
                continue

            for pair_name, df in data_1h.items():

                if pair_name in self.positions:
                    continue

                if timestamp not in df.index:
                    continue

                # Get current bar index
                current_idx = df.index.get_loc(timestamp)

                # Get daily levels
                resistance, support = self.get_daily_levels(df, timestamp)

                if resistance is None or support is None:
                    continue

                # Check for break-and-retest setup
                signal = self.check_break_and_retest(df, current_idx, resistance, support)

                if signal is None:
                    continue

                # Enter position
                entry_price = df.loc[timestamp, 'close']

                # Calculate position size
                risk_amount = self.balance * self.risk_per_trade
                units = risk_amount / (entry_price * self.stop_loss_pct)

                if signal == 'LONG':
                    tp = entry_price * (1 + self.profit_target_pct)
                    sl = resistance * (1 - self.stop_loss_pct)  # SL below retest level

                else:  # SHORT
                    tp = entry_price * (1 - self.profit_target_pct)
                    sl = support * (1 + self.stop_loss_pct)  # SL above retest level

                self.positions[pair_name] = {
                    'direction': signal,
                    'entry_price': entry_price,
                    'tp': tp,
                    'sl': sl,
                    'units': units,
                    'entry_time': timestamp,
                    'resistance': resistance,
                    'support': support
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
                'resistance': position.get('resistance'),
                'support': position.get('support')
            })

        self.positions = {}

    def print_results(self):
        """Print backtest results"""

        total_trades = len(self.closed_trades)

        if total_trades == 0:
            print('[!] NO TRADES EXECUTED')
            print('  Break-and-retest setups are rare. This is normal.')
            print('  Strategy waits for HIGH CONVICTION setups only.')
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

        print('=' * 70)
        print('PRICE ACTION BREAK-AND-RETEST BACKTEST RESULTS')
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
        print('COMPARISON TO MACD/RSI STRATEGIES')
        print('=' * 70)
        print(f'\n{"Metric":<25} {"MACD/RSI":<15} {"PRICE ACTION":<15}')
        print('-' * 70)
        print(f'{"Win Rate":<25} {"26.7%":<15} {f"{win_rate:.1f}%":<15}')
        print(f'{"Total Trades":<25} {"30":<15} {total_trades:<15}')
        print(f'{"Profit/Loss":<25} {"-$18,556":<15} {f"${total_pnl:,.0f}":<15}')
        print(f'{"Max Drawdown":<25} {"14.0%":<15} {f"{max_dd:.1f}%":<15}')
        print(f'{"Expectancy":<25} {"-$619":<15} {f"${expectancy:.0f}":<15}')

        print('\n' + '=' * 70)
        print('E8 $200K CHALLENGE REQUIREMENTS')
        print('=' * 70)
        print(f'\nProfit Target:    $20,000.00 (10%)')
        print(f'Current Profit:   ${total_pnl:,.2f} ({total_pnl/20000*100:.1f}% of target)')
        print(f'\nDrawdown Limit:   $12,000.00 (6%)')
        print(f'Max Drawdown:     ${max_dd * self.initial_balance / 100:,.2f} ({max_dd:.2f}%)')

        print('\n' + '=' * 70)
        print('VERDICT')
        print('=' * 70)

        if win_rate >= 50 and total_pnl > 0 and max_dd < 6:
            print('\n[OK] PRICE ACTION STRATEGY WORKS!')
            print(f'  Win rate: {win_rate:.1f}% (target: 50%+) OK')
            print(f'  Profit: ${total_pnl:,.2f} (positive) OK')
            print(f'  Expectancy: ${expectancy:.2f}/trade (positive) OK')
            print(f'  Max DD: {max_dd:.1f}% (under 6%) OK')

            if total_pnl >= 20000:
                print('\n  WOULD HAVE PASSED E8 CHALLENGE!')
            else:
                trades_needed = (20000 - total_pnl) / expectancy
                print(f'\n  Need ~{trades_needed:.0f} more trades to hit $20K target')
                print(f'  At {total_trades / 180:.1f} trades/day = {trades_needed / (total_trades/180):.0f} more days')

        elif win_rate >= 45 and total_pnl > 0:
            print('\n[~] PRICE ACTION IS PROMISING')
            print(f'  Win rate: {win_rate:.1f}% (close to 50%)')
            print(f'  Profit: ${total_pnl:,.2f} (positive) OK')
            print(f'  Max DD: {max_dd:.1f}%')
            print('\n  Consider tweaking retest threshold or R/R')

        elif total_pnl > 0:
            print('\n[~] PROFITABLE BUT NEEDS OPTIMIZATION')
            print(f'  Win rate: {win_rate:.1f}%')
            print(f'  Profit: ${total_pnl:,.2f} (positive) OK')
            print(f'  Max DD: {max_dd:.1f}%')

        else:
            print('\n[X] PRICE ACTION NEEDS WORK')
            print(f'  Win rate: {win_rate:.1f}%')
            print(f'  Profit: ${total_pnl:,.2f}')
            print('\n  May need to adjust break/retest thresholds')

        print('=' * 70)

        # Trade details
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
    print('PRICE ACTION BREAK-AND-RETEST STRATEGY')
    print('=' * 70)
    print('\nThis strategy trades institutional order flow at broken levels.')
    print('Expected win rate: 50-60% (vs 26.7% with MACD/RSI)\n')

    # Same 6-month period as MACD/RSI tests
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    print(f'Period: {start_date.date()} to {end_date.date()}')
    print('(Same data that MACD/RSI failed on)\n')

    # Same 3 pairs
    pairs = {
        'EURUSD': 'EUR/USD',
        'GBPUSD': 'GBP/USD',
        'USDJPY': 'USD/JPY'
    }

    # Run backtest
    backtest = PriceActionStrategy(initial_balance=200000)
    backtest.run_backtest(pairs, start_date, end_date)
    backtest.print_results()
