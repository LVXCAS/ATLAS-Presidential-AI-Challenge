"""
MAXIMIZE PRICE ACTION ROI
Current: +5.44% in 6 months (44% WR, 25 trades)
Goal: +10% in 3 months to pass E8

Strategy: Test different optimizations to find best ROI
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class OptimizedPriceAction:
    def __init__(self, initial_balance=200000, config=None):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}
        self.closed_trades = []

        # Default config (from original backtest)
        default_config = {
            'profit_target_pct': 0.015,  # 1.5%
            'stop_loss_pct': 0.008,  # 0.8%
            'risk_per_trade': 0.01,  # 1% risk
            'max_positions': 3,
            'break_threshold': 0.0002,  # 2 pips
            'retest_threshold': 0.0003,  # 3 pips
            'max_bars_for_retest': 12,  # 12 hours

            # NEW: Optimization parameters
            'require_trend_confirmation': False,  # Add EMA trend filter?
            'require_volume_spike': False,  # Require volume on break?
            'wider_tp': False,  # Use 2% TP instead of 1.5%?
            'tighter_retest': False,  # Require closer retest (2 pips vs 3)?
            'focus_best_pair': False,  # Only trade EUR/USD (54.5% WR)?
        }

        self.config = {**default_config, **(config or {})}

    def download_data(self, pair, start_date, end_date):
        """Download 1H forex data"""
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

    def check_trend_confirmation(self, df, current_idx, direction):
        """OPTIMIZATION: Check if 4H trend agrees"""
        if not self.config['require_trend_confirmation']:
            return True

        if current_idx < 200:
            return False

        # Calculate 200 EMA on 1H (approximates 4H trend)
        closes = df['close'].iloc[:current_idx+1].values

        if len(closes) < 200:
            return False

        # Simple EMA calculation
        ema_200 = pd.Series(closes).ewm(span=200, adjust=False).mean().iloc[-1]
        current_price = closes[-1]

        if direction == 'LONG':
            return current_price > ema_200  # Price above 200 EMA = uptrend
        else:
            return current_price < ema_200  # Price below 200 EMA = downtrend

    def check_volume_spike(self, df, current_idx):
        """OPTIMIZATION: Check if break had volume spike"""
        if not self.config['require_volume_spike']:
            return True

        if current_idx < 20:
            return False

        # Compare current volume to 20-bar average
        recent_volumes = df['volume'].iloc[current_idx-20:current_idx]
        avg_volume = recent_volumes.mean()
        current_volume = df['volume'].iloc[current_idx]

        # Volume spike = 1.5x average
        return current_volume > avg_volume * 1.5

    def check_break_and_retest(self, df, current_idx, resistance, support):
        """Check for break-and-retest setup with optimizations"""
        if current_idx < 24:
            return None

        lookback_start = max(0, current_idx - self.config['max_bars_for_retest'])
        recent_bars = df.iloc[lookback_start:current_idx+1]

        current_bar = df.iloc[current_idx]
        current_price = current_bar['close']
        current_high = current_bar['high']
        current_low = current_bar['low']

        break_threshold = self.config['break_threshold']
        retest_threshold = self.config['retest_threshold']

        # OPTIMIZATION: Tighter retest = higher quality setups
        if self.config['tighter_retest']:
            retest_threshold = 0.0002  # 2 pips instead of 3

        # LONG SETUP
        if current_price > resistance:
            broke_resistance = False
            break_bar_idx = None

            for i in range(len(recent_bars) - 1):
                prev_bar = recent_bars.iloc[i]
                next_bar = recent_bars.iloc[i + 1]

                if prev_bar['high'] < resistance and next_bar['high'] >= resistance + break_threshold:
                    broke_resistance = True
                    break_bar_idx = lookback_start + i + 1
                    break

            if not broke_resistance:
                return None

            # OPTIMIZATION: Check volume on break
            if not self.check_volume_spike(df, break_bar_idx):
                return None

            # Check retest
            retest_distance = current_low - resistance

            if -retest_threshold <= retest_distance <= retest_threshold:
                if current_price > resistance:

                    # OPTIMIZATION: Check trend confirmation
                    if not self.check_trend_confirmation(df, current_idx, 'LONG'):
                        return None

                    return 'LONG'

        # SHORT SETUP
        elif current_price < support:
            broke_support = False
            break_bar_idx = None

            for i in range(len(recent_bars) - 1):
                prev_bar = recent_bars.iloc[i]
                next_bar = recent_bars.iloc[i + 1]

                if prev_bar['low'] > support and next_bar['low'] <= support - break_threshold:
                    broke_support = True
                    break_bar_idx = lookback_start + i + 1
                    break

            if not broke_support:
                return None

            # OPTIMIZATION: Check volume on break
            if not self.check_volume_spike(df, break_bar_idx):
                return None

            # Check retest
            retest_distance = current_high - support

            if -retest_threshold <= retest_distance <= retest_threshold:
                if current_price < support:

                    # OPTIMIZATION: Check trend confirmation
                    if not self.check_trend_confirmation(df, current_idx, 'SHORT'):
                        return None

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
        """Run backtest"""
        data_1h = {}

        for pair_name, ticker in pairs.items():
            df = self.download_data(ticker, start_date, end_date)
            if df is not None:
                data_1h[pair_name] = df

        all_timestamps = sorted(set().union(*[set(df.index) for df in data_1h.values()]))

        for timestamp in all_timestamps:

            # Check exits
            for pair_name in list(self.positions.keys()):
                if pair_name not in data_1h:
                    continue

                if timestamp not in data_1h[pair_name].index:
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
            if len(self.positions) >= self.config['max_positions']:
                continue

            for pair_name, df in data_1h.items():

                if pair_name in self.positions:
                    continue

                if timestamp not in df.index:
                    continue

                current_idx = df.index.get_loc(timestamp)

                resistance, support = self.get_daily_levels(df, timestamp)

                if resistance is None or support is None:
                    continue

                signal = self.check_break_and_retest(df, current_idx, resistance, support)

                if signal is None:
                    continue

                entry_price = df.loc[timestamp, 'close']

                risk_amount = self.balance * self.config['risk_per_trade']
                units = risk_amount / (entry_price * self.config['stop_loss_pct'])

                profit_target = self.config['profit_target_pct']

                # OPTIMIZATION: Wider TP = bigger winners
                if self.config['wider_tp']:
                    profit_target = 0.02  # 2% instead of 1.5%

                if signal == 'LONG':
                    tp = entry_price * (1 + profit_target)
                    sl = resistance * (1 - self.config['stop_loss_pct'])
                else:
                    tp = entry_price * (1 - profit_target)
                    sl = support * (1 + self.config['stop_loss_pct'])

                self.positions[pair_name] = {
                    'direction': signal,
                    'entry_price': entry_price,
                    'tp': tp,
                    'sl': sl,
                    'units': units,
                    'entry_time': timestamp,
                }

                if len(self.positions) >= self.config['max_positions']:
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

    def get_results(self):
        """Return results as dict"""
        total_trades = len(self.closed_trades)

        if total_trades == 0:
            return None

        wins = [t for t in self.closed_trades if t['pnl'] > 0]
        losses = [t for t in self.closed_trades if t['pnl'] <= 0]

        win_rate = len(wins) / total_trades * 100
        total_pnl = sum(t['pnl'] for t in self.closed_trades)
        roi = (total_pnl / self.initial_balance) * 100

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

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'roi': roi,
            'expectancy': expectancy,
            'max_dd': max_dd,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'rr_ratio': abs(avg_win/avg_loss) if avg_loss != 0 else 0
        }


def test_optimization(name, config, pairs, start_date, end_date):
    """Test a specific optimization"""
    print(f'\nTesting: {name}')
    print('-' * 70)

    backtest = OptimizedPriceAction(initial_balance=200000, config=config)
    backtest.run_backtest(pairs, start_date, end_date)

    results = backtest.get_results()

    if results is None:
        print('  NO TRADES - Config too restrictive')
        return None

    print(f'  Trades: {results["total_trades"]}')
    print(f'  Win Rate: {results["win_rate"]:.1f}%')
    print(f'  ROI: {results["roi"]:.2f}%')
    print(f'  Expectancy: ${results["expectancy"]:.2f}/trade')
    print(f'  Max DD: {results["max_dd"]:.2f}%')
    print(f'  R/R Ratio: {results["rr_ratio"]:.2f}:1')

    return results


if __name__ == '__main__':
    print('=' * 70)
    print('PRICE ACTION ROI OPTIMIZATION')
    print('=' * 70)
    print('\nTesting different optimizations to maximize ROI')
    print('Baseline: 44% WR, 5.44% ROI, 25 trades in 6 months\n')

    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    pairs = {
        'EURUSD': 'EUR/USD',
        'GBPUSD': 'GBP/USD',
        'USDJPY': 'USD/JPY'
    }

    results_table = []

    # BASELINE (original strategy)
    print('=' * 70)
    print('BASELINE')
    print('=' * 70)
    baseline = test_optimization('Baseline (Original)', {}, pairs, start_date, end_date)
    results_table.append(('Baseline', baseline))

    # OPTIMIZATION 1: Add trend confirmation
    print('\n' + '=' * 70)
    print('OPTIMIZATION 1: Trend Confirmation')
    print('=' * 70)
    print('Only enter if 200 EMA agrees with direction')
    opt1 = test_optimization('Trend Confirmation', {'require_trend_confirmation': True}, pairs, start_date, end_date)
    results_table.append(('Trend Confirmation', opt1))

    # OPTIMIZATION 2: Add volume filter
    print('\n' + '=' * 70)
    print('OPTIMIZATION 2: Volume Spike Required')
    print('=' * 70)
    print('Only enter if break had 1.5x volume')
    opt2 = test_optimization('Volume Filter', {'require_volume_spike': True}, pairs, start_date, end_date)
    results_table.append(('Volume Filter', opt2))

    # OPTIMIZATION 3: Wider TP
    print('\n' + '=' * 70)
    print('OPTIMIZATION 3: Wider Take Profit')
    print('=' * 70)
    print('Use 2% TP instead of 1.5% (bigger winners)')
    opt3 = test_optimization('Wider TP (2%)', {'wider_tp': True}, pairs, start_date, end_date)
    results_table.append(('Wider TP', opt3))

    # OPTIMIZATION 4: Tighter retest
    print('\n' + '=' * 70)
    print('OPTIMIZATION 4: Tighter Retest')
    print('=' * 70)
    print('Require 2-pip retest instead of 3 pips (higher quality)')
    opt4 = test_optimization('Tighter Retest (2 pips)', {'tighter_retest': True}, pairs, start_date, end_date)
    results_table.append(('Tighter Retest', opt4))

    # OPTIMIZATION 5: Focus on EUR/USD only
    print('\n' + '=' * 70)
    print('OPTIMIZATION 5: EUR/USD Only')
    print('=' * 70)
    print('Only trade EUR/USD (had 54.5% WR in baseline)')
    eurusd_only = {'EURUSD': 'EUR/USD'}
    opt5 = test_optimization('EUR/USD Only', {}, eurusd_only, start_date, end_date)
    results_table.append(('EUR/USD Only', opt5))

    # OPTIMIZATION 6: Combine best
    print('\n' + '=' * 70)
    print('OPTIMIZATION 6: COMBO (Trend + Volume + Wider TP)')
    print('=' * 70)
    combo = test_optimization('Combo', {
        'require_trend_confirmation': True,
        'require_volume_spike': True,
        'wider_tp': True
    }, pairs, start_date, end_date)
    results_table.append(('Combo', combo))

    # Print comparison table
    print('\n' + '=' * 70)
    print('RESULTS COMPARISON')
    print('=' * 70)
    print(f'\n{"Strategy":<25} {"Trades":<10} {"Win%":<10} {"ROI%":<10} {"Exp":<12} {"DD%":<10}')
    print('-' * 70)

    best_roi = None
    best_config = None

    for name, results in results_table:
        if results is None:
            print(f'{name:<25} {"N/A":<10} {"N/A":<10} {"N/A":<10} {"N/A":<12} {"N/A":<10}')
        else:
            print(f'{name:<25} {results["total_trades"]:<10} {results["win_rate"]:<9.1f}% {results["roi"]:<9.2f}% ${results["expectancy"]:<10.0f} {results["max_dd"]:<9.2f}%')

            # Track best ROI (that passes E8 DD limit)
            if results['max_dd'] < 6 and (best_roi is None or results['roi'] > best_roi):
                best_roi = results['roi']
                best_config = name

    print('\n' + '=' * 70)
    print('RECOMMENDATION')
    print('=' * 70)

    if best_config:
        print(f'\nBest Config: {best_config}')
        print(f'ROI: {best_roi:.2f}%')
        print(f'\nThis configuration should be deployed.')
    else:
        print('\nNo config passed E8 drawdown limit (<6%)')
        print('Consider adjusting position sizing.')
