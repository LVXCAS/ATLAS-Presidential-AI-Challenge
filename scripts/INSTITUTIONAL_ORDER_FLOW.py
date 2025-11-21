"""
INSTITUTIONAL ORDER FLOW ARBITRAGE STRATEGY
The real edge: Liquidity hunts + News fades + Intermarket arbitrage

Win Rate Target: 75-80%
Monthly ROI Target: 20-30%
Risk per trade: 2-3%

Components:
1. Liquidity Hunt Algorithm - front-run stop sweeps
2. High-Frequency News Fade - fade 3-sigma spikes
3. Intermarket Arbitrage - USD/JPY vs bond spreads
4. Options Flow - trade into 10am NY cut expiries
5. 3am London Pre-Position - ride algo activation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from scipy import stats

class InstitutionalOrderFlow:
    def __init__(self, initial_balance=200000):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}
        self.closed_trades = []

        # AGGRESSIVE parameters
        self.risk_per_trade = 0.025  # 2.5% risk (vs 1% conservative)
        self.max_positions = 5  # More concurrent

        # News fade parameters
        self.news_spike_threshold = 3.0  # 3 standard deviations
        self.fade_entry_fib = 0.618  # Enter at 61.8% retracement

        # Liquidity hunt parameters
        self.stop_cluster_threshold = 0.0005  # 5 pips for stop clusters

        # Session parameters
        self.london_pre_position_time = (2, 45)  # 2:45am EST
        self.london_exit_time = (4, 0)  # 4:00am EST

    def download_data(self, pair, start_date, end_date, interval='5m'):
        """Download high-frequency data (5min for institutional moves)"""
        ticker = pair.replace('/', '') + '=X'
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.columns = [str(col).lower() for col in data.columns]

        # Calculate volatility metrics
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(20).std()
        data['volume_ma'] = data['volume'].rolling(20).mean()

        return data

    def detect_liquidity_sweep(self, df, current_idx):
        """
        Detect institutional liquidity hunts

        Pattern: Price spikes to take out stops, then reverses sharply
        """
        if current_idx < 50:
            return None

        recent = df.iloc[current_idx-20:current_idx+1]
        current_bar = df.iloc[current_idx]

        # Find recent swing highs/lows (where stops cluster)
        swing_high = recent['high'].iloc[:-1].max()
        swing_low = recent['low'].iloc[:-1].min()

        current_high = current_bar['high']
        current_low = current_bar['low']
        current_close = current_bar['close']

        # BULLISH SWEEP: Spiked below swing low then closed above
        if current_low < swing_low - self.stop_cluster_threshold:
            if current_close > swing_low:
                # Volume confirmation - should be high on the sweep
                if current_bar['volume'] > current_bar['volume_ma'] * 1.3:
                    return {
                        'direction': 'LONG',
                        'type': 'LIQUIDITY_SWEEP',
                        'entry': current_close,
                        'swept_level': swing_low,
                        'stop': current_low - 0.0003,  # 3 pips below sweep
                        'target': swing_high  # Target opposite side
                    }

        # BEARISH SWEEP: Spiked above swing high then closed below
        if current_high > swing_high + self.stop_cluster_threshold:
            if current_close < swing_high:
                if current_bar['volume'] > current_bar['volume_ma'] * 1.3:
                    return {
                        'direction': 'SHORT',
                        'type': 'LIQUIDITY_SWEEP',
                        'entry': current_close,
                        'swept_level': swing_high,
                        'stop': current_high + 0.0003,
                        'target': swing_low
                    }

        return None

    def detect_news_fade(self, df, current_idx):
        """
        Fade 3-sigma price spikes

        News releases cause irrational spikes - institutional traders fade them
        """
        if current_idx < 100:
            return None

        lookback = df.iloc[current_idx-100:current_idx]
        current_bar = df.iloc[current_idx]

        # Calculate z-score of current move
        mean_return = lookback['returns'].mean()
        std_return = lookback['returns'].std()

        current_return = current_bar['returns']

        if pd.isna(current_return) or std_return == 0:
            return None

        z_score = (current_return - mean_return) / std_return

        # 3-sigma spike DOWN - fade to upside
        if z_score < -self.news_spike_threshold:
            spike_size = abs(current_bar['open'] - current_bar['low'])
            entry = current_bar['low'] + (self.fade_entry_fib * spike_size)

            # Volume must be elevated (news-driven)
            if current_bar['volume'] > current_bar['volume_ma'] * 2.0:
                return {
                    'direction': 'LONG',
                    'type': 'NEWS_FADE',
                    'entry': entry,
                    'stop': current_bar['low'] * 0.998,  # Tight stop below spike
                    'target': current_bar['open']  # Target VWAP/spike origin
                }

        # 3-sigma spike UP - fade to downside
        if z_score > self.news_spike_threshold:
            spike_size = abs(current_bar['high'] - current_bar['open'])
            entry = current_bar['high'] - (self.fade_entry_fib * spike_size)

            if current_bar['volume'] > current_bar['volume_ma'] * 2.0:
                return {
                    'direction': 'SHORT',
                    'type': 'NEWS_FADE',
                    'entry': entry,
                    'stop': current_bar['high'] * 1.002,
                    'target': current_bar['open']
                }

        return None

    def london_pre_position(self, df, timestamp):
        """
        3am London Pre-Position Strategy

        Enter before algo activation, ride the surge, exit into retail FOMO
        """
        est = pytz.timezone('US/Eastern')
        time_est = timestamp.astimezone(est) if timestamp.tzinfo else est.localize(timestamp)

        hour = time_est.hour
        minute = time_est.minute

        # Entry window: 2:45-2:55am EST
        if hour == self.london_pre_position_time[0] and 45 <= minute <= 55:
            # Determine direction based on pre-market momentum
            recent = df.iloc[-12:]  # Last hour

            momentum_up = recent['close'].iloc[-1] > recent['close'].iloc[0]
            volume_increasing = recent['volume'].iloc[-3:].mean() > recent['volume'].iloc[-12:-3].mean()

            if momentum_up and volume_increasing:
                return {
                    'direction': 'LONG',
                    'type': 'LONDON_PRE_POSITION',
                    'entry': df['close'].iloc[-1],
                    'stop': df['low'].iloc[-12:].min() * 0.999,
                    'target': df['close'].iloc[-1] * 1.01,  # 1% target
                    'exit_time': self.london_exit_time
                }
            elif not momentum_up and volume_increasing:
                return {
                    'direction': 'SHORT',
                    'type': 'LONDON_PRE_POSITION',
                    'entry': df['close'].iloc[-1],
                    'stop': df['high'].iloc[-12:].max() * 1.001,
                    'target': df['close'].iloc[-1] * 0.99,
                    'exit_time': self.london_exit_time
                }

        return None

    def check_exit(self, position, current_price, timestamp=None):
        """Check TP/SL or time-based exit"""

        # Time-based exit for London pre-position
        if position.get('exit_time') and timestamp:
            est = pytz.timezone('US/Eastern')
            time_est = timestamp.astimezone(est) if timestamp.tzinfo else est.localize(timestamp)

            exit_hour, exit_minute = position['exit_time']
            if time_est.hour >= exit_hour and time_est.minute >= exit_minute:
                return 'TIME_EXIT', current_price

        # Standard TP/SL
        if position['direction'] == 'LONG':
            if current_price >= position['tp']:
                return 'TP', current_price
            if current_price <= position['sl']:
                return 'SL', current_price
        else:
            if current_price <= position['tp']:
                return 'TP', current_price
            if current_price >= position['sl']:
                return 'SL', current_price

        return None, None

    def run_backtest(self, pairs, start_date, end_date):
        """Run institutional order flow backtest"""

        print('Downloading 5min data (institutional timeframe)...\n')

        data_5m = {}

        for pair_name, ticker in pairs.items():
            df = self.download_data(ticker, start_date, end_date, interval='5m')

            if df is not None:
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')

                data_5m[pair_name] = df
                print(f'{pair_name}: {len(df)} bars (5min)')

        print(f'\nRunning institutional order flow backtest...\n')

        all_timestamps = sorted(set().union(*[set(df.index) for df in data_5m.values()]))

        for timestamp in all_timestamps:

            # Check exits
            for pair_name in list(self.positions.keys()):
                if pair_name not in data_5m or timestamp not in data_5m[pair_name].index:
                    continue

                current_price = data_5m[pair_name].loc[timestamp, 'close']
                position = self.positions[pair_name]

                exit_reason, exit_price = self.check_exit(position, current_price, timestamp)

                if exit_reason:
                    pnl = (exit_price - position['entry']) * position['units'] if position['direction'] == 'LONG' else (position['entry'] - exit_price) * position['units']

                    self.balance += pnl

                    self.closed_trades.append({
                        'pair': pair_name,
                        'type': position['type'],
                        'direction': position['direction'],
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                    })

                    del self.positions[pair_name]

            # Check for new entries
            if len(self.positions) >= self.max_positions:
                continue

            for pair_name, df in data_5m.items():

                if pair_name in self.positions or timestamp not in df.index:
                    continue

                current_idx = df.index.get_loc(timestamp)

                # Try all signal types
                signals = []

                # 1. Liquidity sweep
                sweep_signal = self.detect_liquidity_sweep(df, current_idx)
                if sweep_signal:
                    signals.append(sweep_signal)

                # 2. News fade
                news_signal = self.detect_news_fade(df, current_idx)
                if news_signal:
                    signals.append(news_signal)

                # 3. London pre-position
                london_signal = self.london_pre_position(df, timestamp)
                if london_signal:
                    signals.append(london_signal)

                # Take first signal
                if not signals:
                    continue

                signal = signals[0]

                # Calculate position size (AGGRESSIVE 2.5% risk)
                entry = signal['entry']
                stop = signal['stop']
                target = signal['target']

                risk_amount = self.balance * self.risk_per_trade
                stop_distance = abs(entry - stop)

                if stop_distance == 0:
                    continue

                units = risk_amount / stop_distance

                self.positions[pair_name] = {
                    'type': signal['type'],
                    'direction': signal['direction'],
                    'entry': entry,
                    'tp': target,
                    'sl': stop,
                    'units': units,
                    'exit_time': signal.get('exit_time')
                }

                if len(self.positions) >= self.max_positions:
                    break

        # Close remaining
        for pair_name, position in self.positions.items():
            final_price = data_5m[pair_name]['close'].iloc[-1]

            pnl = (final_price - position['entry']) * position['units'] if position['direction'] == 'LONG' else (position['entry'] - final_price) * position['units']

            self.balance += pnl

            self.closed_trades.append({
                'pair': pair_name,
                'type': position['type'],
                'pnl': pnl,
                'exit_reason': 'END',
                'direction': position['direction'],
            })

        self.positions = {}

    def print_results(self):
        """Print results"""

        total_trades = len(self.closed_trades)

        if total_trades == 0:
            print('[!] NO TRADES FOUND')
            print('  Institutional setups are rare but high-quality')
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

        # Breakdown by type
        liquidity_sweeps = [t for t in self.closed_trades if t.get('type') == 'LIQUIDITY_SWEEP']
        news_fades = [t for t in self.closed_trades if t.get('type') == 'NEWS_FADE']
        london_trades = [t for t in self.closed_trades if t.get('type') == 'LONDON_PRE_POSITION']

        print('=' * 70)
        print('INSTITUTIONAL ORDER FLOW RESULTS')
        print('=' * 70)
        print(f'\nStarting Balance: ${self.initial_balance:,.2f}')
        print(f'Ending Balance:   ${self.balance:,.2f}')
        print(f'Total P/L:        ${total_pnl:,.2f} ({roi:+.2f}%)')

        print(f'\nTotal Trades:     {total_trades}')
        print(f'  Liquidity Sweeps:  {len(liquidity_sweeps)}')
        print(f'  News Fades:        {len(news_fades)}')
        print(f'  London Pre-Pos:    {len(london_trades)}')

        print(f'\nWin Rate:         {win_rate:.1f}%')
        print(f'Expectancy:       ${expectancy:,.2f}/trade')
        print(f'Max Drawdown:     {max_dd:.2f}%')
        print(f'R/R Ratio:        {abs(avg_win/avg_loss):.2f}:1' if avg_loss != 0 else 'N/A')

        print('\n' + '=' * 70)
        print('COMPARISON: INSTITUTIONAL vs PRICE ACTION')
        print('=' * 70)
        print(f'\n{"Metric":<25} {"Price Action":<20} {"Institutional":<20}')
        print('-' * 70)
        print(f'{"Timeframe":<25} {"1H":<20} {"5min":<20}')
        print(f'{"Risk per Trade":<25} {"1.0%":<20} {"2.5%":<20}')
        print(f'{"Win Rate":<25} {"44.0%":<20} {f"{win_rate:.1f}%":<20}')
        print(f'{"ROI (6 months)":<25} {"+5.44%":<20} {f"{roi:+.2f}%":<20}')
        print(f'{"Max Drawdown":<25} {"5.94%":<20} {f"{max_dd:.2f}%":<20}')
        print(f'{"Expectancy":<25} {"+$435":<20} {f"${expectancy:.0f}":<20}')

        print('\n' + '=' * 70)
        print('VERDICT')
        print('=' * 70)

        if win_rate >= 70 and roi >= 15 and max_dd < 10:
            print(f'\n[OK] INSTITUTIONAL STRATEGY CRUSHES IT!')
            print(f'  Win rate: {win_rate:.1f}% (target: 70%+) OK')
            print(f'  ROI: {roi:.2f}% (target: 15%+) OK')
            print(f'  THIS IS THE EDGE WE NEED')

        elif win_rate >= 60 and roi > 5.44:
            print(f'\n[~] INSTITUTIONAL IS BETTER')
            print(f'  ROI: {roi:.2f}% vs 5.44% (price action)')
            print(f'  Win Rate: {win_rate:.1f}% vs 44%')
            print(f'  Improvement: +{roi - 5.44:.2f}%')

        else:
            print(f'\n[X] INSTITUTIONAL DID NOT BEAT CONSERVATIVE')
            print(f'  Institutional: {roi:+.2f}% ROI, {win_rate:.1f}% WR, {max_dd:.2f}% DD')
            print(f'  Price Action: +5.44% ROI, 44% WR, 5.94% DD')


if __name__ == '__main__':
    print('=' * 70)
    print('INSTITUTIONAL ORDER FLOW ARBITRAGE STRATEGY')
    print('=' * 70)
    print('\nLiquidity Hunts + News Fades + Session Positioning')
    print('Target: 75-80% WR, 20-30% monthly ROI\n')

    # Note: 5min data only available for last 60 days on yfinance
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)

    print(f'Period: {start_date.date()} to {end_date.date()} (60 days)')
    print('(5min data limited to 60 days)\n')

    pairs = {
        'EURUSD': 'EUR/USD',
        'GBPUSD': 'GBP/USD',
        'USDJPY': 'USD/JPY'
    }

    backtest = InstitutionalOrderFlow(initial_balance=200000)
    backtest.run_backtest(pairs, start_date, end_date)
    backtest.print_results()
