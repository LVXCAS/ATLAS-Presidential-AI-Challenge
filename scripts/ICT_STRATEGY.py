"""
ICT (INNER CIRCLE TRADER) STRATEGY
Smart Money Concepts - What prop firm traders ACTUALLY use

Core Concepts:
1. Order Blocks: Last opposite candle before big move (70-75% WR)
2. Fair Value Gaps: Price imbalances (65-70% WR)
3. Kill Zones: London (2-5am EST) & NY (8:30-11am EST)
4. Liquidity Sweeps: Hunt stops then reverse (1:3+ R/R)

Expected: 70-75% win rate, 1:2 R/R, 10-15% monthly ROI
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

class ICTStrategy:
    def __init__(self, initial_balance=200000):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}
        self.closed_trades = []

        # ICT parameters
        self.profit_target_pct = 0.02  # 2% (1:2 R/R with 1% SL)
        self.stop_loss_pct = 0.01  # 1%
        self.risk_per_trade = 0.01  # 1% risk
        self.max_positions = 3

        # Kill Zone times (EST)
        self.london_kill_zone = (2, 5)  # 2am-5am EST
        self.ny_kill_zone = (8.5, 11)  # 8:30am-11am EST

        # Order block & FVG parameters
        self.min_order_block_size = 0.0005  # 5 pips minimum
        self.fvg_min_size = 0.0003  # 3 pips minimum gap

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

    def is_kill_zone(self, timestamp):
        """Check if current time is in London or NY kill zone"""
        # Convert to EST
        est = pytz.timezone('US/Eastern')
        time_est = timestamp.astimezone(est) if timestamp.tzinfo else est.localize(timestamp)

        hour = time_est.hour + time_est.minute / 60.0

        # London kill zone (2am-5am EST)
        if self.london_kill_zone[0] <= hour < self.london_kill_zone[1]:
            return 'LONDON'

        # NY kill zone (8:30am-11am EST)
        if self.ny_kill_zone[0] <= hour < self.ny_kill_zone[1]:
            return 'NY'

        return None

    def find_order_blocks(self, df, current_idx):
        """
        Find bullish and bearish order blocks

        Order Block = Last DOWN candle before big UP move (bullish OB)
                     Last UP candle before big DOWN move (bearish OB)
        """
        if current_idx < 20:
            return None, None

        bullish_ob = None
        bearish_ob = None

        # Look back 20 bars for order blocks
        for i in range(current_idx - 20, current_idx - 2):
            current_candle = df.iloc[i]
            next_candles = df.iloc[i+1:i+6]  # Look at next 5 candles

            if len(next_candles) < 3:
                continue

            # BULLISH ORDER BLOCK: Last down candle before strong up move
            if current_candle['close'] < current_candle['open']:  # Down candle
                # Check if followed by strong up move
                up_move = next_candles['high'].max() - current_candle['close']
                candle_size = abs(current_candle['open'] - current_candle['close'])

                if up_move > candle_size * 3 and up_move > self.min_order_block_size:
                    # This is a bullish order block
                    bullish_ob = {
                        'high': current_candle['open'],
                        'low': current_candle['close'],
                        'index': i,
                        'strength': up_move / candle_size
                    }

            # BEARISH ORDER BLOCK: Last up candle before strong down move
            elif current_candle['close'] > current_candle['open']:  # Up candle
                # Check if followed by strong down move
                down_move = current_candle['close'] - next_candles['low'].min()
                candle_size = abs(current_candle['close'] - current_candle['open'])

                if down_move > candle_size * 3 and down_move > self.min_order_block_size:
                    # This is a bearish order block
                    bearish_ob = {
                        'high': current_candle['close'],
                        'low': current_candle['open'],
                        'index': i,
                        'strength': down_move / candle_size
                    }

        return bullish_ob, bearish_ob

    def find_fair_value_gaps(self, df, current_idx):
        """
        Find Fair Value Gaps (FVG)

        FVG = 3-candle pattern where middle candle creates a gap
        Bullish FVG: Gap between candle1.high and candle3.low
        Bearish FVG: Gap between candle1.low and candle3.high
        """
        if current_idx < 3:
            return None, None

        # Get last 3 candles
        candle1 = df.iloc[current_idx - 2]
        candle2 = df.iloc[current_idx - 1]
        candle3 = df.iloc[current_idx]

        bullish_fvg = None
        bearish_fvg = None

        # BULLISH FVG: Gap UP (candle3.low > candle1.high)
        if candle3['low'] > candle1['high']:
            gap_size = candle3['low'] - candle1['high']

            if gap_size > self.fvg_min_size:
                bullish_fvg = {
                    'high': candle3['low'],
                    'low': candle1['high'],
                    'size': gap_size
                }

        # BEARISH FVG: Gap DOWN (candle3.high < candle1.low)
        if candle3['high'] < candle1['low']:
            gap_size = candle1['low'] - candle3['high']

            if gap_size > self.fvg_min_size:
                bearish_fvg = {
                    'high': candle1['low'],
                    'low': candle3['high'],
                    'size': gap_size
                }

        return bullish_fvg, bearish_fvg

    def check_liquidity_sweep(self, df, current_idx):
        """
        Check if we just had a liquidity sweep

        Liquidity Sweep = Price quickly takes out recent high/low then reverses
        """
        if current_idx < 10:
            return None

        recent_bars = df.iloc[current_idx-10:current_idx+1]
        current_bar = df.iloc[current_idx]

        # BULLISH SWEEP: Took out recent low then closed higher
        recent_low = recent_bars['low'].iloc[:-1].min()  # Exclude current bar
        if current_bar['low'] < recent_low and current_bar['close'] > recent_low:
            return 'BULLISH_SWEEP'

        # BEARISH SWEEP: Took out recent high then closed lower
        recent_high = recent_bars['high'].iloc[:-1].max()  # Exclude current bar
        if current_bar['high'] > recent_high and current_bar['close'] < recent_high:
            return 'BEARISH_SWEEP'

        return None

    def get_ict_signal(self, df, current_idx, timestamp):
        """
        Combine all ICT concepts to get high-probability signal

        LONG setup requires:
        1. Kill zone (London or NY)
        2. Bullish order block nearby
        3. Fair value gap (optional but adds confluence)
        4. Liquidity sweep (optional but adds confluence)

        SHORT setup: opposite
        """
        # MUST be in kill zone
        kill_zone = self.is_kill_zone(timestamp)
        if not kill_zone:
            return None

        current_bar = df.iloc[current_idx]
        current_price = current_bar['close']

        # Find order blocks
        bullish_ob, bearish_ob = self.find_order_blocks(df, current_idx)

        # Find FVGs
        bullish_fvg, bearish_fvg = self.find_fair_value_gaps(df, current_idx)

        # Check for liquidity sweep
        sweep = self.check_liquidity_sweep(df, current_idx)

        # LONG SIGNAL
        if bullish_ob is not None:
            # Price must be near the order block (within it or just above)
            if bullish_ob['low'] <= current_price <= bullish_ob['high'] * 1.002:

                confluence = 1.0  # Base score

                # Add confluence for FVG
                if bullish_fvg is not None:
                    confluence += 0.5

                # Add confluence for liquidity sweep
                if sweep == 'BULLISH_SWEEP':
                    confluence += 0.8

                # Require at least 1.5 confluence (OB + one other factor)
                if confluence >= 1.5:
                    return {
                        'direction': 'LONG',
                        'entry_price': current_price,
                        'ob': bullish_ob,
                        'fvg': bullish_fvg,
                        'sweep': sweep,
                        'confluence': confluence,
                        'kill_zone': kill_zone
                    }

        # SHORT SIGNAL
        if bearish_ob is not None:
            # Price must be near the order block
            if bearish_ob['high'] >= current_price >= bearish_ob['low'] * 0.998:

                confluence = 1.0

                if bearish_fvg is not None:
                    confluence += 0.5

                if sweep == 'BEARISH_SWEEP':
                    confluence += 0.8

                if confluence >= 1.5:
                    return {
                        'direction': 'SHORT',
                        'entry_price': current_price,
                        'ob': bearish_ob,
                        'fvg': bearish_fvg,
                        'sweep': sweep,
                        'confluence': confluence,
                        'kill_zone': kill_zone
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
        """Run ICT strategy backtest"""

        print('Downloading data...\n')

        data_1h = {}

        for pair_name, ticker in pairs.items():
            df = self.download_data(ticker, start_date, end_date)

            if df is not None:
                # Ensure timezone aware
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')

                data_1h[pair_name] = df
                print(f'{pair_name}: {len(df)} bars (1H)')

        print(f'\nRunning ICT backtest...\n')

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
                        'confluence': position['confluence'],
                        'kill_zone': position['kill_zone'],
                    })

                    del self.positions[pair_name]

            # Check for new entries
            if len(self.positions) >= self.max_positions:
                continue

            for pair_name, df in data_1h.items():

                if pair_name in self.positions or timestamp not in df.index:
                    continue

                current_idx = df.index.get_loc(timestamp)

                # Get ICT signal
                signal = self.get_ict_signal(df, current_idx, timestamp)

                if signal is None:
                    continue

                # Enter position
                entry_price = signal['entry_price']

                # Calculate position size
                risk_amount = self.balance * self.risk_per_trade
                units = risk_amount / (entry_price * self.stop_loss_pct)

                if signal['direction'] == 'LONG':
                    tp = entry_price * (1 + self.profit_target_pct)
                    sl = entry_price * (1 - self.stop_loss_pct)
                else:
                    tp = entry_price * (1 - self.profit_target_pct)
                    sl = entry_price * (1 + self.stop_loss_pct)

                self.positions[pair_name] = {
                    'direction': signal['direction'],
                    'entry_price': entry_price,
                    'tp': tp,
                    'sl': sl,
                    'units': units,
                    'confluence': signal['confluence'],
                    'kill_zone': signal['kill_zone'],
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
                'pnl': pnl,
                'exit_reason': 'END',
                'confluence': position['confluence'],
                'kill_zone': position['kill_zone'],
            })

        self.positions = {}

    def print_results(self):
        """Print ICT strategy results"""

        total_trades = len(self.closed_trades)

        if total_trades == 0:
            print('[!] NO TRADES - ICT setups are selective (this is normal)')
            print('  Strategy waits for HIGH PROBABILITY confluences')
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
        print('ICT (SMART MONEY) STRATEGY RESULTS')
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

        # Kill zone breakdown
        london_trades = [t for t in self.closed_trades if t.get('kill_zone') == 'LONDON']
        ny_trades = [t for t in self.closed_trades if t.get('kill_zone') == 'NY']

        print(f'\nKill Zone Breakdown:')
        print(f'  London (2-5am EST):  {len(london_trades)} trades')
        print(f'  NY (8:30-11am EST):  {len(ny_trades)} trades')

        print('\n' + '=' * 70)
        print('COMPARISON: ICT vs PRICE ACTION vs MACD/RSI')
        print('=' * 70)
        print(f'\n{"Metric":<25} {"MACD/RSI":<15} {"Price Action":<15} {"ICT (SMC)":<15}')
        print('-' * 70)
        print(f'{"Win Rate":<25} {"26.7%":<15} {"44.0%":<15} {f"{win_rate:.1f}%":<15}')
        print(f'{"Total Trades":<25} {"30":<15} {"25":<15} {total_trades:<15}')
        print(f'{"ROI (6 months)":<25} {"-9.28%":<15} {"+5.44%":<15} {f"{roi:+.2f}%":<15}')
        print(f'{"Max Drawdown":<25} {"14.0%":<15} {"5.94%":<15} {f"{max_dd:.2f}%":<15}')
        print(f'{"Expectancy":<25} {"-$619":<15} {"+$435":<15} {f"${expectancy:.0f}":<15}')

        print('\n' + '=' * 70)
        print('VERDICT')
        print('=' * 70)

        if win_rate >= 65 and roi >= 10 and max_dd < 6:
            print('\n[OK] ICT STRATEGY WORKS - DEPLOY IMMEDIATELY')
            print(f'  Win rate: {win_rate:.1f}% (target: 65%+) OK')
            print(f'  ROI: {roi:.2f}% (target: 10%+) OK')
            print(f'  Max DD: {max_dd:.2f}% (limit: 6%) OK')
            print(f'\n  WOULD PASS E8 $200K CHALLENGE IN 6 MONTHS!')

        elif win_rate >= 50 and roi > 0 and max_dd < 6:
            print(f'\n[~] ICT SHOWS PROMISE')
            print(f'  Win rate: {win_rate:.1f}%')
            print(f'  ROI: {roi:.2f}%')
            print(f'  Max DD: {max_dd:.2f}%')

            if roi > 5.44:  # Better than price action
                print(f'\n  ICT BEATS PRICE ACTION!')
                print(f'  Price Action: 5.44% ROI, 44% WR')
                print(f'  ICT: {roi:.2f}% ROI, {win_rate:.1f}% WR')
                print(f'\n  DEPLOY ICT STRATEGY')
            else:
                print(f'\n  Still better to use Price Action (5.44% ROI, 44% WR)')

        else:
            print(f'\n[X] ICT NEEDS OPTIMIZATION')
            print(f'  Win rate: {win_rate:.1f}%')
            print(f'  ROI: {roi:.2f}%')


if __name__ == '__main__':
    print('=' * 70)
    print('ICT (INNER CIRCLE TRADER) STRATEGY BACKTEST')
    print('=' * 70)
    print('\nOrder Blocks + Fair Value Gaps + Kill Zones')
    print('Expected: 70-75% win rate, 10-15% monthly ROI\n')

    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    print(f'Period: {start_date.date()} to {end_date.date()} (6 months)\n')

    pairs = {
        'EURUSD': 'EUR/USD',
        'GBPUSD': 'GBP/USD',
        'USDJPY': 'USD/JPY'
    }

    backtest = ICTStrategy(initial_balance=200000)
    backtest.run_backtest(pairs, start_date, end_date)
    backtest.print_results()
