"""
E8 STRATEGY BACKTESTER - Using OANDA Historical Data

Validates the 3 market microstructure setups:
1. London Fake-out (stop hunts)
2. NY Absorption (order flow reversal)
3. Tokyo Gap Fill (mechanical reversion)

Uses OANDA data for backtesting, results apply to E8 live trading.
"""

import os
from datetime import datetime, timedelta, time as dt_time
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class E8Backtester:
    """
    Backtest E8 setups using OANDA historical data.
    Simulates detection without execution to prove edge.
    """

    def __init__(self):
        # OANDA credentials for historical data
        self.api = API(access_token=os.getenv('OANDA_API_KEY'))
        self.account_id = os.getenv('OANDA_ACCOUNT_ID')

        # Trading pairs for E8
        self.pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']

        # Results storage
        self.results = {
            'london_fakeout': [],
            'ny_absorption': [],
            'tokyo_gap_fill': []
        }

        print("=" * 70)
        print("E8 STRATEGY BACKTESTER")
        print("=" * 70)
        print("Data Source: OANDA (historical candles)")
        print("Target: E8 Challenge Account")
        print("Pairs: EUR/USD, GBP/USD, USD/JPY")
        print("Timeframe: Last 30 days")
        print("=" * 70)

    def get_candles(self, pair, start, end, granularity='H1'):
        """Fetch historical candles from OANDA"""
        try:
            params = {
                'from': start.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'to': end.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'granularity': granularity
            }

            req = InstrumentsCandles(instrument=pair, params=params)
            response = self.api.request(req)

            candles = []
            for candle in response.get('candles', []):
                if candle['complete']:
                    candles.append({
                        'time': candle['time'],
                        'timestamp': datetime.fromisoformat(candle['time'].replace('Z', '+00:00')),
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': int(candle['volume'])
                    })

            return candles

        except Exception as e:
            print(f"[ERROR] Fetching candles for {pair}: {e}")
            return []

    def is_london_session(self, dt):
        """Check if time is London session (3-5 AM EST)"""
        return dt_time(3, 0) <= dt.time() < dt_time(5, 0)

    def is_ny_session(self, dt):
        """Check if time is NY session (8 AM-12 PM EST)"""
        return dt_time(8, 0) <= dt.time() < dt_time(12, 0)

    def is_tokyo_session(self, dt):
        """Check if time is Tokyo session (7 PM-2 AM EST)"""
        t = dt.time()
        return t >= dt_time(19, 0) or t < dt_time(2, 0)

    def detect_london_fakeout(self, candles, idx):
        """
        Detect London fake-out setup

        Theory: Low volume breakout of Asian range = stop hunt
        Entry: Fade the breakout (reverse direction)
        Edge: Institutions trapping retail before reversing
        """
        if idx < 32:  # Need 8 hours (32 x 15min) of Asian session
            return None

        current = candles[idx]

        # Only trade during London session
        if not self.is_london_session(current['timestamp']):
            return None

        # Calculate Asian range (previous 8 hours)
        asian_candles = candles[idx-32:idx]
        asian_high = max([c['high'] for c in asian_candles])
        asian_low = min([c['low'] for c in asian_candles])
        asian_range = asian_high - asian_low

        # Calculate volume metrics
        avg_volume = sum([c['volume'] for c in candles[idx-10:idx]]) / 10
        current_volume = current['volume']

        # Detection: Low volume breakout = fake
        breakout_buffer = asian_range * 0.02

        # Upside fake-out
        if current['close'] > asian_high + breakout_buffer:
            if current_volume < avg_volume * 0.7:  # Low volume = fake
                return {
                    'setup': 'london_fakeout',
                    'direction': 'SHORT',
                    'entry': current['close'],
                    'stop': asian_high + (asian_range * 0.05),
                    'target': asian_low,
                    'timestamp': current['timestamp'],
                    'asian_range': asian_range,
                    'volume_ratio': current_volume / avg_volume
                }

        # Downside fake-out
        elif current['close'] < asian_low - breakout_buffer:
            if current_volume < avg_volume * 0.7:
                return {
                    'setup': 'london_fakeout',
                    'direction': 'LONG',
                    'entry': current['close'],
                    'stop': asian_low - (asian_range * 0.05),
                    'target': asian_high,
                    'timestamp': current['timestamp'],
                    'asian_range': asian_range,
                    'volume_ratio': current_volume / avg_volume
                }

        return None

    def detect_ny_absorption(self, candles, idx):
        """
        Detect NY absorption setup

        Theory: Price tests previous day high/low, gets absorbed
        Entry: Trade the rejection (order flow imbalance)
        Edge: Institutions defending levels with size
        """
        if idx < 48:  # Need previous day data
            return None

        current = candles[idx]

        # Only trade during NY session
        if not self.is_ny_session(current['timestamp']):
            return None

        # Get previous day high/low
        prev_day_candles = candles[idx-48:idx-24]
        prev_high = max([c['high'] for c in prev_day_candles])
        prev_low = min([c['low'] for c in prev_day_candles])

        # Recent price action (last 4 hours)
        recent = candles[idx-4:idx+1]

        # Check for rejection at previous high
        if current['high'] >= prev_high * 0.999:  # Within 0.1%
            if current['close'] < current['open']:  # Bearish close
                # Confirmed rejection
                return {
                    'setup': 'ny_absorption',
                    'direction': 'SHORT',
                    'entry': current['close'],
                    'stop': prev_high + (prev_high * 0.001),
                    'target': prev_low,
                    'timestamp': current['timestamp'],
                    'level': 'prev_day_high',
                    'rejection_size': current['high'] - current['close']
                }

        # Check for rejection at previous low
        elif current['low'] <= prev_low * 1.001:  # Within 0.1%
            if current['close'] > current['open']:  # Bullish close
                return {
                    'setup': 'ny_absorption',
                    'direction': 'LONG',
                    'entry': current['close'],
                    'stop': prev_low - (prev_low * 0.001),
                    'target': prev_high,
                    'timestamp': current['timestamp'],
                    'level': 'prev_day_low',
                    'rejection_size': current['close'] - current['low']
                }

        return None

    def detect_tokyo_gap_fill(self, candles, idx):
        """
        Detect Tokyo gap fill setup

        Theory: Weekend/session gaps fill mechanically
        Entry: Trade into the gap
        Edge: 70%+ gap fill rate, mean reversion
        """
        if idx < 2:
            return None

        current = candles[idx]

        # Only trade during Tokyo session
        if not self.is_tokyo_session(current['timestamp']):
            return None

        # Check for gap from previous close
        prev_close = candles[idx-1]['close']
        current_open = current['open']

        gap_size = abs(current_open - prev_close)
        gap_threshold = prev_close * 0.0005  # 5 pips minimum

        if gap_size < gap_threshold:
            return None

        # Gap up - expect fill down
        if current_open > prev_close:
            return {
                'setup': 'tokyo_gap_fill',
                'direction': 'SHORT',
                'entry': current['close'],
                'stop': current['high'] + gap_size * 0.5,
                'target': prev_close,
                'timestamp': current['timestamp'],
                'gap_size': gap_size,
                'gap_pips': gap_size * 10000  # Approx pips
            }

        # Gap down - expect fill up
        else:
            return {
                'setup': 'tokyo_gap_fill',
                'direction': 'LONG',
                'entry': current['close'],
                'stop': current['low'] - gap_size * 0.5,
                'target': prev_close,
                'timestamp': current['timestamp'],
                'gap_size': gap_size,
                'gap_pips': gap_size * 10000
            }

    def simulate_trade(self, setup, candles, start_idx):
        """
        Simulate trade outcome

        Returns: {
            'win': True/False,
            'pnl': float,
            'exit_time': datetime,
            'exit_reason': 'target'|'stop'|'time'
        }
        """
        entry = setup['entry']
        stop = setup['stop']
        target = setup['target']
        direction = setup['direction']

        # Check next 24 hours (24 candles if H1)
        for i in range(start_idx + 1, min(start_idx + 25, len(candles))):
            candle = candles[i]

            if direction == 'LONG':
                # Check if stopped out
                if candle['low'] <= stop:
                    return {
                        'win': False,
                        'pnl': stop - entry,
                        'exit_time': candle['timestamp'],
                        'exit_reason': 'stop'
                    }
                # Check if target hit
                if candle['high'] >= target:
                    return {
                        'win': True,
                        'pnl': target - entry,
                        'exit_time': candle['timestamp'],
                        'exit_reason': 'target'
                    }

            else:  # SHORT
                # Check if stopped out
                if candle['high'] >= stop:
                    return {
                        'win': False,
                        'pnl': entry - stop,
                        'exit_time': candle['timestamp'],
                        'exit_reason': 'stop'
                    }
                # Check if target hit
                if candle['low'] <= target:
                    return {
                        'win': True,
                        'pnl': entry - target,
                        'exit_time': candle['timestamp'],
                        'exit_reason': 'target'
                    }

        # Time-based exit after 24 hours
        final_candle = candles[min(start_idx + 24, len(candles) - 1)]
        final_price = final_candle['close']

        if direction == 'LONG':
            pnl = final_price - entry
        else:
            pnl = entry - final_price

        return {
            'win': pnl > 0,
            'pnl': pnl,
            'exit_time': final_candle['timestamp'],
            'exit_reason': 'time'
        }

    def backtest_pair(self, pair, days=30):
        """Run backtest for one pair"""
        print(f"\n{'='*70}")
        print(f"BACKTESTING: {pair}")
        print(f"{'='*70}")

        # Fetch data
        end = datetime.utcnow()
        start = end - timedelta(days=days)

        print(f"Fetching {days} days of hourly data...")
        candles = self.get_candles(pair, start, end, granularity='H1')

        if not candles:
            print("[ERROR] No data received!")
            return

        print(f"Analyzing {len(candles)} candles...")

        # Run detection on each candle
        for i in range(len(candles)):
            # London Fake-out
            setup = self.detect_london_fakeout(candles, i)
            if setup:
                outcome = self.simulate_trade(setup, candles, i)
                setup['pair'] = pair
                setup['outcome'] = outcome
                self.results['london_fakeout'].append(setup)

            # NY Absorption
            setup = self.detect_ny_absorption(candles, i)
            if setup:
                outcome = self.simulate_trade(setup, candles, i)
                setup['pair'] = pair
                setup['outcome'] = outcome
                self.results['ny_absorption'].append(setup)

            # Tokyo Gap Fill
            setup = self.detect_tokyo_gap_fill(candles, i)
            if setup:
                outcome = self.simulate_trade(setup, candles, i)
                setup['pair'] = pair
                setup['outcome'] = outcome
                self.results['tokyo_gap_fill'].append(setup)

        print(f"[OK] Backtest complete for {pair}")

    def run(self, days=30):
        """Run full backtest across all pairs"""
        print(f"\nStarting backtest for last {days} days...\n")

        for pair in self.pairs:
            self.backtest_pair(pair, days)

        # Generate report
        self.print_report()
        self.save_results()

    def print_report(self):
        """Print backtest results"""
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)

        for setup_name, trades in self.results.items():
            if not trades:
                continue

            wins = [t for t in trades if t['outcome']['win']]
            win_rate = len(wins) / len(trades) * 100

            total_pnl = sum([t['outcome']['pnl'] for t in trades])
            avg_win = sum([t['outcome']['pnl'] for t in wins]) / len(wins) if wins else 0
            losses = [t for t in trades if not t['outcome']['win']]
            avg_loss = sum([t['outcome']['pnl'] for t in losses]) / len(losses) if losses else 0

            print(f"\n{setup_name.upper().replace('_', ' ')}")
            print(f"  Total Setups: {len(trades)}")
            print(f"  Win Rate: {win_rate:.1f}% ({len(wins)}/{len(trades)})")
            print(f"  Total P&L: {total_pnl:+.5f}")
            print(f"  Avg Win: {avg_win:+.5f}")
            print(f"  Avg Loss: {avg_loss:+.5f}")

            if avg_loss != 0:
                risk_reward = abs(avg_win / avg_loss)
                print(f"  Risk:Reward: 1:{risk_reward:.2f}")

            # Show best trade
            if wins:
                best = max(wins, key=lambda x: x['outcome']['pnl'])
                print(f"  Best Trade: {best['outcome']['pnl']:+.5f} on {best['pair']} at {best['timestamp']}")

        print("\n" + "=" * 70)

    def save_results(self):
        """Save results to JSON"""
        filename = f"e8_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert datetime objects to strings
        serializable = {}
        for setup_name, trades in self.results.items():
            serializable[setup_name] = []
            for trade in trades:
                t = trade.copy()
                t['timestamp'] = t['timestamp'].isoformat()
                t['outcome']['exit_time'] = t['outcome']['exit_time'].isoformat()
                serializable[setup_name].append(t)

        with open(filename, 'w') as f:
            json.dump(serializable, f, indent=2)

        print(f"\n[SAVED] Results: {filename}")


if __name__ == '__main__':
    backtester = E8Backtester()
    backtester.run(days=30)  # Backtest last 30 days
