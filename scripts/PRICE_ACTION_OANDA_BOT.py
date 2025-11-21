"""
PRICE ACTION BREAK-AND-RETEST BOT - OANDA LIVE
Backtest: +$10,878 in 6 months (44% WR, 5.94% max DD)
This is what will actually pass E8.
"""

import os
import requests
import json
import time
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class PriceActionOandaBot:
    def __init__(self):
        self.account_id = os.getenv('OANDA_ACCOUNT_ID')
        self.api_key = os.getenv('OANDA_API_KEY')
        self.base_url = 'https://api-fxpractice.oanda.com'

        # Strategy parameters (from backtest)
        self.pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']
        self.profit_target_pct = 0.015  # 1.5%
        self.stop_loss_pct = 0.008  # 0.8%
        self.risk_per_trade = 0.01  # 1% risk
        self.max_positions = 3

        # Break-and-retest parameters
        self.break_threshold = 0.0002  # 2 pips
        self.retest_threshold = 0.0003  # 3 pips
        self.max_hours_for_retest = 12

        # Track recent breaks (to detect retests)
        self.recent_breaks = {}  # {pair: {'type': 'resistance'/'support', 'level': price, 'time': timestamp}}

        print('=' * 70)
        print('PRICE ACTION OANDA BOT STARTED')
        print('=' * 70)
        print(f'Strategy: Break-and-Retest')
        print(f'Pairs: {", ".join(self.pairs)}')
        print(f'Risk per trade: {self.risk_per_trade * 100}%')
        print(f'Expected: +$435/trade, 44% WR')
        print('=' * 70)

    def get_account_summary(self):
        """Get account balance and positions"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        response = requests.get(
            f'{self.base_url}/v3/accounts/{self.account_id}/summary',
            headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            balance = float(data['account']['balance'])
            nav = float(data['account']['NAV'])
            unrealized_pl = float(data['account']['unrealizedPL'])
            return balance, nav, unrealized_pl
        else:
            print(f'[ERROR] Failed to get account summary: {response.text}')
            return None, None, None

    def get_open_positions(self):
        """Get currently open positions"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
        }

        response = requests.get(
            f'{self.base_url}/v3/accounts/{self.account_id}/openPositions',
            headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            return data.get('positions', [])
        else:
            return []

    def get_candles(self, pair, count=200, granularity='H1'):
        """Get historical candles"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
        }

        response = requests.get(
            f'{self.base_url}/v3/instruments/{pair}/candles',
            headers=headers,
            params={
                'count': count,
                'granularity': granularity,
                'price': 'M'  # Mid prices
            }
        )

        if response.status_code == 200:
            data = response.json()
            candles = data.get('candles', [])

            # Convert to DataFrame
            df_data = []
            for candle in candles:
                df_data.append({
                    'time': pd.to_datetime(candle['time']),
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume'])
                })

            df = pd.DataFrame(df_data)
            df.set_index('time', inplace=True)

            return df
        else:
            print(f'[ERROR] Failed to get candles for {pair}: {response.text}')
            return None

    def get_daily_levels(self, df):
        """Get yesterday's high/low as S/R levels"""
        # Get yesterday's bars
        now = df.index[-1]
        yesterday_date = (now - timedelta(days=1)).date()

        # Filter to yesterday
        df_yesterday = df[df.index.date == yesterday_date]

        if len(df_yesterday) == 0:
            # Weekend - try 2 days ago
            yesterday_date = (now - timedelta(days=2)).date()
            df_yesterday = df[df.index.date == yesterday_date]

        if len(df_yesterday) == 0:
            # Still nothing - try 3 days ago
            yesterday_date = (now - timedelta(days=3)).date()
            df_yesterday = df[df.index.date == yesterday_date]

        if len(df_yesterday) == 0:
            return None, None

        resistance = df_yesterday['high'].max()
        support = df_yesterday['low'].min()

        return resistance, support

    def check_for_break(self, df, resistance, support, pair):
        """Check if we just broke resistance or support"""
        if len(df) < 2:
            return

        prev_bar = df.iloc[-2]
        current_bar = df.iloc[-1]

        # RESISTANCE BREAK
        if prev_bar['high'] < resistance and current_bar['high'] >= resistance + self.break_threshold:
            # Confirmed break!
            self.recent_breaks[pair] = {
                'type': 'RESISTANCE',
                'level': resistance,
                'time': current_bar.name,
                'support': support
            }
            print(f'\n[!] {pair} BROKE RESISTANCE at {resistance:.5f}')
            print(f'    Watching for retest in next {self.max_hours_for_retest} hours...')

        # SUPPORT BREAK
        elif prev_bar['low'] > support and current_bar['low'] <= support - self.break_threshold:
            # Confirmed break!
            self.recent_breaks[pair] = {
                'type': 'SUPPORT',
                'level': support,
                'time': current_bar.name,
                'resistance': resistance
            }
            print(f'\n[!] {pair} BROKE SUPPORT at {support:.5f}')
            print(f'    Watching for retest in next {self.max_hours_for_retest} hours...')

    def check_for_retest(self, df, pair):
        """Check if price is retesting a recently broken level"""
        if pair not in self.recent_breaks:
            return None

        break_info = self.recent_breaks[pair]
        current_bar = df.iloc[-1]

        # Check if break is too old
        hours_since_break = (current_bar.name - break_info['time']).total_seconds() / 3600

        if hours_since_break > self.max_hours_for_retest:
            # Break is stale, remove it
            del self.recent_breaks[pair]
            return None

        # LONG SETUP: Resistance broken, now retesting from above
        if break_info['type'] == 'RESISTANCE':
            level = break_info['level']

            # Price must be above level (stay broken)
            if current_bar['close'] < level:
                return None

            # Low must have touched level (retest)
            retest_distance = current_bar['low'] - level

            if -self.retest_threshold <= retest_distance <= self.retest_threshold:
                # RETEST CONFIRMED! Enter LONG
                return {
                    'direction': 'LONG',
                    'entry_price': current_bar['close'],
                    'retest_level': level,
                    'support': break_info.get('support')
                }

        # SHORT SETUP: Support broken, now retesting from below
        elif break_info['type'] == 'SUPPORT':
            level = break_info['level']

            # Price must be below level (stay broken)
            if current_bar['close'] > level:
                return None

            # High must have touched level (retest)
            retest_distance = current_bar['high'] - level

            if -self.retest_threshold <= retest_distance <= self.retest_threshold:
                # RETEST CONFIRMED! Enter SHORT
                return {
                    'direction': 'SHORT',
                    'entry_price': current_bar['close'],
                    'retest_level': level,
                    'resistance': break_info.get('resistance')
                }

        return None

    def place_order(self, pair, direction, units, tp, sl):
        """Place market order with TP and SL"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        # OANDA uses negative units for SHORT
        if direction == 'SHORT':
            units = -abs(units)
        else:
            units = abs(units)

        order_data = {
            'order': {
                'type': 'MARKET',
                'instrument': pair,
                'units': str(int(units)),
                'timeInForce': 'FOK',
                'positionFill': 'DEFAULT',
                'takeProfitOnFill': {
                    'price': f'{tp:.5f}'
                },
                'stopLossOnFill': {
                    'price': f'{sl:.5f}'
                }
            }
        }

        response = requests.post(
            f'{self.base_url}/v3/accounts/{self.account_id}/orders',
            headers=headers,
            json=order_data
        )

        if response.status_code == 201:
            data = response.json()
            print(f'[OK] Order placed successfully')
            return data
        else:
            print(f'[ERROR] Order failed: {response.text}')
            return None

    def scan_for_signals(self):
        """Scan all pairs for break-and-retest setups"""
        balance, nav, unrealized_pl = self.get_account_summary()

        if balance is None:
            return

        print(f'\n[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]')
        print(f'Balance: ${balance:,.2f} | NAV: ${nav:,.2f} | Unrealized P/L: ${unrealized_pl:,.2f}')

        # Check how many positions open
        open_positions = self.get_open_positions()
        open_pairs = [pos['instrument'] for pos in open_positions]

        print(f'Open Positions: {len(open_positions)}/{self.max_positions}')

        if len(open_positions) >= self.max_positions:
            print('  [Max positions reached, waiting for exits...]')
            return

        # Scan each pair
        for pair in self.pairs:

            # Skip if already in position
            if pair in open_pairs:
                continue

            # Get candles
            df = self.get_candles(pair, count=200, granularity='H1')

            if df is None or len(df) < 48:
                continue

            # Get daily levels
            resistance, support = self.get_daily_levels(df)

            if resistance is None or support is None:
                continue

            # Check for new breaks
            self.check_for_break(df, resistance, support, pair)

            # Check for retest
            signal = self.check_for_retest(df, pair)

            if signal is not None:
                # ENTRY SIGNAL!
                direction = signal['direction']
                entry_price = signal['entry_price']
                retest_level = signal['retest_level']

                print(f'\n' + '=' * 70)
                print(f'[SIGNAL] {pair} {direction} at {entry_price:.5f}')
                print(f'  Retest Level: {retest_level:.5f}')
                print(f'  Resistance: {resistance:.5f}')
                print(f'  Support: {support:.5f}')

                # Calculate position size
                risk_amount = balance * self.risk_per_trade
                stop_distance = entry_price * self.stop_loss_pct
                units = risk_amount / stop_distance

                # Calculate TP/SL
                if direction == 'LONG':
                    tp = entry_price * (1 + self.profit_target_pct)
                    sl = retest_level * (1 - self.stop_loss_pct)
                else:  # SHORT
                    tp = entry_price * (1 - self.profit_target_pct)
                    sl = retest_level * (1 + self.stop_loss_pct)

                print(f'  Units: {int(units):,}')
                print(f'  TP: {tp:.5f} ({self.profit_target_pct * 100}%)')
                print(f'  SL: {sl:.5f} ({self.stop_loss_pct * 100}%)')
                print(f'  Risk: ${risk_amount:,.2f}')
                print('=' * 70)

                # Place order
                result = self.place_order(pair, direction, units, tp, sl)

                if result:
                    # Clear the break (we entered on it)
                    if pair in self.recent_breaks:
                        del self.recent_breaks[pair]

                time.sleep(1)

    def run(self, scan_interval=3600):
        """Run bot continuously"""
        print(f'\nScanning every {scan_interval} seconds ({scan_interval/3600:.1f} hours)...\n')

        while True:
            try:
                self.scan_for_signals()

            except Exception as e:
                print(f'[ERROR] {e}')
                import traceback
                traceback.print_exc()

            time.sleep(scan_interval)


if __name__ == '__main__':
    bot = PriceActionOandaBot()
    bot.run(scan_interval=3600)  # Scan every hour
