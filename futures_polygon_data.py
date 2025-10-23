#!/usr/bin/env python3
"""
FUTURES POLYGON DATA INTEGRATION
Alternative data source for futures backtesting

Polygon.io provides futures historical data with free tier.
Use this as alternative when Alpaca paper account doesn't have historical access.

Free Tier:
- 5 API calls per minute
- 2 years historical data
- Delayed data (15 min delay)

Get API Key: https://polygon.io (Free tier available)

Usage:
    python futures_polygon_data.py --symbol MES --backtest
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import json


class PolygonFuturesDataFetcher:
    """
    Fetch futures data from Polygon.io

    Supports:
    - MES (Micro E-mini S&P 500) → ES
    - MNQ (Micro E-mini Nasdaq-100) → NQ
    - Historical bars (1min, 5min, 15min, 1hour, 1day)
    - Current quotes
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Polygon data fetcher

        Args:
            api_key: Polygon.io API key (or set POLYGON_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')

        if not self.api_key:
            print("\n⚠ WARNING: No Polygon API key found")
            print("Get free API key at: https://polygon.io")
            print("Set environment variable: POLYGON_API_KEY=your_key")
            print("Or pass api_key parameter\n")

        self.base_url = "https://api.polygon.io"
        self.rate_limit_delay = 12  # 5 calls/min = 12 sec between calls (free tier)

        # Symbol mapping: Micro futures → Full size futures for Polygon
        self.symbol_map = {
            'MES': 'ES',   # Micro E-mini S&P 500 → E-mini S&P 500
            'MNQ': 'NQ',   # Micro E-mini Nasdaq → E-mini Nasdaq
            'M2K': '2K',   # Micro E-mini Russell → E-mini Russell
            'MYM': 'YM'    # Micro E-mini Dow → E-mini Dow
        }

        # Timeframe mapping
        self.timeframe_map = {
            '1Min': {'multiplier': 1, 'timespan': 'minute'},
            '5Min': {'multiplier': 5, 'timespan': 'minute'},
            '15Min': {'multiplier': 15, 'timespan': 'minute'},
            '1H': {'multiplier': 1, 'timespan': 'hour'},
            '1D': {'multiplier': 1, 'timespan': 'day'}
        }

    def convert_symbol(self, micro_symbol: str) -> str:
        """Convert micro futures symbol to full size for Polygon"""
        return self.symbol_map.get(micro_symbol, micro_symbol)

    def get_bars(self, symbol: str, timeframe: str = '15Min', days: int = 90) -> Optional[pd.DataFrame]:
        """
        Get historical bars from Polygon

        Args:
            symbol: Micro futures symbol (e.g., 'MES', 'MNQ')
            timeframe: '1Min', '5Min', '15Min', '1H', '1D'
            days: Number of days of historical data

        Returns:
            DataFrame with OHLCV data
        """
        if not self.api_key:
            print("[ERROR] No API key provided")
            return None

        # Convert symbol
        polygon_symbol = self.convert_symbol(symbol)

        # Get timeframe parameters
        if timeframe not in self.timeframe_map:
            print(f"[ERROR] Invalid timeframe: {timeframe}")
            return None

        tf_params = self.timeframe_map[timeframe]

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Format dates for Polygon
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')

        # Build URL
        # Example: /v2/aggs/ticker/X:ES/range/15/minute/2024-01-01/2024-04-01
        url = (
            f"{self.base_url}/v2/aggs/ticker/X:{polygon_symbol}/range/"
            f"{tf_params['multiplier']}/{tf_params['timespan']}/{from_date}/{to_date}"
        )

        params = {
            'apiKey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        }

        print(f"\n[POLYGON] Fetching {symbol} ({polygon_symbol}) data...")
        print(f"  Timeframe: {timeframe}")
        print(f"  Period: {from_date} to {to_date}")

        try:
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 429:
                print("[ERROR] Rate limit exceeded. Free tier: 5 calls/min")
                return None

            if response.status_code != 200:
                print(f"[ERROR] API returned status {response.status_code}")
                print(f"Response: {response.text}")
                return None

            data = response.json()

            if data.get('status') != 'OK':
                print(f"[ERROR] API status: {data.get('status')}")
                return None

            results = data.get('results', [])

            if not results:
                print("[ERROR] No data returned")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(results)

            # Rename columns to match our format
            df = df.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                't': 'timestamp'
            })

            # Convert timestamp (ms) to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')

            # Select relevant columns
            df = df[['open', 'high', 'low', 'close', 'volume']]

            print(f"[SUCCESS] Fetched {len(df)} bars")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")

            # Rate limit protection
            time.sleep(self.rate_limit_delay)

            return df

        except requests.exceptions.Timeout:
            print("[ERROR] Request timed out")
            return None
        except Exception as e:
            print(f"[ERROR] {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for symbol

        Args:
            symbol: Micro futures symbol

        Returns:
            Current price or None
        """
        if not self.api_key:
            return None

        polygon_symbol = self.convert_symbol(symbol)

        # Get last trade
        url = f"{self.base_url}/v2/last/trade/X:{polygon_symbol}"
        params = {'apiKey': self.api_key}

        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'OK':
                    price = data['results']['p']
                    time.sleep(self.rate_limit_delay)
                    return price

        except:
            pass

        return None


def run_polygon_backtest(symbol: str = 'MES', timeframe: str = '15Min', days: int = 90):
    """
    Run backtest using Polygon data

    Args:
        symbol: Futures symbol
        timeframe: Bar timeframe
        days: Days of historical data
    """
    print("\n" + "="*70)
    print("FUTURES BACKTEST WITH POLYGON DATA")
    print("="*70)

    # Initialize Polygon fetcher
    fetcher = PolygonFuturesDataFetcher()

    # Fetch data
    df = fetcher.get_bars(symbol, timeframe, days)

    if df is None or df.empty:
        print("\n[FAILED] Could not fetch data from Polygon")
        print("\nTroubleshooting:")
        print("  1. Check API key is set: POLYGON_API_KEY")
        print("  2. Verify API key at: https://polygon.io/dashboard/api-keys")
        print("  3. Free tier has 5 calls/min limit")
        return None

    # Import strategy
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from strategies.futures_ema_strategy import FuturesEMAStrategy

    # Run strategy analysis
    strategy = FuturesEMAStrategy()
    opportunity = strategy.analyze_opportunity(df, symbol)

    if opportunity:
        print("\n" + "="*70)
        print("CURRENT OPPORTUNITY")
        print("="*70)
        print(f"Symbol: {opportunity['symbol']}")
        print(f"Direction: {opportunity['direction']}")
        print(f"Score: {opportunity['score']:.2f}/12")
        print(f"Entry: ${opportunity['entry_price']:.2f}")
        print(f"Stop Loss: ${opportunity['stop_loss']:.2f}")
        print(f"Take Profit: ${opportunity['take_profit']:.2f}")
        print(f"Risk/Reward: {opportunity['risk_reward']:.2f}:1")
        print(f"Risk per contract: ${opportunity['risk_per_contract']:.2f}")

        valid = strategy.validate_rules(opportunity)
        print(f"\nPasses All Rules: {'✓ YES' if valid else '✗ NO'}")
        print("="*70)
    else:
        print("\n[NO SIGNAL] No setup detected at this time")

    # Run simple backtest
    print("\n" + "="*70)
    print("SIMPLE BACKTEST")
    print("="*70)

    df_with_indicators = strategy.calculate_indicators(df)

    # Detect all signals
    signals = []
    for i in range(strategy.ema_trend + 10, len(df_with_indicators) - 1):
        window_df = df_with_indicators.iloc[:i+1]
        opp = strategy.analyze_opportunity(window_df.copy(), symbol)

        if opp and strategy.validate_rules(opp):
            signals.append({
                'index': i,
                'timestamp': df_with_indicators.index[i],
                'direction': opp['direction'],
                'entry': opp['entry_price'],
                'stop': opp['stop_loss'],
                'target': opp['take_profit']
            })

    print(f"Signals detected: {len(signals)}")

    if signals:
        # Simulate trades
        trades = []
        for signal in signals:
            entry_idx = signal['index']
            future_bars = df_with_indicators.iloc[entry_idx+1:]

            # Find exit
            exit_idx = None
            exit_price = None
            exit_reason = None

            for j, (timestamp, bar) in enumerate(future_bars.iterrows(), entry_idx + 1):
                if signal['direction'] == 'LONG':
                    if bar['low'] <= signal['stop']:
                        exit_idx = j
                        exit_price = signal['stop']
                        exit_reason = 'STOP'
                        break
                    elif bar['high'] >= signal['target']:
                        exit_idx = j
                        exit_price = signal['target']
                        exit_reason = 'TARGET'
                        break
                else:  # SHORT
                    if bar['high'] >= signal['stop']:
                        exit_idx = j
                        exit_price = signal['stop']
                        exit_reason = 'STOP'
                        break
                    elif bar['low'] <= signal['target']:
                        exit_idx = j
                        exit_price = signal['target']
                        exit_reason = 'TARGET'
                        break

                # Max holding period: 100 bars
                if j - entry_idx > 100:
                    exit_idx = j
                    exit_price = bar['close']
                    exit_reason = 'TIME'
                    break

            if exit_idx:
                if signal['direction'] == 'LONG':
                    pnl = exit_price - signal['entry']
                else:
                    pnl = signal['entry'] - exit_price

                point_value = 5.0 if 'MES' in symbol else 2.0
                pnl_dollars = pnl * point_value

                trades.append({
                    'entry_time': signal['timestamp'],
                    'exit_time': df_with_indicators.index[exit_idx],
                    'direction': signal['direction'],
                    'entry': signal['entry'],
                    'exit': exit_price,
                    'pnl_points': pnl,
                    'pnl_dollars': pnl_dollars,
                    'exit_reason': exit_reason,
                    'outcome': 'WIN' if pnl > 0 else 'LOSS'
                })

        # Calculate statistics
        wins = [t for t in trades if t['outcome'] == 'WIN']
        losses = [t for t in trades if t['outcome'] == 'LOSS']

        print(f"\nCompleted Trades: {len(trades)}")
        print(f"  Wins: {len(wins)}")
        print(f"  Losses: {len(losses)}")

        if trades:
            win_rate = len(wins) / len(trades)
            total_pnl = sum(t['pnl_dollars'] for t in trades)
            avg_win = sum(t['pnl_dollars'] for t in wins) / len(wins) if wins else 0
            avg_loss = sum(t['pnl_dollars'] for t in losses) / len(losses) if losses else 0

            print(f"  Win Rate: {win_rate:.1%}")
            print(f"  Total P&L: ${total_pnl:.2f}")
            print(f"  Avg Win: ${avg_win:.2f}")
            print(f"  Avg Loss: ${avg_loss:.2f}")

            if win_rate >= 0.55:
                print("\n✓ STRATEGY VALIDATED (Win Rate ≥55%)")
            else:
                print("\n✗ STRATEGY NEEDS IMPROVEMENT")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'polygon_backtest_{symbol}_{timestamp}.json'

        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'days': days,
            'bars_analyzed': len(df),
            'signals_detected': len(signals),
            'trades_completed': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate if trades else 0,
            'total_pnl': total_pnl if trades else 0,
            'trades': trades
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n[RESULTS SAVED] {filename}")

    print("="*70)

    return df


def main():
    """Run Polygon data demo"""
    import argparse

    parser = argparse.ArgumentParser(description='Futures Polygon Data Integration')
    parser.add_argument('--symbol', type=str, default='MES', help='Futures symbol (MES, MNQ)')
    parser.add_argument('--timeframe', type=str, default='15Min', help='Timeframe (1Min, 5Min, 15Min, 1H, 1D)')
    parser.add_argument('--days', type=int, default=90, help='Days of historical data')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--check-price', action='store_true', help='Check current price only')

    args = parser.parse_args()

    if args.check_price:
        # Just check current price
        fetcher = PolygonFuturesDataFetcher()
        price = fetcher.get_current_price(args.symbol)

        if price:
            print(f"\n{args.symbol} Current Price: ${price:.2f}\n")
        else:
            print(f"\n[ERROR] Could not fetch current price for {args.symbol}\n")

    elif args.backtest:
        # Run full backtest
        run_polygon_backtest(args.symbol, args.timeframe, args.days)

    else:
        # Just fetch data
        fetcher = PolygonFuturesDataFetcher()
        df = fetcher.get_bars(args.symbol, args.timeframe, args.days)

        if df is not None:
            print("\n[SUCCESS] Data fetched successfully")
            print(f"Shape: {df.shape}")
            print(f"\nFirst few rows:")
            print(df.head())
            print(f"\nLast few rows:")
            print(df.tail())


if __name__ == "__main__":
    main()
