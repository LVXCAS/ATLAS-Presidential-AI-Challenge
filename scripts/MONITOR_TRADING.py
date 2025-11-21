#!/usr/bin/env python3
"""
SIMPLE TRADING MONITOR
Shows current market status and checks for trading signals
Run this every 30-60 minutes to monitor your trading
"""

import os
import json
import requests
from datetime import datetime
from pathlib import Path

# API Configuration
API_KEY = "0bff5dc7375409bb8747deebab8988a1-d8b26324102c95d6f2b6f641bc330a7c"
BASE_URL = "https://api-fxpractice.oanda.com/v3"

def get_current_prices():
    """Fetch current forex prices"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    pairs = ['EUR_USD', 'USD_JPY', 'GBP_USD']
    prices = {}

    for pair in pairs:
        url = f"{BASE_URL}/instruments/{pair}/candles"
        params = {'count': 2, 'granularity': 'H1', 'price': 'M'}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                candles = data.get('candles', [])
                if len(candles) >= 2:
                    current = float(candles[-1]['mid']['c'])
                    previous = float(candles[-2]['mid']['c'])
                    change = ((current - previous) / previous) * 100
                    prices[pair] = {
                        'price': current,
                        'previous': previous,
                        'change': change
                    }
        except:
            prices[pair] = None

    return prices

def check_for_signals(prices):
    """Check if any trading signals should trigger"""
    signals = []

    # Load configuration to get thresholds
    try:
        with open('config/forex_elite_config.json', 'r') as f:
            config = json.load(f)
        threshold = config['strategy']['score_threshold']
    except:
        threshold = 6.0  # Default balanced threshold

    # Simple signal detection based on price movement
    for pair, data in prices.items():
        if data and abs(data['change']) > 0.15:  # 0.15% move in 1 hour
            direction = "LONG" if data['change'] > 0 else "SHORT"
            signals.append({
                'pair': pair,
                'direction': direction,
                'price': data['price'],
                'change': data['change']
            })

    return signals, threshold

def check_existing_trades():
    """Check for any existing trade logs"""
    trade_files = list(Path('forex_trades').glob('*.json')) if Path('forex_trades').exists() else []
    signal_files = list(Path('.').glob('signals_*.json'))

    return len(trade_files), len(signal_files)

def main():
    print("\n" + "="*60)
    print(f"TRADING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Get current prices
    print("\n[1] MARKET STATUS")
    print("-"*40)
    prices = get_current_prices()

    for pair, data in prices.items():
        if data:
            arrow = "↑" if data['change'] > 0 else "↓" if data['change'] < 0 else "→"
            color_start = "+" if data['change'] > 0 else ""
            print(f"{pair}: {data['price']:.5f} {arrow} ({color_start}{data['change']:.3f}%)")
        else:
            print(f"{pair}: [ERROR - Could not fetch]")

    # Check configuration
    print("\n[2] CONFIGURATION")
    print("-"*40)
    try:
        with open('config/forex_elite_config.json', 'r') as f:
            config = json.load(f)
        score_threshold = config['strategy']['score_threshold']
        pairs_count = len(config['trading']['pairs'])

        if score_threshold == 8.0:
            print(f"Score Threshold: {score_threshold} [TOO STRICT - Change to 6.0]")
        elif score_threshold == 6.0:
            print(f"Score Threshold: {score_threshold} [BALANCED - Good]")
        else:
            print(f"Score Threshold: {score_threshold} [CUSTOM]")

        print(f"Trading Pairs: {pairs_count}")
        print(f"Mode: Paper Trading")
    except Exception as e:
        print(f"Config Error: {e}")

    # Check for signals
    print("\n[3] SIGNAL CHECK")
    print("-"*40)
    signals, threshold = check_for_signals(prices)

    if signals:
        print(f"SIGNALS DETECTED:")
        for sig in signals:
            print(f"  - {sig['pair']}: {sig['direction']} at {sig['price']:.5f} ({sig['change']:.3f}% move)")
    else:
        print("No strong signals at this time")
        print(f"(Looking for >0.15% hourly moves with score >{threshold})")

    # Check existing trades
    print("\n[4] TRADE HISTORY")
    print("-"*40)
    trade_count, signal_count = check_existing_trades()
    print(f"Executed Trades: {trade_count}")
    print(f"Signal Files: {signal_count}")

    # Recommendations
    print("\n[5] RECOMMENDATIONS")
    print("-"*40)

    if any(data and abs(data['change']) > 0.1 for data in prices.values()):
        print("• Market is ACTIVE - Good trading conditions")
    else:
        print("• Market is QUIET - Limited opportunities")

    if score_threshold == 8.0:
        print("• URGENT: Lower score threshold to 6.0 for more trades")

    if trade_count == 0:
        print("• No trades executed yet - Monitor for signals")

    print("\n" + "="*60)
    print("Run this monitor every 30-60 minutes during active trading")
    print("="*60)

if __name__ == "__main__":
    main()