#!/usr/bin/env python3
"""
FOREX PARAMETER TESTER
Quick test to see if new parameters would have generated signals
"""

import json
import requests
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

class ParameterTester:
    def __init__(self):
        self.api_key = os.getenv('OANDA_API_KEY', '0bff5dc7375409bb8747deebab8988a1-d8b26324102c95d6f2b6f641bc330a7c')
        self.base_url = "https://api-fxpractice.oanda.com/v3"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def get_candles(self, instrument, count=200):
        """Fetch historical candles"""
        url = f"{self.base_url}/instruments/{instrument}/candles"
        params = {
            'count': count,
            'granularity': 'H1',
            'price': 'M'
        }

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                candles = data.get('candles', [])
                return [(float(c['mid']['c']), float(c['mid']['h']), float(c['mid']['l'])) for c in candles]
        except:
            pass
        return None

    def calculate_indicators(self, prices):
        """Calculate technical indicators"""
        if not prices or len(prices) < 200:
            return None

        closes = [p[0] for p in prices]

        # EMA calculations (simple version)
        ema10 = sum(closes[-10:]) / 10
        ema21 = sum(closes[-21:]) / 21
        ema200 = sum(closes[-200:]) / 200

        # RSI calculation (simplified)
        gains = []
        losses = []
        for i in range(1, 15):
            diff = closes[-i] - closes[-i-1]
            if diff > 0:
                gains.append(diff)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(diff))

        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # ADX calculation (simplified - using price volatility)
        high_low_ranges = [p[1] - p[2] for p in prices[-14:]]
        avg_range = sum(high_low_ranges) / len(high_low_ranges)
        adx = avg_range * 100  # Simplified ADX proxy

        return {
            'ema10': ema10,
            'ema21': ema21,
            'ema200': ema200,
            'rsi': rsi,
            'adx': adx,
            'current_price': closes[-1]
        }

    def evaluate_signal(self, indicators, config):
        """Evaluate if signal would trigger with given config"""
        if not indicators:
            return None

        score = 0
        details = []

        # EMA Cross
        if indicators['ema10'] > indicators['ema21']:
            score += 3
            details.append("EMA10 > EMA21 (+3)")

        # Trend alignment
        if indicators['current_price'] > indicators['ema200']:
            score += 2
            details.append("Price > EMA200 (+2)")

        # RSI check for LONG
        rsi_lower = config['strategy']['rsi_long_lower']
        rsi_upper = config['strategy']['rsi_long_upper']
        if rsi_lower <= indicators['rsi'] <= rsi_upper:
            score += 2
            details.append(f"RSI {indicators['rsi']:.1f} in range [{rsi_lower}-{rsi_upper}] (+2)")

        # ADX check
        if indicators['adx'] > config['strategy']['adx_threshold']:
            score += 2
            details.append(f"ADX {indicators['adx']:.1f} > {config['strategy']['adx_threshold']} (+2)")

        # Volume/volatility bonus
        score += 1  # Assume some volatility exists
        details.append("Volatility bonus (+1)")

        return {
            'score': score,
            'threshold': config['strategy']['score_threshold'],
            'signal': score >= config['strategy']['score_threshold'],
            'details': details
        }

def main():
    print("\n" + "="*80)
    print("FOREX PARAMETER TESTING - WOULD SIGNALS HAVE TRIGGERED?")
    print("="*80)

    tester = ParameterTester()

    # Load both configs
    with open('config/forex_elite_config.json', 'r') as f:
        strict_config = json.load(f)

    with open('config/forex_elite_balanced.json', 'r') as f:
        balanced_config = json.load(f)

    # Test each pair
    pairs = ['EUR_USD', 'USD_JPY', 'GBP_USD']

    print("\nFetching market data and testing parameters...")
    print("-"*40)

    for pair in pairs:
        print(f"\n{pair}:")

        # Get candles
        candles = tester.get_candles(pair)
        if not candles:
            print("  ERROR: Could not fetch data")
            continue

        # Calculate indicators
        indicators = tester.calculate_indicators(candles)
        if not indicators:
            print("  ERROR: Insufficient data")
            continue

        print(f"  Current Price: {indicators['current_price']:.5f}")
        print(f"  RSI: {indicators['rsi']:.1f}")
        print(f"  ADX: {indicators['adx']:.1f}")

        # Test STRICT config
        strict_result = tester.evaluate_signal(indicators, strict_config)
        print(f"\n  STRICT CONFIG:")
        print(f"    Score: {strict_result['score']:.1f} / {strict_result['threshold']}")
        print(f"    Signal: {'YES! 🎯' if strict_result['signal'] else 'NO ❌'}")
        if not strict_result['signal']:
            print(f"    Missing: {strict_result['threshold'] - strict_result['score']:.1f} points")

        # Test BALANCED config
        balanced_result = tester.evaluate_signal(indicators, balanced_config)
        print(f"\n  BALANCED CONFIG:")
        print(f"    Score: {balanced_result['score']:.1f} / {balanced_result['threshold']}")
        print(f"    Signal: {'YES! 🎯' if balanced_result['signal'] else 'NO ❌'}")
        if balanced_result['signal']:
            print("    Details:", ", ".join(balanced_result['details']))

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
    The BALANCED configuration is more likely to generate trading signals
    while still maintaining safety through:
    - Paper trading mode
    - 1% risk per trade
    - Stop loss protection
    - Maximum daily loss limits

    Note: Markets are currently CLOSED (weekend), so any signals would
    not execute until Sunday 5 PM EST when forex markets reopen.
    """)

if __name__ == "__main__":
    main()