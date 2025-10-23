#!/usr/bin/env python3
"""
OPTIONS FLOW DETECTOR - Week 5+ Feature
========================================
Detect unusual options activity (smart money flow)

Signals:
- Large premium paid (>$50k per trade) = institutional
- Unusual volume (>3x average) = smart money positioning
- Aggressive fills (buying ask, selling bid) = urgency
- Large OTM purchases = directional bets

Strategy: Follow the smart money
"""

import alpaca_trade_api as tradeapi
import yfinance as yf
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class OptionsFlowDetector:
    """Detect unusual options activity for trading signals"""

    def __init__(self):
        self.api = tradeapi.REST(
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

    def detect_unusual_volume(self, symbol, volume_multiplier=3.0):
        """
        Detect options with unusually high volume

        Args:
            symbol: Stock symbol
            volume_multiplier: Multiple of average volume to flag (3.0 = 3x average)

        Returns:
            List of unusual options contracts
        """

        print(f"\n{'='*80}")
        print(f"UNUSUAL OPTIONS FLOW: {symbol}")
        print(f"Volume Threshold: {volume_multiplier}x average")
        print(f"{'='*80}")

        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options

            if not expirations:
                print(f"  [NO DATA] No options available for {symbol}")
                return []

            unusual_activity = []

            # Scan first 3 expirations
            for exp in expirations[:3]:
                opt_chain = ticker.option_chain(exp)

                # Check calls
                for _, call in opt_chain.calls.iterrows():
                    volume = call.get('volume', 0)
                    open_interest = call.get('openInterest', 1)

                    if volume > 0 and open_interest > 0:
                        vol_oi_ratio = volume / open_interest

                        # Unusual if volume > 3x open interest
                        if vol_oi_ratio > volume_multiplier:
                            premium_value = volume * call['lastPrice'] * 100

                            if premium_value > 10000:  # >$10k premium
                                unusual_activity.append({
                                    'symbol': symbol,
                                    'type': 'CALL',
                                    'strike': call['strike'],
                                    'expiration': exp,
                                    'volume': volume,
                                    'open_interest': open_interest,
                                    'ratio': vol_oi_ratio,
                                    'premium_value': premium_value,
                                    'signal': 'BULLISH'
                                })

                # Check puts
                for _, put in opt_chain.puts.iterrows():
                    volume = put.get('volume', 0)
                    open_interest = put.get('openInterest', 1)

                    if volume > 0 and open_interest > 0:
                        vol_oi_ratio = volume / open_interest

                        if vol_oi_ratio > volume_multiplier:
                            premium_value = volume * put['lastPrice'] * 100

                            if premium_value > 10000:  # >$10k premium
                                unusual_activity.append({
                                    'symbol': symbol,
                                    'type': 'PUT',
                                    'strike': put['strike'],
                                    'expiration': exp,
                                    'volume': volume,
                                    'open_interest': open_interest,
                                    'ratio': vol_oi_ratio,
                                    'premium_value': premium_value,
                                    'signal': 'BEARISH'
                                })

            # Sort by premium value (largest first)
            unusual_activity.sort(key=lambda x: x['premium_value'], reverse=True)

            print(f"\n  Found {len(unusual_activity)} unusual flow events")

            for i, flow in enumerate(unusual_activity[:5], 1):  # Top 5
                print(f"\n  {i}. {flow['type']} ${flow['strike']:.0f} exp {flow['expiration']}")
                print(f"     Volume: {flow['volume']:,} | OI: {flow['open_interest']:,} | Ratio: {flow['ratio']:.1f}x")
                print(f"     Premium: ${flow['premium_value']:,.0f}")
                print(f"     Signal: {flow['signal']}")

            return unusual_activity

        except Exception as e:
            print(f"  [ERROR] Failed to detect flow: {e}")
            return []

    def detect_large_trades(self, symbol, min_premium=50000):
        """
        Detect large single trades (institutional activity)

        Args:
            symbol: Stock symbol
            min_premium: Minimum premium value to flag ($50k default)
        """

        print(f"\n{'='*80}")
        print(f"LARGE TRADE DETECTION: {symbol}")
        print(f"Minimum Premium: ${min_premium:,}")
        print(f"{'='*80}")

        # Note: Alpaca doesn't provide real-time options trade data
        # This would require a premium data provider (Bloomberg, IVolatility, etc.)

        print(f"  [INFO] Large trade detection requires premium data feed")
        print(f"  [INFO] Would integrate with: Trade Alert, Unusual Whales, or Flow Algo")
        print(f"  [INFO] Fallback: Using volume/OI analysis (see detect_unusual_volume)")

        return []

    def scan_multiple_symbols(self, symbols, volume_multiplier=3.0):
        """Scan multiple symbols for unusual flow"""

        print(f"\n{'='*80}")
        print(f"SCANNING {len(symbols)} SYMBOLS FOR UNUSUAL FLOW")
        print(f"{'='*80}")

        all_flows = []

        for symbol in symbols:
            try:
                flows = self.detect_unusual_volume(symbol, volume_multiplier)
                all_flows.extend(flows)
            except:
                continue

        # Sort by premium value
        all_flows.sort(key=lambda x: x['premium_value'], reverse=True)

        print(f"\n{'='*80}")
        print(f"TOP 10 UNUSUAL FLOWS ACROSS ALL SYMBOLS")
        print(f"{'='*80}")

        for i, flow in enumerate(all_flows[:10], 1):
            print(f"\n{i}. {flow['symbol']} {flow['type']} ${flow['strike']:.0f}")
            print(f"   Premium: ${flow['premium_value']:,.0f} | Signal: {flow['signal']}")

        return all_flows


def test_options_flow_detector():
    """Test options flow detector"""

    detector = OptionsFlowDetector()

    print("="*80)
    print("TEST: OPTIONS FLOW DETECTION")
    print("="*80)

    # Test on single symbol
    test_symbol = 'NVDA'
    detector.detect_unusual_volume(test_symbol, volume_multiplier=3.0)

    # Test on multiple symbols
    print("\n\n")
    test_symbols = ['AAPL', 'NVDA', 'TSLA', 'AMD', 'MSFT']
    detector.scan_multiple_symbols(test_symbols, volume_multiplier=2.5)


if __name__ == "__main__":
    test_options_flow_detector()
