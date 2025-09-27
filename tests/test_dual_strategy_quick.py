#!/usr/bin/env python3
"""
Quick test of dual strategy without options execution
"""

from adaptive_dual_options_engine import AdaptiveDualOptionsEngine

def quick_test():
    engine = AdaptiveDualOptionsEngine()

    # Test strike calculation for your proven symbols
    test_symbols = ['AAPL', 'GOOGL', 'SPY', 'META']

    print("DUAL STRATEGY QUICK TEST")
    print("=" * 60)
    print("Testing strike calculation for your proven dual strategy")
    print("=" * 60)

    for symbol in test_symbols:
        try:
            # Get current price
            bars = engine.api.get_latest_bar(symbol)
            current_price = float(bars.c)

            # Detect market regime
            regime = engine.detect_market_regime(symbol)

            # Calculate strikes
            strike_data = engine.calculate_adaptive_strikes(symbol, current_price, regime)

            put_strike = strike_data['put_strike']
            call_strike = strike_data['call_strike']
            volatility_factor = strike_data['volatility_factor']

            # Calculate position size
            contracts = engine.calculate_position_size(symbol, 500000, 0.25, volatility_factor)

            put_percent = ((put_strike / current_price - 1) * 100)
            call_percent = ((call_strike / current_price - 1) * 100)

            print(f"\n{symbol}:")
            print(f"  Current Price: ${current_price:.2f}")
            print(f"  Market Regime: {regime.upper()}")
            print(f"  PUT Strike: ${put_strike:.0f} ({put_percent:+.1f}%) - SELL {contracts} contracts")
            print(f"  CALL Strike: ${call_strike:.0f} ({call_percent:+.1f}%) - BUY {contracts} contracts")
            print(f"  Volatility Factor: {volatility_factor:.1f}x")

            # Show strategy logic
            if put_percent < -5:
                print(f"  Strategy: Collect premium from OTM puts + capture upside with calls")
            else:
                print(f"  Strategy: Conservative put selling + aggressive call buying")

        except Exception as e:
            print(f"\n{symbol}: Error - {e}")

    print(f"\n" + "=" * 60)
    print("DUAL STRATEGY ANALYSIS COMPLETE")
    print("Ready for Monday market open with your proven 68.3% ROI method")
    print("=" * 60)

if __name__ == "__main__":
    quick_test()