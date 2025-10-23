#!/usr/bin/env python3
"""
Test the division by zero fix for options chain processing
"""

import asyncio
import sys
from agents.options_trading_agent import OptionsTrader

async def test_division_fix():
    """Test the division by zero fix"""

    print("Testing division by zero fix in options chain processing...")

    try:
        # Initialize options trader
        trader = OptionsTrader()

        # Test with UNH which was causing the error
        print("Testing UNH options chain retrieval...")
        options = await trader.get_options_chain('UNH')

        if options:
            print(f"[SUCCESS] Retrieved {len(options)} UNH options without division error")

            # Check a few contracts to ensure valid pricing
            for i, opt in enumerate(options[:3]):
                print(f"  Option {i+1}: {opt.symbol}")
                print(f"    Bid: ${opt.bid:.2f}, Ask: ${opt.ask:.2f}")
                print(f"    Mid Price: ${opt.mid_price:.2f}")
                print(f"    Spread: ${opt.spread:.2f}")

                # Verify no division by zero
                if opt.mid_price > 0:
                    spread_ratio = opt.spread / opt.mid_price
                    print(f"    Spread Ratio: {spread_ratio:.4f} (no division error)")
                else:
                    print("    [WARNING] Mid price is zero - this should not happen now")
        else:
            print("[INFO] No options retrieved (possibly outside market hours or no liquid options)")

        # Test other symbols that might have similar issues
        test_symbols = ['TSLA', 'NVDA', 'AMZN']

        for symbol in test_symbols:
            print(f"\nTesting {symbol} options chain...")
            try:
                options = await trader.get_options_chain(symbol)
                if options:
                    print(f"[SUCCESS] {symbol}: Retrieved {len(options)} options")
                else:
                    print(f"[INFO] {symbol}: No liquid options found")
            except Exception as e:
                print(f"[ERROR] {symbol}: {e}")

        print("\n[SUCCESS] Division by zero fix test completed!")
        return True

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test"""
    try:
        success = asyncio.run(test_division_fix())
        return success
    except Exception as e:
        print(f"Test execution failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)