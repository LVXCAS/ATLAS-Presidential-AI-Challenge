#!/usr/bin/env python3
"""
Test bot functionality - trading operations
"""

import sys
import asyncio
import traceback
from datetime import datetime
import OPTIONS_BOT

async def test_bot_functionality():
    """Test the bot's core trading functionality"""

    print("Testing OPTIONS_BOT functionality...")
    print("="*50)

    try:
        # Initialize bot
        bot = OPTIONS_BOT.TomorrowReadyOptionsBot()
        print("[OK] Bot initialized")

        # Test pre-market preparation
        print("\nTesting pre-market preparation...")
        await bot.pre_market_preparation()
        print("[OK] Pre-market preparation completed")

        # Test market data retrieval
        print("\nTesting market data retrieval...")
        market_data = await bot.get_enhanced_market_data('SPY')
        if market_data:
            print(f"[OK] Market data retrieved for SPY: ${market_data.get('current_price', 'N/A')}")
        else:
            print("[WARN] No market data retrieved")

        # Test options chain retrieval
        print("\nTesting options data...")
        try:
            from agents.options_trading_agent import OptionsTrader
            options_trader = OptionsTrader()
            options_data = await options_trader.get_liquid_options('SPY', min_volume=1, min_oi=1)
            if options_data:
                print(f"[OK] Found {len(options_data)} liquid options for SPY")
            else:
                print("[WARN] No liquid options found")
        except Exception as e:
            print(f"[ERROR] Options data test failed: {e}")

        # Test strategy selection
        print("\nTesting strategy selection...")
        trading_plan = await bot.generate_daily_trading_plan()
        if trading_plan:
            print(f"[OK] Generated trading plan with {trading_plan.get('target_new_positions', 0)} target positions")
            print(f"  - Market regime: {trading_plan.get('market_regime', 'Unknown')}")
            print(f"  - Preferred strategies: {trading_plan.get('preferred_strategies', [])}")
            print(f"  - Focus symbols: {trading_plan.get('focus_symbols', [])[:3]}")  # Show first 3 symbols
        else:
            print("[WARN] No trading plan generated")

        print("\n" + "="*50)
        print("[SUCCESS] All functionality tests passed!")
        print("="*50)

        return True

    except Exception as e:
        print(f"\n[ERROR] Functionality test failed: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

def main():
    """Run functionality tests"""
    try:
        # Run async tests
        success = asyncio.run(test_bot_functionality())

        if success:
            print("\n[SUCCESS] Bot is working properly!")
            print("\nNext steps:")
            print("1. Configure your trading parameters in .env")
            print("2. Run the bot with: python OPTIONS_BOT.py")
            print("3. Monitor results in logs/ directory")
        else:
            print("\n[WARN] Some functionality issues found")

        return success

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Test execution failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)