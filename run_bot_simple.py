#!/usr/bin/env python3
"""
Simple bot execution test
"""

import OPTIONS_BOT
import asyncio

async def main():
    print("Initializing OPTIONS_BOT...")
    bot = OPTIONS_BOT.TomorrowReadyOptionsBot()

    print("Running pre-market preparation...")
    await bot.pre_market_preparation()

    print("Bot is ready for trading!")
    print("Trading configuration:")
    print(f"- Paper Trading: {getattr(bot, 'paper_trading', True)}")
    print(f"- Available components:")
    print(f"  - Exit Strategy: {bot.exit_agent is not None}")
    print(f"  - ML Engine: {bot.advanced_ml is not None}")
    print(f"  - Options Pricing: {bot.options_pricing is not None}")
    print(f"  - Monte Carlo: {bot.monte_carlo_engine is not None}")

    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        print("\n[SUCCESS] Bot test completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Bot test failed: {e}")
        import traceback
        traceback.print_exc()