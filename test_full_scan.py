#!/usr/bin/env python3
"""
Test the full 31-symbol scanning
"""
import asyncio
import sys
sys.path.append('.')

async def test_full_scan():
    try:
        from OPTIONS_BOT import TomorrowReadyOptionsBot
        
        print("Testing full symbol scanning...")
        
        # Create and initialize bot
        bot = TomorrowReadyOptionsBot()
        await bot.initialize_all_systems()
        
        print(f"Bot has {len(bot.tier1_stocks)} symbols in universe:")
        print(f"Symbols: {bot.tier1_stocks}")
        
        # Test the scan method
        print(f"\nTesting scan_for_new_opportunities...")
        await bot.scan_for_new_opportunities()
        
        print("Full scan test completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_full_scan())