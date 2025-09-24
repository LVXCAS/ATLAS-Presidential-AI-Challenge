#!/usr/bin/env python3
"""
Debug the options trader initialization issue
"""
import asyncio
import sys
sys.path.append('.')

async def debug_trader_init():
    try:
        print("Testing OPTIONS_BOT initialization...")
        
        from OPTIONS_BOT import TomorrowReadyOptionsBot
        
        # Create bot
        bot = TomorrowReadyOptionsBot()
        print(f"Bot created - options_trader: {bot.options_trader}")
        
        # Initialize systems  
        print("Calling initialize_all_systems...")
        result = await bot.initialize_all_systems()
        
        print(f"Initialization result: {result}")
        print(f"After init - options_trader: {bot.options_trader}")
        print(f"After init - broker: {bot.broker}")
        print(f"After init - options_broker: {bot.options_broker}")
        
        # Test the specific method that's failing
        if bot.options_trader:
            print("Testing get_options_chain method...")
            try:
                chain = await bot.options_trader.get_options_chain('AAPL')
                print(f"Options chain loaded: {len(chain)} contracts")
            except Exception as e:
                print(f"get_options_chain failed: {e}")
        else:
            print("options_trader is None - cannot test get_options_chain")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_trader_init())