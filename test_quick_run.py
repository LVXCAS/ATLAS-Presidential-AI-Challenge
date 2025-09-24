#!/usr/bin/env python3
"""
Test a quick run of the bot to see if the options_trader issue occurs
"""
import asyncio
import sys
sys.path.append('.')

async def test_quick_run():
    try:
        print("Starting quick bot test...")
        
        from OPTIONS_BOT import TomorrowReadyOptionsBot
        
        # Create and initialize bot
        bot = TomorrowReadyOptionsBot()
        print(f"Bot created - options_trader before init: {bot.options_trader}")
        
        # Initialize systems
        init_result = await bot.initialize_all_systems()
        print(f"Initialization result: {init_result}")
        print(f"After init - options_trader: {bot.options_trader}")
        
        if bot.options_trader is None:
            print("ERROR: options_trader is None after initialization!")
            return
            
        # Test scanning for opportunities (this is what the real bot does)
        print("Testing opportunity scanning...")
        test_symbols = ['AAPL', 'GOOGL']
        
        opportunities = []
        for symbol in test_symbols:
            try:
                print(f"Checking {symbol}...")
                opp = await bot.find_high_quality_opportunity(symbol)
                if opp:
                    opportunities.append(opp)
                    print(f"Found opportunity: {symbol} - {opp.get('confidence', 0):.1%}")
            except Exception as e:
                print(f"Error with {symbol}: {e}")
        
        print(f"Found {len(opportunities)} total opportunities")
        
        # Test execution path (what was failing)
        if opportunities:
            opp = opportunities[0]
            print(f"Testing execution for {opp['symbol']}...")
            
            # Check trader state before execution
            print(f"Before execution - options_trader: {bot.options_trader}")
            
            # This is where the error was occurring
            try:
                success = await bot.execute_new_position(opp)
                print(f"Execution result: {success}")
            except Exception as e:
                print(f"Execution error: {e}")
                import traceback
                traceback.print_exc()
        
        print("Quick test completed successfully!")
        
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_quick_run())