#!/usr/bin/env python3
"""
Test why AAPL 75% opportunity isn't executing
"""
import asyncio
import sys
sys.path.append('.')

async def test_aapl_execution():
    try:
        from OPTIONS_BOT import TomorrowReadyOptionsBot
        
        print("TESTING AAPL EXECUTION")
        print("=" * 25)
        
        bot = TomorrowReadyOptionsBot()
        await bot.initialize_all_systems()
        
        print(f"Current positions: {len(bot.active_positions)}")
        print(f"Max positions: {bot.daily_risk_limits.get('max_positions', 5)}")
        
        # Get the AAPL opportunity
        print("\nFinding AAPL opportunity...")
        aapl_opp = await bot.find_high_quality_opportunity('AAPL')
        
        if aapl_opp:
            conf = aapl_opp.get('confidence', 0)
            print(f"Found AAPL: {aapl_opp['strategy']} at {conf:.1%} confidence")
            
            if conf >= 0.75:
                print("SUCCESS: Confidence >= 75% - should execute")
                
                print("\nTesting execution...")
                success = await bot.execute_new_position(aapl_opp)
                
                if success:
                    print("SUCCESS: AAPL trade executed!")
                    print(f"Active positions now: {len(bot.active_positions)}")
                    
                    # Show the position details
                    for pos_id, pos_data in bot.active_positions.items():
                        print(f"Position {pos_id}:")
                        print(f"  Symbol: {pos_data.get('symbol')}")
                        print(f"  Strategy: {pos_data.get('strategy')}")
                        print(f"  Real trade: {pos_data.get('real_trade', False)}")
                else:
                    print("EXECUTION FAILED")
            else:
                print(f"FAIL: Confidence too low: {conf:.1%} < 75%")
        else:
            print("FAIL: No AAPL opportunity found")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_aapl_execution())