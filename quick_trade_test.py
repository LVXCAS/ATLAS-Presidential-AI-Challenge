#!/usr/bin/env python3
"""
Quick test to verify the bot can find and attempt to execute trades
"""

import asyncio
import sys
sys.path.append('.')

from OPTIONS_BOT import TomorrowReadyOptionsBot

async def quick_trade_test():
    """Test if bot can find and execute trades"""
    print("QUICK TRADE TEST")
    print("=" * 30)
    
    bot = TomorrowReadyOptionsBot()
    
    # Initialize broker
    print("Initializing broker...")
    try:
        success = await bot.initialize_all_systems()
        if success:
            print(f"[OK] Broker connected - Account: ${bot.risk_manager.account_value:,.2f}")
        else:
            print("[FAIL] Broker connection failed")
            return
    except Exception as e:
        print(f"[ERROR] Broker init error: {e}")
        return
    
    # Test one complete trading cycle
    print("\nTesting complete trading cycle...")
    
    # Set risk limits
    await bot.set_daily_risk_limits()
    print(f"Risk limits: Max positions {bot.daily_risk_limits['max_positions']}, Max risk ${bot.daily_risk_limits['max_position_risk']}")
    
    # Generate trading plan
    trading_plan = await bot.generate_daily_trading_plan()
    print(f"Trading plan: {trading_plan['target_new_positions']} positions, focus on {trading_plan['focus_symbols'][:3]}")
    
    # Scan for opportunities
    print("\nScanning for opportunities...")
    if len(bot.active_positions) >= bot.daily_risk_limits.get('max_positions', 5):
        print("[SKIP] Position limit reached")
        return
    
    opportunities = []
    test_symbols = ['AAPL', 'SPY', 'MSFT', 'QQQ']  # Test specific symbols
    
    for symbol in test_symbols:
        print(f"  Checking {symbol}...")
        try:
            opportunity = await bot.find_high_quality_opportunity(symbol)
            if opportunity:
                opportunities.append(opportunity)
                print(f"    [OPPORTUNITY] {opportunity['strategy']} - {opportunity['confidence']:.0%} confidence")
                print(f"    Reasoning: {opportunity.get('reasoning', 'N/A')}")
            else:
                print(f"    [NO OPPORTUNITY]")
        except Exception as e:
            print(f"    [ERROR] {e}")
    
    print(f"\nTotal opportunities found: {len(opportunities)}")
    
    if opportunities:
        # Select best opportunity
        best = max(opportunities, key=lambda x: x.get('confidence', 0) * x.get('expected_return', 0))
        print(f"\nBest opportunity: {best['symbol']} {best['strategy']} ({best['confidence']:.0%})")
        
        # Test execution (dry run)
        print(f"\nTesting trade execution for {best['symbol']}...")
        try:
            # Check risk limits before execution
            position_risk = best['max_loss'] * 100
            if position_risk <= bot.daily_risk_limits.get('max_position_risk', 500):
                print(f"  [OK] Risk check passed: ${position_risk} <= ${bot.daily_risk_limits['max_position_risk']}")
                print(f"  [SIMULATE] Would execute {best['symbol']} {best['strategy']}")
                print(f"  [SIMULATE] Max profit: ${best['max_profit']}, Max loss: ${best['max_loss']}")
                
                # In a real execution, this would call execute_new_position(best)
                print("  [SUCCESS] Trade would be executed!")
                
            else:
                print(f"  [FAIL] Risk too high: ${position_risk} > ${bot.daily_risk_limits['max_position_risk']}")
        
        except Exception as e:
            print(f"  [ERROR] Execution test failed: {e}")
    else:
        print("\n[NO TRADES] No opportunities found - this is normal with conservative criteria")
    
    print("\nTest completed!")

if __name__ == "__main__":
    asyncio.run(quick_trade_test())