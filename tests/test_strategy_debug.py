#!/usr/bin/env python3
"""
Debug strategy selection specifically
"""

import asyncio
import sys

sys.path.append('.')

async def test_strategy_debug():
    """Debug strategy selection with various parameters"""
    print("STRATEGY SELECTION DEBUG")
    print("=" * 40)
    
    from agents.options_trading_agent import OptionsTrader
    
    # Initialize trader
    trader = OptionsTrader(None)
    
    # Get AAPL contracts
    print("Getting AAPL options chain...")
    contracts = await trader.get_options_chain("AAPL")
    print(f"Found {len(contracts)} contracts")
    
    if not contracts:
        print("No contracts found - exiting")
        return
    
    # Test with various market conditions
    test_cases = [
        # price, volatility, rsi, price_change, description
        (150.0, 25.0, 50.0, 0.02, "Strong bullish (2% up)"),
        (150.0, 25.0, 70.0, 0.01, "Moderate bullish (1% up)"),
        (150.0, 25.0, 30.0, -0.02, "Strong bearish (2% down)"),
        (150.0, 35.0, 50.0, 0.0, "High volatility, neutral"),
        (150.0, 15.0, 50.0, 0.005, "Low volatility, slight up"),
        (220.0, 30.0, 60.0, 0.015, "Current price, moderate bullish"),
    ]
    
    for i, (price, vol, rsi, change, desc) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {desc}")
        print(f"  Parameters: Price=${price}, Vol={vol}%, RSI={rsi}, Change={change*100:.1f}%")
        
        strategy_result = trader.find_best_options_strategy("AAPL", price, vol, rsi, change)
        
        if strategy_result:
            strategy, selected_contracts = strategy_result
            print(f"  [SUCCESS] Strategy: {strategy}")
            print(f"  [INFO] Contracts selected: {len(selected_contracts)}")
            for j, contract in enumerate(selected_contracts):
                print(f"    Contract {j+1}: {contract.symbol} (Strike: ${contract.strike})")
        else:
            print(f"  [FAIL] No strategy selected")
            
            # Debug why it failed - check calls and puts availability
            calls = [c for c in contracts if c.option_type == 'call']
            puts = [c for c in contracts if c.option_type == 'put']
            print(f"  [DEBUG] Available: {len(calls)} calls, {len(puts)} puts")
            
            if calls:
                strikes = sorted([c.strike for c in calls])
                print(f"  [DEBUG] Call strikes: ${strikes[0]:.0f} to ${strikes[-1]:.0f}")
                
                # Check specific conditions for bull call spread
                if change > 0.005 and rsi < 75:
                    long_target = price * 1.02
                    short_target = price * 1.08
                    print(f"  [DEBUG] Bull spread targets: Long=${long_target:.0f}, Short=${short_target:.0f}")
                    
                    long_calls = [c for c in calls if price * 0.98 <= c.strike <= price * 1.05]
                    short_calls = [c for c in calls if price * 1.05 < c.strike <= price * 1.12]
                    print(f"  [DEBUG] Eligible: {len(long_calls)} long calls, {len(short_calls)} short calls")

if __name__ == "__main__":
    asyncio.run(test_strategy_debug())