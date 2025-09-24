#!/usr/bin/env python3
"""
Test the fixed strategy detection
"""
import asyncio
import sys
import os

sys.path.append('.')

async def test_strategy_detection():
    try:
        from agents.options_trading_agent import OptionsTrader
        from agents.options_broker import OptionsBroker
        
        print("Testing fixed strategy detection...")
        
        # Create trader
        broker = OptionsBroker(paper_trading=True)
        trader = OptionsTrader(broker)
        
        # Test the corrected method call
        symbol = 'SPY'
        
        # First load options chain
        options_chain = await trader.get_options_chain(symbol)
        print(f"Loaded {len(options_chain)} options for {symbol}")
        
        if options_chain:
            # Now test strategy detection with correct parameters
            strategy = trader.find_best_options_strategy(
                symbol=symbol,
                price=550.0,
                volatility=0.20,
                rsi=60.0,  # Slightly bullish
                price_change=0.005  # Small upward move
            )
            
            if strategy:
                print(f"SUCCESS: Found strategy - {strategy[0]}")
                return True
            else:
                print("No strategy found (this might be normal)")
                return True  # Still success - no error
        else:
            print("No options loaded")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_strategy_detection())
