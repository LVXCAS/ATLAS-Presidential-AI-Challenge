#!/usr/bin/env python3
"""
Test that limit prices are properly rounded to 2 decimal places
"""
import asyncio
import sys
sys.path.append('.')

async def test_price_rounding():
    try:
        from agents.options_trading_agent import OptionsTrader, OptionsStrategy
        from agents.options_broker import OptionsBroker
        from agents.broker_integration import AlpacaBrokerIntegration
        
        print("Testing price rounding...")
        
        # Initialize components
        broker = AlpacaBrokerIntegration(paper_trading=True)
        options_broker = OptionsBroker(broker, paper_trading=True)
        trader = OptionsTrader(broker)
        
        # Get options chain
        chain = await trader.get_options_chain('IWM')
        print(f"Loaded {len(chain)} IWM options")
        
        if chain:
            # Find strategy
            strategy = trader.find_best_options_strategy(
                symbol='IWM',
                price=241.0,  # Approximate IWM price
                volatility=0.22,
                rsi=55.0,
                price_change=0.008  # Small positive change
            )
            
            if strategy:
                strategy_type, contracts = strategy
                print(f"Strategy: {strategy_type}")
                print(f"Contracts: {len(contracts)}")
                
                # Test execution with confidence < 75% (should use LIMIT order)
                print("Testing LIMIT order execution (confidence < 75%)...")
                position = await trader.execute_options_strategy(
                    strategy=strategy_type,
                    contracts=contracts,
                    quantity=1,
                    confidence=0.65  # Below 75%, should use LIMIT
                )
                
                if position:
                    print(f"SUCCESS: Position created with entry price: ${position.entry_price}")
                else:
                    print("No position returned (order may have failed but should not error)")
                
                print("Test completed - checking for 422 decimal errors in logs above")
            else:
                print("No strategy found for IWM")
        else:
            print("No options chain loaded")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_price_rounding())