#!/usr/bin/env python3
"""
Simple Options Trading Debug Test
No emojis, just pure functionality testing
"""

import asyncio
import sys
from datetime import datetime

# Add current directory to path
sys.path.append('.')

async def test_options_debug():
    """Debug options trading step by step"""
    print("SIMPLE OPTIONS TRADING DEBUG")
    print("=" * 50)
    
    try:
        # Step 1: Test imports
        print("Step 1: Testing imports...")
        from agents.options_broker import OptionsBroker, OptionsOrderRequest, OptionsOrderType
        from agents.broker_integration import OrderSide
        from agents.options_trading_agent import OptionsTrader
        print("  [OK] All imports successful")
        
        # Step 2: Initialize broker
        print("\nStep 2: Initializing options broker...")
        broker = OptionsBroker(None, paper_trading=True)
        print("  [OK] Options broker initialized")
        
        # Step 3: Initialize trader
        print("\nStep 3: Initializing options trader...")
        trader = OptionsTrader(None)
        print("  [OK] Options trader initialized")
        
        # Step 4: Test options chain retrieval
        print("\nStep 4: Testing options chain retrieval for AAPL...")
        contracts = await trader.get_options_chain("AAPL")
        print(f"  [INFO] Found {len(contracts)} options contracts")
        
        if contracts:
            print("  [OK] Options chain retrieved successfully")
            # Show first few contracts
            for i, contract in enumerate(contracts[:3]):
                print(f"    Contract {i+1}: {contract.symbol} - Strike: ${contract.strike} - Exp: {contract.expiration.strftime('%Y-%m-%d')}")
        else:
            print("  [FAIL] No options contracts found")
            print("  [DEBUG] Trying to understand why...")
            
            # Debug: Check if AAPL has any options at all
            import yfinance as yf
            ticker = yf.Ticker("AAPL")
            try:
                expirations = ticker.options
                print(f"  [DEBUG] AAPL has {len(expirations)} expiration dates: {expirations[:3] if expirations else 'None'}")
                
                if expirations:
                    # Try to get options for first expiration
                    exp_date = expirations[0]
                    option_chain = ticker.option_chain(exp_date)
                    print(f"  [DEBUG] Calls for {exp_date}: {len(option_chain.calls)} contracts")
                    print(f"  [DEBUG] Puts for {exp_date}: {len(option_chain.puts)} contracts")
                    
                    if len(option_chain.calls) > 0:
                        print("  [DEBUG] Sample call option:")
                        print(f"    Symbol: {option_chain.calls.iloc[0]['contractSymbol']}")
                        print(f"    Strike: ${option_chain.calls.iloc[0]['strike']}")
                        print(f"    Last Price: ${option_chain.calls.iloc[0]['lastPrice']}")
                        
            except Exception as e:
                print(f"  [DEBUG] Error accessing Yahoo Finance options: {e}")
        
        # Step 5: Test strategy selection if we have contracts
        if contracts:
            print("\nStep 5: Testing strategy selection...")
            strategy_result = trader.find_best_options_strategy(
                "AAPL", 150.0, 25.0, 50.0, 0.02  # symbol, price, volatility, rsi, price_change
            )
            
            if strategy_result:
                strategy, selected_contracts = strategy_result
                print(f"  [OK] Strategy selected: {strategy}")
                print(f"  [INFO] Selected {len(selected_contracts)} contracts")
            else:
                print("  [FAIL] No strategy selected")
        else:
            print("\nStep 5: Skipping strategy test - no contracts available")
        
        # Step 6: Test paper trading order
        print("\nStep 6: Testing paper trading order...")
        buy_order = OptionsOrderRequest(
            symbol="AAPL250321C00150000",  # AAPL March 21, 2025 $150 Call
            underlying="AAPL",
            qty=1,
            side=OrderSide.BUY,
            type=OptionsOrderType.MARKET,
            option_type='call',
            strike=150.0,
            expiration=datetime(2025, 3, 21)
        )
        
        buy_response = await broker.submit_options_order(buy_order)
        
        if buy_response and buy_response.status == "filled":
            print(f"  [SUCCESS] Paper trade executed!")
            print(f"    Order ID: {buy_response.id}")
            print(f"    Symbol: {buy_response.symbol}")
            print(f"    Quantity: {buy_response.qty}")
            print(f"    Price: ${buy_response.avg_fill_price}")
        else:
            print("  [FAIL] Paper trade failed")
            if buy_response:
                print(f"    Status: {buy_response.status}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed with exception: {e}")
        import traceback
        print(f"[DEBUG] Full traceback:\n{traceback.format_exc()}")
        return False

async def main():
    """Main test function"""
    success = await test_options_debug()
    
    if success:
        print("\n" + "=" * 50)
        print("OPTIONS DEBUG TEST COMPLETED")
        print("Check results above for any issues")
    else:
        print("\n" + "=" * 50)
        print("OPTIONS DEBUG TEST FAILED")
        print("See error details above")

if __name__ == "__main__":
    asyncio.run(main())