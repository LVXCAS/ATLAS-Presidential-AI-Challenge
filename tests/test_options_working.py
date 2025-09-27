#!/usr/bin/env python3
"""
Working Options Trading Test - Uses Real Current Prices
"""

import asyncio
import sys
import yfinance as yf

sys.path.append('.')

async def test_options_with_real_prices():
    """Test options trading with real current stock prices"""
    print("REAL PRICE OPTIONS TRADING TEST")
    print("=" * 45)
    
    from agents.options_trading_agent import OptionsTrader
    from agents.options_broker import OptionsBroker, OptionsOrderRequest, OptionsOrderType
    from agents.broker_integration import OrderSide
    from datetime import datetime
    
    try:
        # Step 1: Get current stock price
        print("Step 1: Getting current AAPL price...")
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="1d")
        current_price = float(hist['Close'].iloc[-1])
        print(f"  Current AAPL price: ${current_price:.2f}")
        
        # Step 2: Initialize systems
        print("\nStep 2: Initializing trading systems...")
        trader = OptionsTrader(None)
        broker = OptionsBroker(None, paper_trading=True)
        print("  [OK] Systems initialized")
        
        # Step 3: Get options chain
        print("\nStep 3: Getting options chain...")
        contracts = await trader.get_options_chain("AAPL")
        print(f"  Found {len(contracts)} liquid contracts")
        
        if contracts:
            calls = [c for c in contracts if c.option_type == 'call']
            puts = [c for c in contracts if c.option_type == 'put']
            print(f"  Calls: {len(calls)}, Puts: {len(puts)}")
            
            # Show available strikes
            call_strikes = sorted([c.strike for c in calls])
            put_strikes = sorted([c.strike for c in puts])
            print(f"  Call strikes: ${call_strikes[0]:.0f} to ${call_strikes[-1]:.0f}")
            print(f"  Put strikes: ${put_strikes[0]:.0f} to ${put_strikes[-1]:.0f}")
        
        # Step 4: Test strategy selection with realistic scenarios
        print("\nStep 4: Testing strategy selection...")
        
        test_scenarios = [
            (current_price, 25.0, 55.0, 0.015, "Bullish scenario"),
            (current_price, 30.0, 45.0, -0.015, "Bearish scenario"),
            (current_price, 35.0, 50.0, 0.005, "High volatility scenario"),
        ]
        
        successful_strategies = []
        
        for price, vol, rsi, change, desc in test_scenarios:
            print(f"\n  Testing: {desc}")
            print(f"    Price: ${price:.2f}, Vol: {vol}%, RSI: {rsi}, Change: {change*100:+.1f}%")
            
            strategy_result = trader.find_best_options_strategy("AAPL", price, vol, rsi, change)
            
            if strategy_result:
                strategy, selected_contracts = strategy_result
                print(f"    [SUCCESS] Strategy: {strategy}")
                print(f"    [INFO] Selected {len(selected_contracts)} contracts:")
                for i, contract in enumerate(selected_contracts):
                    print(f"      {i+1}. {contract.symbol} - ${contract.strike} {contract.option_type} (${contract.mid_price:.2f})")
                successful_strategies.append((strategy, selected_contracts, desc))
            else:
                print(f"    [FAIL] No strategy found")
        
        # Step 5: Execute a real options strategy if we found any
        if successful_strategies:
            print(f"\nStep 5: Executing options strategy...")
            strategy, selected_contracts, desc = successful_strategies[0]
            print(f"  Executing: {strategy} ({desc})")
            
            # Execute the strategy
            position = await trader.execute_options_strategy(strategy, selected_contracts, quantity=1)
            
            if position:
                print(f"  [SUCCESS] Options position created!")
                print(f"    Position ID: {position.symbol}")
                print(f"    Strategy: {position.strategy}")
                print(f"    Entry Price: ${position.entry_price:.2f}")
                print(f"    Max Loss: ${position.max_loss:.2f}" if position.max_loss else "    Max Loss: Unlimited")
            else:
                print(f"  [FAIL] Strategy execution failed")
        else:
            print(f"\nStep 5: No strategies to execute")
        
        # Step 6: Test direct paper trading
        print(f"\nStep 6: Testing direct paper trading...")
        
        # Create a simple call option order
        if contracts:
            # Find a reasonable call option
            atm_calls = [c for c in contracts if c.option_type == 'call' and abs(c.strike - current_price) <= 20]
            if atm_calls:
                contract = atm_calls[0]
                
                buy_order = OptionsOrderRequest(
                    symbol=contract.symbol,
                    underlying="AAPL",
                    qty=1,
                    side=OrderSide.BUY,
                    type=OptionsOrderType.MARKET,
                    option_type='call',
                    strike=contract.strike,
                    expiration=contract.expiration
                )
                
                response = await broker.submit_options_order(buy_order)
                
                if response and response.status == "filled":
                    print(f"  [SUCCESS] Direct option trade executed!")
                    print(f"    Symbol: {response.symbol}")
                    print(f"    Price: ${response.avg_fill_price:.2f}")
                    print(f"    Total Cost: ${response.avg_fill_price * 100:.2f}")
                else:
                    print(f"  [FAIL] Direct trade failed")
            else:
                print(f"  [INFO] No suitable ATM calls found for direct trade")
        
        print(f"\n" + "=" * 45)
        print(f"REAL PRICE OPTIONS TEST COMPLETED")
        print(f"Strategy Success Rate: {len(successful_strategies)}/{len(test_scenarios)}")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_options_with_real_prices())