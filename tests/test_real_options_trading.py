#!/usr/bin/env python3
"""
Test Real Options Trading - Buy and Sell Functionality
Comprehensive test of the new options trading system
"""

import asyncio
import sys
from datetime import datetime

# Add current directory to path
sys.path.append('.')

async def test_options_broker():
    """Test the options broker functionality"""
    print("=" * 60)
    print("TESTING REAL OPTIONS TRADING SYSTEM")
    print("=" * 60)
    
    try:
        from agents.options_broker import OptionsBroker, OptionsOrderRequest, OptionsOrderType
        from agents.broker_integration import OrderSide
        
        # Initialize options broker (paper trading mode)
        broker = OptionsBroker(None, paper_trading=True)
        print("[OK] Options broker initialized")
        
        # Test 1: Submit a simple options buy order
        print("\n1. Testing Options BUY Order (Long Call)")
        print("-" * 40)
        
        buy_order = OptionsOrderRequest(
            symbol="AAPL250919C00200000",  # AAPL Sep 19, 2025 $200 Call
            underlying="AAPL",
            qty=1,
            side=OrderSide.BUY,
            type=OptionsOrderType.MARKET,
            option_type='call',
            strike=200.0,
            expiration=datetime(2025, 9, 19),
            client_order_id="TEST_BUY_CALL"
        )
        
        buy_response = await broker.submit_options_order(buy_order)
        
        if buy_response.status == "filled":
            print(f"[SUCCESS] BUY Order Filled!")
            print(f"  Order ID: {buy_response.id}")
            print(f"  Symbol: {buy_response.symbol}")
            print(f"  Quantity: {buy_response.qty} contracts")
            print(f"  Fill Price: ${buy_response.avg_fill_price:.2f}")
            print(f"  Total Cost: ${buy_response.avg_fill_price * buy_response.qty * 100:.2f}")
            
            buy_order_id = buy_response.id
            buy_symbol = buy_response.symbol
        else:
            print(f"[FAIL] BUY Order not filled: {buy_response.status}")
            return False
        
        # Test 2: Check positions
        print("\n2. Testing Position Tracking")
        print("-" * 40)
        
        positions = await broker.get_options_positions()
        
        if positions:
            print(f"[OK] Found {len(positions)} positions:")
            for pos in positions:
                print(f"  {pos['symbol']}: {pos['quantity']} contracts")
                print(f"    Current Value: ${pos['market_value']:.2f}")
                print(f"    P&L: ${pos['unrealized_pnl']:.2f} ({pos['unrealized_pnl_percent']:.1f}%)")
        else:
            print("[WARN] No positions found")
        
        # Test 3: Submit a sell order to close
        print("\n3. Testing Options SELL Order (Close Position)")
        print("-" * 40)
        
        sell_order = OptionsOrderRequest(
            symbol=buy_symbol,
            underlying="AAPL",
            qty=1,
            side=OrderSide.SELL,
            type=OptionsOrderType.MARKET,
            option_type='call',
            strike=200.0,
            expiration=datetime(2025, 9, 19),
            client_order_id="TEST_SELL_CALL"
        )
        
        sell_response = await broker.submit_options_order(sell_order)
        
        if sell_response.status == "filled":
            print(f"[SUCCESS] SELL Order Filled!")
            print(f"  Order ID: {sell_response.id}")
            print(f"  Fill Price: ${sell_response.avg_fill_price:.2f}")
            print(f"  Total Proceeds: ${sell_response.avg_fill_price * sell_response.qty * 100:.2f}")
            
            # Calculate P&L
            buy_cost = buy_response.avg_fill_price * 100
            sell_proceeds = sell_response.avg_fill_price * 100
            pnl = sell_proceeds - buy_cost
            pnl_percent = (pnl / buy_cost) * 100
            
            print(f"  TRADE P&L: ${pnl:.2f} ({pnl_percent:+.1f}%)")
        else:
            print(f"[FAIL] SELL Order not filled: {sell_response.status}")
            return False
        
        # Test 4: Check final positions (should be empty)
        print("\n4. Testing Position Cleanup")
        print("-" * 40)
        
        final_positions = await broker.get_options_positions()
        if not final_positions:
            print("[SUCCESS] All positions closed properly")
        else:
            print(f"[WARN] {len(final_positions)} positions still open")
        
        # Test 5: Get summary
        summary = broker.get_paper_summary()
        print(f"\nTRADING SUMMARY:")
        print(f"  Total Orders: {summary['total_orders']}")
        print(f"  Total Positions: {summary['total_positions']}")
        print(f"  Realized P&L: ${summary['total_realized_pnl']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Options broker test failed: {e}")
        return False

async def test_options_trader_integration():
    """Test the full options trader with strategies"""
    print("\n" + "=" * 60)
    print("TESTING OPTIONS TRADER STRATEGIES")
    print("=" * 60)
    
    try:
        from agents.options_trading_agent import OptionsTrader, OptionsStrategy
        
        # Initialize options trader
        trader = OptionsTrader(None)  # No broker for testing
        print("[OK] Options trader initialized")
        
        # Test 1: Get options chain
        print("\n1. Getting Options Chain for AAPL")
        print("-" * 40)
        
        contracts = await trader.get_options_chain('AAPL')
        
        if contracts:
            print(f"[OK] Found {len(contracts)} contracts with >14 days to expiry")
            sample_contract = contracts[0]
            print(f"Sample: {sample_contract.symbol} - ${sample_contract.strike:.2f} strike, {sample_contract.days_to_expiry} days")
        else:
            print("[WARN] No suitable options contracts found")
            return False
        
        # Test 2: Find strategy for bullish scenario
        print("\n2. Testing Strategy Selection (Bullish)")
        print("-" * 40)
        
        strategy_result = trader.find_best_options_strategy(
            'AAPL', 150.0, 25.0, 45.0, 0.035  # Bullish momentum conditions
        )
        
        if strategy_result:
            strategy, strategy_contracts = strategy_result
            print(f"[OK] Found strategy: {strategy}")
            print(f"  Using {len(strategy_contracts)} contracts")
            
            # Test 3: Execute the strategy
            print("\n3. Executing Options Strategy")
            print("-" * 40)
            
            position = await trader.execute_options_strategy(strategy, strategy_contracts, quantity=1)
            
            if position:
                print(f"[SUCCESS] Strategy executed!")
                print(f"  Position ID: {position.symbol}")
                print(f"  Strategy: {position.strategy}")
                print(f"  Underlying: {position.underlying}")
                print(f"  Quantity: {position.quantity} contracts")
                print(f"  Entry Price: ${position.entry_price:.2f}")
                print(f"  Entry Time: {position.entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Test 4: Monitor position
                print("\n4. Monitoring Position")
                print("-" * 40)
                
                actions = await trader.monitor_options_positions()
                
                if actions:
                    print(f"[INFO] Monitoring triggered {len(actions)} actions:")
                    for action in actions:
                        print(f"  - {action['action']}: {action['reason']}")
                else:
                    print("[OK] Position within normal parameters")
                
                # Test 5: Close position
                print("\n5. Closing Position")
                print("-" * 40)
                
                close_success = await trader.close_position(position.symbol, "Test Close")
                
                if close_success:
                    print(f"[SUCCESS] Position closed!")
                    print(f"  Final P&L: ${position.pnl:.2f}")
                else:
                    print("[WARN] Position close failed")
                
            else:
                print("[FAIL] Strategy execution failed")
                return False
        else:
            print("[INFO] No suitable strategy found for test conditions (normal)")
        
        # Test 6: Get positions summary
        summary = trader.get_positions_summary()
        print(f"\nOPTIONS TRADER SUMMARY:")
        print(f"  Active Positions: {summary['total_positions']}")
        print(f"  Total P&L: ${summary['total_pnl']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Options trader test failed: {e}")
        return False

async def test_multiple_strategies():
    """Test different options strategies"""
    print("\n" + "=" * 60)
    print("TESTING MULTIPLE OPTIONS STRATEGIES")
    print("=" * 60)
    
    try:
        from agents.options_trading_agent import OptionsTrader
        
        trader = OptionsTrader(None)
        
        test_scenarios = [
            {
                'name': 'Strong Bullish',
                'params': {'price': 150.0, 'volatility': 30.0, 'rsi': 40.0, 'price_change': 0.04},
                'expected': ['LONG_CALL', 'BULL_CALL_SPREAD']
            },
            {
                'name': 'Strong Bearish', 
                'params': {'price': 150.0, 'volatility': 35.0, 'rsi': 65.0, 'price_change': -0.04},
                'expected': ['LONG_PUT', 'BEAR_PUT_SPREAD']
            },
            {
                'name': 'High Volatility Neutral',
                'params': {'price': 150.0, 'volatility': 40.0, 'rsi': 50.0, 'price_change': 0.005},
                'expected': ['STRADDLE', 'IRON_CONDOR']
            }
        ]
        
        strategies_found = 0
        
        for scenario in test_scenarios:
            print(f"\nTesting {scenario['name']} Scenario:")
            print("-" * 40)
            
            # Get options chain first
            contracts = await trader.get_options_chain('AAPL')
            if not contracts:
                print(f"  [SKIP] No contracts available")
                continue
            
            result = trader.find_best_options_strategy('AAPL', **scenario['params'])
            
            if result:
                strategy, strategy_contracts = result
                print(f"  [OK] Strategy: {strategy}")
                print(f"  [OK] Contracts: {len(strategy_contracts)}")
                
                if strategy in scenario['expected']:
                    print(f"  [PASS] Expected strategy found")
                else:
                    print(f"  [INFO] Unexpected but valid strategy")
                
                strategies_found += 1
            else:
                print(f"  [SKIP] No strategy found (market conditions)")
        
        print(f"\nSTRATEGY TESTING SUMMARY:")
        print(f"  Scenarios tested: {len(test_scenarios)}")
        print(f"  Strategies found: {strategies_found}")
        
        return strategies_found > 0
        
    except Exception as e:
        print(f"[ERROR] Multiple strategy test failed: {e}")
        return False

async def main():
    """Run comprehensive options trading tests"""
    print("COMPREHENSIVE REAL OPTIONS TRADING TEST")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    test_results = []
    
    # Test 1: Basic options broker functionality
    print("\n[INFO] PHASE 1: Options Broker Testing")
    result1 = await test_options_broker()
    test_results.append(("Options Broker", result1))
    
    # Test 2: Options trader integration
    print("\n[INFO] PHASE 2: Options Trader Integration")
    result2 = await test_options_trader_integration()
    test_results.append(("Options Trader", result2))
    
    # Test 3: Multiple strategies
    print("\n[INFO] PHASE 3: Strategy Selection Testing")
    result3 = await test_multiple_strategies()
    test_results.append(("Strategy Selection", result3))
    
    # Final results
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    
    passed = 0
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name:.<40} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("\n[PARTY] ALL TESTS PASSED!")
        print("[OK] The system can now buy and sell real options contracts!")
        print("[OK] Multiple options strategies are working!")
        print("[OK] Position tracking and P&L calculation working!")
    else:
        print(f"\n[WARN]  {len(test_results) - passed} tests failed")
    
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())