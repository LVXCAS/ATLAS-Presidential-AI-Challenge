#!/usr/bin/env python3
"""
Debug Monte Carlo - Simple version to find issues
"""

import asyncio
import sys
import random
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append('.')

from agents.options_trading_agent import OptionsTrader

async def test_single_scenario():
    """Test a single trading scenario"""
    
    print("TESTING SINGLE TRADING SCENARIO")
    print("=" * 50)
    
    # Create a bullish scenario that should trigger trading
    scenario = {
        'symbol': 'AAPL',
        'current_price': 225.0,  # Realistic AAPL price
        'price_change_pct': 3.5,  # Strong bullish move
        'volatility': 0.25,       # 25% volatility
        'rsi': 65.0,             # Bullish but not overbought
        'regime': 'trending_up'
    }
    
    print(f"Scenario: {scenario['symbol']} - {scenario['regime']}")
    print(f"  Price: ${scenario['current_price']:.2f}")
    print(f"  Change: {scenario['price_change_pct']:.1f}%")
    print(f"  Volatility: {scenario['volatility']:.1%}")
    print(f"  RSI: {scenario['rsi']:.1f}")
    
    try:
        # Initialize trader
        trader = OptionsTrader(None)
        print("[OK] Options trader initialized")
        
        # Test strategy finding
        print("\nTesting strategy selection...")
        strategy_result = trader.find_best_options_strategy(
            scenario['symbol'],
            scenario['current_price'],
            scenario['volatility'] * 100,  # Convert to percentage
            scenario['rsi'],
            scenario['price_change_pct'] / 100  # Convert to decimal
        )
        
        if strategy_result:
            strategy, contracts = strategy_result
            print(f"[SUCCESS] Strategy found: {strategy}")
            print(f"[SUCCESS] Contracts available: {len(contracts)}")
            
            # Show contract details
            for i, contract in enumerate(contracts[:3]):  # Show first 3
                print(f"  Contract {i+1}: {contract.symbol}")
                print(f"    Strike: ${contract.strike}")
                print(f"    Type: {contract.option_type}")
                print(f"    Delta: {contract.delta:.3f}")
        else:
            print("[WARN] No strategy found")
            
            # Debug: Let's check what get_options_chain returns
            print("\nDebugging options chain...")
            contracts = await trader.get_options_chain(scenario['symbol'])
            
            if contracts:
                print(f"[INFO] Options chain has {len(contracts)} contracts")
                
                # Show sample contracts
                for i, contract in enumerate(contracts[:5]):
                    days_to_expiry = (contract.expiration - datetime.now()).days
                    print(f"  Contract {i+1}: {contract.symbol}")
                    print(f"    Strike: ${contract.strike}, Type: {contract.option_type}")
                    print(f"    Days to expiry: {days_to_expiry}")
                    print(f"    Delta: {contract.delta:.3f}, Volume: {getattr(contract, 'volume', 'N/A')}")
                
                # Check filtering criteria
                suitable_calls = [c for c in contracts if 
                                c.option_type == 'call' and 
                                (datetime.now() + timedelta(days=14)) <= c.expiration <= (datetime.now() + timedelta(days=45))]
                
                print(f"\n[INFO] Calls with 14-45 days to expiry: {len(suitable_calls)}")
                
                if suitable_calls:
                    # Check price/volume filters
                    liquid_calls = [c for c in suitable_calls if 
                                  hasattr(c, 'volume') and c.volume and c.volume > 10]
                    print(f"[INFO] Liquid calls (volume > 10): {len(liquid_calls)}")
                    
                    reasonably_priced = [c for c in suitable_calls if 
                                       hasattr(c, 'last_price') and c.last_price and 0.05 <= c.last_price <= 50.0]
                    print(f"[INFO] Reasonably priced calls ($0.05-$50): {len(reasonably_priced)}")
                
            else:
                print("[ERROR] No options contracts returned from get_options_chain")
                print("This indicates an issue with options data retrieval")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_multiple_scenarios():
    """Test multiple scenarios"""
    
    print("\n\nTESTING MULTIPLE SCENARIOS")
    print("=" * 50)
    
    scenarios = [
        {'symbol': 'AAPL', 'price': 225.0, 'change': 5.0, 'vol': 0.20, 'rsi': 70.0, 'name': 'Strong Bullish'},
        {'symbol': 'MSFT', 'price': 280.0, 'change': -4.0, 'vol': 0.25, 'rsi': 30.0, 'name': 'Strong Bearish'},
        {'symbol': 'SPY', 'price': 450.0, 'change': 0.5, 'vol': 0.15, 'rsi': 50.0, 'name': 'Neutral'},
        {'symbol': 'TSLA', 'price': 200.0, 'change': 8.0, 'vol': 0.40, 'rsi': 75.0, 'name': 'High Vol Bullish'},
        {'symbol': 'NVDA', 'price': 450.0, 'change': -6.0, 'vol': 0.35, 'rsi': 25.0, 'name': 'High Vol Bearish'}
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\nScenario {i+1}: {scenario['name']} - {scenario['symbol']}")
        
        try:
            trader = OptionsTrader(None)
            
            strategy_result = trader.find_best_options_strategy(
                scenario['symbol'],
                scenario['price'],
                scenario['vol'] * 100,
                scenario['rsi'],
                scenario['change'] / 100
            )
            
            if strategy_result:
                strategy, contracts = strategy_result
                print(f"  [SUCCESS] {strategy} with {len(contracts)} contracts")
                results.append({'scenario': scenario['name'], 'strategy': strategy, 'contracts': len(contracts)})
            else:
                print(f"  [WARN] No strategy found")
                results.append({'scenario': scenario['name'], 'strategy': None, 'contracts': 0})
                
        except Exception as e:
            print(f"  [ERROR] Failed: {e}")
            results.append({'scenario': scenario['name'], 'strategy': 'ERROR', 'contracts': 0})
    
    print(f"\n\nSUMMARY:")
    print(f"-" * 30)
    successful_strategies = [r for r in results if r['strategy'] and r['strategy'] != 'ERROR']
    print(f"Successful strategies: {len(successful_strategies)}/{len(results)}")
    
    for result in results:
        status = "OK" if result['strategy'] and result['strategy'] != 'ERROR' else "FAIL"
        print(f"  {result['scenario']:20} [{status:4}] {result['strategy'] or 'None'}")
    
    return len(successful_strategies) > 0

async def main():
    """Run debug tests"""
    
    print("MONTE CARLO DEBUG TEST")
    print("=" * 60)
    
    # Test 1: Single detailed scenario
    success1 = await test_single_scenario()
    
    # Test 2: Multiple scenarios
    success2 = await test_multiple_scenarios()
    
    print(f"\n" + "=" * 60)
    print("DEBUG TEST RESULTS")
    print(f"=" * 60)
    
    if success1 and success2:
        print("[SUCCESS] Bot is finding trading opportunities!")
        print("Monte Carlo test should work properly.")
    elif success1 or success2:
        print("[PARTIAL] Some scenarios work, others don't.")
        print("Bot may have limited trading conditions.")
    else:
        print("[FAILURE] Bot is not finding any trading opportunities.")
        print("Need to debug strategy selection logic.")
    
    print(f"=" * 60)

if __name__ == "__main__":
    asyncio.run(main())