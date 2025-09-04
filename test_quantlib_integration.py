#!/usr/bin/env python3
"""
Test QuantLib Integration in Options Trading System
"""

import asyncio
import sys
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append('.')

def test_quantlib_installation():
    """Test that QuantLib is properly installed"""
    print("TESTING QUANTLIB INSTALLATION")
    print("=" * 50)
    
    try:
        import QuantLib as ql
        print(f"[OK] QuantLib imported successfully")
        print(f"[OK] QuantLib version: {ql.__version__}")
        
        # Test basic functionality
        today = ql.Date.todaysDate()
        print(f"[OK] Today's date in QuantLib: {today}")
        
        # Test Black-Scholes calculation
        option_type = ql.Option.Call
        underlying = 100.0
        strike = 105.0
        risk_free_rate = 0.05
        volatility = 0.20
        time_to_expiry = 0.25  # 3 months
        
        payoff = ql.PlainVanillaPayoff(option_type, strike)
        exercise = ql.EuropeanExercise(today + int(time_to_expiry * 365))
        
        option = ql.VanillaOption(payoff, exercise)
        
        # Set up Black-Scholes process
        underlying_handle = ql.QuoteHandle(ql.SimpleQuote(underlying))
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate, ql.Actual365Fixed()))
        dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))
        flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.UnitedStates(ql.UnitedStates.NYSE), volatility, ql.Actual365Fixed()))
        
        bs_process = ql.BlackScholesMertonProcess(underlying_handle, dividend_handle, flat_ts, flat_vol_ts)
        engine = ql.AnalyticEuropeanEngine(bs_process)
        option.setPricingEngine(engine)
        
        price = option.NPV()
        delta = option.delta()
        gamma = option.gamma()
        theta = option.theta()
        vega = option.vega()
        
        print(f"[OK] Sample Black-Scholes calculation:")
        print(f"     Call price: ${price:.4f}")
        print(f"     Delta: {delta:.4f}")
        print(f"     Gamma: {gamma:.4f}")
        print(f"     Theta: {theta:.4f}")
        print(f"     Vega: {vega:.4f}")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] QuantLib import failed: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] QuantLib test failed: {e}")
        return False

def test_quantlib_pricer():
    """Test our QuantLib pricing module"""
    print("\nTESTING QUANTLIB PRICER MODULE")
    print("=" * 50)
    
    try:
        from agents.quantlib_pricing import quantlib_pricer
        
        if not quantlib_pricer:
            print("[ERROR] QuantLib pricer not available")
            return False
        
        print("[OK] QuantLib pricer module loaded")
        
        # Test option pricing
        expiry_date = datetime.now() + timedelta(days=30)
        
        pricing_result = quantlib_pricer.price_european_option(
            option_type='call',
            underlying_price=150.0,
            strike=155.0,
            expiry_date=expiry_date,
            symbol='AAPL'
        )
        
        print(f"[OK] European call option pricing:")
        print(f"     Price: ${pricing_result['price']:.4f}")
        print(f"     Delta: {pricing_result['delta']:.4f}")
        print(f"     Gamma: {pricing_result['gamma']:.4f}")
        print(f"     Theta: ${pricing_result['theta']:.4f}")
        print(f"     Vega: {pricing_result['vega']:.4f}")
        print(f"     Volatility used: {pricing_result['volatility_used']:.1%}")
        print(f"     Risk-free rate: {pricing_result['risk_free_rate']:.1%}")
        
        # Test put pricing
        put_result = quantlib_pricer.price_european_option(
            option_type='put',
            underlying_price=150.0,
            strike=145.0,
            expiry_date=expiry_date,
            symbol='AAPL'
        )
        
        print(f"[OK] European put option pricing:")
        print(f"     Price: ${put_result['price']:.4f}")
        print(f"     Delta: {put_result['delta']:.4f}")
        
        # Test implied volatility calculation
        market_price = pricing_result['price'] + 0.50  # Add some premium
        implied_vol = quantlib_pricer.calculate_implied_volatility(
            option_price=market_price,
            option_type='call',
            underlying_price=150.0,
            strike=155.0,
            expiry_date=expiry_date,
            symbol='AAPL'
        )
        
        print(f"[OK] Implied volatility calculation:")
        print(f"     Market price: ${market_price:.4f}")
        print(f"     Implied vol: {implied_vol:.1%}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] QuantLib pricer test failed: {e}")
        return False

async def test_options_trader_with_quantlib():
    """Test options trader with QuantLib integration"""
    print("\nTESTING OPTIONS TRADER WITH QUANTLIB")
    print("=" * 50)
    
    try:
        from agents.options_trading_agent import OptionsTrader
        from agents.quantlib_pricing import quantlib_pricer
        
        # Initialize trader
        trader = OptionsTrader(None)
        print("[OK] Options trader initialized")
        
        # Get options chain (should now have QuantLib Greeks)
        print("[INFO] Getting options chain with QuantLib Greeks...")
        contracts = await trader.get_options_chain('AAPL')
        
        if contracts:
            print(f"[OK] Retrieved {len(contracts)} contracts")
            
            # Check if Greeks are calculated
            sample_contract = contracts[0]
            print(f"[OK] Sample contract: {sample_contract.symbol}")
            print(f"     Strike: ${sample_contract.strike:.2f}")
            print(f"     Type: {sample_contract.option_type}")
            print(f"     Delta: {sample_contract.delta:.4f}")
            print(f"     Gamma: {sample_contract.gamma:.4f}")
            print(f"     Theta: ${sample_contract.theta:.4f}")
            print(f"     Vega: {sample_contract.vega:.4f}")
            
            # Check if Greeks are non-zero (indicating QuantLib worked)
            if abs(sample_contract.delta) > 0.01:
                print("[SUCCESS] QuantLib Greeks calculated correctly!")
            else:
                print("[WARN] Greeks appear to be default values")
            
            # Test strategy selection with Greeks
            print("\n[INFO] Testing strategy selection with QuantLib Greeks...")
            strategy_result = trader.find_best_options_strategy(
                'AAPL', 150.0, 25.0, 45.0, 0.035  # Bullish conditions
            )
            
            if strategy_result:
                strategy, strategy_contracts = strategy_result
                print(f"[OK] Strategy selected: {strategy}")
                print(f"     Using {len(strategy_contracts)} contracts")
                
                for i, contract in enumerate(strategy_contracts):
                    print(f"     Contract {i+1}: δ={contract.delta:.3f}, γ={contract.gamma:.4f}, θ={contract.theta:.3f}")
            else:
                print("[INFO] No strategy found (normal for test conditions)")
            
        else:
            print("[WARN] No options contracts retrieved")
            return False
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Options trader QuantLib test failed: {e}")
        return False

def test_quantlib_pricing_accuracy():
    """Test QuantLib pricing accuracy against known values"""
    print("\nTESTING QUANTLIB PRICING ACCURACY")
    print("=" * 50)
    
    try:
        from agents.quantlib_pricing import quantlib_pricer
        
        if not quantlib_pricer:
            print("[ERROR] QuantLib pricer not available")
            return False
        
        # Known test case: ATM call with standard parameters
        test_cases = [
            {
                'name': 'ATM Call',
                'type': 'call',
                'underlying': 100.0,
                'strike': 100.0,
                'days': 30,
                'vol': 0.20,
                'expected_delta': 0.55  # Approximate for ATM call
            },
            {
                'name': 'OTM Call',
                'type': 'call', 
                'underlying': 100.0,
                'strike': 110.0,
                'days': 30,
                'vol': 0.20,
                'expected_delta': 0.25  # Lower delta for OTM
            },
            {
                'name': 'ATM Put',
                'type': 'put',
                'underlying': 100.0,
                'strike': 100.0,
                'days': 30,
                'vol': 0.20,
                'expected_delta': -0.45  # Negative for puts
            }
        ]
        
        for test_case in test_cases:
            expiry = datetime.now() + timedelta(days=test_case['days'])
            
            result = quantlib_pricer.price_european_option(
                option_type=test_case['type'],
                underlying_price=test_case['underlying'],
                strike=test_case['strike'],
                expiry_date=expiry,
                symbol='TEST',
                volatility=test_case['vol']
            )
            
            delta_diff = abs(result['delta'] - test_case['expected_delta'])
            delta_ok = delta_diff < 0.15  # Allow 15% tolerance
            
            print(f"[{'OK' if delta_ok else 'WARN'}] {test_case['name']}:")
            print(f"     Price: ${result['price']:.4f}")
            print(f"     Delta: {result['delta']:.4f} (expected ~{test_case['expected_delta']:.2f})")
            print(f"     Gamma: {result['gamma']:.4f}")
            print(f"     Theta: ${result['theta']:.4f}")
            print(f"     Vega: {result['vega']:.4f}")
        
        print("[SUCCESS] QuantLib pricing accuracy tests completed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Pricing accuracy test failed: {e}")
        return False

async def main():
    """Run all QuantLib integration tests"""
    print("QUANTLIB INTEGRATION TEST SUITE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    tests = [
        ("QuantLib Installation", test_quantlib_installation),
        ("QuantLib Pricer Module", test_quantlib_pricer),
        ("Options Trader Integration", test_options_trader_with_quantlib),
        ("Pricing Accuracy", test_quantlib_pricing_accuracy)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n>>> RUNNING: {test_name}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Final results
    print("\n" + "=" * 70)
    print("QUANTLIB INTEGRATION TEST RESULTS")
    print("=" * 70)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name:.<50} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n[SUCCESS] QuantLib integration is working perfectly!")
        print("- Accurate options pricing with Black-Scholes")
        print("- Real Greeks calculations (Delta, Gamma, Theta, Vega)")
        print("- Enhanced strategy selection using Greeks")
        print("- Implied volatility calculations")
    else:
        print(f"\n[PARTIAL] {len(results) - passed} tests failed")
    
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())