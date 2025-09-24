#!/usr/bin/env python3
"""
Test script to validate enhanced libraries integration
"""

import asyncio
import sys
import os
sys.path.append('.')

from agents.enhanced_technical_analysis import enhanced_technical_analysis
from agents.enhanced_options_pricing import enhanced_options_pricing

async def test_enhanced_features():
    """Test the enhanced technical analysis and options pricing"""
    print("=" * 60)
    print("TESTING ENHANCED LIBRARIES INTEGRATION")
    print("=" * 60)
    
    # Test symbols
    test_symbols = ['AAPL', 'SPY', 'MSFT']
    
    for symbol in test_symbols:
        print(f"\n[TEST] Testing Enhanced Analysis for {symbol}")
        print("-" * 40)
        
        try:
            # Test enhanced technical analysis
            print("1. Enhanced Technical Analysis...")
            tech_analysis = await enhanced_technical_analysis.get_comprehensive_analysis(symbol)
            
            if tech_analysis and tech_analysis['current_price'] > 0:
                print(f"[OK] Technical Analysis Success!")
                print(f"   Current Price: ${tech_analysis['current_price']:.2f}")
                print(f"   Overall Signal: {tech_analysis['signals']['overall_signal']}")
                print(f"   Signal Strength: {tech_analysis['signals']['signal_strength']:.1%}")
                print(f"   Confidence: {tech_analysis['signals']['confidence']:.1%}")
                
                # Show key indicators
                indicators = tech_analysis['technical_indicators']
                if indicators:
                    print(f"   RSI: {indicators.get('rsi', 'N/A')}")
                    print(f"   Volatility: {tech_analysis['volatility_analysis'].get('realized_vol_20d', 'N/A'):.1f}%")
                    print(f"   Momentum: {tech_analysis['momentum_analysis'].get('momentum_strength', 'N/A')}")
                
                # Test enhanced options pricing
                print("\n2. Enhanced Options Pricing...")
                current_price = tech_analysis['current_price']
                strike_price = round(current_price)  # ATM strike
                volatility = tech_analysis['volatility_analysis'].get('realized_vol_20d', 25)
                
                # Test call option pricing
                call_analysis = await enhanced_options_pricing.get_comprehensive_option_analysis(
                    underlying_price=current_price,
                    strike_price=strike_price,
                    time_to_expiry_days=21,  # 3 weeks
                    volatility=volatility,
                    option_type='call'
                )
                
                if call_analysis:
                    print(f"[OK] Options Pricing Success!")
                    pricing = call_analysis['pricing']
                    greeks = call_analysis['greeks']
                    
                    print(f"   Call Price: ${pricing['theoretical_price']:.2f}")
                    print(f"   Intrinsic: ${pricing['intrinsic_value']:.2f}")
                    print(f"   Time Value: ${pricing['time_value']:.2f}")
                    print(f"   Method: {pricing['pricing_method']}")
                    print(f"   Delta: {greeks['delta']:.3f}")
                    print(f"   Theta: ${greeks['theta']:.3f}")
                    print(f"   Vega: {greeks['vega']:.3f}")
                    
                    # Test profit scenarios
                    profit_analysis = call_analysis.get('profitability_analysis', {})
                    if profit_analysis:
                        print(f"\n   Profit Scenarios:")
                        for scenario, data in list(profit_analysis.items())[:3]:  # Show first 3
                            print(f"     {scenario}: ${data['option_value']:.2f} (P&L: {data['pnl_percent']:+.1f}%)")
                
                else:
                    print("[FAIL] Options Pricing Failed")
                    
            else:
                print("[FAIL] Technical Analysis Failed")
                
        except Exception as e:
            print(f"[ERROR] Error testing {symbol}: {e}")
    
    print("\n" + "=" * 60)
    print("LIBRARY AVAILABILITY CHECK")
    print("=" * 60)
    
    # Check library availability
    libraries = {
        'finta': 'Technical Analysis Indicators',
        'py_vollib': 'Professional Options Pricing',
        'mibian': 'Alternative Options Pricing', 
        'scipy': 'Statistical Functions',
        'statsmodels': 'Time Series Analysis',
        'scikit-learn': 'Machine Learning'
    }
    
    for lib_name, description in libraries.items():
        try:
            __import__(lib_name)
            print(f"[OK] {lib_name:15} - {description}")
        except ImportError:
            print(f"[MISSING] {lib_name:15} - {description} (NOT AVAILABLE)")
    
    print("\n[COMPLETE] Enhanced Libraries Integration Test Complete!")
    print("\nKey Upgrades Implemented:")
    print("• Professional technical analysis with 15+ indicators")
    print("• Advanced options pricing using Black-Scholes models")
    print("• Support/resistance level detection")
    print("• Volatility regime analysis")
    print("• Enhanced momentum and trend analysis")
    print("• Professional Greeks calculation")
    print("• Profit/loss scenario modeling")

if __name__ == "__main__":
    asyncio.run(test_enhanced_features())