#!/usr/bin/env python3
"""
Test FRED API Integration
"""

import asyncio
import sys
import os
sys.path.append('.')

from agents.economic_data_agent import economic_data_agent

async def test_fred_api():
    """Test FRED API functionality"""
    print("=" * 60)
    print("TESTING FRED API INTEGRATION")
    print("=" * 60)
    
    try:
        # Test comprehensive economic analysis
        print("\n[TEST] Getting comprehensive economic analysis...")
        economic_data = await economic_data_agent.get_comprehensive_economic_analysis()
        
        if economic_data:
            print("[OK] FRED API Connected Successfully!")
            print(f"  Timestamp: {economic_data['timestamp']}")
            print(f"  Market Regime: {economic_data['market_regime']}")
            print(f"  Volatility Environment: {economic_data['volatility_environment']}")
            print(f"  Options Strategy Bias: {economic_data['options_strategy_bias']}")
            
            # Fed Policy Data
            fed_policy = economic_data.get('fed_policy', {})
            if fed_policy:
                print(f"\n[FED POLICY]")
                print(f"  Fed Funds Rate: {fed_policy.get('fed_funds_rate', 0):.2f}%")
                print(f"  6-Month Change: {fed_policy.get('fed_funds_change_6m', 0):+.2f}%")
                print(f"  Policy Stance: {fed_policy.get('policy_stance', 'N/A')}")
                print(f"  Next Meeting Impact: {fed_policy.get('next_meeting_impact', 'N/A')}")
            
            # Inflation Data
            inflation = economic_data.get('inflation_data', {})
            if inflation:
                print(f"\n[INFLATION]")
                print(f"  CPI Year-over-Year: {inflation.get('cpi_yoy', 0):.1f}%")
                print(f"  Core CPI YoY: {inflation.get('core_cpi_yoy', 0):.1f}%")
                print(f"  Fed Target Distance: {inflation.get('fed_target_distance', 0):+.1f}%")
                print(f"  Inflation Trend: {inflation.get('inflation_trend', 'N/A')}")
            
            # Economic Stress
            stress = economic_data.get('economic_stress', {})
            if stress:
                print(f"\n[ECONOMIC STRESS]")
                print(f"  Recession Probability: {stress.get('recession_probability', 'N/A')}")
                print(f"  Leading Indicators: {stress.get('leading_indicators_trend', 'N/A')}")
                print(f"  Consumer Confidence: {stress.get('consumer_confidence', 'N/A')}")
                
                # Credit Spreads
                credit = stress.get('credit_spreads', {})
                if credit:
                    print(f"\n[CREDIT MARKETS]")
                    print(f"  High Yield Spread: {credit.get('high_yield', 0):.0f} bps")
                    print(f"  Credit Stress Level: {credit.get('stress_level', 'N/A')}")
                
                # Dollar Strength
                dollar = stress.get('dollar_strength', {})
                if dollar:
                    print(f"\n[DOLLAR STRENGTH]")
                    print(f"  Dollar Index: {dollar.get('index_value', 0):.1f}")
                    print(f"  Trend: {dollar.get('trend', 'N/A')}")
                
                # Yield Curve
                yield_curve = stress.get('yield_curve', {})
                if yield_curve:
                    print(f"\n[YIELD CURVE]")
                    print(f"  10Y-2Y Spread: {yield_curve.get('spread_10y2y', 0):.2f}%")
                    print(f"  Curve Shape: {yield_curve.get('shape', 'N/A')}")
            
            # Options Implications
            implications = economic_data.get('options_implications', {})
            if implications:
                print(f"\n[OPTIONS IMPLICATIONS]")
                print(f"  Volatility Bias: {implications.get('volatility_bias', 'N/A')}")
                print(f"  Event Risk: {implications.get('event_risk_assessment', 'N/A')}")
                print(f"  Liquidity Environment: {implications.get('liquidity_environment', 'N/A')}")
            
            print(f"\n[TRADING RECOMMENDATIONS]")
            print(f"  Recommended Strategy Bias: {economic_data['options_strategy_bias']}")
            
            # Strategy explanations
            strategy_explanations = {
                'PROTECTIVE_PUTS': 'Buy puts to hedge against market downturns',
                'DEFENSIVE_SPREADS': 'Conservative spreads during financial stress',
                'VOLATILITY_SELLING': 'Sell options to profit from high implied volatility',
                'BEAR_PUT_SPREADS': 'Bearish spreads for hawkish Fed environment',
                'BULL_CALL_SPREADS': 'Bullish spreads for dovish Fed environment',
                'EXPORT_SECTOR_PUTS': 'Target exporters hurt by strong dollar',
                'NEUTRAL': 'Balanced approach, no strong directional bias'
            }
            
            explanation = strategy_explanations.get(economic_data['options_strategy_bias'], 'No specific bias')
            print(f"  Strategy Explanation: {explanation}")
            
        else:
            print("[FAIL] No economic data received")
            
    except Exception as e:
        print(f"[ERROR] FRED API test failed: {e}")
        print("\nThis could be due to:")
        print("- Invalid API key")
        print("- Network connectivity issues")
        print("- FRED API rate limits")
        print("- Missing fredapi library")

    print("\n" + "=" * 60)
    print("FRED API PROFITABILITY BENEFITS")
    print("=" * 60)
    print("\nWhy FRED API makes your bot more profitable:")
    print("• Fed Policy Awareness: Avoid trading before Fed meetings")
    print("• Inflation Regime Detection: Adjust volatility strategies")
    print("• Recession Probability: Switch to defensive positions early")
    print("• Economic Stress Monitoring: Increase vol selling in calm periods")
    print("• Strategy Bias Optimization: Use macro trends for direction")
    
    print("\n[COMPLETE] FRED Integration Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_fred_api())