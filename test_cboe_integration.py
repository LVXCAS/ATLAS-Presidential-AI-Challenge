#!/usr/bin/env python3
"""
Test CBOE API Integration - VIX Term Structure & Options Flow
"""

import asyncio
import sys
import os
sys.path.append('.')

from agents.cboe_data_agent import cboe_data_agent

async def test_cboe_apis():
    """Test CBOE API functionality"""
    print("=" * 60)
    print("TESTING CBOE VOLATILITY INTELLIGENCE")
    print("=" * 60)
    
    try:
        # Test VIX term structure analysis
        print("\n[TEST] Getting VIX term structure analysis...")
        vix_data = await cboe_data_agent.get_vix_term_structure_analysis()
        
        if vix_data:
            print("[OK] VIX Term Structure Analysis Complete!")
            print(f"  Timestamp: {vix_data['timestamp']}")
            print(f"  VIX Current: {vix_data['vix_current']:.1f}")
            print(f"  VIX 9D: {vix_data['vix9d']:.1f}")
            print(f"  VIX 3M: {vix_data['vix3m']:.1f}")
            print(f"  VIX 6M: {vix_data['vix6m']:.1f}")
            
            print(f"\n[VIX TERM STRUCTURE]")
            print(f"  Term Structure: {vix_data['term_structure']}")
            print(f"  Volatility Regime: {vix_data['volatility_regime']}")
            print(f"  Volatility Percentile: {vix_data['volatility_percentile']:.1f}%")
            print(f"  Backwardation Signal: {vix_data['backwardation_signal']}")
            
            # VIX implications
            vix_implications = vix_data.get('options_implications', {})
            if vix_implications:
                print(f"\n[VIX TRADING IMPLICATIONS]")
                print(f"  Volatility Trade Bias: {vix_implications.get('volatility_trade_bias', 'N/A')}")
                print(f"  Term Structure Trade: {vix_implications.get('term_structure_trade', 'N/A')}")
                print(f"  Volatility Timing: {vix_implications.get('volatility_timing', 'N/A')}")
        
        # Test options flow analysis
        print(f"\n[TEST] Getting options flow analysis...")
        flow_data = await cboe_data_agent.get_options_flow_analysis()
        
        if flow_data:
            print("[OK] Options Flow Analysis Complete!")
            
            print(f"\n[OPTIONS FLOW DATA]")
            print(f"  Put/Call Ratio: {flow_data['put_call_ratio']:.2f}")
            print(f"  VIX Call/Put Ratio: {flow_data['vix_call_put_ratio']:.2f}")
            print(f"  SKEW Index: {flow_data['skew_index']:.1f}")
            print(f"  Volatility Demand: {flow_data['volatility_demand']}")
            
            print(f"\n[MARKET SENTIMENT]")
            print(f"  Market Sentiment: {flow_data['market_sentiment']}")
            print(f"  Fear/Greed Indicator: {flow_data['fear_greed_indicator']}")
            
            # Flow implications
            flow_implications = flow_data.get('options_implications', {})
            if flow_implications:
                print(f"\n[FLOW TRADING IMPLICATIONS]")
                print(f"  Positioning Bias: {flow_implications.get('positioning_bias', 'N/A')}")
                print(f"  Volatility Strategy: {flow_implications.get('volatility_strategy', 'N/A')}")
                print(f"  Market Timing: {flow_implications.get('market_timing', 'N/A')}")
        
        # Combined volatility intelligence
        print(f"\n[COMBINED VOLATILITY INTELLIGENCE]")
        
        # VIX signals
        if vix_data['backwardation_signal']:
            print("  ALERT: VIX backwardation detected - volatility selling opportunity")
        
        if vix_data['volatility_regime'] in ['HIGH_VOLATILITY', 'EXTREME_VOLATILITY']:
            print("  ALERT: High volatility regime - consider volatility selling strategies")
        elif vix_data['volatility_regime'] == 'LOW_VOLATILITY':
            print("  ALERT: Low volatility regime - consider volatility buying strategies")
        
        # Flow signals  
        if flow_data['market_sentiment'] == 'EXTREME_FEAR':
            print("  SIGNAL: Extreme fear detected - potential contrarian opportunity")
        elif flow_data['market_sentiment'] == 'EXTREME_GREED':
            print("  SIGNAL: Extreme greed detected - potential correction ahead")
        
        if flow_data['put_call_ratio'] > 1.3:
            print("  SIGNAL: Elevated put buying - defensive positioning")
        elif flow_data['put_call_ratio'] < 0.7:
            print("  SIGNAL: Low put buying - potential complacency")
            
    except Exception as e:
        print(f"[ERROR] CBOE API test failed: {e}")
        print("\nNote: CBOE APIs are simulated in this implementation")
        print("Real implementation would require CBOE data subscriptions")

    print("\n" + "=" * 60)
    print("CBOE VOLATILITY INTELLIGENCE BENEFITS")
    print("=" * 60)
    print("\nWhy CBOE volatility data makes your bot more profitable:")
    print("• VIX Term Structure: Identify volatility selling opportunities")
    print("• Backwardation Detection: Time volatility trades perfectly")
    print("• Options Flow Analysis: Follow smart money positioning")
    print("• Market Sentiment: Use fear/greed as contrarian signals")
    print("• Volatility Regime: Adjust strategies to vol environment")
    print("• SKEW Analysis: Understand tail risk premiums")
    
    print("\n[COMPLETE] CBOE Volatility Intelligence Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_cboe_apis())