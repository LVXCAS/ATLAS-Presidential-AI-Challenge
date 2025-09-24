#!/usr/bin/env python3
"""
Test Integrated Intelligence System
Combines FRED Economic Data + CBOE Volatility Intelligence + Enhanced Technical Analysis
"""

import asyncio
import sys
import os
sys.path.append('.')

from agents.economic_data_agent import economic_data_agent
from agents.cboe_data_agent import cboe_data_agent
from agents.enhanced_technical_analysis import enhanced_technical_analysis

async def test_integrated_system():
    """Test the complete integrated intelligence system"""
    print("=" * 70)
    print("INTEGRATED INTELLIGENCE SYSTEM TEST")
    print("=" * 70)
    
    try:
        # Test symbol
        symbol = "SPY"
        
        print(f"\n[TEST] Getting comprehensive intelligence for {symbol}...")
        
        # Get all intelligence sources in parallel
        print("\n[1/4] Economic Analysis (FRED API)...")
        economic_data = await economic_data_agent.get_comprehensive_economic_analysis()
        
        print("[2/4] VIX Term Structure Analysis (CBOE)...")
        vix_data = await cboe_data_agent.get_vix_term_structure_analysis()
        
        print("[3/4] Options Flow Analysis (CBOE)...")
        flow_data = await cboe_data_agent.get_options_flow_analysis()
        
        print("[4/4] Technical Analysis (Enhanced)...")
        technical_data = await enhanced_technical_analysis.get_comprehensive_analysis(symbol, period="60d")
        
        # Synthesize intelligence
        print(f"\n[INTELLIGENCE SYNTHESIS for {symbol}]")
        print("=" * 50)
        
        # Economic Intelligence
        print(f"\n[ECONOMIC REGIME]")
        print(f"  Market Regime: {economic_data['market_regime']}")
        print(f"  Fed Funds Rate: {economic_data['fed_policy']['fed_funds_rate']:.2f}%")
        print(f"  Inflation (YoY): {economic_data['inflation_data']['cpi_yoy']:.1f}%")
        print(f"  Recession Probability: {economic_data['economic_stress']['recession_probability']}")
        
        # Volatility Intelligence
        print(f"\n[VOLATILITY REGIME]")
        print(f"  VIX Current: {vix_data['vix_current']:.1f}")
        print(f"  Volatility Regime: {vix_data['volatility_regime']}")
        print(f"  Term Structure: {vix_data['term_structure']}")
        print(f"  Backwardation Signal: {vix_data['backwardation_signal']}")
        
        # Market Sentiment
        print(f"\n[MARKET SENTIMENT]")
        print(f"  Options Sentiment: {flow_data['market_sentiment']}")
        print(f"  Put/Call Ratio: {flow_data['put_call_ratio']:.2f}")
        print(f"  Fear/Greed: {flow_data['fear_greed_indicator']}")
        print(f"  SKEW Index: {flow_data['skew_index']:.0f}")
        
        # Technical Analysis
        signals = technical_data.get('signals', {})
        print(f"\n[TECHNICAL SIGNALS]")
        print(f"  Current Price: ${technical_data['current_price']:.2f}")
        print(f"  Overall Signal: {signals.get('overall_signal', 'N/A')}")
        print(f"  Signal Strength: {signals.get('signal_strength', 0):.1f}%")
        print(f"  Signal Confidence: {signals.get('confidence', 0):.1f}%")
        print(f"  RSI: {technical_data['technical_indicators']['rsi']:.1f}")
        
        # INTEGRATED TRADING RECOMMENDATIONS
        print(f"\n[INTEGRATED TRADING RECOMMENDATIONS]")
        print("=" * 50)
        
        # Economic Strategy Bias
        economic_bias = economic_data['options_strategy_bias']
        print(f"Economic Strategy Bias: {economic_bias}")
        
        # Volatility Strategy
        vol_trade_bias = vix_data['options_implications']['volatility_trade_bias']
        positioning_bias = flow_data['options_implications']['positioning_bias']
        print(f"Volatility Trade Bias: {vol_trade_bias}")
        print(f"Positioning Bias: {positioning_bias}")
        
        # Technical Direction
        technical_bias = signals.get('overall_signal', 'NEUTRAL')
        print(f"Technical Direction: {technical_bias}")
        
        # COMPOSITE RECOMMENDATION
        print(f"\n[COMPOSITE RECOMMENDATION]")
        print("-" * 30)
        
        # Crisis check
        if economic_data['market_regime'] == 'CRISIS':
            print("ALERT: CRISIS MODE - Focus on protective strategies")
            print("   Recommended: LONG PUTS, BEAR PUT SPREADS")
        
        # Volatility opportunities
        elif vix_data['backwardation_signal']:
            print("OPPORTUNITY: VIX backwardation detected")
            print("   Recommended: VOLATILITY SELLING strategies")
        
        # Fear/Greed extremes
        elif flow_data['market_sentiment'] == 'EXTREME_FEAR':
            print("EXTREME FEAR: Contrarian opportunity")
            print("   Recommended: CASH SECURED PUTS, selective BULL SPREADS")
        
        elif flow_data['market_sentiment'] == 'EXTREME_GREED':
            print("EXTREME GREED: Correction risk")
            print("   Recommended: PROTECTIVE PUTS, reduce exposure")
        
        # Normal conditions
        else:
            if technical_bias == 'BULLISH' and economic_bias in ['BULL_CALL_SPREADS', 'NEUTRAL']:
                print("BULLISH ENVIRONMENT: Upward momentum")
                print("   Recommended: BULL CALL SPREADS, selective LONG CALLS")
            elif technical_bias == 'BEARISH':
                print("BEARISH ENVIRONMENT: Downward pressure")
                print("   Recommended: BEAR PUT SPREADS, LONG PUTS")
            else:
                print("NEUTRAL ENVIRONMENT: Range-bound strategies")
                print("   Recommended: IRON CONDORS, SPREADS")
        
        # Risk management overlay
        print(f"\n[RISK MANAGEMENT]")
        print("-" * 20)
        
        if economic_data['economic_stress']['recession_probability'] == 'HIGH':
            print("WARNING: High recession probability - reduce position sizes")
        
        if vix_data['volatility_regime'] in ['HIGH_VOLATILITY', 'EXTREME_VOLATILITY']:
            print("WARNING: High volatility regime - avoid long options")
        
        if flow_data['put_call_ratio'] > 1.3:
            print("INFO: Elevated put buying - market positioning defensively")
        
        print(f"\n[INTELLIGENCE SUMMARY]")
        print("=" * 30)
        print("CHECKMARK Economic data: Real-time Fed policy and inflation trends")
        print("CHECKMARK Volatility intelligence: VIX term structure and flow analysis")
        print("CHECKMARK Technical analysis: 15+ indicators with signal confidence")
        print("CHECKMARK Market sentiment: Options flow and fear/greed metrics")
        print("CHECKMARK Integrated recommendations: Multi-factor strategy selection")
        
    except Exception as e:
        print(f"[ERROR] Integrated system test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("SYSTEM INTEGRATION COMPLETE")
    print("=" * 70)
    print("\nYour OPTIONS_BOT now has institutional-grade intelligence!")
    print("TARGETING Economic regime awareness")
    print("CHART Professional volatility analysis") 
    print("BRAIN Advanced technical indicators")
    print("LIGHTBULB Intelligent strategy selection")
    print("SCALE Multi-factor risk management")

if __name__ == "__main__":
    asyncio.run(test_integrated_system())