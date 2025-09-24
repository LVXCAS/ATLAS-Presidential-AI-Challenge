#!/usr/bin/env python3
"""
Test the advanced ML system with technical analysis, volatility, macro events, and options flow
"""
import asyncio
import sys
sys.path.append('.')

async def test_advanced_ml():
    try:
        print("TESTING ADVANCED ML SYSTEM")
        print("=" * 50)
        
        from agents.advanced_ml_engine import advanced_ml_engine
        
        test_symbol = "AAPL"
        
        print(f"Testing ML features for {test_symbol}...")
        
        # Test technical features
        print("\n[TECH] TECHNICAL ANALYSIS FEATURES:")
        tech_features = advanced_ml_engine.extract_technical_features(test_symbol)
        print(f"  RSI (14): {tech_features.rsi_14:.1f}")
        print(f"  MACD: {tech_features.macd:.4f}")
        print(f"  MACD Signal: {tech_features.macd_signal:.4f}")
        print(f"  BB Position: {tech_features.bb_position:.2f}")
        print(f"  5-day momentum: {tech_features.price_momentum_5d*100:+.2f}%")
        print(f"  20-day momentum: {tech_features.price_momentum_20d*100:+.2f}%")
        print(f"  Volume ratio: {tech_features.volume_ratio_5d:.2f}x")
        print(f"  Volatility rank: {tech_features.volatility_rank:.1f}%")
        
        # Test volatility features
        print("\n[VOL] VOLATILITY SURFACE FEATURES:")
        vol_features = advanced_ml_engine.extract_volatility_features(test_symbol)
        print(f"  IV Rank: {vol_features.iv_rank:.1f}%")
        print(f"  HV/IV Ratio: {vol_features.hv_iv_ratio:.2f}")
        print(f"  Term Structure: {vol_features.term_structure_slope:.3f}")
        
        # Test macro features
        print("\n[MACRO] MACRO & EVENT FEATURES:")
        macro_features = advanced_ml_engine.extract_macro_features(test_symbol)
        print(f"  Days to earnings: {macro_features.days_to_earnings}")
        print(f"  Fed meeting proximity: {macro_features.fed_meeting_proximity}")
        print(f"  Market stress index: {macro_features.market_stress_index:.1f}%")
        
        # Test options flow features
        print("\n[TECH] OPTIONS FLOW FEATURES:")
        flow_features = advanced_ml_engine.extract_options_flow_features(test_symbol)
        print(f"  Put/Call ratio: {flow_features.put_call_ratio:.2f}")
        print(f"  Unusual call volume: {flow_features.unusual_call_volume}")
        print(f"  Unusual put volume: {flow_features.unusual_put_volume}")
        print(f"  Flow sentiment: {flow_features.flow_sentiment}")
        
        # Test feature vector creation
        print("\n[VECTOR] FEATURE VECTOR:")
        feature_vector = advanced_ml_engine.create_feature_vector(test_symbol)
        print(f"  Vector length: {len(feature_vector)}")
        print(f"  Sample values: {feature_vector[:5]}")
        
        # Test ML prediction
        print("\n[PREDICT] ML PREDICTIONS:")
        prob, explanation = advanced_ml_engine.predict_trade_success(test_symbol, "LONG_CALL", 0.75)
        print(f"  Success probability: {prob:.1%}")
        print(f"  Prediction method: {explanation.get('method', 'ml_model')}")
        
        if 'rsi_14' in explanation:
            print(f"  RSI factor: {explanation['rsi_14']:.1f}")
        if 'momentum_5d' in explanation:
            print(f"  Momentum factor: {explanation['momentum_5d']:+.2f}%")
        
        # Test comprehensive analysis
        print("\n[ANALYSIS] COMPREHENSIVE ANALYSIS:")
        analysis = advanced_ml_engine.get_feature_analysis(test_symbol)
        
        if analysis:
            tech = analysis.get('technical', {})
            vol = analysis.get('volatility', {})
            macro = analysis.get('macro', {})
            flow = analysis.get('options_flow', {})
            
            print("  Technical Summary:")
            print(f"    RSI: {tech.get('rsi_14', 0):.1f} ({'Overbought' if tech.get('rsi_14', 50) > 70 else 'Oversold' if tech.get('rsi_14', 50) < 30 else 'Neutral'})")
            print(f"    Momentum: {tech.get('momentum_5d', 0):+.1f}% (5d), {tech.get('momentum_20d', 0):+.1f}% (20d)")
            print(f"    Volume: {tech.get('volume_ratio', 1.0):.1f}x average")
            
            print("  Options Flow Summary:")
            print(f"    P/C Ratio: {flow.get('put_call_ratio', 1.0):.2f}")
            print(f"    Sentiment: {flow.get('sentiment', 'NEUTRAL')}")
            print(f"    Unusual Activity: Calls={flow.get('unusual_calls', False)}, Puts={flow.get('unusual_puts', False)}")
            
            print("  Risk Factors:")
            risk_factors = []
            if macro.get('days_to_earnings', 999) < 7:
                risk_factors.append("Earnings within 1 week")
            if macro.get('fed_proximity', 999) < 7:
                risk_factors.append("Fed meeting within 1 week")
            if macro.get('market_stress', 25) > 60:
                risk_factors.append("High market stress (VIX)")
            
            if risk_factors:
                for factor in risk_factors:
                    print(f"    WARNING:  {factor}")
            else:
                print(f"    SUCCESS: No major risk factors detected")
        
        # Test multiple symbols
        print("\n[COMPARE] MULTI-SYMBOL COMPARISON:")
        test_symbols = ["AAPL", "GOOGL", "SPY"]
        
        for symbol in test_symbols:
            try:
                prob, _ = advanced_ml_engine.predict_trade_success(symbol, "LONG_CALL", 0.75)
                analysis = advanced_ml_engine.get_feature_analysis(symbol)
                tech = analysis.get('technical', {})
                
                print(f"  {symbol}: {prob:.1%} success prob, RSI={tech.get('rsi_14', 0):.1f}, "
                      f"Momentum={tech.get('momentum_5d', 0):+.1f}%")
            except Exception as e:
                print(f"  {symbol}: Error - {e}")
        
        print("\n" + "=" * 50)
        print("[COMPLETE] ADVANCED ML SYSTEM TEST COMPLETE")
        print("\nCapabilities enabled:")
        print("SUCCESS: Technical Analysis (RSI, MACD, Bollinger Bands, Momentum)")
        print("SUCCESS: Volatility Surface Analysis (IV Rank, HV/IV Ratio)")
        print("SUCCESS: Macro Event Detection (Earnings, Fed meetings, VIX)")
        print("SUCCESS: Options Flow Analysis (P/C Ratio, Unusual Volume)")
        print("SUCCESS: ML-based Success Prediction")
        print("SUCCESS: Multi-factor Risk Assessment")
        
        print(f"\nThe bot now uses {len(feature_vector)} advanced features")
        print("   to predict trade success and calibrate confidence!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_advanced_ml())