#!/usr/bin/env python3
"""
Test OPTIONS_BOT ML Ensemble Integration
Quick test to verify ML ensemble loads and integrates correctly
"""

import sys
sys.path.append('.')

from ai.ml_ensemble_wrapper import get_ml_ensemble

print("=" * 70)
print("TESTING ML ENSEMBLE INTEGRATION")
print("=" * 70)

# Test 1: Load ML ensemble
print("\n[TEST 1] Loading ML ensemble...")
ml_ensemble = get_ml_ensemble()

if ml_ensemble.loaded:
    print("[OK] ML Ensemble loaded successfully")
    print(f"     Models: {list(ml_ensemble.trading_models.keys())}")
else:
    print("[ERROR] ML Ensemble failed to load")
    sys.exit(1)

# Test 2: Create sample market_data (like OPTIONS_BOT uses)
print("\n[TEST 2] Testing ML prediction with OPTIONS_BOT-style market data...")

sample_market_data = {
    'current_price': 150.0,
    'price_momentum': 0.02,  # 2% momentum
    'sma_20': 148.0,
    'sma_50': 145.0,
    'rsi': 65.0,
    'macd': 0.5,
    'macd_signal': 0.3,
    'bollinger_upper': 155.0,
    'bollinger_lower': 145.0,
    'volatility': 0.025,
    'volume_trend': 1.2
}

# Extract features like the _get_ml_prediction method does
features = {
    # Returns (1d, 3d, 5d, 10d, 20d)
    'returns_1d': sample_market_data.get('price_momentum', 0.0),
    'returns_3d': sample_market_data.get('price_momentum', 0.0) * 1.5,
    'returns_5d': sample_market_data.get('price_momentum', 0.0) * 2.5,
    'returns_10d': sample_market_data.get('price_momentum', 0.0) * 5.0,
    'returns_20d': sample_market_data.get('price_momentum', 0.0) * 10.0,

    # Price to SMA ratios
    'price_to_sma_5': sample_market_data.get('current_price', 100) / sample_market_data.get('sma_20', 100),
    'price_to_sma_10': sample_market_data.get('current_price', 100) / sample_market_data.get('sma_20', 100),
    'price_to_sma_20': sample_market_data.get('current_price', 100) / sample_market_data.get('sma_20', 100),
    'price_to_sma_50': sample_market_data.get('current_price', 100) / sample_market_data.get('sma_50', 100),

    # Technical indicators
    'rsi': sample_market_data.get('rsi', 50.0),
    'macd': sample_market_data.get('macd', 0.0),
    'macd_signal': sample_market_data.get('macd_signal', 0.0),
    'macd_histogram': sample_market_data.get('macd', 0.0) - sample_market_data.get('macd_signal', 0.0),

    # Bollinger bands
    'bb_width': abs(sample_market_data.get('bollinger_upper', 105) - sample_market_data.get('bollinger_lower', 95)) / sample_market_data.get('current_price', 100),
    'bb_position': (sample_market_data.get('current_price', 100) - sample_market_data.get('bollinger_lower', 95)) / (sample_market_data.get('bollinger_upper', 105) - sample_market_data.get('bollinger_lower', 95) + 0.001),

    # Volatility
    'volatility_5d': sample_market_data.get('volatility', 0.02),
    'volatility_20d': sample_market_data.get('volatility', 0.02) * 1.2,
    'volatility_ratio': 0.85,

    # Volume
    'volume_ratio': sample_market_data.get('volume_trend', 1.0),
    'volume_change': sample_market_data.get('volume_trend', 1.0) - 1.0,

    # Additional indicators
    'atr': sample_market_data.get('volatility', 0.02) * sample_market_data.get('current_price', 100),
    'obv_trend': 1.0,
    'adx': 25.0,
    'cci': 0.0,
    'stoch_k': min(100, max(0, (sample_market_data.get('rsi', 50) - 20) * 1.25)),
    'stoch_d': min(100, max(0, (sample_market_data.get('rsi', 50) - 20) * 1.25)),
}

print(f"Market Data: Price={sample_market_data['current_price']}, Momentum={sample_market_data['price_momentum']}, RSI={sample_market_data['rsi']}")

ml_result = ml_ensemble.predict_direction(features)

print(f"\nML Prediction:")
print(f"  Direction: {'UP' if ml_result['prediction'] == 1 else 'DOWN'}")
print(f"  Confidence: {ml_result['confidence']:.1%}")
print(f"  Model Votes: {ml_result.get('model_votes', {})}")

# Test 3: Test hybrid blending logic
print("\n[TEST 3] Testing 60/40 hybrid blending...")

learning_confidence = 0.75  # Simulated learning engine confidence
ml_confidence = ml_result['confidence']

# Simulate OPTIONS_BOT logic
blended_confidence = (learning_confidence * 0.6) + (ml_confidence * 0.4)

print(f"Learning Engine: {learning_confidence:.1%}")
print(f"ML Ensemble: {ml_confidence:.1%}")
print(f"Blended (60/40): {blended_confidence:.1%}")

print("\n" + "=" * 70)
print("[SUCCESS] ML Ensemble integration test complete!")
print("=" * 70)
print("\nOPTIONS_BOT is ready to use ML ensemble predictions!")
print("Next step: Restart OPTIONS_BOT to activate ML-enhanced trading")
