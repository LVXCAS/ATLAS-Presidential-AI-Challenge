#!/usr/bin/env python3
"""Test ML Ensemble Model Loading and Prediction"""

from ai.ml_ensemble_wrapper import get_ml_ensemble
import numpy as np

print("=" * 70)
print("TESTING ML ENSEMBLE MODEL LOADING")
print("=" * 70)

# Initialize ensemble
ml_ensemble = get_ml_ensemble()

if ml_ensemble.loaded:
    print("\n[OK] ML Ensemble loaded successfully!")
    print(f"     Trading models: {list(ml_ensemble.trading_models.keys())}")

    # Test prediction with sample features
    print("\n" + "=" * 70)
    print("TESTING PREDICTION")
    print("=" * 70)

    # All 26 features required by the model
    test_features = {
        'returns_1d': 0.02,
        'returns_3d': 0.04,
        'returns_5d': 0.05,
        'returns_10d': 0.08,
        'returns_20d': 0.10,
        'price_to_sma_5': 1.01,
        'price_to_sma_10': 1.02,
        'price_to_sma_20': 1.03,
        'price_to_sma_50': 1.05,
        'rsi': 65.0,
        'macd': 0.5,
        'macd_signal': 0.3,
        'macd_histogram': 0.2,
        'bb_width': 0.05,
        'bb_position': 0.6,
        'volatility_5d': 0.015,
        'volatility_20d': 0.020,
        'volatility_ratio': 0.75,
        'volume_ratio': 1.2,
        'volume_change': 0.1,
        'atr': 2.5,
        'obv_trend': 1.05,
        'adx': 25.0,
        'cci': 100.0,
        'stoch_k': 70.0,
        'stoch_d': 65.0
    }

    print(f"\nTest Features: {test_features}")

    result = ml_ensemble.predict_direction(test_features)

    print(f"\nPrediction Result:")
    print(f"  Direction: {'UP' if result['prediction'] == 1 else 'DOWN'}")
    print(f"  Confidence: {result['confidence']:.1%}")
    if 'model_votes' in result:
        print(f"  Model Votes: {result['model_votes']}")
    if 'model_confidences' in result:
        print(f"  Model Confidences: {result['model_confidences']}")

    print("\n[OK] ML Ensemble is working correctly!")
    print("\nReady for integration into OPTIONS_BOT!")

else:
    print("\n[ERROR] Failed to load ML ensemble")
    print("Check that models/trading_models.pkl exists")

print("\n" + "=" * 70)
