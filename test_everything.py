#!/usr/bin/env python3
"""
Comprehensive System Test
Verify all components are working correctly
"""

import sys
import os

print("=" * 70)
print("COMPREHENSIVE SYSTEM TEST")
print("=" * 70)

tests_passed = 0
tests_failed = 0

# Test 1: ML Ensemble
print("\n[TEST 1] ML Ensemble Integration")
try:
    from ai.ml_ensemble_wrapper import get_ml_ensemble
    ml_ensemble = get_ml_ensemble()
    assert ml_ensemble.loaded, "ML ensemble not loaded"
    assert 'trading_rf_clf' in ml_ensemble.trading_models, "RF model not found"
    assert 'trading_xgb_clf' in ml_ensemble.trading_models, "XGB model not found"
    print("[PASS] ML Ensemble loaded with RF and XGB models")
    tests_passed += 1
except Exception as e:
    print(f"[FAIL] {e}")
    tests_failed += 1

# Test 2: ML Prediction
print("\n[TEST 2] ML Prediction")
try:
    test_features = {
        'returns_1d': 0.02, 'returns_3d': 0.03, 'returns_5d': 0.05, 'returns_10d': 0.08, 'returns_20d': 0.10,
        'price_to_sma_5': 1.01, 'price_to_sma_10': 1.02, 'price_to_sma_20': 1.03, 'price_to_sma_50': 1.05,
        'rsi': 65.0, 'macd': 0.5, 'macd_signal': 0.3, 'macd_histogram': 0.2,
        'bb_width': 0.05, 'bb_position': 0.6,
        'volatility_5d': 0.015, 'volatility_20d': 0.020, 'volatility_ratio': 0.75,
        'volume_ratio': 1.2, 'volume_change': 0.1,
        'atr': 2.5, 'obv_trend': 1.05, 'adx': 25.0, 'cci': 100.0, 'stoch_k': 70.0, 'stoch_d': 65.0
    }
    result = ml_ensemble.predict_direction(test_features)
    assert 'prediction' in result, "No prediction in result"
    assert 'confidence' in result, "No confidence in result"
    assert 'model_votes' in result, "No model votes in result"
    print(f"[PASS] Prediction: {'UP' if result['prediction'] == 1 else 'DOWN'}, Confidence: {result['confidence']:.1%}")
    tests_passed += 1
except Exception as e:
    print(f"[FAIL] {e}")
    tests_failed += 1

# Test 3: Model Files
print("\n[TEST 3] Model Files")
try:
    model_file = 'models/trading_models.pkl'
    assert os.path.exists(model_file), f"{model_file} not found"
    size_mb = os.path.getsize(model_file) / (1024 * 1024)
    assert size_mb > 30, f"Model file too small ({size_mb:.1f} MB)"
    print(f"[PASS] trading_models.pkl exists ({size_mb:.1f} MB)")
    tests_passed += 1
except Exception as e:
    print(f"[FAIL] {e}")
    tests_failed += 1

# Test 4: Jupyter Notebook
print("\n[TEST 4] Jupyter Notebook Installation")
try:
    import notebook
    print(f"[PASS] Jupyter Notebook {notebook.__version__} installed")
    tests_passed += 1
except Exception as e:
    print(f"[FAIL] {e}")
    tests_failed += 1

# Test 5: Visualization Libraries
print("\n[TEST 5] Visualization Libraries")
try:
    import matplotlib
    import seaborn
    print(f"[PASS] Matplotlib {matplotlib.__version__}, Seaborn {seaborn.__version__}")
    tests_passed += 1
except Exception as e:
    print(f"[FAIL] {e}")
    tests_failed += 1

# Test 6: ML Experimentation Notebook
print("\n[TEST 6] ML Experimentation Notebook")
try:
    notebook_file = 'ML_Experimentation.ipynb'
    assert os.path.exists(notebook_file), f"{notebook_file} not found"
    size_kb = os.path.getsize(notebook_file) / 1024
    assert size_kb > 10, f"Notebook file too small ({size_kb:.1f} KB)"
    print(f"[PASS] ML_Experimentation.ipynb exists ({size_kb:.1f} KB)")
    tests_passed += 1
except Exception as e:
    print(f"[FAIL] {e}")
    tests_failed += 1

# Test 7: OPTIONS_BOT Import
print("\n[TEST 7] OPTIONS_BOT ML Integration")
try:
    # Just test the import works
    import importlib.util
    spec = importlib.util.spec_from_file_location("options_bot", "OPTIONS_BOT.py")
    # Don't actually load it (would start the bot)
    assert spec is not None, "Cannot load OPTIONS_BOT.py"

    # Check that ML import is in the file
    with open('OPTIONS_BOT.py', 'r', encoding='utf-8') as f:
        content = f.read()
        assert 'from ai.ml_ensemble_wrapper import get_ml_ensemble' in content, "ML import not found"
        assert 'self.ml_ensemble' in content, "ML ensemble not initialized"
        assert '_get_ml_prediction' in content, "ML prediction method not found"

    print("[PASS] OPTIONS_BOT has ML ensemble integration")
    tests_passed += 1
except Exception as e:
    print(f"[FAIL] {e}")
    tests_failed += 1

# Test 8: Documentation Files
print("\n[TEST 8] Documentation")
try:
    docs = ['JUPYTER_QUICKSTART.md', 'ML_INTEGRATION_COMPLETE.md', 'ML_TRAINING_SESSION.md']
    for doc in docs:
        assert os.path.exists(doc), f"{doc} not found"
    print(f"[PASS] All documentation files present")
    tests_passed += 1
except Exception as e:
    print(f"[FAIL] {e}")
    tests_failed += 1

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print(f"Tests Passed: {tests_passed}")
print(f"Tests Failed: {tests_failed}")
print(f"Success Rate: {tests_passed / (tests_passed + tests_failed) * 100:.1f}%")

if tests_failed == 0:
    print("\n[SUCCESS] All systems operational!")
    print("\nReady for:")
    print("  1. ML experimentation with Jupyter")
    print("  2. OPTIONS_BOT trading with ML ensemble")
    print("=" * 70)
    sys.exit(0)
else:
    print("\n[WARNING] Some tests failed. Check errors above.")
    print("=" * 70)
    sys.exit(1)
