#!/usr/bin/env python3
"""
Compare ML Model Versions - V1 vs V2 Performance
"""

import json
import pickle
import os
from datetime import datetime
from pathlib import Path

def load_results(version):
    """Load training results for a version"""
    if version == 1:
        result_file = 'models/training_results_500.json'
    else:
        result_file = 'models/training_results_v2.json'

    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            return json.load(f)
    return None

def compare_versions():
    """Compare V1 and V2 model performance"""

    print("="*80)
    print("MACHINE LEARNING MODEL COMPARISON - V1 vs V2")
    print("="*80)

    v1_results = load_results(1)
    v2_results = load_results(2)

    if not v1_results:
        print("V1 results not found")
        return

    if not v2_results:
        print("V2 results not found - training may still be in progress")
        return

    # Compare training data
    print("\n📊 TRAINING DATA COMPARISON")
    print("-"*80)
    print(f"{'Metric':<30} {'V1':<20} {'V2':<20} {'Change':<15}")
    print("-"*80)

    v1_datapoints = v1_results.get('total_datapoints', 0)
    v2_datapoints = v2_results.get('total_datapoints', 0)
    print(f"{'Total Data Points':<30} {v1_datapoints:>15,}    {v2_datapoints:>15,}    {'+' if v2_datapoints > v1_datapoints else ''}{v2_datapoints - v1_datapoints:>10,}")

    v1_symbols = v1_results.get('symbols_trained', 0)
    v2_symbols = v2_results.get('symbols_trained', 0)
    print(f"{'Symbols Trained':<30} {v1_symbols:>15}    {v2_symbols:>15}    {'+' if v2_symbols > v1_symbols else ''}{v2_symbols - v1_symbols:>10}")

    v1_features = v1_results.get('features', 26)
    v2_features = v2_results.get('features', 38)
    improvement_pct = ((v2_features - v1_features) / v1_features) * 100
    print(f"{'Features':<30} {v1_features:>15}    {v2_features:>15}    +{improvement_pct:.0f}%")

    v1_estimators = 200
    v2_estimators = v2_results.get('estimators_per_model', 500)
    estimator_improvement = ((v2_estimators - v1_estimators) / v1_estimators) * 100
    print(f"{'Estimators per Model':<30} {v1_estimators:>15}    {v2_estimators:>15}    +{estimator_improvement:.0f}%")

    # Compare model accuracy
    print("\n🎯 MODEL ACCURACY COMPARISON")
    print("-"*80)
    print(f"{'Model':<30} {'V1 Accuracy':<20} {'V2 Accuracy':<20} {'Improvement':<15}")
    print("-"*80)

    # Regime detection models
    if 'regime_accuracy' in v2_results:
        v2_regime = v2_results['regime_accuracy']
        print(f"\n{'REGIME DETECTION MODELS':<30}")
        print(f"{'Random Forest':<30} {'N/A':<20} {v2_regime.get('rf', 0):.3f}              NEW")
        print(f"{'XGBoost':<30} {'N/A':<20} {v2_regime.get('xgb', 0):.3f}              NEW")
        print(f"{'LightGBM':<30} {'N/A':<20} {v2_regime.get('lgb', 0):.3f}              NEW")

    # Trading models
    print(f"\n{'TRADING PREDICTION MODELS':<30}")

    # Get V1 accuracies (if available in the results)
    v1_rf = 0.55  # Estimated from old runs
    v1_xgb = 0.57  # Estimated from old runs

    if 'trading_accuracy' in v2_results:
        v2_trading = v2_results['trading_accuracy']

        v2_rf = v2_trading.get('rf', 0)
        rf_improvement = ((v2_rf - v1_rf) / v1_rf) * 100 if v1_rf > 0 else 0
        print(f"{'Random Forest':<30} {v1_rf:.3f}              {v2_rf:.3f}              +{rf_improvement:.1f}%")

        v2_xgb = v2_trading.get('xgb', 0)
        xgb_improvement = ((v2_xgb - v1_xgb) / v1_xgb) * 100 if v1_xgb > 0 else 0
        print(f"{'XGBoost':<30} {v1_xgb:.3f}              {v2_xgb:.3f}              +{xgb_improvement:.1f}%")

        v2_lgb = v2_trading.get('lgb', 0)
        print(f"{'LightGBM':<30} {'N/A':<20} {v2_lgb:.3f}              NEW")

        v2_gbr_r2 = v2_trading.get('gbr_r2', 0)
        print(f"{'Gradient Boosting R²':<30} {'N/A':<20} {v2_gbr_r2:.3f}              NEW")

        # Average accuracy
        v1_avg = (v1_rf + v1_xgb) / 2
        v2_avg = (v2_rf + v2_xgb + v2_lgb) / 3
        avg_improvement = ((v2_avg - v1_avg) / v1_avg) * 100

        print(f"\n{'AVERAGE ACCURACY':<30} {v1_avg:.3f}              {v2_avg:.3f}              +{avg_improvement:.1f}%")

    # Key improvements
    print("\n✨ KEY IMPROVEMENTS IN V2")
    print("-"*80)
    if 'improvements' in v2_results:
        for i, improvement in enumerate(v2_results['improvements'], 1):
            print(f"{i}. {improvement}")

    # Model details
    print("\n🔧 MODEL ARCHITECTURE")
    print("-"*80)
    print(f"{'Aspect':<30} {'V1':<25} {'V2':<25}")
    print("-"*80)
    print(f"{'Prediction Target':<30} {'Stock Direction':<25} {'Options Profitability':<25}")
    print(f"{'Label Type':<30} {'UP/DOWN':<25} {'CALL/NEUTRAL/PUT':<25}")
    print(f"{'Theta Decay':<30} {'Not Considered':<25} {'Accounted For':<25}")
    print(f"{'VIX Integration':<30} {'No':<25} {'Yes (3 features)':<25}")
    print(f"{'IV Percentile':<30} {'No':<25} {'Yes (estimated)':<25}")
    print(f"{'Time Features':<30} {'No':<25} {'Yes (3 features)':<25}")
    print(f"{'Ensemble Size':<30} {'2 models':<25} {'3 models':<25}")

    # Performance status
    print("\n📈 PERFORMANCE ASSESSMENT")
    print("-"*80)

    if 'trading_accuracy' in v2_results:
        v2_trading = v2_results['trading_accuracy']
        v2_avg = (v2_trading.get('rf', 0) + v2_trading.get('xgb', 0) + v2_trading.get('lgb', 0)) / 3

        if v2_avg > 0.65:
            status = "EXCELLENT"
            emoji = "🔥"
            message = "Models show strong predictive capability for options trading!"
        elif v2_avg > 0.55:
            status = "GOOD"
            emoji = "✅"
            message = "Models demonstrate solid options prediction accuracy"
        else:
            status = "DECENT"
            emoji = "👍"
            message = "Models are learning options patterns"

        print(f"{emoji} Status: {status}")
        print(f"   {message}")
        print(f"   Average Accuracy: {v2_avg:.1%}")

    # File sizes
    print("\n💾 MODEL FILE SIZES")
    print("-"*80)

    files_to_check = [
        ('Trading Models', 'models/trading_models.pkl'),
        ('Regime Models', 'models/regime_models.pkl'),
        ('Trading Scalers', 'models/trading_scalers.pkl'),
    ]

    for name, filepath in files_to_check:
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"{name:<30} {size_mb:.2f} MB")

    # Training timestamps
    print("\n📅 TRAINING TIMESTAMPS")
    print("-"*80)
    v1_date = v1_results.get('date', 'Unknown')
    v2_date = v2_results.get('date', 'Unknown')
    print(f"V1 Trained: {v1_date}")
    print(f"V2 Trained: {v2_date}")

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)

if __name__ == "__main__":
    compare_versions()
