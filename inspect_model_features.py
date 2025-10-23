#!/usr/bin/env python3
"""Inspect what features the trained model expects"""

import pickle

with open('models/trading_models.pkl', 'rb') as f:
    models = pickle.load(f)

print("Models in file:", list(models.keys()))

# Check RF model
rf_model = models['trading_rf_clf']
print(f"\nRandomForest expects {rf_model.n_features_in_} features")

if hasattr(rf_model, 'feature_names_in_'):
    print("\nFeature names:")
    for i, name in enumerate(rf_model.feature_names_in_, 1):
        print(f"  {i}. {name}")
else:
    print("\n(Feature names not stored in model)")
