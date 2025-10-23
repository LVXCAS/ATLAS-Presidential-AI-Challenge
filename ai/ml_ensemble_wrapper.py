"""
ML Ensemble Wrapper for OPTIONS_BOT Integration
Loads pre-trained models and provides prediction interface
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MLEnsemblePredictor:
    """
    Wrapper for pre-trained ML ensemble models
    Provides predictions without needing to retrain
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.trading_models = None
        self.regime_models = None
        self.trading_scalers = None
        self.regime_scalers = None
        self.loaded = False

    def load_models(self) -> bool:
        """Load all trained models from disk"""
        try:
            # Load trading models
            trading_path = os.path.join(self.model_dir, "trading_models.pkl")
            if os.path.exists(trading_path):
                with open(trading_path, 'rb') as f:
                    self.trading_models = pickle.load(f)
                logger.info(f"Loaded trading models: {list(self.trading_models.keys())}")
            else:
                logger.warning(f"Trading models not found at {trading_path}")
                return False

            # Load trading scalers
            scaler_path = os.path.join(self.model_dir, "trading_scalers.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.trading_scalers = pickle.load(f)
                logger.info("Loaded trading scalers")

            # Load regime models (optional)
            regime_path = os.path.join(self.model_dir, "regime_models.pkl")
            if os.path.exists(regime_path):
                with open(regime_path, 'rb') as f:
                    self.regime_models = pickle.load(f)
                logger.info(f"Loaded regime models: {list(self.regime_models.keys())}")

            self.loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            return False

    def predict_direction(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict trade direction with confidence

        Args:
            features: Dict of technical indicators (RSI, MACD, volume_ratio, etc.)

        Returns:
            {
                'prediction': int (0=down, 1=up),
                'confidence': float (0-1),
                'model_votes': {
                    'rf': float,
                    'xgb': float
                }
            }
        """
        if not self.loaded:
            logger.warning("Models not loaded - call load_models() first")
            return {'prediction': 0, 'confidence': 0.5, 'model_votes': {}}

        try:
            # Convert features to DataFrame
            feature_df = pd.DataFrame([features])

            # Get predictions from each model
            predictions = {}
            confidences = {}

            # Random Forest prediction
            if 'trading_rf_clf' in self.trading_models:
                rf_model = self.trading_models['trading_rf_clf']

                # Scale features if scaler available
                if self.trading_scalers and 'trading' in self.trading_scalers:
                    scaler = self.trading_scalers['trading']
                    feature_cols = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else list(features.keys())
                    features_scaled = scaler.transform(feature_df[feature_cols])
                else:
                    features_scaled = feature_df.values

                rf_proba = rf_model.predict_proba(features_scaled)[0]
                predictions['rf'] = int(np.argmax(rf_proba))
                confidences['rf'] = float(np.max(rf_proba))

            # XGBoost prediction
            if 'trading_xgb_clf' in self.trading_models:
                xgb_model = self.trading_models['trading_xgb_clf']

                if self.trading_scalers and 'trading' in self.trading_scalers:
                    scaler = self.trading_scalers['trading']
                    feature_cols = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else list(features.keys())
                    features_scaled = scaler.transform(feature_df[feature_cols])
                else:
                    features_scaled = feature_df.values

                xgb_proba = xgb_model.predict_proba(features_scaled)[0]
                predictions['xgb'] = int(np.argmax(xgb_proba))
                confidences['xgb'] = float(np.max(xgb_proba))

            # LightGBM prediction (NEW in V2!)
            if 'trading_lgb_clf' in self.trading_models:
                lgb_model = self.trading_models['trading_lgb_clf']

                if self.trading_scalers and 'trading' in self.trading_scalers:
                    scaler = self.trading_scalers['trading']
                    feature_cols = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else list(features.keys())
                    features_scaled = scaler.transform(feature_df[feature_cols])
                else:
                    features_scaled = feature_df.values

                lgb_proba = lgb_model.predict_proba(features_scaled)[0]
                predictions['lgb'] = int(np.argmax(lgb_proba))
                confidences['lgb'] = float(np.max(lgb_proba))

            # Ensemble prediction (weighted average)
            if predictions:
                # Weight: RF 33%, XGB 33%, LGB 34% (equal weighting for V2)
                # If V1 models (no LGB), use RF 55%, XGB 45%
                if 'lgb' in predictions:
                    weights = {'rf': 0.33, 'xgb': 0.33, 'lgb': 0.34}  # V2 weights
                else:
                    weights = {'rf': 0.55, 'xgb': 0.45}  # V1 weights

                # Detect if V2 models (3-class: 0=PUT, 1=NEUTRAL, 2=CALL) or V1 (2-class: 0=DOWN, 1=UP)
                # Check number of probability outputs from first model
                first_model_name = list(predictions.keys())[0]
                if first_model_name == 'rf' and 'trading_rf_clf' in self.trading_models:
                    n_classes = len(rf_proba) if 'rf_proba' in locals() else 2
                elif first_model_name == 'xgb' and 'trading_xgb_clf' in self.trading_models:
                    n_classes = len(xgb_proba) if 'xgb_proba' in locals() else 2
                elif first_model_name == 'lgb' and 'trading_lgb_clf' in self.trading_models:
                    n_classes = len(lgb_proba) if 'lgb_proba' in locals() else 2
                else:
                    n_classes = 2

                ensemble_proba = np.zeros(n_classes)
                total_weight = 0

                for model_name, pred in predictions.items():
                    weight = weights.get(model_name, 0.33)
                    if model_name in confidences:
                        # Add weighted vote for the predicted class
                        ensemble_proba[pred] += weight * confidences[model_name]
                        total_weight += weight

                if total_weight > 0:
                    ensemble_proba /= total_weight

                final_prediction = int(np.argmax(ensemble_proba))
                final_confidence = float(np.max(ensemble_proba))

                # For V2 models (3-class), map to binary for backwards compatibility
                # 0 (PUT) = 0 (DOWN), 1 (NEUTRAL) = ignore, 2 (CALL) = 1 (UP)
                if n_classes == 3:
                    if final_prediction == 2:
                        binary_prediction = 1  # CALL -> UP
                    elif final_prediction == 0:
                        binary_prediction = 0  # PUT -> DOWN
                    else:
                        binary_prediction = 1 if ensemble_proba[2] > ensemble_proba[0] else 0  # NEUTRAL -> compare CALL vs PUT
                else:
                    binary_prediction = final_prediction

                return {
                    'prediction': binary_prediction,  # Use binary for backwards compatibility
                    'raw_prediction': final_prediction,  # Original 3-class prediction
                    'confidence': final_confidence,
                    'model_votes': predictions,
                    'model_confidences': confidences,
                    'ensemble_proba': ensemble_proba.tolist(),
                    'n_classes': n_classes,  # 2 for V1, 3 for V2
                    'model_version': 'V2' if n_classes == 3 else 'V1'
                }

            # No models available
            return {
                'prediction': 0,
                'confidence': 0.5,
                'model_votes': {},
                'error': 'No models loaded'
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'prediction': 0,
                'confidence': 0.5,
                'model_votes': {},
                'error': str(e)
            }

    def get_regime(self, market_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect current market regime

        Args:
            market_features: Market-level features (returns, volatility, etc.)

        Returns:
            {
                'regime': str ('bull', 'bear', 'sideways', 'volatile'),
                'confidence': float
            }
        """
        if not self.loaded or not self.regime_models:
            return {'regime': 'unknown', 'confidence': 0.5}

        try:
            feature_df = pd.DataFrame([market_features])
            regime_map = {0: 'bear', 1: 'sideways', 2: 'bull', 3: 'volatile'}

            # Ensemble regime predictions from all available models
            regime_votes = []
            regime_confidences = []

            # Random Forest regime
            if 'regime_rf' in self.regime_models:
                rf_model = self.regime_models['regime_rf']

                if self.regime_scalers and 'regime' in self.regime_scalers:
                    scaler = self.regime_scalers['regime']
                    features_scaled = scaler.transform(feature_df)
                else:
                    features_scaled = feature_df.values

                regime_pred = rf_model.predict(features_scaled)[0]
                regime_proba = rf_model.predict_proba(features_scaled)[0]
                regime_votes.append(regime_pred)
                regime_confidences.append(np.max(regime_proba))

            # XGBoost regime
            if 'regime_xgb' in self.regime_models:
                xgb_model = self.regime_models['regime_xgb']

                if self.regime_scalers and 'regime' in self.regime_scalers:
                    scaler = self.regime_scalers['regime']
                    features_scaled = scaler.transform(feature_df)
                else:
                    features_scaled = feature_df.values

                regime_pred = xgb_model.predict(features_scaled)[0]
                regime_proba = xgb_model.predict_proba(features_scaled)[0]
                regime_votes.append(regime_pred)
                regime_confidences.append(np.max(regime_proba))

            # LightGBM regime (NEW in V2!)
            if 'regime_lgb' in self.regime_models:
                lgb_model = self.regime_models['regime_lgb']

                if self.regime_scalers and 'regime' in self.regime_scalers:
                    scaler = self.regime_scalers['regime']
                    features_scaled = scaler.transform(feature_df)
                else:
                    features_scaled = feature_df.values

                regime_pred = lgb_model.predict(features_scaled)[0]
                regime_proba = lgb_model.predict_proba(features_scaled)[0]
                regime_votes.append(regime_pred)
                regime_confidences.append(np.max(regime_proba))

            if regime_votes:
                # Majority vote
                final_regime = int(np.bincount(regime_votes).argmax())
                avg_confidence = float(np.mean(regime_confidences))

                regime_name = regime_map.get(final_regime, 'unknown')

                return {
                    'regime': regime_name,
                    'confidence': avg_confidence,
                    'regime_votes': regime_votes,
                    'n_models': len(regime_votes)
                }

            return {'regime': 'unknown', 'confidence': 0.5}

        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return {'regime': 'unknown', 'confidence': 0.5}


# Singleton instance
_ml_ensemble = None

def get_ml_ensemble(model_dir: str = "models") -> MLEnsemblePredictor:
    """Get singleton ML ensemble instance"""
    global _ml_ensemble
    if _ml_ensemble is None:
        _ml_ensemble = MLEnsemblePredictor(model_dir)
        _ml_ensemble.load_models()
    return _ml_ensemble
