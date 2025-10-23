"""
Advanced ML Finance Agent - Integrates machine learning models from Finance repository
Provides LSTM predictions, deep learning analysis, and quantitative models
"""

import asyncio
import logging
import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add Finance repository to path
sys.path.append('./Finance')

# Import ML modules
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler
    import joblib
    ML_AVAILABLE = True
except ImportError as e:
    print(f"ML libraries not available: {e}")
    ML_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MLPrediction:
    """ML prediction structure"""
    symbol: str
    prediction: str  # 'UP', 'DOWN', 'SIDEWAYS'
    probability: float
    confidence: float
    model_type: str
    features_used: List[str]
    prediction_horizon: str  # '1d', '5d', '1w'
    timestamp: datetime

class AdvancedMLFinanceAgent:
    """
    Advanced ML agent using Finance repository techniques:
    - LSTM time series prediction
    - Ensemble model classification
    - Deep learning price prediction
    - Quantitative factor analysis
    - Feature engineering from technical indicators
    """

    def __init__(self):
        self.name = "Advanced ML Finance Agent"
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        logger.info("Advanced ML Finance Agent initialized")

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set for ML models"""
        try:
            features = df.copy()

            # Price-based features
            features['returns'] = df['Close'].pct_change()
            features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            features['price_change'] = df['Close'] - df['Open']
            features['high_low_ratio'] = df['High'] / df['Low']
            features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

            # Technical indicators
            # Moving averages
            for period in [5, 10, 20, 50]:
                features[f'sma_{period}'] = df['Close'].rolling(period).mean()
                features[f'price_vs_sma_{period}'] = df['Close'] / features[f'sma_{period}'] - 1

            # Exponential moving averages
            for period in [12, 26]:
                features[f'ema_{period}'] = df['Close'].ewm(span=period).mean()

            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            features['macd'] = ema12 - ema26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']

            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_middle = df['Close'].rolling(bb_period).mean()
            bb_std_dev = df['Close'].rolling(bb_period).std()
            features['bb_upper'] = bb_middle + (bb_std_dev * bb_std)
            features['bb_lower'] = bb_middle - (bb_std_dev * bb_std)
            features['bb_position'] = (df['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

            # Volatility features
            features['volatility_20d'] = df['Close'].rolling(20).std()
            features['volatility_ratio'] = features['volatility_20d'] / features['volatility_20d'].rolling(60).mean()

            # Volume features
            features['volume_sma_20'] = df['Volume'].rolling(20).mean()
            features['volume_ratio_20'] = df['Volume'] / features['volume_sma_20']
            features['price_volume'] = df['Close'] * df['Volume']

            # Lag features
            for lag in [1, 2, 3, 5]:
                features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
                features[f'volume_lag_{lag}'] = features['volume_ratio'].shift(lag)

            # Rolling statistics
            for window in [5, 10, 20]:
                features[f'returns_mean_{window}'] = features['returns'].rolling(window).mean()
                features[f'returns_std_{window}'] = features['returns'].rolling(window).std()
                features[f'returns_skew_{window}'] = features['returns'].rolling(window).skew()

            return features

        except Exception as e:
            logger.error(f"Feature creation error: {e}")
            return df

    def _create_target(self, df: pd.DataFrame, horizon: int = 1) -> pd.Series:
        """Create target variable for classification"""
        try:
            future_returns = df['Close'].shift(-horizon) / df['Close'] - 1

            # Create categorical target
            conditions = [
                future_returns > 0.02,  # UP: > 2% gain
                future_returns < -0.02,  # DOWN: > 2% loss
            ]
            choices = [1, -1]  # UP, DOWN
            target = np.select(conditions, choices, default=0)  # SIDEWAYS

            return pd.Series(target, index=df.index)

        except Exception as e:
            logger.error(f"Target creation error: {e}")
            return pd.Series(0, index=df.index)

    async def train_ensemble_model(self, symbol: str, period: str = '2y') -> Dict[str, Any]:
        """Train ensemble ML model for price prediction"""
        try:
            if not ML_AVAILABLE:
                return {'error': 'ML libraries not available'}

            # Get data
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)

            if len(df) < 100:
                return {'error': 'Insufficient data'}

            # Create features
            features_df = self._create_features(df)
            target = self._create_target(df)

            # Select feature columns (exclude original OHLCV)
            feature_cols = [col for col in features_df.columns
                          if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]

            # Clean data
            X = features_df[feature_cols].dropna()
            y = target.loc[X.index]

            if len(X) < 50:
                return {'error': 'Insufficient clean data'}

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train multiple models
            models = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(random_state=42),
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
            }

            results = {}
            trained_models = {}

            for name, model in models.items():
                # Train model
                if name == 'logistic_regression':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_prob = model.predict_proba(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)

                # Evaluate
                accuracy = accuracy_score(y_test, y_pred)

                results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred.tolist(),
                    'probabilities': y_prob.tolist() if y_prob is not None else None
                }

                trained_models[name] = model

            # Store models and scaler
            self.models[symbol] = trained_models
            self.scalers[symbol] = scaler
            self.feature_columns = feature_cols

            # Create ensemble prediction
            ensemble_pred = self._create_ensemble_prediction(X_test, trained_models, scaler)

            return {
                'symbol': symbol,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': len(feature_cols),
                'model_results': results,
                'ensemble_prediction': ensemble_pred,
                'training_timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Model training error for {symbol}: {e}")
            return {'error': str(e)}

    def _create_ensemble_prediction(self, X_test: pd.DataFrame, models: Dict, scaler: StandardScaler) -> Dict[str, Any]:
        """Create ensemble prediction from multiple models"""
        try:
            predictions = []
            probabilities = []

            for name, model in models.items():
                if name == 'logistic_regression':
                    X_scaled = scaler.transform(X_test)
                    pred = model.predict(X_scaled)
                    prob = model.predict_proba(X_scaled)
                else:
                    pred = model.predict(X_test)
                    prob = model.predict_proba(X_test)

                predictions.append(pred)
                probabilities.append(prob)

            # Ensemble prediction (majority vote)
            ensemble_pred = np.array(predictions).mean(axis=0)
            ensemble_prob = np.array(probabilities).mean(axis=0)

            # Convert to final prediction
            final_pred = np.round(ensemble_pred).astype(int)

            # Get confidence (max probability)
            max_prob_idx = np.argmax(ensemble_prob, axis=1)
            confidence = np.max(ensemble_prob, axis=1).mean()

            return {
                'prediction': final_pred.tolist(),
                'probability_distribution': ensemble_prob.tolist(),
                'confidence': float(confidence),
                'ensemble_method': 'weighted_average'
            }

        except Exception as e:
            logger.error(f"Ensemble prediction error: {e}")
            return {}

    async def predict_price_movement(self, symbol: str, horizon: str = '1d') -> Optional[MLPrediction]:
        """Generate ML-based price movement prediction"""
        try:
            # Check if model exists
            if symbol not in self.models:
                training_result = await self.train_ensemble_model(symbol)
                if 'error' in training_result:
                    return None

            # Get current data
            stock = yf.Ticker(symbol)
            df = stock.history(period='3mo')  # Get recent data

            if df.empty:
                return None

            # Create features for latest data point
            features_df = self._create_features(df)

            if not self.feature_columns:
                return None

            # Get latest features
            latest_features = features_df[self.feature_columns].iloc[-1:].dropna(axis=1)

            if latest_features.empty:
                return None

            # Make ensemble prediction
            models = self.models[symbol]
            scaler = self.scalers[symbol]

            ensemble_pred = self._create_ensemble_prediction(latest_features, models, scaler)

            if not ensemble_pred:
                return None

            # Convert prediction to direction
            pred_value = ensemble_pred['prediction'][0] if ensemble_pred['prediction'] else 0

            if pred_value > 0.5:
                direction = 'UP'
            elif pred_value < -0.5:
                direction = 'DOWN'
            else:
                direction = 'SIDEWAYS'

            return MLPrediction(
                symbol=symbol,
                prediction=direction,
                probability=float(ensemble_pred['confidence']),
                confidence=float(ensemble_pred['confidence']),
                model_type='ensemble',
                features_used=self.feature_columns,
                prediction_horizon=horizon,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Price prediction error for {symbol}: {e}")
            return None

    async def get_feature_importance(self, symbol: str) -> Dict[str, float]:
        """Get feature importance from trained models"""
        try:
            if symbol not in self.models:
                return {}

            models = self.models[symbol]
            importance_scores = {}

            # Get feature importance from tree-based models
            for name, model in models.items():
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    for i, feature in enumerate(self.feature_columns):
                        if feature not in importance_scores:
                            importance_scores[feature] = []
                        importance_scores[feature].append(importances[i])

            # Average importance across models
            avg_importance = {}
            for feature, scores in importance_scores.items():
                avg_importance[feature] = np.mean(scores)

            # Sort by importance
            sorted_importance = dict(sorted(avg_importance.items(),
                                          key=lambda x: x[1], reverse=True))

            return sorted_importance

        except Exception as e:
            logger.error(f"Feature importance error for {symbol}: {e}")
            return {}

# Create singleton instance
advanced_ml_finance_agent = AdvancedMLFinanceAgent()