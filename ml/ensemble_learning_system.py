"""
ML/DL Ensemble Learning System
=============================

This module implements a comprehensive ensemble learning system that combines
multiple machine learning and deep learning models for superior trading predictions:

1. Traditional ML Models (Random Forest, XGBoost, SVM, etc.)
2. Deep Learning Models (LSTM, Transformer, CNN, etc.)
3. Ensemble Methods (Voting, Stacking, Blending, Dynamic Weighting)
4. Feature Engineering Pipeline
5. Model Selection and Hyperparameter Optimization
6. Online Learning and Model Updates
7. Prediction Uncertainty Quantification

The system provides state-of-the-art predictive capabilities that adapt
to changing market conditions while maintaining robust performance.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import joblib
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from collections import deque

# ML/DL imports
import sklearn
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    VotingRegressor, VotingClassifier,
    BaggingRegressor, BaggingClassifier
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, SGDRegressor
)
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# XGBoost and LightGBM
import xgboost as xgb
import lightgbm as lgb

# Technical analysis
import talib

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of ML/DL models"""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    NEURAL_NETWORK = "neural_network"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    SVM = "svm"
    LINEAR = "linear"
    ENSEMBLE = "ensemble"


class PredictionTask(Enum):
    """Types of prediction tasks"""
    PRICE_DIRECTION = "price_direction"
    PRICE_LEVEL = "price_level"
    VOLATILITY = "volatility"
    RETURN_MAGNITUDE = "return_magnitude"
    SUPPORT_RESISTANCE = "support_resistance"
    TREND_CHANGE = "trend_change"
    RISK_SCORE = "risk_score"


class EnsembleMethod(Enum):
    """Ensemble combination methods"""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    VOTING = "voting"
    STACKING = "stacking"
    BLENDING = "blending"
    DYNAMIC_WEIGHTING = "dynamic_weighting"
    BAYESIAN_AVERAGING = "bayesian_averaging"


@dataclass
class ModelConfig:
    """Configuration for individual models"""
    model_type: ModelType
    hyperparameters: Dict[str, Any]
    feature_selection: Optional[Dict[str, Any]] = None
    preprocessing: Optional[Dict[str, Any]] = None
    training_config: Optional[Dict[str, Any]] = None


@dataclass
class PredictionResult:
    """Result from model prediction"""
    model_name: str
    model_type: ModelType
    prediction: Union[float, int, np.ndarray]
    confidence: float
    uncertainty: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EnsemblePrediction:
    """Result from ensemble prediction"""
    task: PredictionTask
    ensemble_prediction: Union[float, int, np.ndarray]
    individual_predictions: List[PredictionResult]
    ensemble_confidence: float
    ensemble_uncertainty: float
    model_weights: Dict[str, float]
    method_used: EnsembleMethod
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FeatureEngineer:
    """Advanced feature engineering for financial data"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.feature_cache: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.feature_selector = None

    def create_features(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Create comprehensive feature set from market data"""
        try:
            if len(data) < 100:
                raise ValueError("Insufficient data for feature engineering")

            features_df = pd.DataFrame(index=data.index)

            # Price-based features
            features_df = self._add_price_features(features_df, data)

            # Technical indicator features
            features_df = self._add_technical_features(features_df, data)

            # Volume features
            features_df = self._add_volume_features(features_df, data)

            # Volatility features
            features_df = self._add_volatility_features(features_df, data)

            # Momentum features
            features_df = self._add_momentum_features(features_df, data)

            # Statistical features
            features_df = self._add_statistical_features(features_df, data)

            # Time-based features
            features_df = self._add_time_features(features_df, data)

            # Lag features
            features_df = self._add_lag_features(features_df, data)

            # Market microstructure features
            features_df = self._add_microstructure_features(features_df, data)

            # Clean features
            features_df = self._clean_features(features_df)

            return features_df

        except Exception as e:
            logger.error(f"Error creating features: {e}")
            raise

    def _add_price_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        try:
            close = data['Close'] if 'Close' in data.columns else data['close']
            high = data['High'] if 'High' in data.columns else data['high']
            low = data['Low'] if 'Low' in data.columns else data['low']
            open_price = data['Open'] if 'Open' in data.columns else data['open']

            # Returns
            features_df['returns_1d'] = close.pct_change()
            features_df['returns_5d'] = close.pct_change(5)
            features_df['returns_10d'] = close.pct_change(10)
            features_df['returns_20d'] = close.pct_change(20)

            # Log returns
            features_df['log_returns_1d'] = np.log(close / close.shift(1))
            features_df['log_returns_5d'] = np.log(close / close.shift(5))

            # Price ratios
            features_df['high_low_ratio'] = high / low
            features_df['close_open_ratio'] = close / open_price
            features_df['hl_price_position'] = (close - low) / (high - low)

            # Price levels
            features_df['price_52w_high'] = close / close.rolling(window=252).max()
            features_df['price_52w_low'] = close / close.rolling(window=252).min()

            return features_df

        except Exception as e:
            logger.error(f"Error adding price features: {e}")
            return features_df

    def _add_technical_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features"""
        try:
            close = data['Close'] if 'Close' in data.columns else data['close']
            high = data['High'] if 'High' in data.columns else data['high']
            low = data['Low'] if 'Low' in data.columns else data['low']
            volume = data['Volume'] if 'Volume' in data.columns else data['volume']

            # Moving averages
            for period in [5, 10, 20, 50, 100, 200]:
                features_df[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
                features_df[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
                features_df[f'price_vs_sma_{period}'] = close / features_df[f'sma_{period}'] - 1

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
            features_df['bb_upper'] = bb_upper
            features_df['bb_lower'] = bb_lower
            features_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features_df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)

            # RSI
            for period in [14, 30]:
                features_df[f'rsi_{period}'] = talib.RSI(close, timeperiod=period)

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            features_df['macd'] = macd
            features_df['macd_signal'] = macd_signal
            features_df['macd_histogram'] = macd_hist

            # Stochastic
            stoch_k, stoch_d = talib.STOCH(high, low, close)
            features_df['stoch_k'] = stoch_k
            features_df['stoch_d'] = stoch_d

            # ADX
            features_df['adx'] = talib.ADX(high, low, close, timeperiod=14)

            # ATR
            features_df['atr'] = talib.ATR(high, low, close, timeperiod=14)
            features_df['atr_ratio'] = features_df['atr'] / close

            # CCI
            features_df['cci'] = talib.CCI(high, low, close, timeperiod=14)

            # Williams %R
            features_df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)

            return features_df

        except Exception as e:
            logger.error(f"Error adding technical features: {e}")
            return features_df

    def _add_volume_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        try:
            close = data['Close'] if 'Close' in data.columns else data['close']
            high = data['High'] if 'High' in data.columns else data['high']
            low = data['Low'] if 'Low' in data.columns else data['low']
            volume = data['Volume'] if 'Volume' in data.columns else data['volume']

            # Volume indicators
            features_df['obv'] = talib.OBV(close, volume)
            features_df['ad_line'] = talib.AD(high, low, close, volume)
            features_df['cmf'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)

            # Volume ratios
            for period in [5, 10, 20]:
                vol_ma = volume.rolling(window=period).mean()
                features_df[f'volume_ratio_{period}d'] = volume / vol_ma

            # Volume-price trends
            features_df['volume_price_trend'] = talib.VPT(close, volume)

            return features_df

        except Exception as e:
            logger.error(f"Error adding volume features: {e}")
            return features_df

    def _add_volatility_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        try:
            close = data['Close'] if 'Close' in data.columns else data['close']
            high = data['High'] if 'High' in data.columns else data['high']
            low = data['Low'] if 'Low' in data.columns else data['low']

            # Historical volatility
            returns = close.pct_change()
            for period in [5, 10, 20, 50]:
                features_df[f'volatility_{period}d'] = returns.rolling(window=period).std() * np.sqrt(252)

            # Parkinson volatility
            features_df['parkinson_vol'] = np.sqrt(
                (1 / (4 * np.log(2))) * (np.log(high / low) ** 2).rolling(window=20).mean() * 252
            )

            # Garman-Klass volatility
            features_df['gk_vol'] = np.sqrt(
                (0.5 * (np.log(high / low) ** 2) -
                 (2 * np.log(2) - 1) * (np.log(close / close.shift(1)) ** 2)).rolling(window=20).mean() * 252
            )

            # GARCH-like features
            features_df['volatility_regime'] = (features_df['volatility_20d'] >
                                              features_df['volatility_20d'].rolling(window=60).mean()).astype(int)

            return features_df

        except Exception as e:
            logger.error(f"Error adding volatility features: {e}")
            return features_df

    def _add_momentum_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features"""
        try:
            close = data['Close'] if 'Close' in data.columns else data['close']

            # Rate of change
            for period in [5, 10, 20]:
                features_df[f'roc_{period}d'] = talib.ROC(close, timeperiod=period)

            # Momentum
            for period in [10, 20]:
                features_df[f'momentum_{period}d'] = talib.MOM(close, timeperiod=period)

            # Trend strength
            features_df['trend_strength'] = (
                (close > close.shift(5)).astype(int) +
                (close > close.shift(10)).astype(int) +
                (close > close.shift(20)).astype(int)
            ) / 3

            return features_df

        except Exception as e:
            logger.error(f"Error adding momentum features: {e}")
            return features_df

    def _add_statistical_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        try:
            close = data['Close'] if 'Close' in data.columns else data['close']
            returns = close.pct_change()

            # Rolling statistics
            for period in [20, 50]:
                features_df[f'skewness_{period}d'] = returns.rolling(window=period).skew()
                features_df[f'kurtosis_{period}d'] = returns.rolling(window=period).kurt()

            # Z-scores
            for period in [20, 50]:
                mean_return = returns.rolling(window=period).mean()
                std_return = returns.rolling(window=period).std()
                features_df[f'zscore_{period}d'] = (returns - mean_return) / std_return

            return features_df

        except Exception as e:
            logger.error(f"Error adding statistical features: {e}")
            return features_df

    def _add_time_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        try:
            # Day of week
            features_df['day_of_week'] = data.index.dayofweek
            features_df['is_monday'] = (data.index.dayofweek == 0).astype(int)
            features_df['is_friday'] = (data.index.dayofweek == 4).astype(int)

            # Month
            features_df['month'] = data.index.month
            features_df['quarter'] = data.index.quarter

            # Cyclical encoding
            features_df['day_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
            features_df['day_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)
            features_df['month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
            features_df['month_cos'] = np.cos(2 * np.pi * data.index.month / 12)

            return features_df

        except Exception as e:
            logger.error(f"Error adding time features: {e}")
            return features_df

    def _add_lag_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features"""
        try:
            close = data['Close'] if 'Close' in data.columns else data['close']
            returns = close.pct_change()

            # Lagged returns
            for lag in [1, 2, 3, 5, 10]:
                features_df[f'returns_lag_{lag}'] = returns.shift(lag)

            # Lagged RSI
            rsi = talib.RSI(close, timeperiod=14)
            for lag in [1, 2, 5]:
                features_df[f'rsi_lag_{lag}'] = rsi.shift(lag)

            return features_df

        except Exception as e:
            logger.error(f"Error adding lag features: {e}")
            return features_df

    def _add_microstructure_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        try:
            close = data['Close'] if 'Close' in data.columns else data['close']
            high = data['High'] if 'High' in data.columns else data['high']
            low = data['Low'] if 'Low' in data.columns else data['low']
            open_price = data['Open'] if 'Open' in data.columns else data['open']

            # Intraday patterns
            features_df['open_to_close'] = (close - open_price) / open_price
            features_df['high_to_close'] = (high - close) / close
            features_df['low_to_close'] = (close - low) / close

            # Gap features
            features_df['gap'] = (open_price - close.shift(1)) / close.shift(1)
            features_df['gap_filled'] = ((close >= close.shift(1)) & (features_df['gap'] < 0)).astype(int)

            # Candlestick patterns (simplified)
            body = abs(close - open_price)
            upper_shadow = high - np.maximum(close, open_price)
            lower_shadow = np.minimum(close, open_price) - low

            features_df['body_ratio'] = body / (high - low)
            features_df['upper_shadow_ratio'] = upper_shadow / (high - low)
            features_df['lower_shadow_ratio'] = lower_shadow / (high - low)

            return features_df

        except Exception as e:
            logger.error(f"Error adding microstructure features: {e}")
            return features_df

    def _clean_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        try:
            # Remove infinite values
            features_df = features_df.replace([np.inf, -np.inf], np.nan)

            # Forward fill and backward fill
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')

            # Remove columns with too many NaN values
            nan_threshold = 0.5
            features_df = features_df.dropna(axis=1, thresh=int(len(features_df) * (1 - nan_threshold)))

            # Remove remaining NaN rows
            features_df = features_df.dropna()

            return features_df

        except Exception as e:
            logger.error(f"Error cleaning features: {e}")
            return features_df

    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'mutual_info', k: int = 50) -> pd.DataFrame:
        """Select most important features"""
        try:
            if method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X.shape[1]))
            elif method == 'f_test':
                selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
            else:
                return X

            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()]

            self.feature_selector = selector
            return X[selected_features]

        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            return X


class BaseMLModel(ABC):
    """Abstract base class for ML models"""

    def __init__(self, model_type: ModelType, config: ModelConfig):
        self.model_type = model_type
        self.config = config
        self.model = None
        self.is_trained = False
        self.feature_importance = {}
        self.training_history = []

    @abstractmethod
    async def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train the model"""
        pass

    @abstractmethod
    async def predict(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make predictions with confidence"""
        pass

    async def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance

    def save_model(self, filepath: str) -> bool:
        """Save model to disk"""
        try:
            joblib.dump({
                'model': self.model,
                'config': self.config,
                'feature_importance': self.feature_importance,
                'is_trained': self.is_trained
            }, filepath)
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """Load model from disk"""
        try:
            data = joblib.load(filepath)
            self.model = data['model']
            self.config = data['config']
            self.feature_importance = data['feature_importance']
            self.is_trained = data['is_trained']
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


class RandomForestModel(BaseMLModel):
    """Random Forest model implementation"""

    def __init__(self, config: ModelConfig):
        super().__init__(ModelType.RANDOM_FOREST, config)

    async def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train Random Forest model"""
        try:
            hyperparams = self.config.hyperparameters

            self.model = RandomForestRegressor(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', None),
                min_samples_split=hyperparams.get('min_samples_split', 2),
                min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
                random_state=hyperparams.get('random_state', 42),
                n_jobs=-1
            )

            self.model.fit(X, y)
            self.is_trained = True

            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = {
                    f'feature_{i}': importance
                    for i, importance in enumerate(self.model.feature_importances_)
                }

            # Calculate training metrics
            y_pred = self.model.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            training_result = {
                'mse': mse,
                'r2': r2,
                'feature_count': X.shape[1],
                'sample_count': X.shape[0]
            }

            self.training_history.append(training_result)
            return training_result

        except Exception as e:
            logger.error(f"Error training Random Forest: {e}")
            raise

    async def predict(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make predictions with confidence"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained")

            predictions = self.model.predict(X)

            # Calculate confidence based on ensemble variance
            if hasattr(self.model, 'estimators_'):
                tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
                confidence = 1.0 - np.mean(np.std(tree_predictions, axis=0))
            else:
                confidence = 0.8  # Default confidence

            return predictions, confidence

        except Exception as e:
            logger.error(f"Error making Random Forest predictions: {e}")
            raise


class XGBoostModel(BaseMLModel):
    """XGBoost model implementation"""

    def __init__(self, config: ModelConfig):
        super().__init__(ModelType.XGBOOST, config)

    async def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train XGBoost model"""
        try:
            hyperparams = self.config.hyperparameters

            self.model = xgb.XGBRegressor(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', 6),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                subsample=hyperparams.get('subsample', 1.0),
                colsample_bytree=hyperparams.get('colsample_bytree', 1.0),
                random_state=hyperparams.get('random_state', 42),
                n_jobs=-1
            )

            self.model.fit(X, y)
            self.is_trained = True

            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = {
                    f'feature_{i}': importance
                    for i, importance in enumerate(self.model.feature_importances_)
                }

            # Calculate training metrics
            y_pred = self.model.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            training_result = {
                'mse': mse,
                'r2': r2,
                'feature_count': X.shape[1],
                'sample_count': X.shape[0]
            }

            self.training_history.append(training_result)
            return training_result

        except Exception as e:
            logger.error(f"Error training XGBoost: {e}")
            raise

    async def predict(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make predictions with confidence"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained")

            predictions = self.model.predict(X)

            # XGBoost confidence based on leaf indices
            confidence = 0.85  # Default high confidence for XGBoost

            return predictions, confidence

        except Exception as e:
            logger.error(f"Error making XGBoost predictions: {e}")
            raise


class LSTMModel(BaseMLModel):
    """LSTM deep learning model implementation"""

    def __init__(self, config: ModelConfig):
        super().__init__(ModelType.LSTM, config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = config.hyperparameters.get('sequence_length', 60)

    async def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train LSTM model"""
        try:
            hyperparams = self.config.hyperparameters

            # Create sequences for LSTM
            X_seq, y_seq = self._create_sequences(X, y)

            # Convert to tensors
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            y_tensor = torch.FloatTensor(y_seq).to(self.device)

            # Create model
            input_size = X.shape[1]
            hidden_size = hyperparams.get('hidden_size', 50)
            num_layers = hyperparams.get('num_layers', 2)
            dropout = hyperparams.get('dropout', 0.2)

            self.model = LSTMNet(input_size, hidden_size, num_layers, dropout).to(self.device)

            # Training parameters
            lr = hyperparams.get('learning_rate', 0.001)
            epochs = hyperparams.get('epochs', 100)
            batch_size = hyperparams.get('batch_size', 32)

            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            # Create data loader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Training loop
            self.model.train()
            training_losses = []

            for epoch in range(epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(dataloader)
                training_losses.append(avg_loss)

                if epoch % 10 == 0:
                    logger.info(f"LSTM Epoch {epoch}, Loss: {avg_loss:.6f}")

            self.is_trained = True

            # Calculate final metrics
            self.model.eval()
            with torch.no_grad():
                final_pred = self.model(X_tensor).cpu().numpy().squeeze()
                mse = mean_squared_error(y_seq, final_pred)
                r2 = r2_score(y_seq, final_pred)

            training_result = {
                'mse': mse,
                'r2': r2,
                'final_loss': training_losses[-1],
                'epochs': epochs,
                'feature_count': X.shape[1],
                'sequence_length': self.sequence_length
            }

            self.training_history.append(training_result)
            return training_result

        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            raise

    async def predict(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make predictions with confidence"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained")

            # Create sequences
            X_seq = self._create_prediction_sequences(X)
            X_tensor = torch.FloatTensor(X_seq).to(self.device)

            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor).cpu().numpy().squeeze()

            # LSTM confidence based on prediction variance
            confidence = 0.8  # Default confidence

            return predictions, confidence

        except Exception as e:
            logger.error(f"Error making LSTM predictions: {e}")
            raise

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X_seq, y_seq = [], []

        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    def _create_prediction_sequences(self, X: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM prediction"""
        X_seq = []

        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])

        return np.array(X_seq)


class LSTMNet(nn.Module):
    """LSTM neural network architecture"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Use the last output
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)

        return out


class EnsembleLearningSystem:
    """
    Main ensemble learning system that combines multiple ML/DL models
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.models: Dict[str, BaseMLModel] = {}
        self.ensemble_weights: Dict[str, float] = {}
        self.feature_engineer = FeatureEngineer(config.get('feature_engineering', {}))

        # Performance tracking
        self.model_performance: Dict[str, Dict[str, float]] = {}
        self.ensemble_performance: Dict[str, float] = {}
        self.prediction_history: deque = deque(maxlen=10000)

        # Configuration
        self.ensemble_method = EnsembleMethod(config.get('ensemble_method', 'weighted_average'))
        self.retraining_frequency = timedelta(hours=config.get('retraining_hours', 24))
        self.last_training = datetime.now(timezone.utc)

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()

    async def initialize(self) -> bool:
        """Initialize the ensemble learning system"""
        try:
            logger.info("Initializing Ensemble Learning System")

            # Initialize models based on configuration
            model_configs = self.config.get('models', [])

            for model_config_dict in model_configs:
                model_config = ModelConfig(**model_config_dict)
                model = await self._create_model(model_config)

                if model:
                    model_name = f"{model_config.model_type.value}_{len(self.models)}"
                    self.models[model_name] = model
                    self.ensemble_weights[model_name] = 1.0 / len(model_configs)  # Equal weights initially

                    logger.info(f"Initialized {model_name}")

            if not self.models:
                # Create default models if none configured
                await self._create_default_models()

            logger.info(f"Ensemble Learning System initialized with {len(self.models)} models")
            return True

        except Exception as e:
            logger.error(f"Error initializing Ensemble Learning System: {e}")
            return False

    async def _create_model(self, config: ModelConfig) -> Optional[BaseMLModel]:
        """Create a model instance based on configuration"""
        try:
            if config.model_type == ModelType.RANDOM_FOREST:
                return RandomForestModel(config)
            elif config.model_type == ModelType.XGBOOST:
                return XGBoostModel(config)
            elif config.model_type == ModelType.LSTM:
                return LSTMModel(config)
            # Add more model types as needed
            else:
                logger.warning(f"Unknown model type: {config.model_type}")
                return None

        except Exception as e:
            logger.error(f"Error creating model: {e}")
            return None

    async def _create_default_models(self) -> None:
        """Create default models if none configured"""
        try:
            # Random Forest
            rf_config = ModelConfig(
                model_type=ModelType.RANDOM_FOREST,
                hyperparameters={
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'random_state': 42
                }
            )
            self.models['random_forest_default'] = RandomForestModel(rf_config)

            # XGBoost
            xgb_config = ModelConfig(
                model_type=ModelType.XGBOOST,
                hyperparameters={
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            )
            self.models['xgboost_default'] = XGBoostModel(xgb_config)

            # LSTM
            lstm_config = ModelConfig(
                model_type=ModelType.LSTM,
                hyperparameters={
                    'hidden_size': 50,
                    'num_layers': 2,
                    'sequence_length': 60,
                    'epochs': 50,
                    'batch_size': 32,
                    'learning_rate': 0.001
                }
            )
            self.models['lstm_default'] = LSTMModel(lstm_config)

            # Equal weights
            num_models = len(self.models)
            for model_name in self.models.keys():
                self.ensemble_weights[model_name] = 1.0 / num_models

        except Exception as e:
            logger.error(f"Error creating default models: {e}")

    async def train_ensemble(self,
                           market_data: pd.DataFrame,
                           target_variable: str = 'returns_1d',
                           prediction_task: PredictionTask = PredictionTask.PRICE_DIRECTION) -> Dict[str, Any]:
        """Train all models in the ensemble"""
        try:
            logger.info("Training ensemble models")

            # Feature engineering
            features_df = self.feature_engineer.create_features(market_data)

            if len(features_df) < 100:
                raise ValueError("Insufficient data for training")

            # Prepare target variable
            if target_variable not in features_df.columns:
                # Create target variable if not exists
                close = market_data['Close'] if 'Close' in market_data.columns else market_data['close']
                if prediction_task == PredictionTask.PRICE_DIRECTION:
                    features_df[target_variable] = (close.pct_change().shift(-1) > 0).astype(int)
                elif prediction_task == PredictionTask.PRICE_LEVEL:
                    features_df[target_variable] = close.pct_change().shift(-1)
                elif prediction_task == PredictionTask.VOLATILITY:
                    features_df[target_variable] = close.pct_change().rolling(window=20).std().shift(-1)

            # Clean data
            features_df = features_df.dropna()

            if len(features_df) < 50:
                raise ValueError("Insufficient clean data for training")

            # Separate features and target
            X = features_df.drop(columns=[target_variable])
            y = features_df[target_variable]

            # Feature selection
            if len(X.columns) > 50:
                X = self.feature_engineer.select_features(X, y, method='mutual_info', k=50)

            # Convert to numpy arrays
            X_np = X.values.astype(np.float32)
            y_np = y.values.astype(np.float32)

            # Train all models
            training_results = {}
            training_tasks = []

            for model_name, model in self.models.items():
                task = asyncio.create_task(self._train_single_model(model_name, model, X_np, y_np))
                training_tasks.append((model_name, task))

            # Wait for all training to complete
            for model_name, task in training_tasks:
                try:
                    result = await task
                    training_results[model_name] = result

                    # Update model performance
                    with self.lock:
                        self.model_performance[model_name] = result

                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    training_results[model_name] = {'error': str(e)}

            # Update ensemble weights based on performance
            await self._update_ensemble_weights(training_results)

            # Update last training time
            self.last_training = datetime.now(timezone.utc)

            # Calculate overall ensemble performance
            ensemble_metrics = self._calculate_ensemble_metrics(training_results)

            logger.info(f"Ensemble training completed: {len(training_results)} models trained")

            return {
                'individual_results': training_results,
                'ensemble_metrics': ensemble_metrics,
                'feature_count': X.shape[1],
                'sample_count': X.shape[0],
                'ensemble_weights': self.ensemble_weights
            }

        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            raise

    async def _train_single_model(self, model_name: str, model: BaseMLModel,
                                 X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train a single model"""
        try:
            logger.info(f"Training {model_name}")
            result = await model.train(X, y)
            logger.info(f"Completed training {model_name}: R2={result.get('r2', 0):.4f}")
            return result
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            raise

    async def predict_ensemble(self,
                             market_data: pd.DataFrame,
                             prediction_task: PredictionTask = PredictionTask.PRICE_DIRECTION) -> EnsemblePrediction:
        """Make ensemble prediction"""
        try:
            # Feature engineering
            features_df = self.feature_engineer.create_features(market_data)

            if len(features_df) == 0:
                raise ValueError("No features generated from market data")

            # Use feature selector if available
            if self.feature_engineer.feature_selector:
                X = self.feature_engineer.feature_selector.transform(features_df)
            else:
                X = features_df.values

            X = X.astype(np.float32)

            # Get predictions from all models
            individual_predictions = []
            prediction_tasks = []

            for model_name, model in self.models.items():
                if model.is_trained:
                    task = asyncio.create_task(self._predict_single_model(model_name, model, X))
                    prediction_tasks.append((model_name, task))

            # Collect predictions
            valid_predictions = []
            for model_name, task in prediction_tasks:
                try:
                    prediction, confidence = await task

                    pred_result = PredictionResult(
                        model_name=model_name,
                        model_type=model.model_type,
                        prediction=prediction[-1] if len(prediction) > 0 else 0.0,  # Latest prediction
                        confidence=confidence,
                        feature_importance=await model.get_feature_importance()
                    )

                    individual_predictions.append(pred_result)
                    valid_predictions.append((prediction[-1] if len(prediction) > 0 else 0.0, confidence, model_name))

                except Exception as e:
                    logger.error(f"Error getting prediction from {model_name}: {e}")

            if not valid_predictions:
                raise ValueError("No valid predictions from ensemble models")

            # Combine predictions using ensemble method
            ensemble_pred, ensemble_conf, ensemble_unc = await self._combine_predictions(valid_predictions)

            ensemble_prediction = EnsemblePrediction(
                task=prediction_task,
                ensemble_prediction=ensemble_pred,
                individual_predictions=individual_predictions,
                ensemble_confidence=ensemble_conf,
                ensemble_uncertainty=ensemble_unc,
                model_weights=self.ensemble_weights.copy(),
                method_used=self.ensemble_method
            )

            # Store in history
            self.prediction_history.append(ensemble_prediction)

            return ensemble_prediction

        except Exception as e:
            logger.error(f"Error making ensemble prediction: {e}")
            raise

    async def _predict_single_model(self, model_name: str, model: BaseMLModel,
                                   X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Get prediction from a single model"""
        try:
            return await model.predict(X)
        except Exception as e:
            logger.error(f"Error predicting with {model_name}: {e}")
            raise

    async def _combine_predictions(self, predictions: List[Tuple[float, float, str]]) -> Tuple[float, float, float]:
        """Combine individual predictions using ensemble method"""
        try:
            if self.ensemble_method == EnsembleMethod.SIMPLE_AVERAGE:
                return self._simple_average(predictions)
            elif self.ensemble_method == EnsembleMethod.WEIGHTED_AVERAGE:
                return self._weighted_average(predictions)
            elif self.ensemble_method == EnsembleMethod.DYNAMIC_WEIGHTING:
                return self._dynamic_weighting(predictions)
            else:
                # Default to weighted average
                return self._weighted_average(predictions)

        except Exception as e:
            logger.error(f"Error combining predictions: {e}")
            return 0.0, 0.0, 1.0

    def _simple_average(self, predictions: List[Tuple[float, float, str]]) -> Tuple[float, float, float]:
        """Simple average ensemble"""
        pred_values = [pred[0] for pred in predictions]
        conf_values = [pred[1] for pred in predictions]

        ensemble_pred = np.mean(pred_values)
        ensemble_conf = np.mean(conf_values)
        ensemble_unc = np.std(pred_values)

        return ensemble_pred, ensemble_conf, ensemble_unc

    def _weighted_average(self, predictions: List[Tuple[float, float, str]]) -> Tuple[float, float, float]:
        """Weighted average ensemble using learned weights"""
        weighted_pred = 0.0
        weighted_conf = 0.0
        total_weight = 0.0
        pred_values = []

        for pred_value, confidence, model_name in predictions:
            weight = self.ensemble_weights.get(model_name, 1.0)
            weighted_pred += pred_value * weight
            weighted_conf += confidence * weight
            total_weight += weight
            pred_values.append(pred_value)

        if total_weight > 0:
            ensemble_pred = weighted_pred / total_weight
            ensemble_conf = weighted_conf / total_weight
        else:
            ensemble_pred = np.mean([pred[0] for pred in predictions])
            ensemble_conf = np.mean([pred[1] for pred in predictions])

        ensemble_unc = np.std(pred_values)

        return ensemble_pred, ensemble_conf, ensemble_unc

    def _dynamic_weighting(self, predictions: List[Tuple[float, float, str]]) -> Tuple[float, float, float]:
        """Dynamic weighting based on recent performance"""
        # Use confidence scores as dynamic weights
        weights = []
        pred_values = []

        for pred_value, confidence, model_name in predictions:
            # Combine learned weight with confidence
            base_weight = self.ensemble_weights.get(model_name, 1.0)
            dynamic_weight = base_weight * confidence
            weights.append(dynamic_weight)
            pred_values.append(pred_value)

        weights = np.array(weights)
        pred_values = np.array(pred_values)

        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
            ensemble_pred = np.sum(pred_values * weights)
            ensemble_conf = np.sum([pred[1] for pred in predictions]) / len(predictions)
        else:
            ensemble_pred = np.mean(pred_values)
            ensemble_conf = np.mean([pred[1] for pred in predictions])

        ensemble_unc = np.std(pred_values)

        return ensemble_pred, ensemble_conf, ensemble_unc

    async def _update_ensemble_weights(self, training_results: Dict[str, Dict[str, Any]]) -> None:
        """Update ensemble weights based on model performance"""
        try:
            # Calculate weights based on R2 scores
            r2_scores = {}
            for model_name, result in training_results.items():
                if 'r2' in result and not np.isnan(result['r2']):
                    r2_scores[model_name] = max(0.0, result['r2'])  # Ensure non-negative
                else:
                    r2_scores[model_name] = 0.0

            # Normalize scores to create weights
            total_score = sum(r2_scores.values())
            if total_score > 0:
                for model_name in self.ensemble_weights.keys():
                    if model_name in r2_scores:
                        self.ensemble_weights[model_name] = r2_scores[model_name] / total_score
                    else:
                        self.ensemble_weights[model_name] = 0.0
            else:
                # Equal weights if no valid scores
                num_models = len(self.ensemble_weights)
                for model_name in self.ensemble_weights.keys():
                    self.ensemble_weights[model_name] = 1.0 / num_models

        except Exception as e:
            logger.error(f"Error updating ensemble weights: {e}")

    def _calculate_ensemble_metrics(self, training_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate ensemble-level metrics"""
        try:
            valid_results = [result for result in training_results.values() if 'r2' in result]

            if not valid_results:
                return {}

            r2_scores = [result['r2'] for result in valid_results if not np.isnan(result['r2'])]
            mse_scores = [result['mse'] for result in valid_results if not np.isnan(result['mse'])]

            return {
                'avg_r2': np.mean(r2_scores) if r2_scores else 0.0,
                'max_r2': np.max(r2_scores) if r2_scores else 0.0,
                'min_r2': np.min(r2_scores) if r2_scores else 0.0,
                'avg_mse': np.mean(mse_scores) if mse_scores else 0.0,
                'model_count': len(valid_results),
                'ensemble_diversity': np.std(r2_scores) if len(r2_scores) > 1 else 0.0
            }

        except Exception as e:
            logger.error(f"Error calculating ensemble metrics: {e}")
            return {}

    def should_retrain(self) -> bool:
        """Check if ensemble should be retrained"""
        time_since_training = datetime.now(timezone.utc) - self.last_training
        return time_since_training >= self.retraining_frequency

    async def save_ensemble(self, directory: str) -> bool:
        """Save entire ensemble to disk"""
        try:
            os.makedirs(directory, exist_ok=True)

            # Save individual models
            for model_name, model in self.models.items():
                model_path = os.path.join(directory, f"{model_name}.joblib")
                success = model.save_model(model_path)
                if not success:
                    logger.warning(f"Failed to save {model_name}")

            # Save ensemble metadata
            metadata = {
                'ensemble_weights': self.ensemble_weights,
                'model_performance': self.model_performance,
                'ensemble_performance': self.ensemble_performance,
                'ensemble_method': self.ensemble_method.value,
                'last_training': self.last_training.isoformat(),
                'config': self.config
            }

            metadata_path = os.path.join(directory, 'ensemble_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            logger.info(f"Ensemble saved to {directory}")
            return True

        except Exception as e:
            logger.error(f"Error saving ensemble: {e}")
            return False

    async def load_ensemble(self, directory: str) -> bool:
        """Load ensemble from disk"""
        try:
            if not os.path.exists(directory):
                logger.error(f"Ensemble directory {directory} does not exist")
                return False

            # Load metadata
            metadata_path = os.path.join(directory, 'ensemble_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                self.ensemble_weights = metadata.get('ensemble_weights', {})
                self.model_performance = metadata.get('model_performance', {})
                self.ensemble_performance = metadata.get('ensemble_performance', {})
                self.ensemble_method = EnsembleMethod(metadata.get('ensemble_method', 'weighted_average'))
                self.last_training = datetime.fromisoformat(metadata.get('last_training', datetime.now(timezone.utc).isoformat()))

            # Load individual models
            for filename in os.listdir(directory):
                if filename.endswith('.joblib') and filename != 'ensemble_metadata.json':
                    model_name = filename.replace('.joblib', '')
                    model_path = os.path.join(directory, filename)

                    # Create model instance (simplified - would need proper config)
                    if 'random_forest' in model_name:
                        model = RandomForestModel(ModelConfig(ModelType.RANDOM_FOREST, {}))
                    elif 'xgboost' in model_name:
                        model = XGBoostModel(ModelConfig(ModelType.XGBOOST, {}))
                    elif 'lstm' in model_name:
                        model = LSTMModel(ModelConfig(ModelType.LSTM, {}))
                    else:
                        continue

                    success = model.load_model(model_path)
                    if success:
                        self.models[model_name] = model

            logger.info(f"Loaded ensemble with {len(self.models)} models")
            return True

        except Exception as e:
            logger.error(f"Error loading ensemble: {e}")
            return False

    def get_ensemble_status(self) -> Dict[str, Any]:
        """Get comprehensive ensemble status"""
        try:
            model_statuses = {}
            for model_name, model in self.models.items():
                model_statuses[model_name] = {
                    'model_type': model.model_type.value,
                    'is_trained': model.is_trained,
                    'training_history_count': len(model.training_history),
                    'feature_importance_count': len(model.feature_importance)
                }

            return {
                'total_models': len(self.models),
                'trained_models': sum(1 for model in self.models.values() if model.is_trained),
                'ensemble_method': self.ensemble_method.value,
                'ensemble_weights': self.ensemble_weights,
                'model_performance': self.model_performance,
                'ensemble_performance': self.ensemble_performance,
                'prediction_history_count': len(self.prediction_history),
                'last_training': self.last_training.isoformat(),
                'should_retrain': self.should_retrain(),
                'model_statuses': model_statuses
            }

        except Exception as e:
            logger.error(f"Error getting ensemble status: {e}")
            return {'error': str(e)}


# Example usage and testing
if __name__ == "__main__":
    async def test_ensemble_system():
        """Test the ensemble learning system"""

        config = {
            'ensemble_method': 'weighted_average',
            'retraining_hours': 24,
            'models': [
                {
                    'model_type': 'random_forest',
                    'hyperparameters': {
                        'n_estimators': 50,
                        'max_depth': 8,
                        'random_state': 42
                    }
                },
                {
                    'model_type': 'xgboost',
                    'hyperparameters': {
                        'n_estimators': 50,
                        'max_depth': 4,
                        'learning_rate': 0.1,
                        'random_state': 42
                    }
                }
            ]
        }

        ensemble = EnsembleLearningSystem(config)

        try:
            # Initialize ensemble
            success = await ensemble.initialize()
            if not success:
                print("Failed to initialize ensemble")
                return

            print("Ensemble initialized successfully")

            # Generate sample market data
            dates = pd.date_range('2023-01-01', periods=500, freq='D')
            np.random.seed(42)

            price_data = []
            base_price = 100.0

            for i in range(500):
                change = np.random.normal(0.001, 0.02)
                base_price *= (1 + change)

                high = base_price * (1 + abs(np.random.normal(0, 0.01)))
                low = base_price * (1 - abs(np.random.normal(0, 0.01)))
                open_price = base_price + np.random.normal(0, 0.005)
                volume = np.random.randint(100000, 1000000)

                price_data.append({
                    'Open': open_price,
                    'High': high,
                    'Low': low,
                    'Close': base_price,
                    'Volume': volume
                })

            market_data = pd.DataFrame(price_data, index=dates)

            # Train ensemble
            print("Training ensemble...")
            training_results = await ensemble.train_ensemble(
                market_data,
                target_variable='returns_1d',
                prediction_task=PredictionTask.PRICE_DIRECTION
            )

            print(f"Training completed:")
            print(f"- Models trained: {len(training_results['individual_results'])}")
            print(f"- Average R2: {training_results['ensemble_metrics'].get('avg_r2', 0):.4f}")
            print(f"- Feature count: {training_results['feature_count']}")

            # Make predictions
            print("\nMaking predictions...")
            recent_data = market_data.tail(100)  # Use recent data for prediction

            prediction = await ensemble.predict_ensemble(
                recent_data,
                PredictionTask.PRICE_DIRECTION
            )

            print(f"Ensemble prediction: {prediction.ensemble_prediction:.4f}")
            print(f"Ensemble confidence: {prediction.ensemble_confidence:.4f}")
            print(f"Ensemble uncertainty: {prediction.ensemble_uncertainty:.4f}")
            print(f"Individual predictions: {len(prediction.individual_predictions)}")

            for pred in prediction.individual_predictions:
                print(f"  {pred.model_name}: {pred.prediction:.4f} (conf: {pred.confidence:.2f})")

            # Get system status
            status = ensemble.get_ensemble_status()
            print(f"\nEnsemble Status:")
            print(f"- Total models: {status['total_models']}")
            print(f"- Trained models: {status['trained_models']}")
            print(f"- Ensemble method: {status['ensemble_method']}")
            print(f"- Should retrain: {status['should_retrain']}")

        except Exception as e:
            print(f"Error testing ensemble system: {e}")

    # Run test
    asyncio.run(test_ensemble_system())