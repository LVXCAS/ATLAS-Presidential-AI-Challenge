"""
Machine Learning Prediction Engine
Uses XGBoost, LightGBM, CatBoost for volatility and price predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
import warnings
warnings.filterwarnings('ignore')

# ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
from config.logging_config import get_logger

logger = get_logger(__name__)

class MLPredictionEngine:
    """Advanced ML engine for market predictions"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache
        
    async def get_price_prediction(self, symbol: str, horizon_days: int = 5) -> Dict:
        """Get price prediction using ensemble ML models"""
        
        cache_key = f"{symbol}_price_pred_{horizon_days}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            # Get training data
            training_data = await self._prepare_training_data(symbol, period="2y")
            if training_data is None or len(training_data) < 100:
                return self._get_default_prediction()
            
            # Train ensemble models
            predictions = {}
            
            # XGBoost prediction
            if XGBOOST_AVAILABLE:
                xgb_pred = await self._train_and_predict_xgboost(training_data, horizon_days)
                predictions['xgboost'] = xgb_pred
            
            # LightGBM prediction
            if LIGHTGBM_AVAILABLE:
                lgb_pred = await self._train_and_predict_lightgbm(training_data, horizon_days)
                predictions['lightgbm'] = lgb_pred
            
            # CatBoost prediction
            if CATBOOST_AVAILABLE:
                cb_pred = await self._train_and_predict_catboost(training_data, horizon_days)
                predictions['catboost'] = cb_pred
            
            # Ensemble prediction
            ensemble_pred = await self._create_ensemble_prediction(predictions, training_data)
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'horizon_days': horizon_days,
                'current_price': float(training_data['close'].iloc[-1]),
                'predictions': predictions,
                'ensemble_prediction': ensemble_pred,
                'feature_importance': self.feature_importance.get(symbol, {}),
                'model_confidence': self._calculate_model_confidence(predictions),
                'prediction_intervals': self._calculate_prediction_intervals(predictions),
                'trading_signals': self._generate_ml_signals(ensemble_pred, training_data)
            }
            
            # Cache result
            self.cache[cache_key] = result
            self.cache_expiry = datetime.now() + timedelta(seconds=self.cache_duration)
            
            return result
            
        except Exception as e:
            logger.error(f"Price prediction error for {symbol}: {e}")
            return self._get_default_prediction()
    
    async def get_volatility_prediction(self, symbol: str, horizon_days: int = 10) -> Dict:
        """Get volatility prediction using ML models"""
        
        cache_key = f"{symbol}_vol_pred_{horizon_days}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            # Get training data with volatility features
            training_data = await self._prepare_volatility_data(symbol, period="2y")
            if training_data is None or len(training_data) < 100:
                return self._get_default_volatility_prediction()
            
            # Train volatility models
            vol_predictions = {}
            
            if XGBOOST_AVAILABLE:
                xgb_vol = await self._train_volatility_xgboost(training_data, horizon_days)
                vol_predictions['xgboost'] = xgb_vol
            
            if LIGHTGBM_AVAILABLE:
                lgb_vol = await self._train_volatility_lightgbm(training_data, horizon_days)
                vol_predictions['lightgbm'] = lgb_vol
            
            # Ensemble volatility prediction
            ensemble_vol = np.mean([pred for pred in vol_predictions.values() if pred is not None])
            
            current_vol = float(training_data['realized_vol'].iloc[-1])
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'horizon_days': horizon_days,
                'current_volatility': current_vol,
                'predicted_volatility': ensemble_vol,
                'volatility_change': (ensemble_vol - current_vol) / current_vol * 100,
                'volatility_regime_prediction': self._predict_volatility_regime(ensemble_vol),
                'confidence': self._calculate_volatility_confidence(vol_predictions),
                'risk_metrics': self._calculate_risk_metrics(training_data, ensemble_vol)
            }
            
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Volatility prediction error for {symbol}: {e}")
            return self._get_default_volatility_prediction()
    
    async def _prepare_training_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Prepare training data with features for price prediction"""
        
        try:
            # Get price data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return None
            
            # Calculate features
            df = pd.DataFrame()
            df['close'] = hist['Close']
            df['high'] = hist['High']
            df['low'] = hist['Low']
            df['volume'] = hist['Volume']
            
            # Price-based features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                df[f'sma_{window}'] = df['close'].rolling(window).mean()
                df[f'price_to_sma_{window}'] = df['close'] / df[f'sma_{window}']
            
            # Volatility features
            df['realized_vol'] = df['returns'].rolling(20).std() * np.sqrt(252)
            df['high_low_ratio'] = df['high'] / df['low']
            df['price_range'] = (df['high'] - df['low']) / df['close']
            
            # Momentum features
            for window in [5, 10, 20]:
                df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
                df[f'rsi_{window}'] = self._calculate_rsi(df['close'], window)
            
            # Volume features
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Technical indicators
            df['macd'] = self._calculate_macd(df['close'])
            df['bollinger_position'] = self._calculate_bollinger_position(df['close'])
            
            # Lag features
            for lag in [1, 2, 3, 5]:
                df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
                df[f'vol_lag_{lag}'] = df['realized_vol'].shift(lag)
            
            # Target variable (future returns)
            df['target'] = df['returns'].shift(-1)  # Next day return
            
            # Remove NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Training data preparation error: {e}")
            return None
    
    async def _prepare_volatility_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Prepare training data for volatility prediction"""
        
        try:
            # Get base data
            df = await self._prepare_training_data(symbol, period)
            if df is None:
                return None
            
            # Add volatility-specific features
            df['vol_of_vol'] = df['realized_vol'].rolling(20).std()
            df['vol_momentum'] = df['realized_vol'] / df['realized_vol'].shift(5) - 1
            df['vol_mean_reversion'] = df['realized_vol'] / df['realized_vol'].rolling(60).mean() - 1
            
            # GARCH-like features
            df['vol_squared'] = df['returns'] ** 2
            df['vol_squared_ma'] = df['vol_squared'].rolling(20).mean()
            
            # Volatility clustering
            df['abs_returns'] = abs(df['returns'])
            df['vol_clustering'] = df['abs_returns'].rolling(10).mean()
            
            # Target: future realized volatility
            df['vol_target'] = df['realized_vol'].shift(-5)  # 5-day ahead volatility
            
            df = df.dropna()
            return df
            
        except Exception as e:
            logger.error(f"Volatility data preparation error: {e}")
            return None
    
    async def _train_and_predict_xgboost(self, data: pd.DataFrame, horizon: int) -> Optional[float]:
        """Train XGBoost model and make prediction"""
        
        if not XGBOOST_AVAILABLE:
            return None
        
        try:
            # Prepare features and target
            feature_cols = [col for col in data.columns if col not in ['target', 'vol_target', 'close']]
            X = data[feature_cols].fillna(0)
            y = data['target'].fillna(0)
            
            # Split data
            train_size = int(len(data) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Train model
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                verbosity=0
            )
            
            model.fit(X_train, y_train)
            
            # Store feature importance
            importance_dict = dict(zip(feature_cols, model.feature_importances_))
            self.feature_importance[f'xgboost'] = importance_dict
            
            # Make prediction
            latest_features = X.iloc[-1:].fillna(0)
            prediction = model.predict(latest_features)[0]
            
            # Calculate model performance
            test_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, test_pred)
            self.model_performance['xgboost'] = {'mse': mse, 'rmse': np.sqrt(mse)}
            
            return float(prediction)
            
        except Exception as e:
            logger.warning(f"XGBoost training error: {e}")
            return None
    
    async def _train_and_predict_lightgbm(self, data: pd.DataFrame, horizon: int) -> Optional[float]:
        """Train LightGBM model and make prediction"""
        
        if not LIGHTGBM_AVAILABLE:
            return None
        
        try:
            feature_cols = [col for col in data.columns if col not in ['target', 'vol_target', 'close']]
            X = data[feature_cols].fillna(0)
            y = data['target'].fillna(0)
            
            train_size = int(len(data) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Train model
            train_data = lgb.Dataset(X_train, label=y_train)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'verbosity': -1
            }
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.log_evaluation(0)]
            )
            
            # Store feature importance
            importance_dict = dict(zip(feature_cols, model.feature_importance()))
            self.feature_importance[f'lightgbm'] = importance_dict
            
            # Make prediction
            latest_features = X.iloc[-1:].fillna(0)
            prediction = model.predict(latest_features)[0]
            
            return float(prediction)
            
        except Exception as e:
            logger.warning(f"LightGBM training error: {e}")
            return None
    
    async def _train_and_predict_catboost(self, data: pd.DataFrame, horizon: int) -> Optional[float]:
        """Train CatBoost model and make prediction"""
        
        if not CATBOOST_AVAILABLE:
            return None
        
        try:
            feature_cols = [col for col in data.columns if col not in ['target', 'vol_target', 'close']]
            X = data[feature_cols].fillna(0)
            y = data['target'].fillna(0)
            
            train_size = int(len(data) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Train model
            model = cb.CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                verbose=0,
                random_seed=42
            )
            
            model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)
            
            # Store feature importance
            importance_dict = dict(zip(feature_cols, model.feature_importances_))
            self.feature_importance[f'catboost'] = importance_dict
            
            # Make prediction
            latest_features = X.iloc[-1:].fillna(0)
            prediction = model.predict(latest_features)[0]
            
            return float(prediction)
            
        except Exception as e:
            logger.warning(f"CatBoost training error: {e}")
            return None
    
    async def _train_volatility_xgboost(self, data: pd.DataFrame, horizon: int) -> Optional[float]:
        """Train XGBoost for volatility prediction"""
        
        if not XGBOOST_AVAILABLE:
            return None
        
        try:
            feature_cols = [col for col in data.columns if col not in ['vol_target', 'target', 'close']]
            X = data[feature_cols].fillna(0)
            y = data['vol_target'].fillna(data['realized_vol'].mean())
            
            train_size = int(len(data) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
            
            model.fit(X_train, y_train)
            
            latest_features = X.iloc[-1:].fillna(0)
            prediction = model.predict(latest_features)[0]
            
            return float(prediction)
            
        except Exception as e:
            logger.warning(f"Volatility XGBoost error: {e}")
            return None
    
    async def _train_volatility_lightgbm(self, data: pd.DataFrame, horizon: int) -> Optional[float]:
        """Train LightGBM for volatility prediction"""
        
        if not LIGHTGBM_AVAILABLE:
            return None
        
        try:
            feature_cols = [col for col in data.columns if col not in ['vol_target', 'target', 'close']]
            X = data[feature_cols].fillna(0)
            y = data['vol_target'].fillna(data['realized_vol'].mean())
            
            train_size = int(len(data) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1
            }
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                callbacks=[lgb.log_evaluation(0)]
            )
            
            latest_features = X.iloc[-1:].fillna(0)
            prediction = model.predict(latest_features)[0]
            
            return float(prediction)
            
        except Exception as e:
            logger.warning(f"Volatility LightGBM error: {e}")
            return None
    
    async def _create_ensemble_prediction(self, predictions: Dict, data: pd.DataFrame) -> Dict:
        """Create ensemble prediction from multiple models"""
        
        valid_predictions = {k: v for k, v in predictions.items() if v is not None}
        
        if not valid_predictions:
            return {'ensemble_return': 0.0, 'ensemble_price': data['close'].iloc[-1]}
        
        # Weight models based on recent performance (if available)
        weights = {}
        total_weight = 0
        
        for model_name in valid_predictions.keys():
            if model_name in self.model_performance:
                # Lower RMSE = higher weight
                rmse = self.model_performance[model_name].get('rmse', 1.0)
                weight = 1.0 / (rmse + 0.001)  # Add small epsilon to avoid division by zero
            else:
                weight = 1.0  # Default weight
            
            weights[model_name] = weight
            total_weight += weight
        
        # Normalize weights
        for model_name in weights:
            weights[model_name] /= total_weight
        
        # Calculate weighted ensemble prediction
        ensemble_return = sum(pred * weights[model_name] for model_name, pred in valid_predictions.items())
        
        current_price = data['close'].iloc[-1]
        ensemble_price = current_price * (1 + ensemble_return)
        
        return {
            'ensemble_return': ensemble_return,
            'ensemble_price': ensemble_price,
            'individual_predictions': valid_predictions,
            'model_weights': weights,
            'confidence_score': self._calculate_ensemble_confidence(valid_predictions)
        }
    
    def _calculate_model_confidence(self, predictions: Dict) -> float:
        """Calculate overall model confidence based on agreement"""
        
        valid_preds = [v for v in predictions.values() if v is not None]
        
        if len(valid_preds) < 2:
            return 0.5  # Low confidence with only one model
        
        # Calculate standard deviation of predictions
        std_pred = np.std(valid_preds)
        mean_pred = np.mean(valid_preds)
        
        # Lower relative standard deviation = higher confidence
        if abs(mean_pred) > 0.001:
            cv = std_pred / abs(mean_pred)  # Coefficient of variation
            confidence = max(0.0, 1.0 - cv * 10)  # Scale and invert
        else:
            confidence = 0.5
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_prediction_intervals(self, predictions: Dict) -> Dict:
        """Calculate prediction intervals based on model spread"""
        
        valid_preds = [v for v in predictions.values() if v is not None]
        
        if not valid_preds:
            return {'lower_bound': 0.0, 'upper_bound': 0.0}
        
        mean_pred = np.mean(valid_preds)
        std_pred = np.std(valid_preds)
        
        # 95% confidence interval (assuming normal distribution)
        lower_bound = mean_pred - 1.96 * std_pred
        upper_bound = mean_pred + 1.96 * std_pred
        
        return {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'std_deviation': std_pred
        }
    
    def _generate_ml_signals(self, ensemble_pred: Dict, data: pd.DataFrame) -> Dict:
        """Generate trading signals based on ML predictions"""
        
        signals = {}
        
        try:
            ensemble_return = ensemble_pred.get('ensemble_return', 0.0)
            confidence = ensemble_pred.get('confidence_score', 0.5)
            
            # Signal strength based on prediction magnitude and confidence
            signal_strength = abs(ensemble_return) * confidence * 100
            
            # Signal direction
            if ensemble_return > 0.005 and confidence > 0.6:  # > 0.5% with high confidence
                signals['ml_signal'] = 'BULLISH'
            elif ensemble_return < -0.005 and confidence > 0.6:  # < -0.5% with high confidence
                signals['ml_signal'] = 'BEARISH'
            else:
                signals['ml_signal'] = 'NEUTRAL'
            
            signals['signal_strength'] = signal_strength
            signals['confidence'] = confidence
            signals['expected_return'] = ensemble_return * 100  # Convert to percentage
            
            # Risk-adjusted signal
            current_vol = data['realized_vol'].iloc[-1] if 'realized_vol' in data.columns else 0.2
            risk_adjusted_return = ensemble_return / current_vol if current_vol > 0 else 0
            signals['risk_adjusted_signal'] = risk_adjusted_return
            
            return signals
            
        except Exception as e:
            logger.warning(f"ML signal generation error: {e}")
            return {'ml_signal': 'NEUTRAL', 'confidence': 0.5}
    
    def _predict_volatility_regime(self, predicted_vol: float) -> str:
        """Predict volatility regime based on predicted volatility"""
        
        if predicted_vol > 0.4:  # 40% annualized
            return 'EXTREME_VOLATILITY'
        elif predicted_vol > 0.25:  # 25% annualized
            return 'HIGH_VOLATILITY'
        elif predicted_vol > 0.15:  # 15% annualized
            return 'NORMAL_VOLATILITY'
        else:
            return 'LOW_VOLATILITY'
    
    def _calculate_volatility_confidence(self, vol_predictions: Dict) -> float:
        """Calculate confidence in volatility predictions"""
        
        valid_preds = [v for v in vol_predictions.values() if v is not None]
        
        if len(valid_preds) < 2:
            return 0.6
        
        cv = np.std(valid_preds) / np.mean(valid_preds) if np.mean(valid_preds) > 0 else 1.0
        confidence = max(0.0, 1.0 - cv)
        
        return min(1.0, confidence)
    
    def _calculate_risk_metrics(self, data: pd.DataFrame, predicted_vol: float) -> Dict:
        """Calculate risk metrics based on predictions"""
        
        current_vol = data['realized_vol'].iloc[-1] if 'realized_vol' in data.columns else 0.2
        
        return {
            'volatility_change': (predicted_vol - current_vol) / current_vol * 100,
            'var_95': -1.96 * predicted_vol / np.sqrt(252),  # Daily VaR
            'expected_range': predicted_vol / np.sqrt(252) * 2,  # ±1 std dev daily
            'volatility_risk': 'HIGH' if predicted_vol > current_vol * 1.5 else 'NORMAL'
        }
    
    def _calculate_ensemble_confidence(self, predictions: Dict) -> float:
        """Calculate confidence score for ensemble prediction"""
        
        if len(predictions) < 2:
            return 0.5
        
        values = list(predictions.values())
        agreement = 1.0 - (np.std(values) / (np.mean(np.abs(values)) + 0.001))
        
        return max(0.0, min(1.0, agreement))
    
    # Technical indicator calculations
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        return ema12 - ema26
    
    def _calculate_bollinger_position(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return (prices - lower) / (upper - lower)
    
    def _get_default_prediction(self) -> Dict:
        """Return default prediction when models fail"""
        return {
            'symbol': 'UNKNOWN',
            'timestamp': datetime.now(),
            'ensemble_prediction': {
                'ensemble_return': 0.0,
                'ensemble_price': 100.0,
                'confidence_score': 0.5
            },
            'model_confidence': 0.5,
            'trading_signals': {
                'ml_signal': 'NEUTRAL',
                'confidence': 0.5
            }
        }
    
    def _get_default_volatility_prediction(self) -> Dict:
        """Return default volatility prediction when models fail"""
        return {
            'symbol': 'UNKNOWN',
            'timestamp': datetime.now(),
            'current_volatility': 0.2,
            'predicted_volatility': 0.2,
            'volatility_regime_prediction': 'NORMAL_VOLATILITY',
            'confidence': 0.5
        }
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        return (cache_key in self.cache and 
                hasattr(self, 'cache_expiry') and 
                datetime.now() < self.cache_expiry)

# Singleton instance
ml_prediction_engine = MLPredictionEngine()