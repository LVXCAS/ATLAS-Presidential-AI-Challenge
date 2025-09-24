"""
Machine Learning for Trading (ML4T) Agent
Implements advanced ML techniques from Stefan Jansen's "Machine Learning for Trading" book
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor,
        GradientBoostingClassifier, AdaBoostClassifier,
        ExtraTreesClassifier
    )
    from sklearn.linear_model import (
        LogisticRegression, Ridge, Lasso, ElasticNet,
        LinearRegression
    )
    from sklearn.svm import SVC, SVR
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.model_selection import (
        train_test_split, cross_val_score, GridSearchCV,
        TimeSeriesSplit
    )
    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, RobustScaler,
        LabelEncoder, OneHotEncoder
    )
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score,
        classification_report, confusion_matrix
    )
    from sklearn.feature_selection import (
        SelectKBest, f_classif, mutual_info_classif,
        RFE, RFECV
    )
    from sklearn.decomposition import PCA, FastICA
    from sklearn.cluster import KMeans
    import joblib
    ML4T_AVAILABLE = True
except ImportError as e:
    print(f"ML4T libraries not available: {e}")
    ML4T_AVAILABLE = False

# Additional ML libraries
try:
    import xgboost as xgb
    import lightgbm as lgb
    BOOSTING_AVAILABLE = True
except ImportError:
    BOOSTING_AVAILABLE = False

try:
    from scipy import stats
    from scipy.optimize import minimize
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, coint
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ML4TPrediction:
    """ML4T prediction structure"""
    symbol: str
    prediction_type: str  # 'classification', 'regression', 'signal'
    prediction: Any  # The actual prediction
    confidence: float
    probability_distribution: Optional[Dict] = None
    feature_importance: Optional[Dict] = None
    model_metrics: Optional[Dict] = None
    ensemble_components: Optional[List] = None
    timestamp: datetime = None

@dataclass
class FactorModel:
    """Factor model for return prediction"""
    factors: List[str]
    loadings: Dict[str, float]
    residual_variance: float
    r_squared: float
    factor_returns: Dict[str, float]

class ML4TAgent:
    """
    Advanced ML agent implementing techniques from Stefan Jansen's ML4T:
    - Factor Models (Fama-French, custom factors)
    - Alternative Data Integration (sentiment, news, etc.)
    - Advanced Feature Engineering (technical, fundamental, macro)
    - Ensemble Methods (stacking, blending, voting)
    - Time Series Cross-Validation
    - Regime Detection
    - Portfolio Construction with ML
    - Risk Models
    """

    def __init__(self):
        self.name = "ML4T Agent"
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.factor_models = {}
        self.regime_models = {}
        self.ensemble_weights = {}

        # ML4T Configuration
        self.lookback_periods = [5, 10, 20, 60, 252]  # 1w, 2w, 1m, 3m, 1y
        self.prediction_horizons = [1, 5, 20]  # 1d, 1w, 1m
        self.rebalancing_frequency = 20  # Every 20 days

        logger.info("ML4T Agent initialized with advanced ML capabilities")

    def _create_alpha_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create alpha factors using ML4T techniques"""
        try:
            factors = df.copy()

            # Price-based factors
            factors['returns'] = df['Close'].pct_change()
            factors['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

            # Momentum factors (ML4T Chapter 4)
            for period in [5, 10, 20, 60]:
                factors[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
                factors[f'mean_reversion_{period}'] = (
                    df['Close'] / df['Close'].rolling(period).mean() - 1
                )

            # Volatility factors
            for period in [5, 20, 60]:
                factors[f'volatility_{period}'] = (
                    factors['returns'].rolling(period).std() * np.sqrt(252)
                )
                factors[f'vol_ratio_{period}'] = (
                    factors[f'volatility_{period}'] /
                    factors['volatility_60'].rolling(60).mean()
                )

            # Volume factors
            factors['volume_sma_20'] = df['Volume'].rolling(20).mean()
            factors['volume_ratio'] = df['Volume'] / factors['volume_sma_20']
            factors['dollar_volume'] = df['Close'] * df['Volume']

            # Technical factors
            # Bollinger Bands
            bb_period = 20
            bb_middle = df['Close'].rolling(bb_period).mean()
            bb_std = df['Close'].rolling(bb_period).std()
            factors['bb_position'] = (df['Close'] - bb_middle) / (2 * bb_std)

            # RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            factors['rsi'] = 100 - (100 / (1 + rs))
            factors['rsi_normalized'] = (factors['rsi'] - 50) / 50

            # Price patterns (ML4T Chapter 13)
            factors['price_vs_high_52w'] = df['Close'] / df['High'].rolling(252).max()
            factors['price_vs_low_52w'] = df['Close'] / df['Low'].rolling(252).min()

            # Regime indicators
            factors['trend_strength'] = abs(
                df['Close'].rolling(20).mean() / df['Close'].rolling(60).mean() - 1
            )

            # Cross-sectional factors (ML4T Chapter 5)
            # Will be enhanced with market data

            return factors

        except Exception as e:
            logger.error(f"Alpha factor creation error: {e}")
            return df

    def _create_regime_features(self, df: pd.DataFrame, market_data: pd.DataFrame = None) -> pd.DataFrame:
        """Create regime detection features (ML4T Chapter 9)"""
        try:
            regime_features = pd.DataFrame(index=df.index)

            # Volatility regime
            vol_20 = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
            vol_60 = df['Close'].pct_change().rolling(60).std() * np.sqrt(252)
            regime_features['vol_regime'] = vol_20 / vol_60

            # Trend regime
            sma_20 = df['Close'].rolling(20).mean()
            sma_60 = df['Close'].rolling(60).mean()
            regime_features['trend_regime'] = sma_20 / sma_60 - 1

            # Market breadth (if market data available)
            if market_data is not None:
                regime_features['market_trend'] = (
                    market_data['Close'] / market_data['Close'].rolling(20).mean() - 1
                )

            # VIX-like measure
            returns = df['Close'].pct_change()
            regime_features['implied_vol'] = returns.rolling(20).std() * np.sqrt(252)

            return regime_features

        except Exception as e:
            logger.error(f"Regime feature creation error: {e}")
            return pd.DataFrame(index=df.index)

    def _create_factor_model(self, returns: pd.Series, factors: pd.DataFrame) -> FactorModel:
        """Create factor model using ML4T methodology"""
        try:
            if not STATS_AVAILABLE:
                return None

            # Align data
            aligned_data = pd.concat([returns, factors], axis=1).dropna()
            if len(aligned_data) < 30:
                return None

            y = aligned_data.iloc[:, 0]  # Returns
            X = aligned_data.iloc[:, 1:]  # Factors

            # Add constant for intercept
            X = sm.add_constant(X)

            # Fit factor model
            model = sm.OLS(y, X).fit()

            # Extract results
            factor_loadings = {}
            factor_returns = {}

            for i, factor in enumerate(X.columns):
                if factor != 'const':
                    factor_loadings[factor] = model.params[factor]
                    # Calculate factor return contribution
                    factor_returns[factor] = model.params[factor] * X[factor].mean()

            return FactorModel(
                factors=list(factor_loadings.keys()),
                loadings=factor_loadings,
                residual_variance=model.mse_resid,
                r_squared=model.rsquared,
                factor_returns=factor_returns
            )

        except Exception as e:
            logger.error(f"Factor model creation error: {e}")
            return None

    def _create_ensemble_model(self, X: pd.DataFrame, y: pd.Series,
                             prediction_type: str = 'classification') -> Dict[str, Any]:
        """Create ensemble model using ML4T techniques"""
        try:
            if not ML4T_AVAILABLE:
                return {}

            # Split data with time series considerations
            tscv = TimeSeriesSplit(n_splits=5)

            # Define base models
            if prediction_type == 'classification':
                base_models = {
                    'rf': RandomForestClassifier(n_estimators=100, random_state=42),
                    'gb': GradientBoostingClassifier(random_state=42),
                    'lr': LogisticRegression(random_state=42, max_iter=1000),
                    'svm': SVC(probability=True, random_state=42),
                    'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
                }

                if BOOSTING_AVAILABLE:
                    base_models['xgb'] = xgb.XGBClassifier(random_state=42)
                    base_models['lgb'] = lgb.LGBMClassifier(random_state=42)

            else:  # regression
                base_models = {
                    'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                    'gb': GradientBoostingClassifier(random_state=42),
                    'ridge': Ridge(random_state=42),
                    'svr': SVR(),
                    'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
                }

                if BOOSTING_AVAILABLE:
                    base_models['xgb'] = xgb.XGBRegressor(random_state=42)
                    base_models['lgb'] = lgb.LGBMRegressor(random_state=42)

            # Scale features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

            # Train models with cross-validation
            model_scores = {}
            trained_models = {}

            for name, model in base_models.items():
                try:
                    # Time series cross-validation
                    scores = cross_val_score(model, X_scaled, y, cv=tscv,
                                           scoring='accuracy' if prediction_type == 'classification' else 'neg_mean_squared_error')
                    model_scores[name] = scores.mean()

                    # Train on full dataset
                    model.fit(X_scaled, y)
                    trained_models[name] = model

                except Exception as e:
                    logger.warning(f"Model {name} training failed: {e}")
                    continue

            # Create ensemble weights based on performance
            if model_scores:
                total_score = sum(abs(score) for score in model_scores.values())
                ensemble_weights = {
                    name: abs(score) / total_score
                    for name, score in model_scores.items()
                }
            else:
                ensemble_weights = {}

            return {
                'models': trained_models,
                'scaler': scaler,
                'weights': ensemble_weights,
                'scores': model_scores,
                'prediction_type': prediction_type
            }

        except Exception as e:
            logger.error(f"Ensemble model creation error: {e}")
            return {}

    async def train_ml4t_model(self, symbol: str, features: List[str] = None) -> Dict[str, Any]:
        """Train comprehensive ML4T model"""
        try:
            # Get data
            stock = yf.Ticker(symbol)
            df = stock.history(period='2y')

            if len(df) < 100:
                return {'error': 'Insufficient data'}

            # Create alpha factors
            factors_df = self._create_alpha_factors(df)

            # Get market data for regime detection
            spy = yf.Ticker('SPY')
            market_df = spy.history(period='2y')
            regime_features = self._create_regime_features(df, market_df)

            # Combine features
            all_features = pd.concat([factors_df, regime_features], axis=1)

            # Select features
            if features is None:
                feature_cols = [col for col in all_features.columns
                              if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            else:
                feature_cols = features

            # Create targets for different horizons
            results = {}

            for horizon in self.prediction_horizons:
                # Classification target (up/down)
                future_returns = df['Close'].shift(-horizon) / df['Close'] - 1
                classification_target = (future_returns > 0).astype(int)

                # Regression target (actual returns)
                regression_target = future_returns

                # Clean data
                X = all_features[feature_cols].dropna()
                y_class = classification_target.loc[X.index].dropna()
                y_reg = regression_target.loc[X.index].dropna()

                # Align all data
                common_index = X.index.intersection(y_class.index).intersection(y_reg.index)
                X = X.loc[common_index]
                y_class = y_class.loc[common_index]
                y_reg = y_reg.loc[common_index]

                if len(X) < 50:
                    continue

                # Feature selection
                if len(feature_cols) > 20:
                    selector = SelectKBest(score_func=f_classif, k=20)
                    X_selected = selector.fit_transform(X, y_class)
                    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
                    X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

                # Train ensemble models
                classification_ensemble = self._create_ensemble_model(X, y_class, 'classification')
                regression_ensemble = self._create_ensemble_model(X, y_reg, 'regression')

                # Create factor model
                returns = df['Close'].pct_change().loc[common_index]
                factor_model = self._create_factor_model(returns, X)

                results[f'horizon_{horizon}d'] = {
                    'classification_ensemble': classification_ensemble,
                    'regression_ensemble': regression_ensemble,
                    'factor_model': factor_model,
                    'features': list(X.columns),
                    'training_samples': len(X)
                }

            # Store models
            self.models[symbol] = results

            return {
                'symbol': symbol,
                'horizons': list(results.keys()),
                'total_features': len(feature_cols),
                'training_completed': datetime.now(),
                'results': results
            }

        except Exception as e:
            logger.error(f"ML4T model training error for {symbol}: {e}")
            return {'error': str(e)}

    async def predict_ml4t(self, symbol: str, horizon: int = 1) -> Optional[ML4TPrediction]:
        """Generate ML4T prediction"""
        try:
            if symbol not in self.models:
                # Train model first
                training_result = await self.train_ml4t_model(symbol)
                if 'error' in training_result:
                    return None

            horizon_key = f'horizon_{horizon}d'
            if horizon_key not in self.models[symbol]:
                return None

            model_data = self.models[symbol][horizon_key]

            # Get current data
            stock = yf.Ticker(symbol)
            df = stock.history(period='6mo')

            if df.empty:
                return None

            # Create features
            factors_df = self._create_alpha_factors(df)
            spy = yf.Ticker('SPY')
            market_df = spy.history(period='6mo')
            regime_features = self._create_regime_features(df, market_df)
            all_features = pd.concat([factors_df, regime_features], axis=1)

            # Get latest features
            latest_features = all_features[model_data['features']].iloc[-1:].dropna(axis=1)

            if latest_features.empty:
                return None

            # Make predictions
            classification_ensemble = model_data['classification_ensemble']
            regression_ensemble = model_data['regression_ensemble']

            predictions = {}

            # Classification prediction
            if classification_ensemble and 'models' in classification_ensemble:
                class_preds = []
                class_probs = []

                scaler = classification_ensemble['scaler']
                X_scaled = scaler.transform(latest_features)

                for name, model in classification_ensemble['models'].items():
                    try:
                        pred = model.predict(X_scaled)[0]
                        prob = model.predict_proba(X_scaled)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]

                        weight = classification_ensemble['weights'].get(name, 1.0)
                        class_preds.append(pred * weight)
                        class_probs.append(prob * weight)
                    except:
                        continue

                if class_preds:
                    ensemble_pred = np.sum(class_preds) / len(class_preds)
                    ensemble_prob = np.mean(class_probs, axis=0)
                    predictions['classification'] = {
                        'prediction': int(ensemble_pred > 0.5),
                        'probability': ensemble_prob,
                        'confidence': max(ensemble_prob)
                    }

            # Regression prediction
            if regression_ensemble and 'models' in regression_ensemble:
                reg_preds = []

                scaler = regression_ensemble['scaler']
                X_scaled = scaler.transform(latest_features)

                for name, model in regression_ensemble['models'].items():
                    try:
                        pred = model.predict(X_scaled)[0]
                        weight = regression_ensemble['weights'].get(name, 1.0)
                        reg_preds.append(pred * weight)
                    except:
                        continue

                if reg_preds:
                    ensemble_pred = np.mean(reg_preds)
                    predictions['regression'] = {
                        'predicted_return': ensemble_pred,
                        'confidence': min(abs(ensemble_pred) * 10, 1.0)  # Simple confidence measure
                    }

            # Combine predictions
            if 'classification' in predictions and 'regression' in predictions:
                class_pred = predictions['classification']['prediction']
                reg_pred = predictions['regression']['predicted_return']
                class_conf = predictions['classification']['confidence']
                reg_conf = predictions['regression']['confidence']

                # Ensemble confidence
                final_confidence = (class_conf + reg_conf) / 2

                # Final prediction
                if class_pred == 1 and reg_pred > 0:
                    final_prediction = 'STRONG_BUY'
                elif class_pred == 1 or reg_pred > 0.01:
                    final_prediction = 'BUY'
                elif class_pred == 0 and reg_pred < 0:
                    final_prediction = 'STRONG_SELL'
                elif class_pred == 0 or reg_pred < -0.01:
                    final_prediction = 'SELL'
                else:
                    final_prediction = 'HOLD'

                return ML4TPrediction(
                    symbol=symbol,
                    prediction_type='ensemble',
                    prediction=final_prediction,
                    confidence=final_confidence,
                    probability_distribution=predictions,
                    model_metrics={
                        'classification_scores': classification_ensemble.get('scores', {}),
                        'regression_scores': regression_ensemble.get('scores', {})
                    },
                    timestamp=datetime.now()
                )

            return None

        except Exception as e:
            logger.error(f"ML4T prediction error for {symbol}: {e}")
            return None

    async def get_factor_exposure(self, symbol: str, horizon: int = 1) -> Optional[Dict[str, float]]:
        """Get factor exposure analysis"""
        try:
            if symbol not in self.models:
                return None

            horizon_key = f'horizon_{horizon}d'
            if horizon_key not in self.models[symbol]:
                return None

            factor_model = self.models[symbol][horizon_key].get('factor_model')
            if not factor_model:
                return None

            return {
                'factor_loadings': factor_model.loadings,
                'factor_returns': factor_model.factor_returns,
                'r_squared': factor_model.r_squared,
                'residual_variance': factor_model.residual_variance
            }

        except Exception as e:
            logger.error(f"Factor exposure error for {symbol}: {e}")
            return None

# Create singleton instance
ml4t_agent = ML4TAgent()