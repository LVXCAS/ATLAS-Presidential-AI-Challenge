"""
Hive Trade Enhanced AI Models V2 - Options-Focused Improvements
Major upgrades:
1. Options-specific prediction labels (profitability, not just direction)
2. VIX and market regime features
3. IV percentile and options volume features
4. Time-based features (day of week, month, etc.)
5. LightGBM model added to ensemble
6. Increased estimators (200 -> 500)
7. Better feature engineering for options trading
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import xgboost as xgb
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging
import json
import pickle
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Try to import LightGBM (free and powerful)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available - installing...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "lightgbm"], check=False)
    try:
        import lightgbm as lgb
        LIGHTGBM_AVAILABLE = True
    except:
        LIGHTGBM_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """AI model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    mse: float
    r2_score: float
    feature_importance: Dict[str, float]
    training_time: float
    inference_time: float

class MarketRegimeDetector:
    """
    Advanced market regime detection using multiple ML models
    Now includes VIX-based features
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.regime_labels = {0: 'bear', 1: 'sideways', 2: 'bull', 3: 'volatile'}
        self.vix_data = None

    def load_vix_data(self):
        """Load VIX data for market regime detection"""
        try:
            vix = yf.Ticker("^VIX")
            self.vix_data = vix.history(period="5y", auto_adjust=True)
            logger.info(f"Loaded VIX data: {len(self.vix_data)} days")
        except Exception as e:
            logger.warning(f"Could not load VIX data: {e}")
            self.vix_data = None

    def prepare_regime_features(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Prepare features for regime detection with VIX"""
        df = data.copy()

        # Price-based features
        df['returns_1d'] = df['Close'].pct_change()
        df['returns_5d'] = df['Close'].pct_change(5)
        df['returns_20d'] = df['Close'].pct_change(20)
        df['returns_60d'] = df['Close'].pct_change(60)

        # Volatility features
        df['vol_5d'] = df['returns_1d'].rolling(5).std()
        df['vol_20d'] = df['returns_1d'].rolling(20).std()
        df['vol_60d'] = df['returns_1d'].rolling(60).std()

        # Trend features
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_60'] = df['Close'].rolling(60).mean()
        df['trend_5_20'] = df['sma_5'] / df['sma_20'] - 1
        df['trend_20_60'] = df['sma_20'] / df['sma_60'] - 1

        # Volume features
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']

        # Price position features
        df['high_52w'] = df['High'].rolling(252).max()
        df['low_52w'] = df['Low'].rolling(252).min()
        df['price_position'] = (df['Close'] - df['low_52w']) / (df['high_52w'] - df['low_52w'])

        # Momentum features
        df['rsi'] = self.calculate_rsi(df['Close'])
        df['macd'] = df['Close'].ewm(12).mean() - df['Close'].ewm(26).mean()
        df['macd_signal'] = df['macd'].ewm(9).mean()

        # VIX features (NEW!)
        if self.vix_data is not None:
            # Align VIX data with stock data
            df['vix_level'] = 0.0
            df['vix_change'] = 0.0
            df['vix_percentile'] = 0.5

            for idx in df.index:
                if idx in self.vix_data.index:
                    df.loc[idx, 'vix_level'] = self.vix_data.loc[idx, 'Close']

            # Calculate VIX features
            df['vix_change'] = df['vix_level'].pct_change()
            df['vix_sma_20'] = df['vix_level'].rolling(20).mean()
            df['vix_percentile'] = df['vix_level'].rolling(252).apply(
                lambda x: (x.iloc[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5
            )
        else:
            df['vix_level'] = 20.0  # Default mid-level VIX
            df['vix_change'] = 0.0
            df['vix_percentile'] = 0.5

        return df

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def label_market_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Label market regimes based on price, volatility, and VIX"""
        df = data.copy()

        # Calculate regime indicators
        trend_strength = df['returns_20d'].rolling(20).mean()
        volatility_level = df['vol_20d'].rolling(20).mean()
        vix_level = df.get('vix_level', 20.0)

        # Define thresholds
        trend_threshold = 0.02  # 2% monthly trend
        vol_threshold = df['vol_20d'].quantile(0.7)  # 70th percentile volatility
        vix_high = 25  # High VIX threshold

        # Regime mapping: 0=bear, 1=sideways, 2=bull, 3=volatile
        regimes = pd.Series(index=df.index, dtype=int)

        for i in range(len(df)):
            if pd.isna(trend_strength.iloc[i]) or pd.isna(volatility_level.iloc[i]):
                regimes.iloc[i] = 1  # sideways
                continue

            trend = trend_strength.iloc[i]
            vol = volatility_level.iloc[i]
            vix = vix_level.iloc[i] if hasattr(vix_level, 'iloc') else vix_level

            # High VIX = volatile regime
            if vix > vix_high or vol > vol_threshold:
                regimes.iloc[i] = 3  # volatile
            elif trend > trend_threshold:
                regimes.iloc[i] = 2  # bull
            elif trend < -trend_threshold:
                regimes.iloc[i] = 0  # bear
            else:
                regimes.iloc[i] = 1  # sideways

        return regimes

    def train_regime_detector(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Train regime detection models"""
        logger.info("Training market regime detection models...")

        # Load VIX data first
        self.load_vix_data()

        # Combine data from all symbols
        combined_features = []
        combined_labels = []

        for symbol, df in data.items():
            # Prepare features (now with VIX)
            features_df = self.prepare_regime_features(df, symbol)

            # Generate labels
            regime_labels = self.label_market_regimes(features_df)

            # Select feature columns
            feature_cols = [
                'returns_1d', 'returns_5d', 'returns_20d', 'returns_60d',
                'vol_5d', 'vol_20d', 'vol_60d',
                'trend_5_20', 'trend_20_60',
                'volume_ratio', 'price_position',
                'rsi', 'macd', 'macd_signal',
                'vix_level', 'vix_change', 'vix_percentile'  # NEW VIX features
            ]

            # Clean data
            feature_data = features_df[feature_cols].dropna()
            regime_data = regime_labels.loc[feature_data.index]

            if len(feature_data) > 100:  # Minimum data requirement
                combined_features.append(feature_data)
                combined_labels.append(regime_data)

        if not combined_features:
            logger.error("No valid data for regime detection training")
            return {}

        # Combine all data
        X = pd.concat(combined_features, ignore_index=True)
        y = pd.concat(combined_labels, ignore_index=True)

        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        logger.info(f"Training regime detector with {len(X)} samples, {len(X.columns)} features")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest (increased estimators)
        rf_model = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)

        # Train XGBoost (increased estimators)
        xgb_model = xgb.XGBClassifier(n_estimators=500, max_depth=8, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train_scaled, y_train)

        # Train LightGBM (NEW!)
        if LIGHTGBM_AVAILABLE:
            lgb_model = lgb.LGBMClassifier(n_estimators=500, max_depth=8, random_state=42, n_jobs=-1, verbose=-1)
            lgb_model.fit(X_train_scaled, y_train)

        # Evaluate models
        rf_score = rf_model.score(X_test_scaled, y_test)
        xgb_score = xgb_model.score(X_test_scaled, y_test)
        lgb_score = lgb_model.score(X_test_scaled, y_test) if LIGHTGBM_AVAILABLE else 0.0

        # Store models
        self.models['regime_rf'] = rf_model
        self.models['regime_xgb'] = xgb_model
        if LIGHTGBM_AVAILABLE:
            self.models['regime_lgb'] = lgb_model
        self.scalers['regime'] = scaler

        logger.info(f"Regime detection trained - RF: {rf_score:.3f}, XGB: {xgb_score:.3f}, LGB: {lgb_score:.3f}")

        return {
            'random_forest_accuracy': rf_score,
            'xgboost_accuracy': xgb_score,
            'lightgbm_accuracy': lgb_score,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(X.columns)
        }

class EnhancedTradingModel:
    """
    Enhanced trading model with options-specific improvements
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.vix_data = None

    def load_vix_data(self):
        """Load VIX data"""
        try:
            vix = yf.Ticker("^VIX")
            self.vix_data = vix.history(period="5y", auto_adjust=True)
            logger.info(f"Loaded VIX data for trading model: {len(self.vix_data)} days")
        except Exception as e:
            logger.warning(f"Could not load VIX data: {e}")
            self.vix_data = None

    def prepare_trading_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive features for trading models - OPTIONS FOCUSED"""
        df = data.copy()

        # Price features
        df['returns_1d'] = df['Close'].pct_change()
        df['returns_3d'] = df['Close'].pct_change(3)
        df['returns_5d'] = df['Close'].pct_change(5)
        df['returns_10d'] = df['Close'].pct_change(10)
        df['returns_20d'] = df['Close'].pct_change(20)

        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(period).mean()
            df[f'price_to_sma_{period}'] = df['Close'] / df[f'sma_{period}'] - 1

        # Technical indicators
        df['rsi'] = self.calculate_rsi(df['Close'])
        df['macd'] = df['Close'].ewm(12).mean() - df['Close'].ewm(26).mean()
        df['macd_signal'] = df['macd'].ewm(9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        bb_mean = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = bb_mean + (bb_std * 2)
        df['bb_lower'] = bb_mean - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['Close']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Volatility features (IMPORTANT for options!)
        df['volatility_5d'] = df['returns_1d'].rolling(5).std()
        df['volatility_20d'] = df['returns_1d'].rolling(20).std()
        df['volatility_60d'] = df['returns_1d'].rolling(60).std()
        df['volatility_ratio'] = df['volatility_5d'] / df['volatility_20d']

        # Annualized realized volatility (for options pricing)
        df['realized_vol_annual'] = df['volatility_20d'] * np.sqrt(252)

        # Volume features
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        df['price_volume'] = df['Close'] * df['Volume']
        df['volume_momentum'] = df['Volume'].pct_change(5)

        # High/Low features
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_to_high'] = df['Close'] / df['High']
        df['close_to_low'] = df['Close'] / df['Low']

        # Momentum and trend
        df['momentum_3d'] = df['Close'] / df['Close'].shift(3) - 1
        df['momentum_10d'] = df['Close'] / df['Close'].shift(10) - 1
        df['trend_strength'] = df['returns_20d'].rolling(10).mean()

        # VIX features (NEW!)
        if self.vix_data is not None:
            df['vix_level'] = 0.0
            df['vix_change'] = 0.0
            df['vix_percentile'] = 0.5

            for idx in df.index:
                if idx in self.vix_data.index:
                    df.loc[idx, 'vix_level'] = self.vix_data.loc[idx, 'Close']

            df['vix_change'] = df['vix_level'].pct_change()
            df['vix_sma_20'] = df['vix_level'].rolling(20).mean()
            df['vix_percentile'] = df['vix_level'].rolling(252).apply(
                lambda x: (x.iloc[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5
            )
        else:
            df['vix_level'] = 20.0
            df['vix_change'] = 0.0
            df['vix_percentile'] = 0.5

        # Time-based features (NEW!)
        df['day_of_week'] = df.index.dayofweek / 4.0  # Normalize 0-1
        df['month'] = df.index.month / 12.0  # Normalize 0-1
        df['quarter'] = df.index.quarter / 4.0  # Normalize 0-1

        # Estimated IV percentile (based on realized vol)
        df['iv_percentile_est'] = df['realized_vol_annual'].rolling(252).apply(
            lambda x: (x.iloc[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5
        )

        return df

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def create_options_labels(self, data: pd.DataFrame, forward_days: int = 5) -> pd.Series:
        """
        Create OPTIONS-SPECIFIC labels based on profitability

        This is a MAJOR IMPROVEMENT - instead of predicting stock direction,
        we predict whether an options trade would be profitable!

        For a 1-week ATM call option:
        - Needs price gain > theta decay (~5% per week)
        - Label 1 = profitable trade
        - Label 0 = unprofitable trade
        """

        # Calculate forward price change
        forward_returns = data['Close'].shift(-forward_days) / data['Close'] - 1

        # Estimate theta decay for 1-week ATM option
        # Theta decay ≈ 5-7% per week for ATM options
        # Add volatility adjustment (higher vol = higher premium = more decay)
        estimated_vol = data.get('realized_vol_annual', 30.0)
        theta_decay = 0.05 + (estimated_vol / 100) * 0.02  # Base 5% + vol adjustment

        # For CALLS: profit if price gains exceed theta decay
        # For PUTS: profit if price drops more than theta decay
        # We'll train on both scenarios

        labels = pd.Series(index=data.index, dtype=int)

        # CALL would be profitable
        call_profitable = forward_returns > theta_decay

        # PUT would be profitable
        put_profitable = forward_returns < -theta_decay

        # Label 2 = STRONG BUY (call profitable)
        # Label 1 = NEUTRAL (neither very profitable)
        # Label 0 = STRONG SELL (put profitable)

        labels[call_profitable] = 2
        labels[put_profitable] = 0
        labels[~(call_profitable | put_profitable)] = 1

        return labels

    def train_trading_models(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train enhanced trading models with options-specific labels"""
        logger.info("Training enhanced trading models with OPTIONS-SPECIFIC labels...")

        # Load VIX data first
        self.load_vix_data()

        # Combine data from all symbols
        combined_features = []
        combined_labels = []

        for symbol, df in data.items():
            # Prepare features
            features_df = self.prepare_trading_features(df)

            # Create OPTIONS-SPECIFIC labels (NEW!)
            labels = self.create_options_labels(features_df)

            # Select feature columns (expanded with new features)
            feature_cols = [
                'returns_1d', 'returns_3d', 'returns_5d', 'returns_10d', 'returns_20d',
                'price_to_sma_5', 'price_to_sma_10', 'price_to_sma_20', 'price_to_sma_50',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_width', 'bb_position',
                'volatility_5d', 'volatility_20d', 'volatility_ratio', 'realized_vol_annual',
                'volume_ratio', 'volume_momentum',
                'high_low_ratio', 'close_to_high', 'close_to_low',
                'momentum_3d', 'momentum_10d', 'trend_strength',
                'vix_level', 'vix_change', 'vix_percentile',  # VIX features
                'day_of_week', 'month', 'quarter',  # Time features
                'iv_percentile_est'  # IV estimate
            ]

            # Clean data
            feature_data = features_df[feature_cols].dropna()
            label_data = labels.loc[feature_data.index].dropna()

            # Align data
            common_index = feature_data.index.intersection(label_data.index)
            if len(common_index) > 100:
                feature_data = feature_data.loc[common_index]
                label_data = label_data.loc[common_index]

                combined_features.append(feature_data)
                combined_labels.append(label_data)

        if not combined_features:
            logger.error("No valid data for trading model training")
            return {}

        # Combine all data
        X = pd.concat(combined_features, ignore_index=True)
        y = pd.concat(combined_labels, ignore_index=True)

        self.feature_columns = X.columns.tolist()

        logger.info(f"Training trading models with {len(X)} samples, {len(self.feature_columns)} features")
        logger.info(f"Using OPTIONS-SPECIFIC profitability labels")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        results = {}

        # Train Random Forest Classifier (500 estimators!)
        logger.info("Training Random Forest (500 estimators)...")
        rf_clf = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
        rf_clf.fit(X_train_scaled, y_train)
        rf_score = rf_clf.score(X_test_scaled, y_test)

        self.models['trading_rf_clf'] = rf_clf
        results['rf_classifier_accuracy'] = rf_score

        # Train XGBoost Classifier (500 estimators!)
        logger.info("Training XGBoost (500 estimators)...")
        xgb_clf = xgb.XGBClassifier(n_estimators=500, max_depth=8, random_state=42, n_jobs=-1)
        xgb_clf.fit(X_train_scaled, y_train)
        xgb_score = xgb_clf.score(X_test_scaled, y_test)

        self.models['trading_xgb_clf'] = xgb_clf
        results['xgb_classifier_accuracy'] = xgb_score

        # Train LightGBM (NEW! 500 estimators!)
        if LIGHTGBM_AVAILABLE:
            logger.info("Training LightGBM (500 estimators)...")
            lgb_clf = lgb.LGBMClassifier(n_estimators=500, max_depth=8, random_state=42, n_jobs=-1, verbose=-1)
            lgb_clf.fit(X_train_scaled, y_train)
            lgb_score = lgb_clf.score(X_test_scaled, y_test)

            self.models['trading_lgb_clf'] = lgb_clf
            results['lgb_classifier_accuracy'] = lgb_score

        # Train Gradient Boosting Regressor for return prediction (500 estimators!)
        logger.info("Training Gradient Boosting Regressor (500 estimators)...")
        y_returns = []
        for symbol, df in data.items():
            returns = df['Close'].shift(-5) / df['Close'] - 1  # 5-day forward returns
            returns = returns.dropna()
            if len(returns) > 100:
                y_returns.append(returns)

        if y_returns:
            y_returns_combined = pd.concat(y_returns, ignore_index=True)

            # Align with features
            min_len = min(len(X), len(y_returns_combined))
            X_returns = X.iloc[:min_len]
            y_returns_aligned = y_returns_combined.iloc[:min_len]

            X_train_ret, X_test_ret, y_train_ret, y_test_ret = train_test_split(
                X_returns, y_returns_aligned, test_size=0.2, random_state=42)

            X_train_ret_scaled = scaler.fit_transform(X_train_ret)
            X_test_ret_scaled = scaler.transform(X_test_ret)

            gbr = GradientBoostingRegressor(n_estimators=500, max_depth=8, random_state=42)
            gbr.fit(X_train_ret_scaled, y_train_ret)

            gbr_pred = gbr.predict(X_test_ret_scaled)
            gbr_mse = mean_squared_error(y_test_ret, gbr_pred)
            gbr_r2 = r2_score(y_test_ret, gbr_pred)

            self.models['trading_gbr'] = gbr
            results['gbr_mse'] = gbr_mse
            results['gbr_r2'] = gbr_r2

        self.scalers['trading'] = scaler

        logger.info(f"Trading models trained:")
        logger.info(f"  RF: {rf_score:.3f}")
        logger.info(f"  XGB: {xgb_score:.3f}")
        if LIGHTGBM_AVAILABLE:
            logger.info(f"  LightGBM: {lgb_score:.3f}")
        logger.info(f"  Features: {len(self.feature_columns)}")

        return results
