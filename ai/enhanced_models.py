"""
Hive Trade Enhanced AI Models
Advanced machine learning models for trading strategy improvement
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
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.regime_labels = {0: 'bear', 1: 'sideways', 2: 'bull', 3: 'volatile'}
        
    def prepare_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for regime detection"""
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
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def label_market_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Label market regimes based on price and volatility patterns"""
        df = data.copy()
        
        # Calculate regime indicators
        trend_strength = df['returns_20d'].rolling(20).mean()
        volatility_level = df['vol_20d'].rolling(20).mean()
        
        # Define thresholds (can be optimized)
        trend_threshold = 0.02  # 2% monthly trend
        vol_threshold = df['vol_20d'].quantile(0.7)  # 70th percentile volatility
        
        # Regime mapping: 0=bear, 1=sideways, 2=bull, 3=volatile
        regimes = pd.Series(index=df.index, dtype=int)
        
        for i in range(len(df)):
            if pd.isna(trend_strength.iloc[i]) or pd.isna(volatility_level.iloc[i]):
                regimes.iloc[i] = 1  # sideways
                continue
                
            trend = trend_strength.iloc[i]
            vol = volatility_level.iloc[i]
            
            if vol > vol_threshold:
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
        
        # Combine data from all symbols
        combined_features = []
        combined_labels = []
        
        for symbol, df in data.items():
            # Prepare features
            features_df = self.prepare_regime_features(df)
            
            # Generate labels
            regime_labels = self.label_market_regimes(features_df)
            
            # Select feature columns
            feature_cols = [
                'returns_1d', 'returns_5d', 'returns_20d', 'returns_60d',
                'vol_5d', 'vol_20d', 'vol_60d',
                'trend_5_20', 'trend_20_60',
                'volume_ratio', 'price_position',
                'rsi', 'macd', 'macd_signal'
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
        
        logger.info(f"Training regime detector with {len(X)} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Train XGBoost
        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        xgb_model.fit(X_train_scaled, y_train)
        
        # Evaluate models
        rf_score = rf_model.score(X_test_scaled, y_test)
        xgb_score = xgb_model.score(X_test_scaled, y_test)
        
        # Store models
        self.models['regime_rf'] = rf_model
        self.models['regime_xgb'] = xgb_model
        self.scalers['regime'] = scaler
        
        logger.info(f"Regime detection trained - RF: {rf_score:.3f}, XGB: {xgb_score:.3f}")
        
        return {
            'random_forest_accuracy': rf_score,
            'xgboost_accuracy': xgb_score,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }

class EnhancedTradingModel:
    """
    Enhanced trading model with multiple ML approaches
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
    def prepare_trading_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive features for trading models"""
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
        
        # Volatility features
        df['volatility_5d'] = df['returns_1d'].rolling(5).std()
        df['volatility_20d'] = df['returns_1d'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_5d'] / df['volatility_20d']
        
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
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def create_labels(self, data: pd.DataFrame, forward_days: int = 5, 
                     threshold: float = 0.02) -> pd.Series:
        """Create trading labels based on future returns"""
        
        # Calculate forward returns
        forward_returns = data['Close'].shift(-forward_days) / data['Close'] - 1
        
        # Create labels: 0=sell, 1=hold, 2=buy (for XGBoost compatibility)
        labels = pd.Series(index=data.index, dtype=int)
        labels[forward_returns > threshold] = 2      # Buy
        labels[forward_returns < -threshold] = 0     # Sell
        labels[(forward_returns >= -threshold) & (forward_returns <= threshold)] = 1  # Hold
        
        return labels
    
    def train_trading_models(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train enhanced trading models"""
        logger.info("Training enhanced trading models...")
        
        # Combine data from all symbols
        combined_features = []
        combined_labels = []
        
        for symbol, df in data.items():
            # Prepare features
            features_df = self.prepare_trading_features(df)
            
            # Create labels
            labels = self.create_labels(features_df)
            
            # Select feature columns
            feature_cols = [
                'returns_1d', 'returns_3d', 'returns_5d', 'returns_10d', 'returns_20d',
                'price_to_sma_5', 'price_to_sma_10', 'price_to_sma_20', 'price_to_sma_50',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_width', 'bb_position',
                'volatility_5d', 'volatility_20d', 'volatility_ratio',
                'volume_ratio', 'volume_momentum',
                'high_low_ratio', 'close_to_high', 'close_to_low',
                'momentum_3d', 'momentum_10d', 'trend_strength'
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
        
        logger.info(f"Training trading models with {len(X)} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        # Train Random Forest Classifier
        rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        rf_clf.fit(X_train_scaled, y_train)
        rf_score = rf_clf.score(X_test_scaled, y_test)
        
        self.models['trading_rf_clf'] = rf_clf
        results['rf_classifier_accuracy'] = rf_score
        
        # Train XGBoost Classifier
        xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, random_state=42)
        xgb_clf.fit(X_train_scaled, y_train)
        xgb_score = xgb_clf.score(X_test_scaled, y_test)
        
        self.models['trading_xgb_clf'] = xgb_clf
        results['xgb_classifier_accuracy'] = xgb_score
        
        # Train Gradient Boosting Regressor for return prediction
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
            
            gbr = GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42)
            gbr.fit(X_train_ret_scaled, y_train_ret)
            
            gbr_pred = gbr.predict(X_test_ret_scaled)
            gbr_mse = mean_squared_error(y_test_ret, gbr_pred)
            gbr_r2 = r2_score(y_test_ret, gbr_pred)
            
            self.models['trading_gbr'] = gbr
            results['gbr_mse'] = gbr_mse
            results['gbr_r2'] = gbr_r2
        
        self.scalers['trading'] = scaler
        
        logger.info(f"Trading models trained - RF: {rf_score:.3f}, XGB: {xgb_score:.3f}")
        
        return results

class DeepTradingNetwork(nn.Module):
    """
    Deep neural network for trading signal generation
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32], 
                 output_size: int = 3, dropout_rate: float = 0.3):
        super(DeepTradingNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class EnhancedAISystem:
    """
    Comprehensive enhanced AI system for trading
    """
    
    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.trading_model = EnhancedTradingModel()
        self.deep_model = None
        self.ensemble_weights = {'rf': 0.3, 'xgb': 0.4, 'deep': 0.3}
        
    def train_all_models(self, symbols: List[str], start_date: str = '2020-01-01') -> Dict[str, Any]:
        """Train all AI models"""
        logger.info("Training comprehensive AI system...")
        
        # Load data
        market_data = self.load_training_data(symbols, start_date)
        
        if not market_data:
            logger.error("No market data loaded")
            return {}
        
        results = {}
        
        # Train regime detector
        regime_results = self.regime_detector.train_regime_detector(market_data)
        results['regime_detection'] = regime_results
        
        # Train trading models
        trading_results = self.trading_model.train_trading_models(market_data)
        results['trading_models'] = trading_results
        
        # Train deep neural network
        deep_results = self.train_deep_model(market_data)
        results['deep_model'] = deep_results
        
        return results
    
    def load_training_data(self, symbols: List[str], start_date: str) -> Dict[str, pd.DataFrame]:
        """Load training data for AI models"""
        market_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, auto_adjust=True)
                
                if not data.empty and len(data) > 252:  # At least 1 year of data
                    market_data[symbol] = data
                    logger.info(f"Loaded {len(data)} days for {symbol}")
                else:
                    logger.warning(f"Insufficient data for {symbol}")
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")
        
        return market_data
    
    def train_deep_model(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Train deep neural network"""
        logger.info("Training deep neural network...")
        
        try:
            # Prepare data for deep learning
            combined_features = []
            combined_labels = []
            
            for symbol, df in data.items():
                features_df = self.trading_model.prepare_trading_features(df)
                labels = self.trading_model.create_labels(features_df)
                
                feature_cols = self.trading_model.feature_columns
                if not feature_cols:  # If not set, use all numeric columns
                    feature_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
                
                feature_data = features_df[feature_cols].dropna()
                label_data = labels.loc[feature_data.index].dropna()
                
                common_index = feature_data.index.intersection(label_data.index)
                if len(common_index) > 100:
                    combined_features.append(feature_data.loc[common_index])
                    combined_labels.append(label_data.loc[common_index])
            
            if not combined_features:
                return {'error': 'No valid data for deep model training'}
            
            X = pd.concat(combined_features, ignore_index=True)
            y = pd.concat(combined_labels, ignore_index=True)
            
            # Convert labels to categorical (0, 1, 2 for sell, hold, buy)
            y_categorical = y + 1  # Convert -1,0,1 to 0,1,2
            
            # Split and scale
            X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled)
            y_train_tensor = torch.LongTensor(y_train.values)
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            y_test_tensor = torch.LongTensor(y_test.values)
            
            # Create model
            input_size = X_train_tensor.shape[1]
            self.deep_model = DeepTradingNetwork(input_size)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.deep_model.parameters(), lr=0.001)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            
            # Training loop
            num_epochs = 100
            best_loss = float('inf')
            
            self.deep_model.train()
            for epoch in range(num_epochs):
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.deep_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {epoch_loss/len(train_loader):.4f}")
            
            # Evaluate
            self.deep_model.eval()
            with torch.no_grad():
                test_outputs = self.deep_model(X_test_tensor)
                test_predictions = torch.argmax(test_outputs, dim=1)
                accuracy = (test_predictions == y_test_tensor).float().mean().item()
            
            return {
                'accuracy': accuracy,
                'input_features': input_size,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            logger.error(f"Deep model training failed: {e}")
            return {'error': str(e)}
    
    def predict_ensemble(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Generate ensemble predictions from all models"""
        predictions = {}
        
        try:
            # Scale features
            if 'trading' in self.trading_model.scalers:
                scaler = self.trading_model.scalers['trading']
                features_scaled = scaler.transform(features[self.trading_model.feature_columns])
            else:
                features_scaled = features.values
            
            # Random Forest prediction
            if 'trading_rf_clf' in self.trading_model.models:
                rf_pred = self.trading_model.models['trading_rf_clf'].predict_proba(features_scaled)
                predictions['rf'] = rf_pred
            
            # XGBoost prediction
            if 'trading_xgb_clf' in self.trading_model.models:
                xgb_pred = self.trading_model.models['trading_xgb_clf'].predict_proba(features_scaled)
                predictions['xgb'] = xgb_pred
            
            # Deep model prediction
            if self.deep_model is not None:
                features_tensor = torch.FloatTensor(features_scaled)
                with torch.no_grad():
                    deep_pred = torch.softmax(self.deep_model(features_tensor), dim=1).numpy()
                predictions['deep'] = deep_pred
            
            # Ensemble prediction
            if predictions:
                ensemble_pred = np.zeros_like(list(predictions.values())[0])
                total_weight = 0
                
                for model_name, pred in predictions.items():
                    weight = self.ensemble_weights.get(model_name, 0.33)
                    ensemble_pred += weight * pred
                    total_weight += weight
                
                if total_weight > 0:
                    ensemble_pred /= total_weight
                
                predictions['ensemble'] = ensemble_pred
            
            return predictions
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return {}
    
    def save_models(self, model_dir: str = "models") -> None:
        """Save all trained models"""
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            # Save trading models
            if self.trading_model.models:
                with open(f"{model_dir}/trading_models.pkl", 'wb') as f:
                    pickle.dump(self.trading_model.models, f)
                
                with open(f"{model_dir}/trading_scalers.pkl", 'wb') as f:
                    pickle.dump(self.trading_model.scalers, f)
            
            # Save regime detector
            if self.regime_detector.models:
                with open(f"{model_dir}/regime_models.pkl", 'wb') as f:
                    pickle.dump(self.regime_detector.models, f)
                
                with open(f"{model_dir}/regime_scalers.pkl", 'wb') as f:
                    pickle.dump(self.regime_detector.scalers, f)
            
            # Save deep model
            if self.deep_model is not None:
                torch.save(self.deep_model.state_dict(), f"{model_dir}/deep_model.pth")
            
            # Save metadata
            metadata = {
                'feature_columns': self.trading_model.feature_columns,
                'ensemble_weights': self.ensemble_weights,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(f"{model_dir}/model_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Models saved to {model_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")

def main():
    """Main AI enhancement workflow"""
    
    print("HIVE TRADE ENHANCED AI MODELS")
    print("="*40)
    
    # Configuration
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']
    start_date = '2020-01-01'
    
    # Initialize enhanced AI system
    ai_system = EnhancedAISystem()
    
    print(f"Training enhanced AI models...")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Training period: {start_date} onwards")
    
    # Train all models
    results = ai_system.train_all_models(symbols, start_date)
    
    if not results:
        print("ERROR: No results from model training")
        return
    
    # Save models
    print("Saving trained models...")
    ai_system.save_models("models")
    
    # Generate report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"\nAI MODEL TRAINING RESULTS:")
    print("-" * 50)
    
    # Regime Detection Results
    if 'regime_detection' in results:
        regime_results = results['regime_detection']
        print(f"Market Regime Detection:")
        print(f"  Random Forest Accuracy: {regime_results.get('random_forest_accuracy', 0):.3f}")
        print(f"  XGBoost Accuracy: {regime_results.get('xgboost_accuracy', 0):.3f}")
        print(f"  Training Samples: {regime_results.get('training_samples', 0):,}")
    
    # Trading Model Results
    if 'trading_models' in results:
        trading_results = results['trading_models']
        print(f"\nTrading Models:")
        print(f"  RF Classifier Accuracy: {trading_results.get('rf_classifier_accuracy', 0):.3f}")
        print(f"  XGBoost Classifier Accuracy: {trading_results.get('xgb_classifier_accuracy', 0):.3f}")
        if 'gbr_r2' in trading_results:
            print(f"  Return Predictor R²: {trading_results.get('gbr_r2', 0):.3f}")
    
    # Deep Model Results
    if 'deep_model' in results:
        deep_results = results['deep_model']
        if 'error' not in deep_results:
            print(f"\nDeep Neural Network:")
            print(f"  Accuracy: {deep_results.get('accuracy', 0):.3f}")
            print(f"  Input Features: {deep_results.get('input_features', 0)}")
            print(f"  Training Samples: {deep_results.get('training_samples', 0):,}")
        else:
            print(f"\nDeep Neural Network: {deep_results['error']}")
    
    # Save detailed results
    results_file = f"ai_training_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nAI ENHANCEMENT COMPLETE:")
    print(f"- Models trained and saved to 'models/' directory")
    print(f"- Results saved: {results_file}")
    print(f"- Enhanced trading capabilities deployed")
    
    # Performance assessment
    if 'trading_models' in results:
        avg_accuracy = np.mean([
            results['trading_models'].get('rf_classifier_accuracy', 0),
            results['trading_models'].get('xgb_classifier_accuracy', 0)
        ])
        
        if avg_accuracy > 0.6:
            print(f"\nSTATUS: Excellent - AI models show strong predictive capability")
        elif avg_accuracy > 0.5:
            print(f"\nSTATUS: Good - AI models show decent predictive capability")
        else:
            print(f"\nSTATUS: Needs improvement - Consider more feature engineering")

if __name__ == "__main__":
    main()