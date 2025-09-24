"""
DEEP LEARNING OPTIONS PREDICTOR
===============================
Advanced ML models to predict options movements and beat 76% win rate baseline
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Try importing deep learning libraries with fallbacks
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow available - Deep learning models enabled")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - Using classical ML models")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available")

class DeepLearningOptionsPredictor:
    """Advanced ML/DL predictor for options trading signals."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Symbols for training
        self.symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'META']
        
        print("DEEP LEARNING OPTIONS PREDICTOR INITIALIZED")
        print("=" * 60)
        print("Mission: Beat 76% win rate with ML/DL models")
        print("Target: 85%+ prediction accuracy for options trades")
        print("=" * 60)
    
    def create_advanced_features(self, data):
        """Create comprehensive feature set for ML models."""
        
        df = data.copy()
        
        # Price-based features
        df['returns_1d'] = df['Close'].pct_change()
        df['returns_2d'] = df['Close'].pct_change(periods=2)
        df['returns_5d'] = df['Close'].pct_change(periods=5)
        df['returns_10d'] = df['Close'].pct_change(periods=10)
        df['returns_20d'] = df['Close'].pct_change(periods=20)
        
        # Volatility features (multiple timeframes)
        df['volatility_5d'] = df['returns_1d'].rolling(5).std() * np.sqrt(252)
        df['volatility_10d'] = df['returns_1d'].rolling(10).std() * np.sqrt(252)
        df['volatility_20d'] = df['returns_1d'].rolling(20).std() * np.sqrt(252)
        df['volatility_50d'] = df['returns_1d'].rolling(50).std() * np.sqrt(252)
        
        # Volatility ratios (key for our strategy)
        df['vol_ratio_5_20'] = df['volatility_5d'] / df['volatility_20d']
        df['vol_ratio_10_50'] = df['volatility_10d'] / df['volatility_50d']
        df['vol_ratio_20_50'] = df['volatility_20d'] / df['volatility_50d']
        
        # Moving averages and trends
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        
        df['price_vs_sma5'] = df['Close'] / df['sma_5'] - 1
        df['price_vs_sma20'] = df['Close'] / df['sma_20'] - 1
        df['price_vs_sma50'] = df['Close'] / df['sma_50'] - 1
        
        # Price position within recent ranges
        df['high_20d'] = df['High'].rolling(20).max()
        df['low_20d'] = df['Low'].rolling(20).min()
        df['price_position'] = (df['Close'] - df['low_20d']) / (df['high_20d'] - df['low_20d'])
        
        # Volume features
        df['volume_sma_10'] = df['Volume'].rolling(10).mean()
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        df['volume_trend'] = df['volume_sma_10'] / df['volume_sma_20']
        
        # Range and compression features (critical for volatility breakout)
        df['daily_range'] = (df['High'] - df['Low']) / df['Close']
        df['range_5d'] = df['daily_range'].rolling(5).mean()
        df['range_20d'] = df['daily_range'].rolling(20).mean()
        df['range_compression'] = df['range_5d'] / df['range_20d']
        
        # Momentum features
        df['momentum_5_20'] = (df['sma_5'] / df['sma_20']) - 1
        df['momentum_10_50'] = (df['sma_10'] / df['sma_50']) - 1
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['Close'])
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def create_volatility_breakout_labels(self, data, lookforward_days=10, profit_threshold=0.08):
        """Create labels for volatility breakout prediction."""
        
        df = data.copy()
        labels = []
        
        for i in range(len(df) - lookforward_days):
            current_price = df['Close'].iloc[i]
            
            # Look forward to see if breakout occurred
            future_prices = df['Close'].iloc[i+1:i+lookforward_days+1]
            max_move = max(
                (future_prices.max() - current_price) / current_price,
                (current_price - future_prices.min()) / current_price
            )
            
            # Label as 1 if breakout > threshold, 0 otherwise
            label = 1 if max_move >= profit_threshold else 0
            labels.append(label)
        
        # Pad the end
        labels.extend([0] * lookforward_days)
        
        return labels
    
    def create_momentum_labels(self, data, lookforward_days=5, profit_threshold=0.05):
        """Create labels for momentum prediction."""
        
        df = data.copy()
        labels = []
        
        for i in range(len(df) - lookforward_days):
            current_price = df['Close'].iloc[i]
            future_price = df['Close'].iloc[i+lookforward_days]
            
            return_pct = (future_price - current_price) / current_price
            
            # Label as 1 if profitable move, 0 otherwise
            label = 1 if abs(return_pct) >= profit_threshold else 0
            labels.append(label)
        
        labels.extend([0] * lookforward_days)
        
        return labels
    
    def prepare_training_data(self):
        """Prepare comprehensive training dataset."""
        
        print("\nPREPARING TRAINING DATA...")
        print("-" * 40)
        
        all_data = []
        
        for symbol in self.symbols:
            print(f"Processing {symbol}...")
            
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="3y", interval="1d")  # 3 years of data
                
                if len(data) < 200:  # Need sufficient data
                    continue
                
                # Create features
                featured_data = self.create_advanced_features(data)
                
                # Create labels for volatility breakout strategy
                vol_labels = self.create_volatility_breakout_labels(featured_data)
                featured_data['vol_breakout_target'] = vol_labels
                
                # Create labels for momentum strategy
                momentum_labels = self.create_momentum_labels(featured_data)
                featured_data['momentum_target'] = momentum_labels
                
                # Add symbol identifier
                featured_data['symbol'] = symbol
                
                all_data.append(featured_data)
                
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        
        # Combine all data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"Total training samples: {len(combined_data)}")
            print(f"Volatility breakout positive rate: {combined_data['vol_breakout_target'].mean()*100:.1f}%")
            print(f"Momentum positive rate: {combined_data['momentum_target'].mean()*100:.1f}%")
            
            return combined_data
        else:
            return None
    
    def train_classical_ml_models(self, data):
        """Train classical ML models (Random Forest, XGBoost, etc.)."""
        
        print("\nTRAINING CLASSICAL ML MODELS...")
        print("-" * 40)
        
        # Select features (exclude targets and metadata)
        feature_cols = [col for col in data.columns if col not in 
                       ['vol_breakout_target', 'momentum_target', 'symbol'] and 
                       not pd.isna(data[col]).all()]
        
        # Clean data
        clean_data = data[feature_cols + ['vol_breakout_target', 'momentum_target']].dropna()
        
        if len(clean_data) < 1000:
            print("Insufficient clean data for training")
            return
        
        X = clean_data[feature_cols].values
        y_vol = clean_data['vol_breakout_target'].values
        y_momentum = clean_data['momentum_target'].values
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['classical'] = scaler
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        models_to_train = {}
        
        # Random Forest
        models_to_train['RandomForest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting
        models_to_train['GradientBoosting'] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
        
        # XGBoost if available
        if XGBOOST_AVAILABLE:
            models_to_train['XGBoost'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        
        results = {}
        
        for strategy in ['vol_breakout', 'momentum']:
            y = y_vol if strategy == 'vol_breakout' else y_momentum
            
            print(f"\nTraining models for {strategy} strategy...")
            
            for model_name, model in models_to_train.items():
                try:
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')
                    
                    # Train final model
                    model.fit(X_scaled, y)
                    
                    # Store results
                    key = f"{model_name}_{strategy}"
                    self.models[key] = model
                    
                    results[key] = {
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'feature_importance': None
                    }
                    
                    # Feature importance for tree-based models
                    if hasattr(model, 'feature_importances_'):
                        importance = pd.DataFrame({
                            'feature': feature_cols,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        results[key]['feature_importance'] = importance.head(10).to_dict('records')
                        self.feature_importance[key] = importance
                    
                    print(f"  {model_name}: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                    
                except Exception as e:
                    print(f"  Error training {model_name}: {e}")
        
        return results
    
    def train_deep_learning_models(self, data):
        """Train deep learning models if TensorFlow is available."""
        
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available - skipping deep learning models")
            return {}
        
        print("\nTRAINING DEEP LEARNING MODELS...")
        print("-" * 40)
        
        # Select features
        feature_cols = [col for col in data.columns if col not in 
                       ['vol_breakout_target', 'momentum_target', 'symbol'] and 
                       not pd.isna(data[col]).all()]
        
        clean_data = data[feature_cols + ['vol_breakout_target', 'momentum_target']].dropna()
        
        if len(clean_data) < 2000:
            print("Insufficient data for deep learning")
            return {}
        
        X = clean_data[feature_cols].values
        y_vol = clean_data['vol_breakout_target'].values
        y_momentum = clean_data['momentum_target'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['deep_learning'] = scaler
        
        # Split data (time series aware)
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        
        results = {}
        
        for strategy in ['vol_breakout', 'momentum']:
            y = y_vol if strategy == 'vol_breakout' else y_momentum
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"\nTraining neural network for {strategy}...")
            
            try:
                # Build neural network
                model = Sequential([
                    Dense(256, activation='relu', input_shape=(X_scaled.shape[1],)),
                    BatchNormalization(),
                    Dropout(0.3),
                    
                    Dense(128, activation='relu'),
                    BatchNormalization(),
                    Dropout(0.3),
                    
                    Dense(64, activation='relu'),
                    Dropout(0.2),
                    
                    Dense(32, activation='relu'),
                    Dropout(0.2),
                    
                    Dense(1, activation='sigmoid')
                ])
                
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Callbacks
                callbacks = [
                    EarlyStopping(patience=20, restore_best_weights=True),
                    ReduceLROnPlateau(patience=10, factor=0.5)
                ]
                
                # Train model
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=200,
                    batch_size=64,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Evaluate
                test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
                
                # Store model
                key = f"DeepLearning_{strategy}"
                self.models[key] = model
                
                results[key] = {
                    'test_accuracy': test_accuracy,
                    'test_loss': test_loss,
                    'epochs_trained': len(history.history['loss'])
                }
                
                print(f"  Test Accuracy: {test_accuracy:.3f}")
                print(f"  Epochs trained: {len(history.history['loss'])}")
                
            except Exception as e:
                print(f"  Error training deep learning model: {e}")
        
        return results
    
    def create_lstm_models(self, data, sequence_length=20):
        """Create LSTM models for time series prediction."""
        
        if not TENSORFLOW_AVAILABLE:
            return {}
        
        print("\nTRAINING LSTM MODELS...")
        print("-" * 40)
        
        results = {}
        
        for symbol in self.symbols[:4]:  # Train on subset for speed
            try:
                symbol_data = data[data['symbol'] == symbol].copy()
                
                if len(symbol_data) < 500:
                    continue
                
                print(f"Training LSTM for {symbol}...")
                
                # Select features for LSTM
                feature_cols = ['returns_1d', 'returns_5d', 'volatility_10d', 'volatility_20d',
                               'vol_ratio_10_50', 'volume_ratio', 'range_compression', 'rsi']
                
                # Create sequences
                X_sequences = []
                y_sequences = []
                
                clean_data = symbol_data[feature_cols + ['vol_breakout_target']].dropna()
                
                for i in range(sequence_length, len(clean_data)):
                    X_sequences.append(clean_data[feature_cols].iloc[i-sequence_length:i].values)
                    y_sequences.append(clean_data['vol_breakout_target'].iloc[i])
                
                if len(X_sequences) < 200:
                    continue
                
                X_seq = np.array(X_sequences)
                y_seq = np.array(y_sequences)
                
                # Split data
                split_idx = int(len(X_seq) * 0.8)
                X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
                y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
                
                # Build LSTM model
                model = Sequential([
                    LSTM(64, return_sequences=True, input_shape=(sequence_length, len(feature_cols))),
                    Dropout(0.2),
                    LSTM(32, return_sequences=False),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])
                
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Train
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,
                    batch_size=32,
                    verbose=0,
                    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
                )
                
                # Evaluate
                test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
                
                key = f"LSTM_{symbol}"
                self.models[key] = model
                
                results[key] = {
                    'test_accuracy': test_accuracy,
                    'test_loss': test_loss
                }
                
                print(f"  {symbol} LSTM Accuracy: {test_accuracy:.3f}")
                
            except Exception as e:
                print(f"  Error training LSTM for {symbol}: {e}")
        
        return results
    
    def run_comprehensive_ml_research(self):
        """Run comprehensive machine learning research."""
        
        print("STARTING COMPREHENSIVE ML RESEARCH")
        print("=" * 70)
        print("Goal: Beat 76% win rate baseline with AI/ML models")
        print("=" * 70)
        
        # Prepare training data
        training_data = self.prepare_training_data()
        
        if training_data is None:
            print("Failed to prepare training data")
            return
        
        # Train all model types
        classical_results = self.train_classical_ml_models(training_data)
        deep_learning_results = self.train_deep_learning_models(training_data)
        lstm_results = self.create_lstm_models(training_data)
        
        # Combine results
        all_results = {
            **classical_results,
            **deep_learning_results,
            **lstm_results
        }
        
        # Find best models
        print(f"\n{'='*70}")
        print("MACHINE LEARNING RESEARCH RESULTS")
        print("=" * 70)
        print("BASELINE TO BEAT: 76.1% (Volatility Breakout Strategy)")
        print("-" * 70)
        
        best_models = []
        
        for model_name, result in all_results.items():
            if 'cv_mean' in result:
                accuracy = result['cv_mean']
            elif 'test_accuracy' in result:
                accuracy = result['test_accuracy']
            else:
                continue
            
            if accuracy > 0.761:  # Beat baseline
                best_models.append((model_name, accuracy))
                status = "[BEATS BASELINE]"
            else:
                status = "[Below baseline]"
            
            print(f"{model_name:30} {accuracy:.3f} ({accuracy*100:.1f}%) {status}")
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        research_results = {
            'timestamp': timestamp,
            'baseline_accuracy': 0.761,
            'total_models_trained': len(all_results),
            'models_beating_baseline': len(best_models),
            'best_models': sorted(best_models, key=lambda x: x[1], reverse=True),
            'detailed_results': all_results,
            'feature_importance': {k: v.head(5).to_dict('records') 
                                 for k, v in self.feature_importance.items()}
        }
        
        with open(f'ml_research_results_{timestamp}.json', 'w') as f:
            json.dump(research_results, f, indent=2, default=str)
        
        print(f"\nML RESEARCH COMPLETE!")
        print(f"Results saved: ml_research_results_{timestamp}.json")
        print(f"Models beating baseline: {len(best_models)}/{len(all_results)}")
        
        if best_models:
            best_model_name, best_accuracy = best_models[0]
            improvement = (best_accuracy - 0.761) * 100
            print(f"Best model: {best_model_name} ({best_accuracy:.1%})")
            print(f"Improvement: +{improvement:.1f}% over baseline")
        
        return research_results

def main():
    """Run comprehensive ML research."""
    
    predictor = DeepLearningOptionsPredictor()
    results = predictor.run_comprehensive_ml_research()
    
    print(f"\nML research completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()