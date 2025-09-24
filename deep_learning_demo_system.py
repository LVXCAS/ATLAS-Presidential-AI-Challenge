"""
DEEP LEARNING DEMO SYSTEM
Demonstrates advanced neural networks with synthetic market data
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.model_selection import train_test_split
    import warnings
    warnings.filterwarnings('ignore')
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('deep_learning_demo.log'),
        logging.StreamHandler()
    ]
)

class DeepLearningDemoSystem:
    """Deep learning demonstration with synthetic market data"""

    def __init__(self):
        self.sequence_length = 60
        self.prediction_horizon = 5
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        self.models = {}
        self.model_performance = {}

    def generate_synthetic_market_data(self, symbol: str, days: int = 500) -> pd.DataFrame:
        """Generate realistic synthetic market data"""

        np.random.seed(hash(symbol) % 2**32)  # Consistent but different per symbol

        # Base parameters
        initial_price = 100 + np.random.random() * 400  # $100-500 starting price
        annual_return = 0.08 + np.random.random() * 0.12  # 8-20% annual return
        annual_volatility = 0.15 + np.random.random() * 0.35  # 15-50% volatility

        # Generate price series using geometric Brownian motion
        dt = 1/252  # Daily time step
        drift = annual_return * dt
        diffusion = annual_volatility * np.sqrt(dt)

        dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')

        # Price evolution with trends and volatility clustering
        returns = []
        volatility = annual_volatility

        for i in range(days):
            # Volatility clustering effect
            if i > 0:
                vol_shock = 0.02 * np.random.randn()
                volatility = max(0.05, volatility + vol_shock)
                volatility = min(1.0, volatility)

            # Trending behavior
            trend_factor = 1.0
            if i > 30:
                momentum = np.mean(returns[-20:]) if len(returns) >= 20 else 0
                trend_factor = 1.0 + 0.1 * momentum

            # Generate return
            daily_return = drift * trend_factor + volatility * np.sqrt(dt) * np.random.randn()
            returns.append(daily_return)

        # Calculate prices
        log_prices = np.cumsum(returns) + np.log(initial_price)
        prices = np.exp(log_prices)

        # Generate volume with realistic patterns
        base_volume = 1000000 + np.random.random() * 5000000
        volume_returns = np.abs(np.array(returns))
        volumes = base_volume * (1 + 2 * volume_returns) * (0.8 + 0.4 * np.random.random(days))

        # Create OHLC data
        df = pd.DataFrame({
            'close': prices,
            'volume': volumes.astype(int),
            'returns': np.concatenate([[0], np.diff(np.log(prices))])
        }, index=dates)

        # Generate OHLC from close prices
        daily_range = 0.02 + 0.03 * np.random.random(days)  # 2-5% daily range
        df['high'] = df['close'] * (1 + daily_range * np.random.random(days))
        df['low'] = df['close'] * (1 - daily_range * np.random.random(days))
        df['open'] = df['close'].shift(1) * (1 + 0.005 * np.random.randn(days))
        df['open'] = df['open'].fillna(df['close'])

        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""

        # Price features
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

        # Volatility features
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']

        # Volume features
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_10']

        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Momentum features
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1

        # Future returns (labels)
        for horizon in [1, 3, 5, 10]:
            df[f'future_return_{horizon}d'] = df['close'].shift(-horizon) / df['close'] - 1
            df[f'future_direction_{horizon}d'] = (df[f'future_return_{horizon}d'] > 0).astype(int)

        return df.dropna()

    async def build_lstm_price_predictor(self, symbol: str, data: pd.DataFrame) -> keras.Model:
        """Build and train LSTM model for price prediction"""

        logging.info(f"Building LSTM Price Predictor for {symbol}")

        # Feature selection
        feature_columns = [
            'close', 'volume', 'returns', 'volatility_20', 'rsi', 'macd',
            'bb_position', 'momentum_5', 'momentum_10', 'price_sma_20_ratio'
        ]

        # Prepare sequences
        X, y = self._prepare_lstm_sequences(data, feature_columns)

        if len(X) < 100:
            logging.warning(f"Insufficient data for {symbol}")
            return None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Build LSTM model
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        # Train with early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )

        # Evaluate
        train_loss = model.evaluate(X_train, y_train, verbose=0)[0]
        test_loss = model.evaluate(X_test, y_test, verbose=0)[0]

        self.model_performance[f"{symbol}_lstm"] = {
            'train_loss': float(train_loss),
            'test_loss': float(test_loss),
            'overfitting_ratio': float(test_loss / train_loss),
            'data_points': len(X),
            'epochs_trained': len(history.history['loss'])
        }

        logging.info(f"✓ LSTM {symbol}: Train Loss {train_loss:.6f}, Test Loss {test_loss:.6f}")
        return model

    async def build_cnn_pattern_recognizer(self, symbol: str, data: pd.DataFrame) -> keras.Model:
        """Build CNN for pattern recognition"""

        logging.info(f"Building CNN Pattern Recognizer for {symbol}")

        feature_columns = [
            'close', 'high', 'low', 'volume', 'rsi', 'macd',
            'bb_position', 'volatility_20', 'momentum_5'
        ]

        # Prepare sequences
        X, y = self._prepare_cnn_sequences(data, feature_columns)

        if len(X) < 100:
            return None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Build CNN model
        model = keras.Sequential([
            layers.Conv1D(64, 3, activation='relu', input_shape=(X.shape[1], X.shape[2])),
            layers.BatchNormalization(),
            layers.Conv1D(64, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Dropout(0.25),

            layers.Conv1D(128, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv1D(128, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Dropout(0.25),

            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Train
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10, restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )

        # Evaluate
        train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
        test_acc = model.evaluate(X_test, y_test, verbose=0)[1]

        self.model_performance[f"{symbol}_cnn"] = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'accuracy_drop': float(train_acc - test_acc),
            'data_points': len(X)
        }

        logging.info(f"✓ CNN {symbol}: Train Acc {train_acc:.3f}, Test Acc {test_acc:.3f}")
        return model

    async def build_transformer_ensemble(self, market_data: Dict[str, pd.DataFrame]) -> keras.Model:
        """Build transformer for market regime detection"""

        logging.info("Building Transformer Market Regime Detector")

        # Combine features from multiple symbols
        combined_features = []
        regime_labels = []

        for symbol, data in market_data.items():
            if len(data) < 100:
                continue

            # Multi-asset features
            features = data[['returns', 'volatility_20', 'volume_ratio', 'rsi']].values

            # Create regime labels based on market conditions
            volatility_regime = pd.qcut(data['volatility_20'], 3, labels=[0, 1, 2], duplicates='drop')

            if len(volatility_regime) > 0:
                combined_features.append(features)
                regime_labels.extend(volatility_regime.values)

        if len(combined_features) == 0:
            return None

        # Prepare data
        X = np.concatenate(combined_features, axis=0)
        y = np.array(regime_labels)

        # Remove NaN values
        valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]

        if len(X) < 200:
            return None

        # Reshape for transformer
        seq_len = min(self.sequence_length, len(X) // 10)
        n_sequences = len(X) // seq_len
        X = X[:n_sequences * seq_len].reshape(n_sequences, seq_len, X.shape[1])
        y = y[:n_sequences * seq_len].reshape(n_sequences, seq_len)[:, -1]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True
        )

        # Build Transformer
        inputs = layers.Input(shape=(X.shape[1], X.shape[2]))

        # Multi-head attention
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
        attention = layers.Dropout(0.1)(attention)
        attention = layers.LayerNormalization()(attention + inputs)

        # Feed forward network
        ff = layers.Dense(128, activation='relu')(attention)
        ff = layers.Dropout(0.1)(ff)
        ff = layers.Dense(X.shape[2])(ff)
        ff = layers.LayerNormalization()(ff + attention)

        # Classification head
        pooled = layers.GlobalAveragePooling1D()(ff)
        outputs = layers.Dense(64, activation='relu')(pooled)
        outputs = layers.Dropout(0.2)(outputs)
        outputs = layers.Dense(3, activation='softmax')(outputs)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train
        history = model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=16,
            validation_data=(X_test, y_test),
            verbose=0
        )

        # Evaluate
        train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
        test_acc = model.evaluate(X_test, y_test, verbose=0)[1]

        self.model_performance["transformer_regime"] = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'data_points': len(X)
        }

        logging.info(f"✓ Transformer: Train Acc {train_acc:.3f}, Test Acc {test_acc:.3f}")
        return model

    def _prepare_lstm_sequences(self, data: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare LSTM sequences"""
        feature_data = data[features].values
        feature_data = self.feature_scaler.fit_transform(feature_data)

        targets = data['future_return_1d'].values

        X, y = [], []
        for i in range(self.sequence_length, len(feature_data) - 1):
            if not np.isnan(targets[i]):
                X.append(feature_data[i-self.sequence_length:i])
                y.append(targets[i])

        return np.array(X), np.array(y)

    def _prepare_cnn_sequences(self, data: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare CNN sequences"""
        feature_data = data[features].values
        feature_data = self.feature_scaler.fit_transform(feature_data)

        targets = data['future_direction_5d'].values

        X, y = [], []
        for i in range(self.sequence_length, len(feature_data) - 5):
            if not np.isnan(targets[i]):
                X.append(feature_data[i-self.sequence_length:i])
                y.append(targets[i])

        return np.array(X), np.array(y)

    async def run_deep_learning_demonstration(self):
        """Run complete deep learning demonstration"""

        if not DEEP_LEARNING_AVAILABLE:
            logging.error("TensorFlow not available")
            return {}

        logging.info("DEEP LEARNING DEMONSTRATION SYSTEM")
        logging.info("Advanced Neural Networks with Synthetic Data")
        logging.info("=" * 55)

        # Target symbols for demonstration
        demo_symbols = ['GOOGL', 'TSLA', 'MSFT', 'AAPL', 'NVDA', 'SPY']

        # Generate synthetic market data
        logging.info("GENERATING SYNTHETIC MARKET DATA")
        logging.info("=" * 35)

        market_data = {}
        for symbol in demo_symbols:
            raw_data = self.generate_synthetic_market_data(symbol)
            enhanced_data = self.add_technical_indicators(raw_data)
            market_data[symbol] = enhanced_data
            logging.info(f"✓ {symbol}: {len(enhanced_data)} samples with {len(enhanced_data.columns)} features")

        # Build and train models
        logging.info("\nTRAINING DEEP LEARNING MODELS")
        logging.info("=" * 33)

        dl_results = {
            'timestamp': datetime.now().isoformat(),
            'symbol_predictions': {},
            'market_regime': {},
            'model_performance': {},
            'ensemble_signals': {}
        }

        # Train models for each symbol
        for symbol, data in market_data.items():
            symbol_predictions = {}

            # LSTM Price Predictor
            lstm_model = await self.build_lstm_price_predictor(symbol, data)
            if lstm_model:
                latest_sequence = self._get_latest_sequence(data, 'lstm')
                if latest_sequence is not None:
                    price_pred = lstm_model.predict(latest_sequence, verbose=0)[0][0]
                    symbol_predictions['lstm_price_prediction'] = float(price_pred)
                    self.models[f"{symbol}_lstm"] = lstm_model

            # CNN Pattern Recognition
            cnn_model = await self.build_cnn_pattern_recognizer(symbol, data)
            if cnn_model:
                latest_sequence = self._get_latest_sequence(data, 'cnn')
                if latest_sequence is not None:
                    direction_pred = cnn_model.predict(latest_sequence, verbose=0)[0][0]
                    symbol_predictions['cnn_direction_probability'] = float(direction_pred)
                    self.models[f"{symbol}_cnn"] = cnn_model

            if symbol_predictions:
                dl_results['symbol_predictions'][symbol] = symbol_predictions

        # Market Regime Transformer
        transformer_model = await self.build_transformer_ensemble(market_data)
        if transformer_model:
            regime_features = self._get_market_regime_features(market_data)
            if regime_features is not None:
                regime_probs = transformer_model.predict(regime_features, verbose=0)[0]
                dl_results['market_regime'] = {
                    'low_volatility_prob': float(regime_probs[0]),
                    'medium_volatility_prob': float(regime_probs[1]),
                    'high_volatility_prob': float(regime_probs[2]),
                    'predicted_regime': ['LOW_VOL', 'MEDIUM_VOL', 'HIGH_VOL'][np.argmax(regime_probs)]
                }

        # Performance summary
        dl_results['model_performance'] = self.model_performance

        # Generate ensemble signals
        ensemble_signals = self._generate_ensemble_signals(dl_results['symbol_predictions'])
        dl_results['ensemble_signals'] = ensemble_signals

        # Save results
        with open('deep_learning_demo_results.json', 'w') as f:
            json.dump(dl_results, f, indent=2)

        # Summary
        logging.info("=" * 55)
        logging.info("DEEP LEARNING DEMONSTRATION COMPLETE")
        logging.info(f"Symbols Analyzed: {len(dl_results['symbol_predictions'])}")
        logging.info(f"Models Trained: {len(self.models)}")
        logging.info(f"Market Regime: {dl_results['market_regime'].get('predicted_regime', 'UNKNOWN')}")

        # Performance metrics
        lstm_models = [k for k in self.model_performance if 'lstm' in k]
        cnn_models = [k for k in self.model_performance if 'cnn' in k]

        if lstm_models:
            avg_lstm_loss = np.mean([self.model_performance[k]['test_loss'] for k in lstm_models])
            logging.info(f"Average LSTM Test Loss: {avg_lstm_loss:.6f}")

        if cnn_models:
            avg_cnn_acc = np.mean([self.model_performance[k]['test_accuracy'] for k in cnn_models])
            logging.info(f"Average CNN Test Accuracy: {avg_cnn_acc:.3f}")

        # Trading signals
        buy_signals = len([s for s in ensemble_signals.values() if s['recommended_action'] == 'BUY'])
        sell_signals = len([s for s in ensemble_signals.values() if s['recommended_action'] == 'SELL'])
        hold_signals = len([s for s in ensemble_signals.values() if s['recommended_action'] == 'HOLD'])

        logging.info(f"Trading Signals - BUY: {buy_signals}, SELL: {sell_signals}, HOLD: {hold_signals}")

        return dl_results

    def _get_latest_sequence(self, data: pd.DataFrame, model_type: str) -> np.ndarray:
        """Get latest sequence for prediction"""
        if model_type == 'lstm':
            features = ['close', 'volume', 'returns', 'volatility_20', 'rsi', 'macd',
                       'bb_position', 'momentum_5', 'momentum_10', 'price_sma_20_ratio']
        else:
            features = ['close', 'high', 'low', 'volume', 'rsi', 'macd',
                       'bb_position', 'volatility_20', 'momentum_5']

        latest_data = data[features].tail(self.sequence_length).values
        latest_data = self.feature_scaler.transform(latest_data)
        return latest_data.reshape(1, latest_data.shape[0], latest_data.shape[1])

    def _get_market_regime_features(self, market_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Get market regime features"""
        all_returns = []
        all_volatility = []
        all_volume_ratio = []
        all_rsi = []

        for symbol, data in market_data.items():
            all_returns.extend(data['returns'].dropna().tail(20).values)
            all_volatility.extend(data['volatility_20'].dropna().tail(20).values)
            all_volume_ratio.extend(data['volume_ratio'].dropna().tail(20).values)
            all_rsi.extend(data['rsi'].dropna().tail(20).values)

        if len(all_returns) < self.sequence_length:
            return None

        market_features = np.column_stack([
            all_returns[:self.sequence_length],
            all_volatility[:self.sequence_length],
            all_volume_ratio[:self.sequence_length],
            all_rsi[:self.sequence_length]
        ])

        market_features = self.feature_scaler.fit_transform(market_features)
        return market_features.reshape(1, market_features.shape[0], market_features.shape[1])

    def _generate_ensemble_signals(self, symbol_predictions: Dict) -> Dict:
        """Generate ensemble trading signals"""
        ensemble_signals = {}

        for symbol, predictions in symbol_predictions.items():
            lstm_pred = predictions.get('lstm_price_prediction', 0)
            cnn_prob = predictions.get('cnn_direction_probability', 0.5)

            signal_strength = abs(lstm_pred) * (2 * abs(cnn_prob - 0.5))
            price_signal = 1 if lstm_pred > 0.02 else (-1 if lstm_pred < -0.02 else 0)
            direction_signal = 1 if cnn_prob > 0.6 else (-1 if cnn_prob < 0.4 else 0)
            combined_signal = (price_signal + direction_signal) / 2

            ensemble_signals[symbol] = {
                'signal_strength': signal_strength,
                'signal_direction': combined_signal,
                'confidence_level': min(signal_strength * 2, 1.0),
                'recommended_action': 'BUY' if combined_signal > 0.3 else ('SELL' if combined_signal < -0.3 else 'HOLD')
            }

        return ensemble_signals

async def main():
    """Run deep learning demonstration"""

    if not DEEP_LEARNING_AVAILABLE:
        print("⚠️  TensorFlow not available")
        return

    system = DeepLearningDemoSystem()
    results = await system.run_deep_learning_demonstration()

    if results:
        print(f"\n🧠 DEEP LEARNING DEMONSTRATION COMPLETE")
        print(f"Advanced Neural Networks: DEPLOYED")
        print(f"Market Pattern Recognition: ACTIVE")
        print(f"Prediction Models: TRAINED & VALIDATED")

if __name__ == "__main__":
    asyncio.run(main())