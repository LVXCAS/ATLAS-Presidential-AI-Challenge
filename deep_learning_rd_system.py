"""
DEEP LEARNING R&D ENHANCEMENT SYSTEM
Advanced neural networks for market pattern recognition and strategy optimization
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import os

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
    logging.warning("TensorFlow not available - using fallback models")

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('deep_learning_rd.log'),
        logging.StreamHandler()
    ]
)

class DeepLearningRDSystem:
    """Advanced deep learning system for R&D strategy enhancement"""

    def __init__(self):
        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # Model parameters
        self.sequence_length = 60  # 60-day lookback for LSTM
        self.prediction_horizon = 5  # 5-day forward prediction

        # Scalers for normalization
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()

        # Model storage
        self.models = {}
        self.model_performance = {}

    async def collect_enhanced_market_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Collect comprehensive market data for deep learning"""

        logging.info("COLLECTING ENHANCED MARKET DATA FOR DEEP LEARNING")
        logging.info("=" * 55)

        enhanced_data = {}

        for symbol in symbols:
            try:
                # Get extended historical data (2 years)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=730)

                bars = self.api.get_bars(
                    symbol,
                    tradeapi.TimeFrame.Day,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    limit=500
                ).df

                if len(bars) < 100:
                    logging.warning(f"Insufficient data for {symbol}")
                    continue

                # Enhanced feature engineering
                df = bars.copy()

                # Price features
                df['returns'] = df['close'].pct_change()
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
                df['price_volume'] = df['close'] * df['volume']

                # Technical indicators
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
                df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
                df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
                df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

                # Future returns (labels)
                for horizon in [1, 3, 5, 10]:
                    df[f'future_return_{horizon}d'] = df['close'].shift(-horizon) / df['close'] - 1
                    df[f'future_direction_{horizon}d'] = (df[f'future_return_{horizon}d'] > 0).astype(int)

                # Drop NaN values
                df = df.dropna()

                enhanced_data[symbol] = df
                logging.info(f"✓ {symbol}: {len(df)} samples with {len(df.columns)} features")

            except Exception as e:
                logging.error(f"Error collecting data for {symbol}: {e}")

        logging.info(f"Enhanced data collected for {len(enhanced_data)} symbols")
        return enhanced_data

    async def build_lstm_price_predictor(self, symbol: str, data: pd.DataFrame) -> keras.Model:
        """Build LSTM model for price prediction"""

        if not DEEP_LEARNING_AVAILABLE:
            logging.warning("TensorFlow not available - cannot build LSTM")
            return None

        logging.info(f"Building LSTM Price Predictor for {symbol}")

        # Prepare features for LSTM
        feature_columns = [
            'close', 'volume', 'returns', 'volatility_20', 'rsi', 'macd',
            'bb_position', 'momentum_5', 'momentum_10', 'price_sma_20_ratio'
        ]

        # Ensure all feature columns exist
        available_features = [col for col in feature_columns if col in data.columns]

        if len(available_features) < 5:
            logging.warning(f"Insufficient features for {symbol}")
            return None

        # Prepare sequences
        X, y = self._prepare_lstm_sequences(data, available_features)

        if len(X) < 100:
            logging.warning(f"Insufficient sequences for {symbol}")
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
            layers.Dense(1, activation='linear')  # Price prediction
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        # Train model
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

        # Evaluate model
        train_loss = model.evaluate(X_train, y_train, verbose=0)[0]
        test_loss = model.evaluate(X_test, y_test, verbose=0)[0]

        self.model_performance[f"{symbol}_lstm"] = {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'overfitting_ratio': test_loss / train_loss,
            'data_points': len(X)
        }

        logging.info(f"LSTM {symbol}: Train Loss {train_loss:.6f}, Test Loss {test_loss:.6f}")
        return model

    async def build_cnn_pattern_recognizer(self, symbol: str, data: pd.DataFrame) -> keras.Model:
        """Build CNN model for pattern recognition"""

        if not DEEP_LEARNING_AVAILABLE:
            return None

        logging.info(f"Building CNN Pattern Recognizer for {symbol}")

        # Prepare 2D sequences for CNN (treat as images)
        feature_columns = [
            'close', 'high', 'low', 'volume', 'rsi', 'macd',
            'bb_position', 'volatility_20', 'momentum_5'
        ]

        available_features = [col for col in feature_columns if col in data.columns]

        if len(available_features) < 6:
            return None

        X, y = self._prepare_cnn_sequences(data, available_features)

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
            layers.Dense(1, activation='sigmoid')  # Direction prediction
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Train model
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

        # Evaluate model
        train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
        test_acc = model.evaluate(X_test, y_test, verbose=0)[1]

        self.model_performance[f"{symbol}_cnn"] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'accuracy_drop': train_acc - test_acc,
            'data_points': len(X)
        }

        logging.info(f"CNN {symbol}: Train Acc {train_acc:.3f}, Test Acc {test_acc:.3f}")
        return model

    async def build_transformer_market_regime_detector(self, market_data: Dict[str, pd.DataFrame]) -> keras.Model:
        """Build transformer model for market regime detection"""

        if not DEEP_LEARNING_AVAILABLE:
            return None

        logging.info("Building Transformer Market Regime Detector")

        # Combine data from all symbols for regime detection
        combined_features = []
        regime_labels = []

        for symbol, data in market_data.items():
            if len(data) < 100:
                continue

            # Market-wide features
            features = data[['returns', 'volatility_20', 'volume_ratio', 'rsi']].values

            # Regime labeling based on volatility and market conditions
            volatility_regime = pd.qcut(data['volatility_20'], 3, labels=[0, 1, 2])  # Low, Medium, High vol

            combined_features.append(features)
            regime_labels.extend(volatility_regime.values)

        if len(combined_features) == 0:
            return None

        # Prepare sequences
        X = np.concatenate(combined_features, axis=0)
        y = np.array(regime_labels)

        # Remove NaN values
        valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]

        if len(X) < 200:
            return None

        # Reshape for transformer
        X = X.reshape((-1, self.sequence_length, X.shape[1]))[:-(len(X) % self.sequence_length)]
        y = y.reshape((-1, self.sequence_length))[:-(len(y) % self.sequence_length)]
        y = y[:, -1]  # Use last label in sequence

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True
        )

        # Build Transformer model
        inputs = layers.Input(shape=(X.shape[1], X.shape[2]))

        # Multi-head attention
        attention = layers.MultiHeadAttention(
            num_heads=4, key_dim=64
        )(inputs, inputs)

        attention = layers.Dropout(0.1)(attention)
        attention = layers.LayerNormalization()(attention + inputs)

        # Feed forward
        ff = layers.Dense(128, activation='relu')(attention)
        ff = layers.Dropout(0.1)(ff)
        ff = layers.Dense(X.shape[2])(ff)
        ff = layers.LayerNormalization()(ff + attention)

        # Global pooling and classification
        pooled = layers.GlobalAveragePooling1D()(ff)
        outputs = layers.Dense(64, activation='relu')(pooled)
        outputs = layers.Dropout(0.2)(outputs)
        outputs = layers.Dense(3, activation='softmax')(outputs)  # 3 regime classes

        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train model
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

        self.model_performance["market_regime_transformer"] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'data_points': len(X)
        }

        logging.info(f"Transformer Regime: Train Acc {train_acc:.3f}, Test Acc {test_acc:.3f}")
        return model

    def _prepare_lstm_sequences(self, data: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""

        # Normalize features
        feature_data = data[features].values
        feature_data = self.feature_scaler.fit_transform(feature_data)

        # Prepare target (next day return)
        targets = data['future_return_1d'].values

        X, y = [], []
        for i in range(self.sequence_length, len(feature_data) - 1):
            if not np.isnan(targets[i]):
                X.append(feature_data[i-self.sequence_length:i])
                y.append(targets[i])

        return np.array(X), np.array(y)

    def _prepare_cnn_sequences(self, data: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for CNN training"""

        # Normalize features
        feature_data = data[features].values
        feature_data = self.feature_scaler.fit_transform(feature_data)

        # Prepare target (direction)
        targets = data['future_direction_5d'].values

        X, y = [], []
        for i in range(self.sequence_length, len(feature_data) - 5):
            if not np.isnan(targets[i]):
                X.append(feature_data[i-self.sequence_length:i])
                y.append(targets[i])

        return np.array(X), np.array(y)

    async def generate_deep_learning_predictions(self, symbols: List[str]) -> Dict:
        """Generate predictions using all deep learning models"""

        logging.info("GENERATING DEEP LEARNING PREDICTIONS")
        logging.info("=" * 40)

        # Collect data
        market_data = await self.collect_enhanced_market_data(symbols)

        if not market_data:
            logging.error("No market data available")
            return {}

        dl_predictions = {
            'timestamp': datetime.now().isoformat(),
            'symbol_predictions': {},
            'market_regime': {},
            'model_performance': {},
            'ensemble_signals': {}
        }

        # Build and train models for each symbol
        for symbol, data in market_data.items():
            if len(data) < 100:
                continue

            symbol_predictions = {}

            # LSTM Price Prediction
            lstm_model = await self.build_lstm_price_predictor(symbol, data)
            if lstm_model:
                # Get latest sequence for prediction
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
                dl_predictions['symbol_predictions'][symbol] = symbol_predictions
                logging.info(f"✓ {symbol}: Generated DL predictions")

        # Market Regime Detection
        transformer_model = await self.build_transformer_market_regime_detector(market_data)
        if transformer_model:
            # Predict current market regime
            regime_features = self._get_market_regime_features(market_data)
            if regime_features is not None:
                regime_probs = transformer_model.predict(regime_features, verbose=0)[0]
                dl_predictions['market_regime'] = {
                    'low_volatility_prob': float(regime_probs[0]),
                    'medium_volatility_prob': float(regime_probs[1]),
                    'high_volatility_prob': float(regime_probs[2]),
                    'predicted_regime': ['LOW_VOL', 'MEDIUM_VOL', 'HIGH_VOL'][np.argmax(regime_probs)]
                }
                self.models["market_regime"] = transformer_model

        # Model performance summary
        dl_predictions['model_performance'] = self.model_performance

        # Generate ensemble signals
        ensemble_signals = self._generate_ensemble_signals(dl_predictions['symbol_predictions'])
        dl_predictions['ensemble_signals'] = ensemble_signals

        # Save predictions
        with open('deep_learning_predictions.json', 'w') as f:
            json.dump(dl_predictions, f, indent=2)

        logging.info("=" * 40)
        logging.info("DEEP LEARNING PREDICTIONS COMPLETE")
        logging.info(f"Symbols Analyzed: {len(dl_predictions['symbol_predictions'])}")
        logging.info(f"Models Trained: {len(self.models)}")
        logging.info(f"Market Regime: {dl_predictions['market_regime'].get('predicted_regime', 'UNKNOWN')}")

        return dl_predictions

    def _get_latest_sequence(self, data: pd.DataFrame, model_type: str) -> np.ndarray:
        """Get latest sequence for prediction"""

        if model_type == 'lstm':
            features = ['close', 'volume', 'returns', 'volatility_20', 'rsi', 'macd',
                       'bb_position', 'momentum_5', 'momentum_10', 'price_sma_20_ratio']
        else:  # CNN
            features = ['close', 'high', 'low', 'volume', 'rsi', 'macd',
                       'bb_position', 'volatility_20', 'momentum_5']

        available_features = [col for col in features if col in data.columns]

        if len(available_features) < 5:
            return None

        latest_data = data[available_features].tail(self.sequence_length).values
        latest_data = self.feature_scaler.transform(latest_data)

        return latest_data.reshape(1, latest_data.shape[0], latest_data.shape[1])

    def _get_market_regime_features(self, market_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Get features for market regime prediction"""

        # Aggregate market features
        all_returns = []
        all_volatility = []
        all_volume_ratio = []
        all_rsi = []

        for symbol, data in market_data.items():
            if len(data) > 0:
                all_returns.extend(data['returns'].dropna().tail(20).values)
                all_volatility.extend(data['volatility_20'].dropna().tail(20).values)
                all_volume_ratio.extend(data['volume_ratio'].dropna().tail(20).values)
                all_rsi.extend(data['rsi'].dropna().tail(20).values)

        if len(all_returns) < self.sequence_length:
            return None

        # Create market-wide features
        market_features = np.column_stack([
            all_returns[:self.sequence_length],
            all_volatility[:self.sequence_length],
            all_volume_ratio[:self.sequence_length],
            all_rsi[:self.sequence_length]
        ])

        market_features = self.feature_scaler.fit_transform(market_features)
        return market_features.reshape(1, market_features.shape[0], market_features.shape[1])

    def _generate_ensemble_signals(self, symbol_predictions: Dict) -> Dict:
        """Generate ensemble trading signals from all models"""

        ensemble_signals = {}

        for symbol, predictions in symbol_predictions.items():
            lstm_pred = predictions.get('lstm_price_prediction', 0)
            cnn_prob = predictions.get('cnn_direction_probability', 0.5)

            # Ensemble signal strength
            signal_strength = abs(lstm_pred) * (2 * abs(cnn_prob - 0.5))

            # Signal direction
            price_signal = 1 if lstm_pred > 0.02 else (-1 if lstm_pred < -0.02 else 0)
            direction_signal = 1 if cnn_prob > 0.6 else (-1 if cnn_prob < 0.4 else 0)

            # Combined signal
            combined_signal = (price_signal + direction_signal) / 2

            ensemble_signals[symbol] = {
                'signal_strength': signal_strength,
                'signal_direction': combined_signal,
                'confidence_level': min(signal_strength * 2, 1.0),
                'recommended_action': 'BUY' if combined_signal > 0.3 else ('SELL' if combined_signal < -0.3 else 'HOLD')
            }

        return ensemble_signals

    async def run_deep_learning_analysis_cycle(self):
        """Run complete deep learning analysis cycle"""

        logging.info("DEEP LEARNING R&D ENHANCEMENT SYSTEM")
        logging.info("Advanced Neural Networks for Market Analysis")
        logging.info("=" * 55)

        # Target symbols (high-quality from R&D analysis)
        target_symbols = ['GOOGL', 'TSLA', 'MSFT', 'AAPL', 'NVDA', 'SPY', 'QQQ', 'TLT']

        # Generate predictions
        dl_results = await self.generate_deep_learning_predictions(target_symbols)

        if dl_results:
            # Integration with existing R&D system
            integration_summary = {
                'deep_learning_symbols': len(dl_results['symbol_predictions']),
                'models_deployed': len(self.models),
                'market_regime': dl_results['market_regime'].get('predicted_regime'),
                'high_confidence_signals': len([
                    s for s in dl_results['ensemble_signals'].values()
                    if s['confidence_level'] > 0.7
                ]),
                'recommended_actions': {
                    action: len([
                        s for s in dl_results['ensemble_signals'].values()
                        if s['recommended_action'] == action
                    ])
                    for action in ['BUY', 'SELL', 'HOLD']
                }
            }

            logging.info("=" * 55)
            logging.info("DEEP LEARNING ANALYSIS COMPLETE")
            logging.info(f"Symbols Analyzed: {integration_summary['deep_learning_symbols']}")
            logging.info(f"Models Deployed: {integration_summary['models_deployed']}")
            logging.info(f"Market Regime: {integration_summary['market_regime']}")
            logging.info(f"High Confidence Signals: {integration_summary['high_confidence_signals']}")
            logging.info(f"Buy Recommendations: {integration_summary['recommended_actions']['BUY']}")

            return {
                'dl_results': dl_results,
                'integration_summary': integration_summary
            }

        return {}

async def main():
    """Run deep learning R&D system"""

    if not DEEP_LEARNING_AVAILABLE:
        print("⚠️  TensorFlow not available - install with: pip install tensorflow")
        print("Using fallback analysis...")
        return

    system = DeepLearningRDSystem()
    results = await system.run_deep_learning_analysis_cycle()

    if results:
        print(f"\n🧠 DEEP LEARNING R&D SYSTEM COMPLETE")
        print(f"Neural Networks: DEPLOYED")
        print(f"Market Analysis: ENHANCED")
        print(f"Prediction Models: ACTIVE")

if __name__ == "__main__":
    asyncio.run(main())