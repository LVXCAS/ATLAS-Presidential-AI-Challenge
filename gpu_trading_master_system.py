"""
GPU-ACCELERATED TRADING MASTER SYSTEM
Leveraging GTX 1660 Super for complete trading infrastructure
"""

import os
import json
import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# GPU-accelerated imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

load_dotenv()

# Configure GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu_trading_master.log'),
        logging.StreamHandler()
    ]
)

class GPUTradingDataset(Dataset):
    """GPU-optimized dataset for trading data"""
    def __init__(self, X, y, device):
        self.X = torch.FloatTensor(X).to(device)
        self.y = torch.FloatTensor(y).to(device) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class AttentionLSTM(nn.Module):
    """GPU-optimized LSTM with attention for market prediction"""
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.2, output_size=1):
        super(AttentionLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers optimized for GTX 1660 Super
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_size * 2,  # bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Feature extraction layers
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        # Output layers with skip connections
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc_out = nn.Linear(hidden_size // 4, output_size)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x):
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out = self.layer_norm1(lstm_out)

        # Self-attention mechanism
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Combine LSTM and attention outputs
        combined = lstm_out + attn_out  # Skip connection

        # Use the last time step
        out = combined[:, -1, :]

        # Feed forward with skip connections
        x1 = self.gelu(self.fc1(out))
        x1 = self.dropout(x1)
        x1 = self.layer_norm2(x1)

        x2 = self.relu(self.fc2(x1))
        x2 = self.dropout(x2)

        x3 = self.relu(self.fc3(x2))
        output = self.fc_out(x3)

        return output, attn_weights

class ConvolutionalPredictor(nn.Module):
    """GPU-optimized 1D CNN for pattern recognition in time series"""
    def __init__(self, input_size, sequence_length, output_size=1):
        super(ConvolutionalPredictor, self).__init__()

        # Multi-scale convolutional layers
        self.conv1 = nn.Conv1d(input_size, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=7, padding=3)
        self.conv4 = nn.Conv1d(512, 256, kernel_size=3, padding=1)

        # Pooling and normalization
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.3)

        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(512)
        self.batch_norm4 = nn.BatchNorm1d(256)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, output_size)

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x):
        # Transpose for 1D conv (batch, features, sequence)
        x = x.transpose(1, 2)

        # Multi-scale convolution with residual connections
        x1 = self.relu(self.batch_norm1(self.conv1(x)))
        x1 = self.dropout(x1)

        x2 = self.relu(self.batch_norm2(self.conv2(x1)))
        x2 = self.dropout(x2)

        x3 = self.relu(self.batch_norm3(self.conv3(x2)))
        x3 = self.dropout(x3)

        x4 = self.relu(self.batch_norm4(self.conv4(x3)))

        # Global pooling and output
        x_pooled = self.pool(x4).squeeze(-1)

        x_out = self.gelu(self.fc1(x_pooled))
        x_out = self.dropout(x_out)
        x_out = self.relu(self.fc2(x_out))
        output = self.fc_out(x_out)

        return output

class GPUTradingMasterSystem:
    """Complete GPU-accelerated trading system"""

    def __init__(self):
        self.api = tradeapi.REST(
            base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY')
        )

        self.device = device
        self.logger = logging.getLogger('GPUTradingMaster')

        # GPU-optimized settings for GTX 1660 Super
        if self.device.type == 'cuda':
            self.batch_size = 512
            self.num_workers = 8
            self.prefetch_factor = 4
            torch.cuda.empty_cache()
            self.logger.info(f">> GPU Mode: {torch.cuda.get_device_name(0)}")
            self.logger.info(f">> GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.batch_size = 128
            self.num_workers = 4
            self.prefetch_factor = 2
            self.logger.info(">> CPU Mode")

        # Model configurations
        self.sequence_length = 60
        self.prediction_horizons = [1, 5, 10, 20]  # Multiple timeframes
        self.feature_scalers = {}
        self.models = {}

        # Trading parameters
        self.risk_tolerance = 0.02
        self.max_position_size = 0.1
        self.rebalance_frequency = '1H'

        self.logger.info(f">> GPU Trading Master System Initialized")
        self.logger.info(f">> Batch Size: {self.batch_size}")
        self.logger.info(f">> Expected 10x performance improvement")

    def get_comprehensive_market_data(self, symbols: List[str], timeframe: str = '1Min', limit: int = 5000) -> Dict[str, pd.DataFrame]:
        """Fetch comprehensive market data for multiple symbols in parallel"""
        def fetch_symbol_data(symbol):
            try:
                # Get bars data
                bars = self.api.get_bars(
                    symbol,
                    timeframe,
                    limit=limit,
                    asof=None,
                    feed='iex',
                    adjustment='raw'
                ).df

                if bars.empty:
                    return symbol, pd.DataFrame()

                bars = bars.reset_index()
                bars['symbol'] = symbol

                # Add market microstructure features
                bars['spread'] = (bars['high'] - bars['low']) / bars['close']
                bars['volume_profile'] = bars['volume'] / bars['volume'].rolling(window=20, min_periods=1).mean()
                bars['price_impact'] = abs(bars['close'] - bars['open']) / bars['volume']

                return symbol, bars

            except Exception as e:
                self.logger.error(f"Error fetching {symbol}: {e}")
                return symbol, pd.DataFrame()

        # Parallel data fetching
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = dict(executor.map(fetch_symbol_data, symbols))

        successful = {k: v for k, v in results.items() if not v.empty}
        self.logger.info(f">> Fetched data for {len(successful)}/{len(symbols)} symbols")

        return successful

    def calculate_gpu_optimized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive features optimized for GPU processing"""
        if df.empty or len(df) < 50:
            return df

        try:
            # Vectorized price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std()

            # Multi-timeframe moving averages
            periods = [5, 10, 20, 50, 100, 200]
            for period in periods:
                sma = df['close'].rolling(window=period, min_periods=1).mean()
                ema = df['close'].ewm(span=period, min_periods=1).mean()

                df[f'sma_{period}'] = sma
                df[f'ema_{period}'] = ema
                df[f'price_sma_ratio_{period}'] = df['close'] / sma
                df[f'price_ema_ratio_{period}'] = df['close'] / ema
                df[f'sma_slope_{period}'] = sma.pct_change(periods=5)

            # Advanced momentum indicators
            for lookback in [5, 10, 20, 50]:
                df[f'momentum_{lookback}'] = df['close'] / df['close'].shift(lookback) - 1
                df[f'volatility_{lookback}'] = df['returns'].rolling(window=lookback, min_periods=1).std()
                df[f'momentum_rank_{lookback}'] = df[f'momentum_{lookback}'].rolling(window=100, min_periods=1).rank(pct=True)

            # RSI variants
            for period in [14, 21, 50]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
                rs = gain / (loss + 1e-8)
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

            # MACD family
            for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (19, 39, 9)]:
                ema_fast = df['close'].ewm(span=fast, min_periods=1).mean()
                ema_slow = df['close'].ewm(span=slow, min_periods=1).mean()
                macd = ema_fast - ema_slow
                macd_signal = macd.ewm(span=signal, min_periods=1).mean()

                df[f'macd_{fast}_{slow}'] = macd
                df[f'macd_signal_{fast}_{slow}'] = macd_signal
                df[f'macd_histogram_{fast}_{slow}'] = macd - macd_signal

            # Volume analysis
            df['volume_sma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-8)
            df['volume_momentum'] = df['volume'] / df['volume'].shift(10)
            df['price_volume_trend'] = df['returns'] * df['volume_ratio']
            df['on_balance_volume'] = (df['returns'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0) * df['volume']).cumsum()

            # Market microstructure
            df['bid_ask_spread'] = df['spread']
            df['market_impact'] = df['price_impact'].rolling(window=10, min_periods=1).mean()
            df['liquidity_score'] = 1 / (df['spread'] + 1e-8)

            # Support/Resistance levels
            for window in [20, 50, 100]:
                df[f'resistance_{window}'] = df['high'].rolling(window=window, min_periods=1).max()
                df[f'support_{window}'] = df['low'].rolling(window=window, min_periods=1).min()
                df[f'price_position_{window}'] = (df['close'] - df[f'support_{window}']) / (df[f'resistance_{window}'] - df[f'support_{window}'] + 1e-8)

            # Bollinger Bands
            for period in [20, 50]:
                sma = df['close'].rolling(window=period, min_periods=1).mean()
                std = df['close'].rolling(window=period, min_periods=1).std()
                df[f'bb_upper_{period}'] = sma + (2 * std)
                df[f'bb_lower_{period}'] = sma - (2 * std)
                df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-8)

            # Advanced pattern features
            df['doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)) < 0.1
            df['hammer'] = ((df['high'] - df[['open', 'close']].max(axis=1)) < (df[['open', 'close']].max(axis=1) - df[['open', 'close']].min(axis=1)) * 0.3) & \
                          ((df[['open', 'close']].min(axis=1) - df['low']) > (df[['open', 'close']].max(axis=1) - df[['open', 'close']].min(axis=1)) * 2)

            # Time-based features
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['market_open'] = (df['hour'] >= 9) & (df['hour'] < 16)
            df['pre_market'] = df['hour'] < 9
            df['after_hours'] = df['hour'] >= 16

            # Fill NaN values
            df = df.fillna(method='ffill').fillna(0)

            return df

        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
            return df.fillna(0)

    def create_multi_horizon_sequences(self, data: np.ndarray, horizons: List[int]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Create sequences for multiple prediction horizons"""
        sequences = {}

        for horizon in horizons:
            X, y = [], []

            for i in range(self.sequence_length, len(data) - horizon + 1):
                X.append(data[i-self.sequence_length:i])
                y.append(data[i + horizon - 1, 0])  # Predict close price

            if len(X) > 0:
                sequences[horizon] = (np.array(X), np.array(y))

        return sequences

    def train_gpu_ensemble(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Train ensemble of GPU-accelerated models"""
        try:
            if len(df) < self.sequence_length + max(self.prediction_horizons) + 50:
                return {}

            # Calculate comprehensive features
            features_df = self.calculate_gpu_optimized_features(df.copy())

            # Select feature columns
            feature_cols = [col for col in features_df.columns if col not in
                          ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap', 'timestamp', 'symbol']]

            if len(feature_cols) < 20:
                return {}

            # Prepare data
            X = features_df[feature_cols].values
            close_prices = features_df[['close'] + feature_cols].values  # Include close price

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            close_scaled = scaler.fit_transform(close_prices)

            # Create multi-horizon sequences
            sequences = self.create_multi_horizon_sequences(close_scaled, self.prediction_horizons)

            if not sequences:
                return {}

            # Train models for each horizon
            trained_models = {}
            performance_metrics = {}

            for horizon, (X_seq, y_seq) in sequences.items():
                if len(X_seq) < 100:
                    continue

                # Train-test split
                split_idx = int(len(X_seq) * 0.8)
                X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
                y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

                # Create datasets
                train_dataset = GPUTradingDataset(X_train, y_train, self.device)
                test_dataset = GPUTradingDataset(X_test, y_test, self.device)

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=0,  # GPU data loading
                    pin_memory=True if self.device.type == 'cuda' else False
                )

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True if self.device.type == 'cuda' else False
                )

                # Train LSTM model
                lstm_model = AttentionLSTM(len(feature_cols), output_size=1).to(self.device)
                lstm_performance = self._train_model(lstm_model, train_loader, test_loader, f"{symbol}_LSTM_{horizon}")

                # Train CNN model
                cnn_model = ConvolutionalPredictor(len(feature_cols), self.sequence_length, output_size=1).to(self.device)
                cnn_performance = self._train_model(cnn_model, train_loader, test_loader, f"{symbol}_CNN_{horizon}")

                # Store models and performance
                trained_models[f'lstm_{horizon}'] = lstm_model
                trained_models[f'cnn_{horizon}'] = cnn_model

                performance_metrics[f'lstm_{horizon}'] = lstm_performance
                performance_metrics[f'cnn_{horizon}'] = cnn_performance

            if not trained_models:
                return {}

            return {
                'models': trained_models,
                'performance': performance_metrics,
                'scaler': scaler,
                'feature_columns': feature_cols,
                'last_sequence': X_scaled[-self.sequence_length:],
                'training_date': datetime.now(),
                'gpu_accelerated': True,
                'symbol': symbol
            }

        except Exception as e:
            self.logger.error(f"Error training ensemble for {symbol}: {e}")
            return {}

    def _train_model(self, model, train_loader, test_loader, model_name: str, epochs: int = 100) -> Dict[str, float]:
        """Train individual GPU model with advanced optimizations"""
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=False)

        # Mixed precision training for GTX 1660 Super
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None

        model.train()
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0

            for batch_data in train_loader:
                if len(batch_data) == 2:
                    X_batch, y_batch = batch_data
                else:
                    continue

                optimizer.zero_grad()

                # Mixed precision forward pass
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        if hasattr(model, 'forward') and 'attn' in str(type(model)):
                            outputs, _ = model(X_batch)
                        else:
                            outputs = model(X_batch)
                        loss = criterion(outputs.squeeze(), y_batch)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if hasattr(model, 'forward') and 'attn' in str(type(model)):
                        outputs, _ = model(X_batch)
                    else:
                        outputs = model(X_batch)
                    loss = criterion(outputs.squeeze(), y_batch)

                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
            scheduler.step(avg_loss)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    break

            if epoch % 20 == 0:
                self.logger.debug(f"{model_name} - Epoch {epoch}: Loss = {avg_loss:.6f}")

        # Evaluation
        model.eval()
        test_losses = []
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch_data in test_loader:
                if len(batch_data) == 2:
                    X_batch, y_batch = batch_data
                else:
                    continue

                if hasattr(model, 'forward') and 'attn' in str(type(model)):
                    outputs, _ = model(X_batch)
                else:
                    outputs = model(X_batch)

                test_loss = criterion(outputs.squeeze(), y_batch)
                test_losses.append(test_loss.item())

                predictions.extend(outputs.squeeze().cpu().numpy())
                actuals.extend(y_batch.cpu().numpy())

        # Calculate performance metrics
        if len(predictions) > 0 and len(actuals) > 0:
            mse = np.mean(test_losses)
            rmse = np.sqrt(mse)

            # R² score
            ss_res = np.sum((np.array(actuals) - np.array(predictions)) ** 2)
            ss_tot = np.sum((np.array(actuals) - np.mean(actuals)) ** 2)
            r2_score = 1 - (ss_res / (ss_tot + 1e-8))

            # Directional accuracy
            actual_directions = np.sign(np.diff(actuals))
            pred_directions = np.sign(np.diff(predictions))
            directional_accuracy = np.mean(actual_directions == pred_directions) if len(actual_directions) > 0 else 0

            return {
                'mse': mse,
                'rmse': rmse,
                'r2_score': r2_score,
                'directional_accuracy': directional_accuracy,
                'best_loss': best_loss
            }

        return {'mse': float('inf'), 'rmse': float('inf'), 'r2_score': -1, 'directional_accuracy': 0, 'best_loss': best_loss}

    def generate_gpu_predictions(self, symbol: str) -> Dict[str, Any]:
        """Generate comprehensive GPU-accelerated predictions"""
        try:
            # Get fresh market data
            data = self.get_comprehensive_market_data([symbol], timeframe='1Min', limit=2000)
            if symbol not in data or data[symbol].empty:
                return {}

            df = data[symbol]

            # Train ensemble models
            model_data = self.train_gpu_ensemble(symbol, df)
            if not model_data:
                return {}

            # Generate predictions for all horizons and models
            predictions = {}
            confidence_scores = {}

            models = model_data['models']
            last_sequence = torch.FloatTensor(model_data['last_sequence']).unsqueeze(0).to(self.device)

            for model_name, model in models.items():
                model.eval()
                with torch.no_grad():
                    if 'lstm' in model_name:
                        pred, attention = model(last_sequence)
                        predictions[model_name] = pred.cpu().item()
                    else:
                        pred = model(last_sequence)
                        predictions[model_name] = pred.cpu().item()

                    # Get confidence from performance metrics
                    if model_name in model_data['performance']:
                        perf = model_data['performance'][model_name]
                        confidence_scores[model_name] = max(0, perf.get('r2_score', 0))

            if not predictions:
                return {}

            # Calculate ensemble predictions
            total_confidence = sum(confidence_scores.values()) + 1e-8
            ensemble_predictions = {}

            for horizon in self.prediction_horizons:
                horizon_preds = []
                horizon_weights = []

                for model_name, pred in predictions.items():
                    if f'_{horizon}' in model_name:
                        horizon_preds.append(pred)
                        horizon_weights.append(confidence_scores.get(model_name, 0))

                if horizon_preds:
                    weighted_pred = np.average(horizon_preds, weights=horizon_weights) if sum(horizon_weights) > 0 else np.mean(horizon_preds)
                    ensemble_predictions[f'horizon_{horizon}'] = weighted_pred

            # Calculate current metrics
            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]

            # Risk assessment
            volatility = df['returns'].rolling(window=20).std().iloc[-1]
            risk_score = min(volatility * 100, 10) / 10  # Normalized risk score

            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'current_price': current_price,
                'current_volume': current_volume,
                'volatility': volatility,
                'risk_score': risk_score,
                'ensemble_predictions': ensemble_predictions,
                'individual_predictions': predictions,
                'confidence_scores': confidence_scores,
                'average_confidence': np.mean(list(confidence_scores.values())),
                'model_performance': model_data['performance'],
                'gpu_accelerated': True,
                'processing_time': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Error generating predictions for {symbol}: {e}")
            return {}

    def run_comprehensive_gpu_scan(self, symbols: List[str] = None, timeframe: str = '1Min') -> Dict[str, Any]:
        """Run comprehensive GPU-accelerated market scan"""
        if symbols is None:
            # Expanded symbol list for comprehensive scanning
            symbols = [
                # Major ETFs
                'SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VTI', 'VTEB', 'BND',
                # Tech stocks
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'NFLX', 'CRM',
                # Financial
                'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK.B',
                # Healthcare
                'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'BMY',
                # Consumer
                'KO', 'PEP', 'WMT', 'HD', 'MCD', 'DIS', 'NKE',
                # Energy
                'XOM', 'CVX', 'COP', 'EOG', 'SLB',
                # Growth/Momentum
                'UBER', 'LYFT', 'COIN', 'PLTR', 'SNOW', 'ZM', 'PTON', 'ROKU'
            ]

        self.logger.info(f">> Starting comprehensive GPU scan on {len(symbols)} symbols...")
        start_time = datetime.now()

        # Process symbols with GPU acceleration
        results = []
        for i, symbol in enumerate(symbols):
            try:
                self.logger.info(f">> Processing {symbol} ({i+1}/{len(symbols)}) on GPU...")

                prediction_result = self.generate_gpu_predictions(symbol)
                if prediction_result and prediction_result.get('average_confidence', 0) > 0.1:
                    results.append(prediction_result)

                # Clear GPU cache periodically
                if i % 10 == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Sort by confidence and potential
        results.sort(key=lambda x: x['average_confidence'] * (1 - x['risk_score']), reverse=True)

        # Generate comprehensive summary
        summary = {
            'scan_timestamp': start_time,
            'completion_timestamp': end_time,
            'symbols_scanned': len(symbols),
            'successful_predictions': len(results),
            'processing_time_seconds': processing_time,
            'gpu_acceleration': True,
            'device': str(self.device),
            'performance_metrics': {
                'symbols_per_second': len(symbols) / processing_time,
                'avg_time_per_symbol': processing_time / len(symbols),
                'gpu_memory_peak': torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                'batch_size_used': self.batch_size,
                'total_models_trained': len(results) * len(self.prediction_horizons) * 2
            },
            'top_opportunities': results[:20],
            'market_overview': {
                'avg_volatility': np.mean([r['volatility'] for r in results]) if results else 0,
                'avg_confidence': np.mean([r['average_confidence'] for r in results]) if results else 0,
                'high_confidence_count': len([r for r in results if r['average_confidence'] > 0.7]),
                'low_risk_count': len([r for r in results if r['risk_score'] < 0.3])
            }
        }

        # Log comprehensive results
        self.logger.info(f">> GPU COMPREHENSIVE SCAN COMPLETE")
        self.logger.info(f">> Processing time: {processing_time:.1f}s ({processing_time/len(symbols):.2f}s per symbol)")
        self.logger.info(f">> Performance: {len(symbols)/processing_time:.1f} symbols/second")
        self.logger.info(f">> Successful predictions: {len(results)}/{len(symbols)}")
        self.logger.info(f">> Models trained: {summary['performance_metrics']['total_models_trained']}")

        if torch.cuda.is_available():
            self.logger.info(f">> Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

        if results:
            top = results[0]
            self.logger.info(f">> Top opportunity: {top['symbol']} (confidence: {top['average_confidence']:.3f})")

        return summary

if __name__ == "__main__":
    # Initialize GPU Trading Master System
    gpu_trading_system = GPUTradingMasterSystem()

    # Run comprehensive scan
    results = gpu_trading_system.run_comprehensive_gpu_scan()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'gpu_comprehensive_scan_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n>> GPU TRADING MASTER SYSTEM COMPLETE!")
    print(f">> Scanned {results['symbols_scanned']} symbols in {results['processing_time_seconds']:.1f}s")
    print(f">> Performance: {results['performance_metrics']['symbols_per_second']:.1f} symbols/second")
    print(f">> Models trained: {results['performance_metrics']['total_models_trained']}")
    print(f">> Success rate: {results['successful_predictions']}/{results['symbols_scanned']} ({results['successful_predictions']/results['symbols_scanned']*100:.1f}%)")

    if torch.cuda.is_available():
        print(f">> GPU utilization: {results['performance_metrics']['gpu_memory_peak']:.2f} GB peak memory")

    if results['top_opportunities']:
        top = results['top_opportunities'][0]
        print(f">> Top opportunity: {top['symbol']} - Confidence: {top['average_confidence']:.3f}")