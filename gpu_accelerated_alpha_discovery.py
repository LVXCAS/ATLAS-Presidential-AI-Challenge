"""
GPU-ACCELERATED ALPHA DISCOVERY SYSTEM
Optimized for GTX 1660 Super with PyTorch CUDA acceleration
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# GPU-accelerated imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

load_dotenv()

# Configure GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu_alpha_discovery.log'),
        logging.StreamHandler()
    ]
)

class AlphaDataset(Dataset):
    """Custom dataset for GPU-accelerated training"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).to(device)
        self.y = torch.FloatTensor(y).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMAlphaModel(nn.Module):
    """GPU-optimized LSTM model for alpha prediction"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMAlphaModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers optimized for GTX 1660 Super
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism for better feature focus
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)

        # Output layers with regularization
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc_out = nn.Linear(hidden_size // 4, 1)

        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)

        # Self-attention for feature importance
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Use the last time step
        out = attn_out[:, -1, :]

        # Feed forward layers
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.fc_out(out)

        return out

class CNNAlphaModel(nn.Module):
    """GPU-optimized CNN model for pattern recognition"""
    def __init__(self, input_size, sequence_length):
        super(CNNAlphaModel, self).__init__()

        # 1D Convolutional layers for time series patterns
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        # Pooling and normalization
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(256)

        # Calculate flattened size after convolutions
        conv_output_size = self._get_conv_output_size(input_size, sequence_length)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc_out = nn.Linear(128, 1)

        self.relu = nn.ReLU()

    def _get_conv_output_size(self, input_size, sequence_length):
        # Simulate forward pass to get output size
        x = torch.randn(1, input_size, sequence_length)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return x.numel()

    def forward(self, x):
        # Transpose for 1D conv (batch, features, sequence)
        x = x.transpose(1, 2)

        # Convolutional layers with batch normalization
        x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))
        x = self.dropout(x)

        x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))
        x = self.dropout(x)

        x = self.pool(self.relu(self.batch_norm3(self.conv3(x))))
        x = self.dropout(x)

        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc_out(x)

        return x

class GPUAlphaDiscovery:
    """GPU-accelerated alpha discovery system"""

    def __init__(self):
        self.api = tradeapi.REST(
            base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY')
        )

        self.device = device
        self.scaler = MinMaxScaler()
        self.models = {}
        self.sequence_length = 60
        self.prediction_horizon = 5

        # GTX 1660 Super optimized settings
        self.batch_size = 256 if device.type == 'cuda' else 64
        self.num_workers = 4
        self.learning_rate = 0.001
        self.epochs = 100

        print(f">> GPU ALPHA DISCOVERY SYSTEM INITIALIZED")
        print(f">> Device: {self.device}")
        print(f">> GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f">> GPU Name: {torch.cuda.get_device_name(0)}")
            print(f">> GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f">> Batch Size: {self.batch_size}")
        print(f">> Expected 5-10x speedup on GTX 1660 Super")

    def get_market_data(self, symbol: str, timeframe: str = '1Day', limit: int = 1000) -> pd.DataFrame:
        """Fetch market data efficiently"""
        try:
            bars = self.api.get_bars(
                symbol,
                timeframe,
                limit=limit,
                asof=None,
                feed='iex',
                adjustment='raw'
            ).df

            if bars.empty:
                return pd.DataFrame()

            bars = bars.reset_index()
            bars['symbol'] = symbol
            return bars

        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_gpu_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features optimized for GPU processing"""
        if df.empty or len(df) < 20:
            return df

        try:
            # Vectorized calculations for GPU efficiency
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Moving averages (vectorized)
            periods = [5, 10, 20, 50]
            for period in periods:
                df[f'sma_{period}'] = df['close'].rolling(window=period, min_periods=1).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period, min_periods=1).mean()
                df[f'price_vs_sma_{period}'] = (df['close'] / df[f'sma_{period}'] - 1) * 100

            # Volatility features
            for window in [5, 10, 20]:
                df[f'volatility_{window}'] = df['returns'].rolling(window=window, min_periods=1).std()
                df[f'vol_rank_{window}'] = df[f'volatility_{window}'].rolling(window=50, min_periods=1).rank(pct=True)

            # Momentum indicators
            df['roc_5'] = (df['close'] / df['close'].shift(5) - 1) * 100
            df['roc_10'] = (df['close'] / df['close'].shift(10) - 1) * 100
            df['roc_20'] = (df['close'] / df['close'].shift(20) - 1) * 100

            # RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            df['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = df['close'].ewm(span=12, min_periods=1).mean()
            ema_26 = df['close'].ewm(span=26, min_periods=1).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9, min_periods=1).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

            # Volume analysis
            df['volume_sma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-8)
            df['price_volume_trend'] = df['returns'] * df['volume_ratio']

            # Support/Resistance
            df['high_20'] = df['high'].rolling(window=20, min_periods=1).max()
            df['low_20'] = df['low'].rolling(window=20, min_periods=1).min()
            df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)

            # Advanced features for deep learning
            df['hl_pct'] = (df['high'] - df['low']) / (df['close'] + 1e-8) * 100
            df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
            df['trend_strength'] = abs(df['ema_10'] - df['ema_20']) / df['close']

            return df.fillna(0)

        except Exception as e:
            logging.error(f"Error calculating features: {e}")
            return df.fillna(0)

    def create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create time series sequences for GPU training"""
        X, y = [], []

        for i in range(self.sequence_length, len(data) - self.prediction_horizon + 1):
            X.append(data[i-self.sequence_length:i])
            y.append(target[i + self.prediction_horizon - 1])  # Predict N days ahead

        return np.array(X), np.array(y)

    def train_gpu_model(self, symbol: str, model_type: str = 'lstm') -> Dict[str, Any]:
        """Train GPU-accelerated model"""
        try:
            # Get data
            df = self.get_market_data(symbol, limit=2000)
            if df.empty or len(df) < self.sequence_length + 50:
                return {}

            # Calculate features
            features_df = self.calculate_gpu_features(df.copy())

            # Select feature columns
            feature_cols = [col for col in features_df.columns if col not in
                          ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap', 'timestamp', 'symbol']]

            if len(feature_cols) < 10:
                return {}

            # Prepare data
            X = features_df[feature_cols].values
            y = features_df['close'].values

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Create sequences
            X_seq, y_seq = self.create_sequences(X_scaled, y)

            if len(X_seq) < 100:
                return {}

            # Train-test split
            split_idx = int(len(X_seq) * 0.8)
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

            # Create datasets
            train_dataset = AlphaDataset(X_train, y_train)
            test_dataset = AlphaDataset(X_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            # Initialize model
            if model_type == 'lstm':
                model = LSTMAlphaModel(len(feature_cols)).to(self.device)
            else:
                model = CNNAlphaModel(len(feature_cols), self.sequence_length).to(self.device)

            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

            # Training loop with GPU acceleration
            model.train()
            train_losses = []

            for epoch in range(self.epochs):
                epoch_loss = 0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()

                    # Forward pass on GPU
                    outputs = model(X_batch).squeeze()
                    loss = criterion(outputs, y_batch)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(train_loader)
                train_losses.append(avg_loss)
                scheduler.step(avg_loss)

                if epoch % 20 == 0:
                    logging.info(f"{symbol} - {model_type} - Epoch {epoch}: Loss = {avg_loss:.6f}")

            # Evaluation
            model.eval()
            test_losses = []
            predictions = []
            actuals = []

            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    outputs = model(X_batch).squeeze()
                    test_loss = criterion(outputs, y_batch)
                    test_losses.append(test_loss.item())

                    predictions.extend(outputs.cpu().numpy())
                    actuals.extend(y_batch.cpu().numpy())

            # Calculate metrics
            test_mse = np.mean(test_losses)
            test_rmse = np.sqrt(test_mse)

            # R² score
            ss_res = np.sum((np.array(actuals) - np.array(predictions)) ** 2)
            ss_tot = np.sum((np.array(actuals) - np.mean(actuals)) ** 2)
            r2_score = 1 - (ss_res / (ss_tot + 1e-8))

            logging.info(f"{symbol} - {model_type} - Training complete: R² = {r2_score:.4f}, RMSE = {test_rmse:.4f}")

            return {
                'model': model,
                'scaler': self.scaler,
                'feature_columns': feature_cols,
                'r2_score': r2_score,
                'rmse': test_rmse,
                'train_losses': train_losses,
                'model_type': model_type,
                'last_sequence': X_scaled[-self.sequence_length:],
                'training_date': datetime.now()
            }

        except Exception as e:
            logging.error(f"Error training GPU model for {symbol}: {e}")
            return {}

    def predict_alpha_gpu(self, symbol: str) -> Dict[str, Any]:
        """Generate GPU-accelerated alpha predictions"""
        try:
            # Train ensemble of models
            lstm_result = self.train_gpu_model(symbol, 'lstm')
            cnn_result = self.train_gpu_model(symbol, 'cnn')

            if not lstm_result or not cnn_result:
                return {}

            # Make predictions with both models
            predictions = {}

            # LSTM prediction
            lstm_model = lstm_result['model']
            lstm_model.eval()
            with torch.no_grad():
                lstm_input = torch.FloatTensor(lstm_result['last_sequence']).unsqueeze(0).to(self.device)
                lstm_pred = lstm_model(lstm_input).cpu().item()
                predictions['lstm'] = lstm_pred

            # CNN prediction
            cnn_model = cnn_result['model']
            cnn_model.eval()
            with torch.no_grad():
                cnn_input = torch.FloatTensor(cnn_result['last_sequence']).unsqueeze(0).to(self.device)
                cnn_pred = cnn_model(cnn_input).cpu().item()
                predictions['cnn'] = cnn_pred

            # Ensemble prediction (weighted by R² scores)
            lstm_weight = lstm_result['r2_score'] / (lstm_result['r2_score'] + cnn_result['r2_score'] + 1e-8)
            cnn_weight = cnn_result['r2_score'] / (lstm_result['r2_score'] + cnn_result['r2_score'] + 1e-8)

            ensemble_pred = lstm_weight * lstm_pred + cnn_weight * cnn_pred

            # Get current price for return calculation
            current_data = self.get_market_data(symbol, limit=1)
            if current_data.empty:
                return {}

            current_price = current_data['close'].iloc[-1]
            predicted_return = (ensemble_pred - current_price) / current_price
            confidence = (lstm_result['r2_score'] + cnn_result['r2_score']) / 2

            return {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': ensemble_pred,
                'predicted_return_pct': predicted_return * 100,
                'confidence': confidence,
                'alpha_strength': abs(predicted_return) * confidence,
                'individual_predictions': predictions,
                'model_scores': {
                    'lstm_r2': lstm_result['r2_score'],
                    'cnn_r2': cnn_result['r2_score']
                },
                'gpu_accelerated': True,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logging.error(f"Error predicting alpha for {symbol}: {e}")
            return {}

    def run_gpu_alpha_scan(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Run GPU-accelerated alpha discovery scan"""
        if symbols is None:
            symbols = ['SPY', 'QQQ', 'TSLA', 'AAPL', 'NVDA', 'AMD', 'MSFT', 'GOOGL']

        logging.info(f">> Starting GPU-accelerated alpha scan on {len(symbols)} symbols...")
        start_time = datetime.now()

        # Process symbols (sequentially due to GPU memory constraints)
        results = []
        for symbol in symbols:
            try:
                logging.info(f">> Processing {symbol} on GPU...")
                result = self.predict_alpha_gpu(symbol)
                if result and result.get('alpha_strength', 0) > 0.01:
                    results.append(result)
            except Exception as e:
                logging.error(f"Error processing {symbol}: {e}")

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Sort by alpha strength
        results.sort(key=lambda x: x['alpha_strength'], reverse=True)

        summary = {
            'scan_timestamp': start_time,
            'symbols_scanned': len(symbols),
            'opportunities_found': len(results),
            'processing_time_seconds': processing_time,
            'gpu_acceleration': True,
            'device': str(self.device),
            'top_opportunities': results[:5],
            'performance_metrics': {
                'avg_time_per_symbol': processing_time / len(symbols),
                'gpu_memory_used': torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            }
        }

        logging.info(f">> GPU Alpha scan complete!")
        logging.info(f">> Processing time: {processing_time:.1f}s ({processing_time/len(symbols):.1f}s per symbol)")
        logging.info(f">> Found {len(results)} high-alpha opportunities")

        if torch.cuda.is_available():
            logging.info(f">> GPU memory used: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

        return summary

if __name__ == "__main__":
    # Initialize GPU alpha discovery
    gpu_alpha = GPUAlphaDiscovery()

    # Run scan
    results = gpu_alpha.run_gpu_alpha_scan()

    # Save results
    with open(f'gpu_alpha_scan_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n>> GPU Alpha Discovery Complete!")
    print(f">> Found {results['opportunities_found']} opportunities in {results['processing_time_seconds']:.1f}s")
    if results['top_opportunities']:
        top = results['top_opportunities'][0]
        print(f">> Top: {top['symbol']} - {top['predicted_return_pct']:.2f}% predicted return")