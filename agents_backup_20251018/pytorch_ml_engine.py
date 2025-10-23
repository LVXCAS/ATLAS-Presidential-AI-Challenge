#!/usr/bin/env python3
"""
PyTorch ML Engine
Advanced machine learning models with Sharpe-based loss functions
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Try to import PyTorch and related libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
    print("+ PyTorch available for advanced ML models")
    
    # Try to import scikit-learn for preprocessing
    try:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        SKLEARN_AVAILABLE = True
    except ImportError:
        SKLEARN_AVAILABLE = False

    class SharpeOptimizedNetwork(nn.Module):
        """Neural network optimized for Sharpe ratio"""
        
        def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32], 
                     output_size: int = 1, dropout_rate: float = 0.2):
            super(SharpeOptimizedNetwork, self).__init__()
            
            self.input_size = input_size
            self.output_size = output_size
            
            # Build network layers
            layers = []
            prev_size = input_size
            
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.BatchNorm1d(hidden_size)
                ])
                prev_size = hidden_size
            
            # Output layer
            layers.append(nn.Linear(prev_size, output_size))
            layers.append(nn.Tanh())  # Output between -1 and 1
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)

    class LSTMPredictor(nn.Module):
        """LSTM model for time series prediction"""
        
        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                     output_size: int = 1, dropout_rate: float = 0.2):
            super(LSTMPredictor, self).__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=dropout_rate)
            self.dropout = nn.Dropout(dropout_rate)
            self.linear = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            # x shape: (batch_size, sequence_length, input_size)
            lstm_out, _ = self.lstm(x)
            # Take the last output
            last_output = lstm_out[:, -1, :]
            output = self.dropout(last_output)
            output = self.linear(output)
            return output

    class SharpeRatioLoss(nn.Module):
        """Custom loss function that optimizes for Sharpe ratio"""
        
        def __init__(self, risk_free_rate: float = 0.02):
            super(SharpeRatioLoss, self).__init__()
            self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate
        
        def forward(self, predictions, returns):
            # Calculate portfolio returns based on predictions
            portfolio_returns = predictions.squeeze() * returns
            
            # Calculate Sharpe ratio
            excess_returns = portfolio_returns - self.risk_free_rate
            mean_return = torch.mean(excess_returns)
            std_return = torch.std(excess_returns)
            
            # Avoid division by zero
            sharpe_ratio = mean_return / (std_return + 1e-8)
            
            # Return negative Sharpe ratio (since we want to maximize)
            return -sharpe_ratio

except ImportError:
    PYTORCH_AVAILABLE = False
    SKLEARN_AVAILABLE = False
    print("- PyTorch not available - using custom implementation")
    
    # Dummy classes when PyTorch is not available
    class SharpeOptimizedNetwork:
        def __init__(self, *args, **kwargs):
            pass
    
    class LSTMPredictor:
        def __init__(self, *args, **kwargs):
            pass
    
    class SharpeRatioLoss:
        def __init__(self, *args, **kwargs):
            pass

class PyTorchMLEngine:
    """Advanced ML engine using PyTorch with financial optimization"""
    
    def __init__(self):
        self.pytorch_available = PYTORCH_AVAILABLE
        self.sklearn_available = SKLEARN_AVAILABLE
        
        # Model cache
        self.models = {}
        self.scalers = {}
        
        if PYTORCH_AVAILABLE:
            print("+ PyTorch ML Engine initialized with GPU support" if torch.cuda.is_available() else "+ PyTorch ML Engine initialized with CPU")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            print("+ PyTorch ML Engine initialized with custom implementation")
            self.device = 'cpu'
    
    async def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        try:
            if data.empty or len(data) < 20:
                return pd.DataFrame()
            
            features = pd.DataFrame(index=data.index)
            
            # Price-based features
            features['returns'] = data['close'].pct_change()
            features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            features['volatility_5d'] = features['returns'].rolling(5).std()
            features['volatility_20d'] = features['returns'].rolling(20).std()
            
            # Technical indicators
            features['rsi_14'] = self._calculate_rsi(data['close'], 14)
            features['sma_10'] = data['close'].rolling(10).mean()
            features['sma_20'] = data['close'].rolling(20).mean()
            features['price_to_sma_20'] = data['close'] / features['sma_20']
            
            # Momentum features
            features['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
            features['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
            features['momentum_20d'] = data['close'] / data['close'].shift(20) - 1
            
            # Volume features
            if 'volume' in data.columns:
                features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
                features['price_volume'] = features['returns'] * np.log(data['volume'] + 1)
            
            # Bollinger Bands
            bb_middle = data['close'].rolling(20).mean()
            bb_std = data['close'].rolling(20).std()
            features['bb_position'] = (data['close'] - (bb_middle - 2*bb_std)) / (4*bb_std)
            
            # Drop NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            print(f"- Feature preparation error: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    async def train_sharpe_model(self, features: pd.DataFrame, returns: pd.Series, 
                                model_name: str = 'sharpe_model') -> Dict:
        """Train Sharpe-optimized neural network"""
        
        if not self.pytorch_available:
            return await self._train_custom_model(features, returns, model_name)
        
        try:
            # Align features and returns
            common_index = features.index.intersection(returns.index)
            if len(common_index) < 50:
                return {'error': 'Insufficient data for training'}
            
            features_aligned = features.loc[common_index]
            returns_aligned = returns.loc[common_index]
            
            # Prepare data
            X = features_aligned.values
            y = returns_aligned.values
            
            # Scale features
            if self.sklearn_available:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                self.scalers[model_name] = scaler
            else:
                X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            
            # Split data
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_test_tensor = torch.FloatTensor(y_test).to(self.device)
            
            # Create model
            model = SharpeOptimizedNetwork(
                input_size=X_train.shape[1],
                hidden_sizes=[64, 32, 16],
                output_size=1
            ).to(self.device)
            
            # Define loss and optimizer
            criterion = SharpeRatioLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
            
            # Training loop
            model.train()
            train_losses = []
            
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                
                # Validation
                if epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_test_tensor)
                        val_loss = criterion(val_outputs, y_test_tensor)
                        scheduler.step(val_loss)
                    model.train()
            
            # Save model
            self.models[model_name] = model
            
            # Calculate final metrics
            model.eval()
            with torch.no_grad():
                train_pred = model(X_train_tensor).cpu().numpy()
                test_pred = model(X_test_tensor).cpu().numpy()
            
            # Calculate Sharpe ratios
            train_portfolio_returns = train_pred.flatten() * y_train
            test_portfolio_returns = test_pred.flatten() * y_test
            
            train_sharpe = np.mean(train_portfolio_returns) / (np.std(train_portfolio_returns) + 1e-8)
            test_sharpe = np.mean(test_portfolio_returns) / (np.std(test_portfolio_returns) + 1e-8)
            
            return {
                'model_name': model_name,
                'train_samples': len(y_train),
                'test_samples': len(y_test),
                'final_loss': train_losses[-1],
                'train_sharpe': float(train_sharpe * np.sqrt(252)),  # Annualized
                'test_sharpe': float(test_sharpe * np.sqrt(252)),   # Annualized
                'features_used': list(features_aligned.columns),
                'training_complete': True
            }
            
        except Exception as e:
            print(f"- Sharpe model training error: {e}")
            return {'error': str(e)}
    
    async def _train_custom_model(self, features: pd.DataFrame, returns: pd.Series, 
                                 model_name: str) -> Dict:
        """Custom model training when PyTorch is not available"""
        try:
            # Simple linear regression using numpy
            X = features.values
            y = returns.values
            
            # Add bias term
            X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
            
            # Normal equation: w = (X'X)^-1 X'y
            try:
                weights = np.linalg.solve(X_with_bias.T @ X_with_bias, X_with_bias.T @ y)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                weights = np.linalg.pinv(X_with_bias) @ y
            
            # Store model
            self.models[model_name] = {
                'weights': weights,
                'features': list(features.columns),
                'type': 'linear_regression'
            }
            
            # Calculate R²
            y_pred = X_with_bias @ weights
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2_score = 1 - (ss_res / ss_tot)
            
            return {
                'model_name': model_name,
                'samples': len(y),
                'r2_score': float(r2_score),
                'features_used': list(features.columns),
                'training_complete': True,
                'model_type': 'custom_linear'
            }
            
        except Exception as e:
            print(f"- Custom model training error: {e}")
            return {'error': str(e)}
    
    async def predict_returns(self, features: pd.DataFrame, model_name: str = 'sharpe_model') -> Dict:
        """Generate return predictions"""
        
        if model_name not in self.models:
            return {'error': f'Model {model_name} not found'}
        
        try:
            if self.pytorch_available and isinstance(self.models[model_name], nn.Module):
                return await self._pytorch_predict(features, model_name)
            else:
                return await self._custom_predict(features, model_name)
                
        except Exception as e:
            print(f"- Prediction error: {e}")
            return {'error': str(e)}
    
    async def _pytorch_predict(self, features: pd.DataFrame, model_name: str) -> Dict:
        """PyTorch model prediction"""
        model = self.models[model_name]
        
        # Scale features
        X = features.values
        if model_name in self.scalers:
            X_scaled = self.scalers[model_name].transform(X)
        else:
            X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Predict
        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy()
        
        return {
            'predictions': predictions.flatten().tolist(),
            'prediction_dates': features.index.tolist(),
            'model_type': 'pytorch',
            'confidence': 0.7  # Placeholder
        }
    
    async def _custom_predict(self, features: pd.DataFrame, model_name: str) -> Dict:
        """Custom model prediction"""
        model_info = self.models[model_name]
        
        # Prepare features
        X = features[model_info['features']].values
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Predict
        predictions = X_with_bias @ model_info['weights']
        
        return {
            'predictions': predictions.tolist(),
            'prediction_dates': features.index.tolist(),
            'model_type': 'custom_linear',
            'confidence': 0.6  # Placeholder
        }
    
    async def calculate_feature_importance(self, model_name: str = 'sharpe_model') -> Dict:
        """Calculate feature importance"""
        
        if model_name not in self.models:
            return {'error': f'Model {model_name} not found'}
        
        try:
            if self.pytorch_available and isinstance(self.models[model_name], nn.Module):
                # For neural networks, use gradient-based importance
                return {'feature_importance': 'gradient_based', 'status': 'not_implemented'}
            else:
                # For linear models, use weight magnitudes
                model_info = self.models[model_name]
                weights = model_info['weights'][1:]  # Exclude bias
                features = model_info['features']
                
                importance = dict(zip(features, np.abs(weights)))
                sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
                
                return {
                    'feature_importance': sorted_importance,
                    'model_type': 'linear',
                    'total_features': len(features)
                }
                
        except Exception as e:
            return {'error': str(e)}
    
    def get_model_status(self) -> Dict:
        """Get status of all models"""
        return {
            'pytorch_available': self.pytorch_available,
            'sklearn_available': self.sklearn_available,
            'device': str(self.device),
            'loaded_models': list(self.models.keys()),
            'model_count': len(self.models)
        }

# Create global instance
pytorch_engine = PyTorchMLEngine()