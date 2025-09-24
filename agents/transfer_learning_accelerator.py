#!/usr/bin/env python3
"""
Transfer Learning Accelerator - Pre-trained Models for Faster Learning
Uses pre-trained models from similar financial datasets to jumpstart learning
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

class TransferLearningAccelerator:
    """Accelerate ML learning using transfer learning techniques"""
    
    def __init__(self):
        self.base_models = {}
        self.transfer_models = {}
        self.feature_cache = {}
        
        # Pre-trained model universe (learn from these first)
        self.universe = [
            'SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
            'JPM', 'BAC', 'XLF', 'XLK', 'XLV', 'XLE', 'GLD', 'TLT', 'VIX'
        ]
        
    async def create_base_models(self):
        """Create pre-trained base models from historical data"""
        print("Creating base models from historical data...")
        
        for symbol in self.universe:
            try:
                print(f"Training base model for {symbol}...")
                
                # Get 5 years of historical data
                data = yf.download(symbol, period="5y", interval="1d")
                if data.empty:
                    continue
                
                # Create features and labels
                features, labels = self.create_training_data(data)
                
                if len(features) > 1000:  # Need sufficient data
                    # Train base model
                    X_train, X_test, y_train, y_test = train_test_split(
                        features, labels, test_size=0.2, random_state=42
                    )
                    
                    model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    model.fit(X_train, y_train)
                    accuracy = model.score(X_test, y_test)
                    
                    if accuracy > 0.52:  # Only keep models better than random
                        self.base_models[symbol] = model
                        print(f"  Base model for {symbol}: {accuracy:.1%} accuracy")
                        
                        # Save model
                        os.makedirs('models/base', exist_ok=True)
                        joblib.dump(model, f'models/base/{symbol}_base.pkl')
                
                await asyncio.sleep(0.1)  # Prevent overload
                
            except Exception as e:
                print(f"Error training base model for {symbol}: {e}")
        
        print(f"Created {len(self.base_models)} base models")
    
    def create_training_data(self, data):
        """Create features and labels for training"""
        # Calculate technical indicators
        data['SMA_5'] = data['Close'].rolling(5).mean()
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['RSI'] = self.calculate_rsi(data['Close'])
        data['MACD'] = data['Close'].ewm(12).mean() - data['Close'].ewm(26).mean()
        data['BB_upper'], data['BB_lower'] = self.calculate_bollinger_bands(data['Close'])
        data['Volume_SMA'] = data['Volume'].rolling(20).mean()
        
        # Price features
        data['Price_Change'] = data['Close'].pct_change()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        data['High_Low_Ratio'] = (data['High'] - data['Low']) / data['Close']
        
        # Create features array
        feature_cols = [
            'SMA_5', 'SMA_20', 'RSI', 'MACD', 'BB_upper', 'BB_lower',
            'Volume_Ratio', 'High_Low_Ratio', 'Price_Change'
        ]
        
        # Create labels (1 = price up next day, 0 = price down/flat)
        data['Next_Return'] = data['Close'].pct_change().shift(-1)
        data['Label'] = (data['Next_Return'] > 0.01).astype(int)  # 1% threshold
        
        # Clean data
        data = data.dropna()
        
        features = data[feature_cols].values
        labels = data['Label'].values
        
        return features, labels
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower
    
    async def create_transfer_model(self, target_symbol):
        """Create transfer model for specific symbol using base models"""
        try:
            print(f"Creating transfer model for {target_symbol}...")
            
            # Get recent data for target symbol
            target_data = yf.download(target_symbol, period="3mo", interval="1d")
            if target_data.empty:
                return None
            
            # Create features and labels
            features, labels = self.create_training_data(target_data)
            
            if len(features) < 30:  # Need minimum data
                print(f"Insufficient data for {target_symbol}")
                return None
            
            # Find most similar base model
            best_model = None
            best_correlation = 0
            
            for base_symbol, base_model in self.base_models.items():
                # Calculate correlation between symbols
                correlation = await self.calculate_symbol_correlation(target_symbol, base_symbol)
                
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_model = base_model
            
            if best_model and best_correlation > 0.3:  # Reasonable correlation
                # Use base model predictions as features for transfer learning
                base_predictions = best_model.predict_proba(features)[:, 1]
                
                # Create enhanced features
                enhanced_features = np.column_stack([features, base_predictions])
                
                # Train transfer model
                if len(enhanced_features) > 20:
                    transfer_model = RandomForestClassifier(
                        n_estimators=50,
                        max_depth=8,
                        random_state=42
                    )
                    
                    transfer_model.fit(enhanced_features, labels)
                    
                    # Test performance
                    if len(enhanced_features) > 10:
                        recent_features = enhanced_features[-10:]
                        recent_labels = labels[-10:]
                        accuracy = transfer_model.score(recent_features, recent_labels)
                        
                        if accuracy > 0.5:
                            self.transfer_models[target_symbol] = transfer_model
                            print(f"Transfer model for {target_symbol}: {accuracy:.1%} accuracy")
                            
                            # Save model
                            os.makedirs('models/transfer', exist_ok=True)
                            joblib.dump(transfer_model, f'models/transfer/{target_symbol}_transfer.pkl')
                            
                            return transfer_model
            
            return None
            
        except Exception as e:
            print(f"Error creating transfer model for {target_symbol}: {e}")
            return None
    
    async def calculate_symbol_correlation(self, symbol1, symbol2, period="1y"):
        """Calculate correlation between two symbols"""
        try:
            data1 = yf.download(symbol1, period=period, interval="1d")['Close']
            data2 = yf.download(symbol2, period=period, interval="1d")['Close']
            
            # Align data
            common_dates = data1.index.intersection(data2.index)
            if len(common_dates) > 50:
                returns1 = data1.loc[common_dates].pct_change().dropna()
                returns2 = data2.loc[common_dates].pct_change().dropna()
                
                correlation = returns1.corr(returns2)
                return abs(correlation) if not np.isnan(correlation) else 0
            
            return 0
            
        except:
            return 0
    
    async def get_transfer_prediction(self, symbol, features):
        """Get prediction from transfer model"""
        try:
            if symbol in self.transfer_models:
                model = self.transfer_models[symbol]
                
                # Need to add base model prediction as feature
                best_base_model = None
                best_correlation = 0
                
                for base_symbol, base_model in self.base_models.items():
                    correlation = await self.calculate_symbol_correlation(symbol, base_symbol, period="3mo")
                    if correlation > best_correlation:
                        best_correlation = correlation
                        best_base_model = base_model
                
                if best_base_model:
                    base_prediction = best_base_model.predict_proba(features.reshape(1, -1))[0, 1]
                    enhanced_features = np.append(features, base_prediction).reshape(1, -1)
                    
                    prediction = model.predict_proba(enhanced_features)[0, 1]
                    confidence = abs(prediction - 0.5) * 2  # Convert to 0-1 confidence
                    
                    return {
                        'prediction': prediction,
                        'confidence': confidence,
                        'signal': 'BUY' if prediction > 0.6 else 'SELL' if prediction < 0.4 else 'HOLD'
                    }
            
            return None
            
        except Exception as e:
            print(f"Transfer prediction error for {symbol}: {e}")
            return None

# Global instance
transfer_accelerator = TransferLearningAccelerator()

async def initialize_transfer_learning():
    """Initialize transfer learning system"""
    print("Initializing Transfer Learning Accelerator...")
    await transfer_accelerator.create_base_models()
    print("Transfer learning system ready!")

if __name__ == "__main__":
    asyncio.run(initialize_transfer_learning())