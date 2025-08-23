#!/usr/bin/env python3
"""
HIVE TRADE - AI Training System (Parallel to Live Trading)
Uses simulated dashboard data to train NLP/ML models with yfinance
"""

import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
import json
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class AITrainingSystem:
    def __init__(self):
        print("=" * 60)
        print("HIVE TRADE - AI TRAINING SYSTEM")
        print("Parallel ML/NLP Training with Simulated Data")
        print("=" * 60)
        
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'SPY', 'QQQ']
        self.crypto_symbols = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD']
        
        # ML Models
        self.stock_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.crypto_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Training data storage
        self.training_data = []
        self.model_accuracy = {'stock': 0.0, 'crypto': 0.0}
        
        print(">> AI Training System initialized")
        print(f">> Monitoring {len(self.symbols)} stocks")
        print(f">> Monitoring {len(self.crypto_symbols)} crypto assets")
        
    def fetch_market_data(self, symbol, period="5d"):
        """Fetch real market data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if len(hist) > 0:
                latest = hist.iloc[-1]
                return {
                    'symbol': symbol,
                    'price': latest['Close'],
                    'volume': latest['Volume'],
                    'high': latest['High'],
                    'low': latest['Low'],
                    'change': (latest['Close'] - latest['Open']) / latest['Open'],
                    'timestamp': datetime.now()
                }
            return None
        except Exception as e:
            print(f">> Data fetch error for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, symbol, period="30d"):
        """Calculate technical indicators for ML features"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if len(hist) < 20:
                return None
            
            # Technical indicators
            prices = hist['Close']
            volumes = hist['Volume']
            
            sma_10 = prices.rolling(window=10).mean().iloc[-1]
            sma_20 = prices.rolling(window=20).mean().iloc[-1]
            rsi = self.calculate_rsi(prices).iloc[-1]
            volatility = prices.rolling(window=10).std().iloc[-1]
            volume_avg = volumes.rolling(window=10).mean().iloc[-1]
            
            current_price = prices.iloc[-1]
            price_momentum = (current_price - prices.iloc[-5]) / prices.iloc[-5]
            
            return {
                'sma_10': sma_10,
                'sma_20': sma_20,
                'rsi': rsi,
                'volatility': volatility,
                'volume_avg': volume_avg,
                'price_momentum': price_momentum,
                'current_price': current_price
            }
        except Exception as e:
            print(f">> Technical analysis error for {symbol}: {e}")
            return None
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def sentiment_analysis_simple(self, symbol):
        """Simple sentiment analysis (placeholder for NLP)"""
        # Simulate sentiment based on recent price action
        data = self.fetch_market_data(symbol, "5d")
        if not data:
            return 0.0
        
        # Simple sentiment: positive change = positive sentiment
        sentiment_score = data['change'] * 2  # Amplify for sentiment
        sentiment_score = np.clip(sentiment_score, -1.0, 1.0)
        
        return sentiment_score
    
    def prepare_training_features(self, symbol):
        """Prepare ML training features"""
        try:
            # Get technical indicators
            tech_data = self.calculate_technical_indicators(symbol)
            if not tech_data:
                return None
            
            # Get sentiment
            sentiment = self.sentiment_analysis_simple(symbol)
            
            # Get current market data
            market_data = self.fetch_market_data(symbol, "1d")
            if not market_data:
                return None
            
            features = [
                tech_data['sma_10'] / tech_data['current_price'],  # SMA ratio
                tech_data['sma_20'] / tech_data['current_price'],  # SMA ratio
                tech_data['rsi'] / 100.0,  # Normalized RSI
                tech_data['volatility'],
                tech_data['price_momentum'],
                sentiment,
                np.log(tech_data['volume_avg']) if tech_data['volume_avg'] > 0 else 0,
                market_data['change']
            ]
            
            # Generate label (1 for buy, 0 for hold/sell)
            label = 1 if tech_data['price_momentum'] > 0.02 else 0
            
            return {
                'symbol': symbol,
                'features': features,
                'label': label,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f">> Feature preparation error for {symbol}: {e}")
            return None
    
    def train_models(self):
        """Train ML models with collected data"""
        if len(self.training_data) < 50:
            print(f">> Insufficient data for training ({len(self.training_data)} samples)")
            return
        
        try:
            # Separate stock and crypto data
            stock_data = [d for d in self.training_data if not d['symbol'].endswith('-USD')]
            crypto_data = [d for d in self.training_data if d['symbol'].endswith('-USD')]
            
            # Train stock model
            if len(stock_data) >= 20:
                X_stock = [d['features'] for d in stock_data]
                y_stock = [d['label'] for d in stock_data]
                
                self.stock_model.fit(X_stock, y_stock)
                stock_pred = self.stock_model.predict(X_stock)
                self.model_accuracy['stock'] = accuracy_score(y_stock, stock_pred)
                
                print(f">> Stock model trained: {len(stock_data)} samples, accuracy: {self.model_accuracy['stock']:.2f}")
            
            # Train crypto model
            if len(crypto_data) >= 20:
                X_crypto = [d['features'] for d in crypto_data]
                y_crypto = [d['label'] for d in crypto_data]
                
                self.crypto_model.fit(X_crypto, y_crypto)
                crypto_pred = self.crypto_model.predict(X_crypto)
                self.model_accuracy['crypto'] = accuracy_score(y_crypto, crypto_pred)
                
                print(f">> Crypto model trained: {len(crypto_data)} samples, accuracy: {self.model_accuracy['crypto']:.2f}")
        
        except Exception as e:
            print(f">> Training error: {e}")
    
    def generate_ai_predictions(self):
        """Generate AI predictions using trained models"""
        predictions = []
        
        # Stock predictions
        for symbol in self.symbols[:3]:  # Test with 3 symbols
            feature_data = self.prepare_training_features(symbol)
            if feature_data and len(feature_data['features']) == 8:
                try:
                    prediction = self.stock_model.predict([feature_data['features']])[0]
                    confidence = max(self.stock_model.predict_proba([feature_data['features']])[0])
                    
                    predictions.append({
                        'symbol': symbol,
                        'prediction': 'BUY' if prediction == 1 else 'HOLD',
                        'confidence': confidence,
                        'model': 'stock_ai'
                    })
                except:
                    pass
        
        # Crypto predictions
        for symbol in self.crypto_symbols[:2]:  # Test with 2 crypto
            feature_data = self.prepare_training_features(symbol)
            if feature_data and len(feature_data['features']) == 8:
                try:
                    prediction = self.crypto_model.predict([feature_data['features']])[0]
                    confidence = max(self.crypto_model.predict_proba([feature_data['features']])[0])
                    
                    predictions.append({
                        'symbol': symbol,
                        'prediction': 'BUY' if prediction == 1 else 'HOLD',
                        'confidence': confidence,
                        'model': 'crypto_ai'
                    })
                except:
                    pass
        
        return predictions
    
    async def training_cycle(self, cycle_num):
        """Execute one training cycle"""
        print(f"\\n>> TRAINING CYCLE #{cycle_num} - {datetime.now().strftime('%H:%M:%S')}")
        
        # Collect training data
        new_samples = 0
        
        # Collect stock data
        for symbol in self.symbols:
            feature_data = self.prepare_training_features(symbol)
            if feature_data:
                self.training_data.append(feature_data)
                new_samples += 1
        
        # Collect crypto data
        for symbol in self.crypto_symbols:
            feature_data = self.prepare_training_features(symbol)
            if feature_data:
                self.training_data.append(feature_data)
                new_samples += 1
        
        print(f">> Collected {new_samples} new training samples")
        print(f">> Total training data: {len(self.training_data)} samples")
        
        # Train models every 5 cycles
        if cycle_num % 5 == 0:
            print(f">> Training models...")
            self.train_models()
        
        # Generate AI predictions
        if self.model_accuracy['stock'] > 0 or self.model_accuracy['crypto'] > 0:
            predictions = self.generate_ai_predictions()
            if predictions:
                print(f">> AI PREDICTIONS ({len(predictions)}):")
                for pred in predictions:
                    print(f"   {pred['symbol']}: {pred['prediction']} ({pred['confidence']:.2f}) [{pred['model']}]")
        
        # Keep training data manageable
        if len(self.training_data) > 1000:
            self.training_data = self.training_data[-500:]  # Keep recent 500
            print(">> Trimmed training data to recent 500 samples")
    
    async def start_training_system(self):
        """Start parallel AI training system"""
        print("\\n>> STARTING AI TRAINING SYSTEM (PARALLEL)")
        print(">> This runs alongside live crypto trading")
        print(">> Training with real market data from yfinance")
        
        cycle = 0
        
        try:
            while True:
                cycle += 1
                await self.training_cycle(cycle)
                
                # Wait between cycles (3 minutes)
                wait_time = 180
                print(f">> Waiting {wait_time}s until next training cycle...")
                await asyncio.sleep(wait_time)
                
        except KeyboardInterrupt:
            print("\\n>> AI TRAINING STOPPED")
            print(f">> Completed {cycle} training cycles")
            print(f">> Final model accuracy - Stock: {self.model_accuracy['stock']:.2f}, Crypto: {self.model_accuracy['crypto']:.2f}")

async def main():
    trainer = AITrainingSystem()
    await trainer.start_training_system()

if __name__ == "__main__":
    print("\\nStarting HIVE TRADE AI Training System (Parallel)...")
    asyncio.run(main())