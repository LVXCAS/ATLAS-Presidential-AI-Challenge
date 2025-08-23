#!/usr/bin/env python3
"""
HIVE TRADE - Enhanced Data Collector
Rapidly collect diverse market data for RL training
"""

import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
import json

class EnhancedDataCollector:
    def __init__(self):
        print("=" * 60)
        print("HIVE TRADE - ENHANCED DATA COLLECTOR")
        print("Rapidly building training dataset for RL")
        print("=" * 60)
        
        # Expanded symbol lists for more data
        self.crypto_symbols = [
            'BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD', 'SOL-USD', 
            'MATIC-USD', 'LINK-USD', 'UNI-USD', 'AVAX-USD', 'ATOM-USD'
        ]
        
        self.stock_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
            'NFLX', 'AMD', 'INTC', 'CRM', 'ADBE', 'PYPL', 'COIN',
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'ARKK'
        ]
        
        self.training_samples = []
        self.market_conditions = []
        
    def get_historical_data(self, symbol, period="30d"):
        """Get rich historical data for training"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if len(hist) < 20:
                return None
                
            # Calculate comprehensive technical indicators
            prices = hist['Close']
            volumes = hist['Volume']
            
            features = {}
            
            # Moving averages
            features['sma_5'] = prices.rolling(5).mean()
            features['sma_10'] = prices.rolling(10).mean()
            features['sma_20'] = prices.rolling(20).mean()
            
            # Price ratios
            features['price_sma5_ratio'] = prices / features['sma_5']
            features['price_sma20_ratio'] = prices / features['sma_20']
            
            # Volatility measures
            features['volatility_10'] = prices.rolling(10).std()
            features['volatility_20'] = prices.rolling(20).std()
            
            # RSI
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # Momentum
            features['momentum_5'] = (prices - prices.shift(5)) / prices.shift(5)
            features['momentum_10'] = (prices - prices.shift(10)) / prices.shift(10)
            
            # Volume indicators
            features['volume_sma'] = volumes.rolling(10).mean()
            features['volume_ratio'] = volumes / features['volume_sma']
            
            # Price position in range
            high_20 = hist['High'].rolling(20).max()
            low_20 = hist['Low'].rolling(20).min()
            features['price_position'] = (prices - low_20) / (high_20 - low_20)
            
            return features
            
        except Exception as e:
            print(f"   Error getting {symbol}: {e}")
            return None
    
    def create_training_samples(self, symbol, features):
        """Create training samples from historical data"""
        samples = []
        
        try:
            # Get last 20 days of complete data
            valid_data = []
            for i in range(len(features['sma_20'])):
                if not pd.isna(features['sma_20'].iloc[i]):
                    sample_features = [
                        features['price_sma5_ratio'].iloc[i] if not pd.isna(features['price_sma5_ratio'].iloc[i]) else 1.0,
                        features['price_sma20_ratio'].iloc[i] if not pd.isna(features['price_sma20_ratio'].iloc[i]) else 1.0,
                        features['rsi'].iloc[i] / 100 if not pd.isna(features['rsi'].iloc[i]) else 0.5,
                        np.clip(features['momentum_5'].iloc[i], -0.2, 0.2) if not pd.isna(features['momentum_5'].iloc[i]) else 0.0,
                        np.clip(features['momentum_10'].iloc[i], -0.3, 0.3) if not pd.isna(features['momentum_10'].iloc[i]) else 0.0,
                        np.clip(features['volatility_10'].iloc[i] / features['sma_10'].iloc[i], 0, 0.1) if not pd.isna(features['volatility_10'].iloc[i]) else 0.02,
                        np.clip(features['volume_ratio'].iloc[i], 0.1, 5.0) if not pd.isna(features['volume_ratio'].iloc[i]) else 1.0,
                        features['price_position'].iloc[i] if not pd.isna(features['price_position'].iloc[i]) else 0.5,
                    ]
                    
                    # Create label based on future price movement
                    if i < len(features['momentum_5']) - 1:
                        future_return = features['momentum_5'].iloc[i+1]
                        if future_return > 0.02:  # >2% gain
                            label = 1  # BUY
                        elif future_return < -0.02:  # >2% loss
                            label = 2  # SELL
                        else:
                            label = 0  # HOLD
                        
                        samples.append({
                            'symbol': symbol,
                            'features': sample_features,
                            'label': label,
                            'future_return': future_return,
                            'timestamp': datetime.now().isoformat()
                        })
            
            return samples[-50:]  # Last 50 samples
            
        except Exception as e:
            print(f"   Sample creation error for {symbol}: {e}")
            return []
    
    def analyze_market_conditions(self):
        """Analyze different market conditions for diverse training"""
        conditions = []
        
        try:
            # Get SPY for market sentiment
            spy = yf.Ticker('SPY')
            spy_hist = spy.history(period='30d')
            
            if len(spy_hist) > 0:
                spy_return = (spy_hist['Close'].iloc[-1] - spy_hist['Close'].iloc[0]) / spy_hist['Close'].iloc[0]
                spy_volatility = spy_hist['Close'].pct_change().std()
                
                if spy_return > 0.05:
                    market_condition = 'BULL'
                elif spy_return < -0.05:
                    market_condition = 'BEAR'
                else:
                    market_condition = 'SIDEWAYS'
                
                volatility_condition = 'HIGH' if spy_volatility > 0.02 else 'LOW'
                
                conditions.append({
                    'market_trend': market_condition,
                    'volatility': volatility_condition,
                    'spy_return': spy_return,
                    'spy_volatility': spy_volatility
                })
        
        except:
            pass
        
        return conditions
    
    async def rapid_data_collection(self):
        """Rapidly collect diverse training data"""
        print("\\n>> RAPID DATA COLLECTION STARTING")
        print(f"   Crypto symbols: {len(self.crypto_symbols)}")
        print(f"   Stock symbols: {len(self.stock_symbols)}")
        
        total_samples = 0
        
        # Collect crypto data
        print("\\n>> COLLECTING CRYPTO DATA:")
        for symbol in self.crypto_symbols[:8]:  # First 8 crypto
            print(f"   Processing {symbol}...")
            features = self.get_historical_data(symbol, "30d")
            
            if features is not None:
                samples = self.create_training_samples(symbol, features)
                self.training_samples.extend(samples)
                total_samples += len(samples)
                print(f"     Added {len(samples)} samples")
            
            await asyncio.sleep(0.5)  # Rate limiting
        
        # Collect stock data
        print("\\n>> COLLECTING STOCK DATA:")
        for symbol in self.stock_symbols[:12]:  # First 12 stocks
            print(f"   Processing {symbol}...")
            features = self.get_historical_data(symbol, "30d")
            
            if features is not None:
                samples = self.create_training_samples(symbol, features)
                self.training_samples.extend(samples)
                total_samples += len(samples)
                print(f"     Added {len(samples)} samples")
            
            await asyncio.sleep(0.3)
        
        # Analyze market conditions
        print("\\n>> ANALYZING MARKET CONDITIONS:")
        self.market_conditions = self.analyze_market_conditions()
        for condition in self.market_conditions:
            print(f"   Market: {condition['market_trend']} | Volatility: {condition['volatility']}")
        
        print(f"\\n>> DATA COLLECTION COMPLETE:")
        print(f"   Total samples: {total_samples}")
        print(f"   Crypto samples: {len([s for s in self.training_samples if 'USD' in s['symbol']])}")
        print(f"   Stock samples: {len([s for s in self.training_samples if 'USD' not in s['symbol']])}")
        print(f"   Market conditions: {len(self.market_conditions)}")
        
        # Save training data
        with open('enhanced_training_data.json', 'w') as f:
            json.dump({
                'samples': self.training_samples,
                'conditions': self.market_conditions,
                'collection_time': datetime.now().isoformat(),
                'total_samples': total_samples
            }, f, indent=2)
        
        print(f"   Data saved to: enhanced_training_data.json")
        
        return {
            'total_samples': total_samples,
            'crypto_samples': len([s for s in self.training_samples if 'USD' in s['symbol']]),
            'stock_samples': len([s for s in self.training_samples if 'USD' not in s['symbol']]),
            'market_conditions': len(self.market_conditions)
        }

async def main():
    collector = EnhancedDataCollector()
    results = await collector.rapid_data_collection()
    
    print("\\n" + "="*60)
    print("ENHANCED TRAINING DATA READY FOR RL!")
    print(f"Your RL system now has {results['total_samples']} samples")
    print("This should dramatically improve training accuracy!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())