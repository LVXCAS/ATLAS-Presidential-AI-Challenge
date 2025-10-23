#!/usr/bin/env python3
"""
Model Loader - Loads pre-trained models for acceleration
"""

import joblib
import os
import json
import numpy as np
from datetime import datetime

class ModelLoader:
    """Load and use pre-trained models"""
    
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load all available pre-trained models"""
        models_dir = 'models/working'
        
        if not os.path.exists(models_dir):
            print("No pre-trained models found")
            return
        
        loaded_count = 0
        for filename in os.listdir(models_dir):
            if filename.endswith('_model.pkl'):
                symbol = filename.replace('_model.pkl', '')
                try:
                    model_path = os.path.join(models_dir, filename)
                    model_data = joblib.load(model_path)
                    
                    self.models[symbol] = model_data
                    loaded_count += 1
                    
                    print(f"Loaded model for {symbol}: {model_data['test_accuracy']:.1%} accuracy")
                    
                except Exception as e:
                    print(f"Error loading model for {symbol}: {e}")
        
        print(f"Total models loaded: {loaded_count}")
        
        # Load summary if available
        summary_path = os.path.join(models_dir, 'training_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                self.training_summary = json.load(f)
        else:
            self.training_summary = {}
    
    def get_prediction(self, symbol, features):
        """Get prediction from pre-trained model"""
        try:
            if symbol not in self.models:
                return None
            
            model_data = self.models[symbol]
            model = model_data['model']
            scaler = model_data['scaler']
            
            # Scale features
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Get prediction
            prediction_proba = model.predict_proba(features_scaled)[0]
            prediction = model.predict(features_scaled)[0]
            
            # Get confidence (distance from 0.5)
            confidence = abs(prediction_proba[1] - 0.5) * 2
            
            return {
                'prediction': int(prediction),
                'probability': float(prediction_proba[1]),
                'confidence': float(confidence),
                'signal': 'BUY' if prediction == 1 else 'SELL',
                'model_accuracy': model_data['test_accuracy'],
                'symbol': symbol
            }
            
        except Exception as e:
            print(f"Prediction error for {symbol}: {e}")
            return None
    
    def get_best_models(self, top_n=5):
        """Get best performing models"""
        if not self.models:
            return []
        
        model_list = []
        for symbol, model_data in self.models.items():
            model_list.append({
                'symbol': symbol,
                'accuracy': model_data['test_accuracy'],
                'samples': model_data['training_samples']
            })
        
        # Sort by accuracy
        model_list.sort(key=lambda x: x['accuracy'], reverse=True)
        
        return model_list[:top_n]
    
    def get_model_info(self):
        """Get information about loaded models"""
        if not self.models:
            return "No pre-trained models available"
        
        info = f"Pre-trained Models Available: {len(self.models)}\n"
        
        for symbol, model_data in self.models.items():
            info += f"  {symbol}: {model_data['test_accuracy']:.1%} accuracy ({model_data['training_samples']} samples)\n"
        
        if self.training_summary:
            avg_acc = self.training_summary.get('average_accuracy', 0)
            info += f"\nAverage Accuracy: {avg_acc:.1%}"
            info += f"\nTotal Data Points: {self.training_summary.get('total_data_points', 0):,}"
        
        return info
    
    def create_features_from_data(self, data):
        """Create features compatible with pre-trained models"""
        try:
            # This should match the features used in training
            features = []
            
            # Returns
            features.append(data.get('return_1d', 0))
            features.append(data.get('return_3d', 0))  
            features.append(data.get('return_5d', 0))
            
            # Price ratios
            features.append(data.get('price_sma5_ratio', 1))
            features.append(data.get('price_sma20_ratio', 1))
            features.append(data.get('sma5_sma20_ratio', 1))
            
            # Other features
            features.append(data.get('volatility', 0.01))
            features.append(data.get('volume_ratio', 1))
            features.append(data.get('hl_range', 0.01))
            features.append(data.get('rsi', 50))
            
            return np.array(features)
            
        except Exception as e:
            print(f"Feature creation error: {e}")
            return None

# Global instance
model_loader = ModelLoader()

def get_accelerated_prediction(symbol, market_data):
    """Get prediction using pre-trained models"""
    try:
        features = model_loader.create_features_from_data(market_data)
        if features is not None:
            return model_loader.get_prediction(symbol, features)
        return None
    except Exception as e:
        print(f"Accelerated prediction error: {e}")
        return None

if __name__ == "__main__":
    loader = ModelLoader()
    print(loader.get_model_info())