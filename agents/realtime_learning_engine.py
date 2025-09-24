#!/usr/bin/env python3
"""
Real-Time Learning Engine - Continuous Learning from Live Trading
Updates models in real-time based on trading outcomes
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import json
from collections import deque
import threading
import time

class RealTimeLearningEngine:
    """Continuously learn and adapt from live trading results"""
    
    def __init__(self):
        self.trade_history = deque(maxlen=10000)  # Keep last 10K trades
        self.learning_buffer = deque(maxlen=1000)  # Real-time learning buffer
        self.model_performance = {}
        self.adaptation_triggers = {
            'accuracy_drop': 0.05,  # Retrain if accuracy drops 5%
            'new_trades': 50,       # Retrain every 50 new trades
            'time_interval': 3600   # Retrain every hour
        }
        
        # Performance tracking
        self.baseline_performance = {}
        self.recent_performance = {}
        self.last_retrain = datetime.now()
        
        # Initialize database
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for trade tracking"""
        self.conn = sqlite3.connect('trading_data.db', check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                strategy TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity INTEGER,
                pnl REAL,
                entry_time TEXT,
                exit_time TEXT,
                predicted_direction TEXT,
                actual_direction TEXT,
                confidence REAL,
                market_regime TEXT,
                volatility_regime TEXT,
                model_version TEXT,
                features TEXT
            )
        ''')
        self.conn.commit()
    
    async def record_trade_entry(self, trade_data):
        """Record trade entry for learning"""
        trade_record = {
            'trade_id': f"{trade_data['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'symbol': trade_data['symbol'],
            'strategy': trade_data.get('strategy', 'UNKNOWN'),
            'entry_price': trade_data['entry_price'],
            'quantity': trade_data['quantity'],
            'entry_time': datetime.now(),
            'predicted_direction': trade_data.get('predicted_direction', 'HOLD'),
            'confidence': trade_data.get('confidence', 0.5),
            'market_regime': trade_data.get('market_regime', 'NEUTRAL'),
            'volatility_regime': trade_data.get('volatility_regime', 'NORMAL'),
            'features': json.dumps(trade_data.get('features', {})),
            'model_version': trade_data.get('model_version', '1.0')
        }
        
        self.learning_buffer.append(trade_record)
        
        # Store in database
        self.conn.execute('''
            INSERT INTO trades (symbol, strategy, entry_price, quantity, entry_time, 
                              predicted_direction, confidence, market_regime, volatility_regime, 
                              features, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_record['symbol'], trade_record['strategy'], trade_record['entry_price'],
            trade_record['quantity'], trade_record['entry_time'].isoformat(),
            trade_record['predicted_direction'], trade_record['confidence'],
            trade_record['market_regime'], trade_record['volatility_regime'],
            trade_record['features'], trade_record['model_version']
        ))
        self.conn.commit()
        
        print(f"Recorded trade entry: {trade_record['symbol']} {trade_record['predicted_direction']}")
    
    async def record_trade_exit(self, trade_id, exit_data):
        """Record trade exit and trigger learning"""
        exit_price = exit_data['exit_price']
        exit_time = datetime.now()
        
        # Update database
        self.conn.execute('''
            UPDATE trades SET exit_price = ?, exit_time = ?, pnl = ?
            WHERE id = (SELECT id FROM trades WHERE symbol || '_' || substr(entry_time, 1, 13) = ?)
        ''', (exit_price, exit_time.isoformat(), exit_data['pnl'], trade_id))
        self.conn.commit()
        
        # Calculate actual direction
        cursor = self.conn.execute('''
            SELECT entry_price, predicted_direction, confidence, features 
            FROM trades WHERE id = (SELECT id FROM trades WHERE symbol || '_' || substr(entry_time, 1, 13) = ?)
        ''', (trade_id,))
        
        result = cursor.fetchone()
        if result:
            entry_price, predicted_direction, confidence, features_json = result
            actual_direction = 'BUY' if exit_price > entry_price else 'SELL'
            
            # Update actual direction
            self.conn.execute('''
                UPDATE trades SET actual_direction = ? 
                WHERE id = (SELECT id FROM trades WHERE symbol || '_' || substr(entry_time, 1, 13) = ?)
            ''', (actual_direction, trade_id))
            self.conn.commit()
            
            # Add to learning buffer for real-time adaptation
            learning_sample = {
                'predicted': predicted_direction,
                'actual': actual_direction,
                'confidence': confidence,
                'features': json.loads(features_json),
                'correct': predicted_direction == actual_direction,
                'timestamp': exit_time
            }
            
            self.trade_history.append(learning_sample)
            
            print(f"Trade completed: {trade_id} - Predicted: {predicted_direction}, Actual: {actual_direction}")
            
            # Check if retraining is needed
            await self.check_retrain_triggers()
    
    async def check_retrain_triggers(self):
        """Check if model retraining should be triggered"""
        if len(self.trade_history) < 20:  # Need minimum trades
            return
        
        # Calculate recent accuracy
        recent_trades = list(self.trade_history)[-50:]  # Last 50 trades
        recent_accuracy = sum(1 for t in recent_trades if t['correct']) / len(recent_trades)
        
        # Calculate baseline accuracy
        if len(self.trade_history) >= 100:
            baseline_trades = list(self.trade_history)[-100:-50]  # Previous 50 trades
            baseline_accuracy = sum(1 for t in baseline_trades if t['correct']) / len(baseline_trades)
        else:
            baseline_accuracy = 0.5  # Assume random baseline
        
        # Trigger conditions
        accuracy_drop = baseline_accuracy - recent_accuracy > self.adaptation_triggers['accuracy_drop']
        new_trades_trigger = len(recent_trades) >= self.adaptation_triggers['new_trades']
        time_trigger = (datetime.now() - self.last_retrain).seconds > self.adaptation_triggers['time_interval']
        
        if accuracy_drop or (new_trades_trigger and time_trigger):
            print(f"Triggering model retrain - Recent accuracy: {recent_accuracy:.1%}, Baseline: {baseline_accuracy:.1%}")
            await self.retrain_models()
    
    async def retrain_models(self):
        """Retrain models based on recent performance"""
        try:
            print("Starting real-time model retraining...")
            
            # Get recent trade data
            cursor = self.conn.execute('''
                SELECT symbol, predicted_direction, actual_direction, confidence, features, market_regime
                FROM trades 
                WHERE exit_time IS NOT NULL 
                ORDER BY exit_time DESC 
                LIMIT 500
            ''')
            
            recent_data = cursor.fetchall()
            
            if len(recent_data) < 50:
                print("Insufficient data for retraining")
                return
            
            # Analyze performance by symbol and strategy
            symbol_performance = {}
            for row in recent_data:
                symbol, predicted, actual, confidence, features_json, market_regime = row
                
                if symbol not in symbol_performance:
                    symbol_performance[symbol] = {
                        'correct': 0, 'total': 0, 'avg_confidence': 0
                    }
                
                symbol_performance[symbol]['total'] += 1
                symbol_performance[symbol]['avg_confidence'] += confidence
                
                if predicted == actual:
                    symbol_performance[symbol]['correct'] += 1
            
            # Calculate accuracy by symbol
            for symbol in symbol_performance:
                perf = symbol_performance[symbol]
                perf['accuracy'] = perf['correct'] / perf['total']
                perf['avg_confidence'] /= perf['total']
                
                print(f"{symbol}: {perf['accuracy']:.1%} accuracy, {perf['avg_confidence']:.2f} avg confidence")
            
            # Identify underperforming models
            underperforming = [
                symbol for symbol, perf in symbol_performance.items() 
                if perf['accuracy'] < 0.5 and perf['total'] >= 10
            ]
            
            if underperforming:
                print(f"Underperforming models: {underperforming}")
                # Trigger model updates for these symbols
                await self.update_underperforming_models(underperforming)
            
            # Update adaptation parameters based on performance
            overall_accuracy = sum(p['correct'] for p in symbol_performance.values()) / sum(p['total'] for p in symbol_performance.values())
            
            if overall_accuracy < 0.55:
                # Increase sensitivity if overall performance is poor
                self.adaptation_triggers['accuracy_drop'] = 0.03  # More sensitive
                self.adaptation_triggers['new_trades'] = 30      # Retrain more frequently
            else:
                # Standard sensitivity if performance is good
                self.adaptation_triggers['accuracy_drop'] = 0.05
                self.adaptation_triggers['new_trades'] = 50
            
            self.last_retrain = datetime.now()
            print(f"Model retraining completed. Overall accuracy: {overall_accuracy:.1%}")
            
        except Exception as e:
            print(f"Error during model retraining: {e}")
    
    async def update_underperforming_models(self, symbols):
        """Update models for underperforming symbols"""
        for symbol in symbols:
            try:
                print(f"Updating model for underperforming symbol: {symbol}")
                
                # Get symbol-specific trade data
                cursor = self.conn.execute('''
                    SELECT features, predicted_direction, actual_direction, market_regime, volatility_regime
                    FROM trades 
                    WHERE symbol = ? AND exit_time IS NOT NULL 
                    ORDER BY exit_time DESC 
                    LIMIT 200
                ''', (symbol,))
                
                symbol_data = cursor.fetchall()
                
                if len(symbol_data) >= 30:
                    # Extract features and create training data
                    features_list = []
                    labels = []
                    
                    for row in symbol_data:
                        features_json, predicted, actual, market_regime, vol_regime = row
                        features = json.loads(features_json)
                        
                        # Add market regime features
                        features['market_regime_encoded'] = hash(market_regime) % 10
                        features['vol_regime_encoded'] = hash(vol_regime) % 5
                        
                        # Convert to feature vector
                        feature_vector = [
                            features.get('price_change', 0),
                            features.get('volume_ratio', 1),
                            features.get('rsi', 50),
                            features.get('signal_strength', 50),
                            features.get('market_regime_encoded', 0),
                            features.get('vol_regime_encoded', 0)
                        ]
                        
                        features_list.append(feature_vector)
                        labels.append(1 if actual == 'BUY' else 0)
                    
                    # Simple online learning update
                    if len(features_list) >= 20:
                        X = np.array(features_list)
                        y = np.array(labels)
                        
                        # Calculate feature importance
                        feature_importance = self.calculate_feature_importance(X, y)
                        
                        print(f"Updated model for {symbol} - Feature importance: {feature_importance}")
                        
                        # Store updated parameters (in real implementation, update actual ML models)
                        self.model_performance[symbol] = {
                            'last_update': datetime.now(),
                            'training_samples': len(features_list),
                            'feature_importance': feature_importance
                        }
                
            except Exception as e:
                print(f"Error updating model for {symbol}: {e}")
    
    def calculate_feature_importance(self, X, y):
        """Calculate simple feature importance"""
        try:
            correlations = []
            for i in range(X.shape[1]):
                corr = np.corrcoef(X[:, i], y)[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
            
            # Normalize
            total = sum(correlations)
            if total > 0:
                return [c / total for c in correlations]
            else:
                return [1.0 / X.shape[1]] * X.shape[1]
                
        except:
            return [1.0 / X.shape[1]] * X.shape[1]
    
    async def get_learning_insights(self):
        """Get insights from learning process"""
        if len(self.trade_history) < 10:
            return "Insufficient data for insights"
        
        recent_trades = list(self.trade_history)[-100:]
        
        # Overall performance
        accuracy = sum(1 for t in recent_trades if t['correct']) / len(recent_trades)
        avg_confidence = sum(t['confidence'] for t in recent_trades) / len(recent_trades)
        
        # Performance by prediction type
        buy_trades = [t for t in recent_trades if t['predicted'] == 'BUY']
        sell_trades = [t for t in recent_trades if t['predicted'] == 'SELL']
        
        buy_accuracy = sum(1 for t in buy_trades if t['correct']) / len(buy_trades) if buy_trades else 0
        sell_accuracy = sum(1 for t in sell_trades if t['correct']) / len(sell_trades) if sell_trades else 0
        
        insights = f"""
Real-Time Learning Insights:
- Overall Accuracy: {accuracy:.1%}
- Average Confidence: {avg_confidence:.2f}
- BUY Signal Accuracy: {buy_accuracy:.1%} ({len(buy_trades)} trades)
- SELL Signal Accuracy: {sell_accuracy:.1%} ({len(sell_trades)} trades)
- Total Trades Analyzed: {len(recent_trades)}
- Last Retrain: {self.last_retrain.strftime('%Y-%m-%d %H:%M')}
"""
        
        return insights

# Global instance
realtime_learner = RealTimeLearningEngine()

async def start_realtime_learning():
    """Start the real-time learning engine"""
    print("Starting Real-Time Learning Engine...")
    
    # Background task for periodic learning
    async def learning_loop():
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await realtime_learner.check_retrain_triggers()
            except Exception as e:
                print(f"Learning loop error: {e}")
    
    # Start background learning
    asyncio.create_task(learning_loop())
    print("Real-Time Learning Engine active!")

if __name__ == "__main__":
    asyncio.run(start_realtime_learning())