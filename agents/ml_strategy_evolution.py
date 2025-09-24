#!/usr/bin/env python3
"""
Machine Learning & Genetic Algorithm Strategy Evolution for OPTIONS_BOT
Uses scikit-learn for ML models and DEAP for evolutionary algorithms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import yfinance as yf
from datetime import datetime, timedelta
import asyncio
import logging

# Machine Learning imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.svm import SVC
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    print("scikit-learn not available - ML features disabled")
    SKLEARN_AVAILABLE = False

# Genetic Algorithm imports
try:
    from deap import base, creator, tools, algorithms
    import random
    DEAP_AVAILABLE = True
except ImportError:
    print("DEAP not available - genetic algorithm features disabled")
    DEAP_AVAILABLE = False

class MLStrategyEvolution:
    """
    Advanced ML and genetic algorithm system for OPTIONS_BOT
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.ml_models = {}
        self.strategy_population = []
        
        # Initialize genetic algorithm components
        if DEAP_AVAILABLE:
            self.setup_genetic_algorithm()
    
    def setup_genetic_algorithm(self):
        """Setup DEAP genetic algorithm framework"""
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize profit
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # Strategy gene definitions
        self.toolbox.register("volatility_threshold", random.uniform, 0.15, 0.35)
        self.toolbox.register("momentum_threshold", random.uniform, 0.01, 0.05)
        self.toolbox.register("volume_threshold", random.uniform, 0.5, 2.0)
        self.toolbox.register("profit_target", random.uniform, 0.15, 0.50)
        self.toolbox.register("stop_loss", random.uniform, 0.10, 0.30)
        self.toolbox.register("days_to_expiry", random.randint, 14, 45)
        self.toolbox.register("delta_target", random.uniform, 0.20, 0.40)
        
        # Individual is a strategy with multiple parameters
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                             (self.toolbox.volatility_threshold,
                              self.toolbox.momentum_threshold,
                              self.toolbox.volume_threshold,
                              self.toolbox.profit_target,
                              self.toolbox.stop_loss,
                              self.toolbox.days_to_expiry,
                              self.toolbox.delta_target), n=1)
        
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_strategy_fitness)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    async def ml_enhanced_market_regime_detection(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Use ML to classify market regimes more accurately than simple VIX thresholds
        """
        if not SKLEARN_AVAILABLE:
            return {'regime': 'UNKNOWN', 'confidence': 0.5, 'model': 'fallback'}
        
        try:
            # Collect market data for ML features
            market_features = []
            
            for symbol in symbols[:5]:  # Use top 5 symbols for regime detection
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="3mo")  # 3 months of data
                
                if len(hist) >= 30:
                    # Calculate technical indicators
                    returns = hist['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)
                    momentum = (hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1) if len(hist) >= 20 else 0
                    volume_ratio = hist['Volume'].iloc[-5:].mean() / hist['Volume'].iloc[-20:-5].mean() if len(hist) >= 20 else 1
                    
                    market_features.append([volatility, momentum, volume_ratio])
            
            if len(market_features) < 3:
                return {'regime': 'INSUFFICIENT_DATA', 'confidence': 0.3, 'model': 'fallback'}
            
            # Use K-means clustering to identify market regimes
            features_array = np.array(market_features)
            features_scaled = self.scaler.fit_transform(features_array)
            
            # 4 clusters: BULL, BEAR, HIGH_VOL, LOW_VOL
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # Interpret clusters based on characteristics
            cluster_centers = self.scaler.inverse_transform(kmeans.cluster_centers_)
            
            regimes = []
            for center in cluster_centers:
                vol, momentum, volume = center
                
                if momentum > 0.02 and vol < 0.25:
                    regimes.append('BULL_MARKET')
                elif momentum < -0.02 and vol < 0.25:
                    regimes.append('BEAR_MARKET')
                elif vol > 0.30:
                    regimes.append('HIGH_VOLATILITY')
                else:
                    regimes.append('NEUTRAL_MARKET')
            
            # Current market classification
            current_features = features_scaled[-1:] if len(features_scaled) > 0 else features_scaled[0:1]
            current_cluster = kmeans.predict(current_features)[0]
            current_regime = regimes[current_cluster] if current_cluster < len(regimes) else 'NEUTRAL_MARKET'
            
            # Calculate confidence based on distance to cluster center
            distances = kmeans.transform(current_features)[0]
            min_distance = np.min(distances)
            confidence = max(0.5, 1.0 - min_distance / np.max(distances))
            
            return {
                'regime': current_regime,
                'confidence': confidence,
                'model': 'kmeans_clustering',
                'cluster_info': {
                    'current_cluster': current_cluster,
                    'all_regimes': regimes,
                    'distance_to_center': min_distance
                }
            }
            
        except Exception as e:
            self.logger.error(f"ML regime detection error: {e}")
            return {'regime': 'ERROR', 'confidence': 0.2, 'model': 'error_fallback'}
    
    async def ml_enhanced_volatility_prediction(self, symbol: str, days_ahead: int = 30) -> Dict[str, float]:
        """
        Use Random Forest to predict volatility more accurately
        """
        if not SKLEARN_AVAILABLE:
            return {'predicted_vol': 25.0, 'confidence': 0.5, 'model': 'fallback'}
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")  # 1 year for training
            
            if len(hist) < 100:
                return {'predicted_vol': 25.0, 'confidence': 0.3, 'model': 'insufficient_data'}
            
            # Prepare features for ML model
            returns = hist['Close'].pct_change().dropna()
            
            # Create features: rolling volatilities, momentum, volume indicators
            features_list = []
            targets_list = []
            
            lookback = 20  # 20-day features
            forecast_days = min(days_ahead, 10)  # Predict up to 10 days ahead reliably
            
            for i in range(lookback, len(returns) - forecast_days):
                # Features: various technical indicators
                recent_returns = returns.iloc[i-lookback:i]
                recent_prices = hist['Close'].iloc[i-lookback:i]
                recent_volumes = hist['Volume'].iloc[i-lookback:i]
                
                vol_5day = recent_returns.iloc[-5:].std() * np.sqrt(252)
                vol_10day = recent_returns.iloc[-10:].std() * np.sqrt(252)
                vol_20day = recent_returns.iloc[-20:].std() * np.sqrt(252)
                momentum = (recent_prices.iloc[-1] / recent_prices.iloc[0] - 1)
                volume_trend = (recent_volumes.iloc[-5:].mean() / recent_volumes.iloc[-10:-5].mean()) if len(recent_volumes) >= 10 else 1
                
                features_list.append([vol_5day, vol_10day, vol_20day, momentum, volume_trend])
                
                # Target: future realized volatility
                future_returns = returns.iloc[i:i+forecast_days]
                future_vol = future_returns.std() * np.sqrt(252) if len(future_returns) > 5 else vol_20day
                targets_list.append(future_vol)
            
            if len(features_list) < 50:
                return {'predicted_vol': 25.0, 'confidence': 0.4, 'model': 'insufficient_training_data'}
            
            # Train Random Forest model
            X = np.array(features_list)
            y = np.array(targets_list)
            
            X_scaled = self.scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            rf_model = GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Calculate model performance
            y_pred = rf_model.predict(X_test)
            mse = np.mean((y_test - y_pred) ** 2)
            confidence = max(0.3, 1.0 - mse / np.var(y_test))
            
            # Make prediction for current data
            current_returns = returns.iloc[-lookback:]
            current_prices = hist['Close'].iloc[-lookback:]
            current_volumes = hist['Volume'].iloc[-lookback:]
            
            current_vol_5 = current_returns.iloc[-5:].std() * np.sqrt(252)
            current_vol_10 = current_returns.iloc[-10:].std() * np.sqrt(252)
            current_vol_20 = current_returns.std() * np.sqrt(252)
            current_momentum = (current_prices.iloc[-1] / current_prices.iloc[0] - 1)
            current_volume_trend = (current_volumes.iloc[-5:].mean() / current_volumes.iloc[-10:-5].mean()) if len(current_volumes) >= 10 else 1
            
            current_features = np.array([[current_vol_5, current_vol_10, current_vol_20, 
                                        current_momentum, current_volume_trend]])
            current_features_scaled = self.scaler.transform(current_features)
            
            predicted_vol = rf_model.predict(current_features_scaled)[0]
            
            return {
                'predicted_vol': predicted_vol * 100,  # Convert to percentage
                'confidence': confidence,
                'model': 'random_forest',
                'current_vol': current_vol_20 * 100,
                'mse': mse,
                'feature_importance': rf_model.feature_importances_.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"ML volatility prediction error for {symbol}: {e}")
            return {'predicted_vol': 25.0, 'confidence': 0.3, 'model': 'error_fallback'}
    
    def evaluate_strategy_fitness(self, individual) -> Tuple[float]:
        """
        Evaluate the fitness of a strategy individual (for genetic algorithm)
        """
        if not individual or len(individual) < 7:
            return (0.0,)
        
        try:
            vol_thresh, mom_thresh, vol_ratio_thresh, profit_target, stop_loss, dte, delta_target = individual
            
            # Simulate strategy performance (simplified)
            # In real implementation, this would backtest the strategy
            
            # Reward balanced parameters
            fitness = 0.0
            
            # Volatility threshold (15-35% range is good)
            if 0.15 <= vol_thresh <= 0.35:
                fitness += 0.2
            
            # Momentum threshold (1-5% range is good)
            if 0.01 <= mom_thresh <= 0.05:
                fitness += 0.2
            
            # Volume ratio (0.5-2.0 range is good)
            if 0.5 <= vol_ratio_thresh <= 2.0:
                fitness += 0.1
            
            # Profit target (15-50% range)
            if 0.15 <= profit_target <= 0.50:
                fitness += 0.2
                
            # Stop loss (10-30% range)
            if 0.10 <= stop_loss <= 0.30:
                fitness += 0.1
            
            # Days to expiry (14-45 days)
            if 14 <= dte <= 45:
                fitness += 0.1
                
            # Delta target (20-40% range)
            if 0.20 <= delta_target <= 0.40:
                fitness += 0.1
            
            # Bonus for risk-reward ratio
            risk_reward_ratio = profit_target / stop_loss if stop_loss > 0 else 0
            if 1.2 <= risk_reward_ratio <= 3.0:
                fitness += 0.2
                
            # Penalty for extreme values
            extreme_penalty = 0
            if vol_thresh < 0.10 or vol_thresh > 0.50:
                extreme_penalty += 0.1
            if mom_thresh < 0.005 or mom_thresh > 0.10:
                extreme_penalty += 0.1
                
            fitness -= extreme_penalty
            
            return (max(0.0, fitness),)
            
        except Exception as e:
            return (0.0,)
    
    async def evolve_optimal_strategy(self, generations: int = 50, population_size: int = 100) -> Dict[str, Any]:
        """
        Use genetic algorithm to evolve optimal trading strategy parameters
        """
        if not DEAP_AVAILABLE:
            return {'error': 'DEAP not available', 'best_strategy': None}
        
        try:
            # Create initial population
            population = self.toolbox.population(n=population_size)
            
            # Statistics tracking
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            # Hall of fame to track best individuals
            hof = tools.HallOfFame(10)
            
            # Evolution parameters
            cxpb, mutpb = 0.7, 0.3  # Crossover and mutation probabilities
            
            # Run genetic algorithm
            population, logbook = algorithms.eaSimple(
                population, self.toolbox, cxpb, mutpb, generations,
                stats=stats, halloffame=hof, verbose=False
            )
            
            # Extract best strategy
            best_individual = hof[0] if hof else population[0]
            best_fitness = best_individual.fitness.values[0]
            
            # Convert to readable strategy
            vol_thresh, mom_thresh, vol_ratio_thresh, profit_target, stop_loss, dte, delta_target = best_individual
            
            evolved_strategy = {
                'volatility_threshold': vol_thresh,
                'momentum_threshold': mom_thresh, 
                'volume_ratio_threshold': vol_ratio_thresh,
                'profit_target': profit_target,
                'stop_loss': stop_loss,
                'days_to_expiry': int(dte),
                'delta_target': delta_target,
                'fitness_score': best_fitness,
                'risk_reward_ratio': profit_target / stop_loss if stop_loss > 0 else 0
            }
            
            return {
                'success': True,
                'best_strategy': evolved_strategy,
                'generations_run': generations,
                'population_size': population_size,
                'final_fitness': best_fitness,
                'evolution_stats': {
                    'final_avg': logbook[-1]['avg'],
                    'final_max': logbook[-1]['max'],
                    'improvement': logbook[-1]['max'] - logbook[0]['max'] if len(logbook) > 0 else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Strategy evolution error: {e}")
            return {'error': str(e), 'best_strategy': None}
    
    async def ml_opportunity_scoring(self, symbol: str, current_vol: float, 
                                   momentum: float, volume_ratio: float) -> Dict[str, float]:
        """
        Use ML to score trading opportunities more accurately
        """
        if not SKLEARN_AVAILABLE:
            return {'score': 0.5, 'confidence': 0.5, 'model': 'fallback'}
        
        try:
            # Create feature vector
            features = np.array([[current_vol, momentum, volume_ratio]])
            
            # Simple scoring based on ML principles (in real implementation, 
            # this would use a trained model with historical success data)
            
            # Normalize features
            vol_score = 1.0 - abs(current_vol - 0.25) / 0.25  # Prefer moderate volatility
            momentum_score = min(1.0, abs(momentum) / 0.03)    # Prefer some momentum
            volume_score = min(1.0, volume_ratio / 1.5)        # Prefer higher volume
            
            # Weighted combination
            composite_score = (vol_score * 0.4 + momentum_score * 0.4 + volume_score * 0.2)
            
            # Confidence based on feature stability
            confidence = min(1.0, (vol_score + momentum_score + volume_score) / 3.0)
            
            return {
                'score': composite_score,
                'confidence': confidence,
                'model': 'ml_composite_scoring',
                'feature_scores': {
                    'volatility': vol_score,
                    'momentum': momentum_score,
                    'volume': volume_score
                }
            }
            
        except Exception as e:
            return {'score': 0.3, 'confidence': 0.3, 'model': 'error_fallback'}

# Global instance
ml_strategy_evolution = MLStrategyEvolution()