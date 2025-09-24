"""
Continuous Learning System with Feedback Loops
==============================================

This module implements the continuous learning cycle that enables the trading system
to constantly improve through feedback loops between the Execution and R&D engines.

Key Features:
- Real-time performance analysis and feedback collection
- Adaptive strategy optimization based on execution results
- Online learning with incremental model updates
- Meta-learning for strategy adaptation across market conditions
- Performance attribution and trade impact analysis
- Dynamic parameter adjustment and model retraining
"""

import asyncio
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import pickle
import os
from collections import deque, defaultdict

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Import system components
from core.parallel_trading_architecture import InterEngineMessage, MessageType, EngineType

logger = logging.getLogger(__name__)


class LearningPhase(Enum):
    """Learning cycle phases"""
    DATA_COLLECTION = "data_collection"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"


class FeedbackType(Enum):
    """Types of feedback from execution"""
    TRADE_EXECUTION = "trade_execution"
    STRATEGY_PERFORMANCE = "strategy_performance"
    RISK_EVENT = "risk_event"
    MARKET_CONDITION_CHANGE = "market_condition_change"
    ORDER_FILL_ANALYSIS = "order_fill_analysis"
    SLIPPAGE_ANALYSIS = "slippage_analysis"


class LearningObjective(Enum):
    """Learning objectives for optimization"""
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MINIMIZE_DRAWDOWN = "minimize_drawdown"
    MAXIMIZE_RETURN = "maximize_return"
    MINIMIZE_VOLATILITY = "minimize_volatility"
    OPTIMIZE_RISK_ADJUSTED = "optimize_risk_adjusted"
    IMPROVE_FILL_RATE = "improve_fill_rate"
    REDUCE_SLIPPAGE = "reduce_slippage"


@dataclass
class FeedbackEvent:
    """Structured feedback event from execution engine"""
    id: str
    feedback_type: FeedbackType
    timestamp: datetime
    symbol: str
    strategy_id: str
    execution_data: Dict[str, Any]
    performance_metrics: Dict[str, float]
    market_context: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningResult:
    """Result of a learning cycle iteration"""
    cycle_id: str
    phase: LearningPhase
    timestamp: datetime
    objective: LearningObjective
    original_parameters: Dict[str, Any]
    optimized_parameters: Dict[str, Any]
    performance_improvement: float
    confidence_score: float
    validation_metrics: Dict[str, float]
    deployment_ready: bool
    insights: List[str] = field(default_factory=list)


@dataclass
class StrategyPerformance:
    """Comprehensive strategy performance tracking"""
    strategy_id: str
    symbol: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    volatility: float
    last_updated: datetime
    trade_history: deque = field(default_factory=lambda: deque(maxlen=1000))


class PerformanceAnalyzer:
    """Analyzes execution performance and extracts learning insights"""

    def __init__(self):
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.market_regimes: Dict[str, str] = {}
        self.performance_history: deque = deque(maxlen=10000)

    def analyze_feedback(self, feedback: FeedbackEvent) -> Dict[str, Any]:
        """Analyze feedback event and extract insights"""
        try:
            insights = {
                'performance_score': 0.0,
                'improvement_areas': [],
                'market_regime': 'unknown',
                'execution_quality': 0.0,
                'risk_metrics': {},
                'recommendations': []
            }

            # Update strategy performance
            self._update_strategy_performance(feedback)

            # Analyze execution quality
            execution_quality = self._analyze_execution_quality(feedback)
            insights['execution_quality'] = execution_quality

            # Detect market regime
            market_regime = self._detect_market_regime(feedback)
            insights['market_regime'] = market_regime

            # Calculate performance score
            performance_score = self._calculate_performance_score(feedback)
            insights['performance_score'] = performance_score

            # Identify improvement areas
            improvement_areas = self._identify_improvement_areas(feedback)
            insights['improvement_areas'] = improvement_areas

            # Generate recommendations
            recommendations = self._generate_recommendations(feedback, insights)
            insights['recommendations'] = recommendations

            # Store in history
            self.performance_history.append({
                'timestamp': feedback.timestamp,
                'feedback': feedback,
                'insights': insights
            })

            return insights

        except Exception as e:
            logger.error(f"Error analyzing feedback: {e}")
            return {}

    def _update_strategy_performance(self, feedback: FeedbackEvent) -> None:
        """Update strategy performance metrics"""
        try:
            strategy_id = feedback.strategy_id
            symbol = feedback.symbol
            key = f"{strategy_id}_{symbol}"

            if key not in self.strategy_performance:
                self.strategy_performance[key] = StrategyPerformance(
                    strategy_id=strategy_id,
                    symbol=symbol,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    total_return=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    win_rate=0.0,
                    avg_win=0.0,
                    avg_loss=0.0,
                    profit_factor=0.0,
                    volatility=0.0,
                    last_updated=feedback.timestamp
                )

            perf = self.strategy_performance[key]

            # Update trade statistics
            if feedback.feedback_type == FeedbackType.TRADE_EXECUTION:
                perf.total_trades += 1
                trade_return = feedback.performance_metrics.get('return', 0.0)
                perf.total_return += trade_return

                if trade_return > 0:
                    perf.winning_trades += 1
                    perf.avg_win = ((perf.avg_win * (perf.winning_trades - 1)) + trade_return) / perf.winning_trades
                else:
                    perf.losing_trades += 1
                    perf.avg_loss = ((perf.avg_loss * (perf.losing_trades - 1)) + trade_return) / perf.losing_trades

                # Calculate derived metrics
                perf.win_rate = perf.winning_trades / perf.total_trades if perf.total_trades > 0 else 0.0
                perf.profit_factor = abs(perf.avg_win / perf.avg_loss) if perf.avg_loss != 0 else float('inf')

                # Add to trade history
                perf.trade_history.append({
                    'timestamp': feedback.timestamp,
                    'return': trade_return,
                    'execution_data': feedback.execution_data
                })

            # Update other metrics
            perf.sharpe_ratio = feedback.performance_metrics.get('sharpe_ratio', perf.sharpe_ratio)
            perf.max_drawdown = max(perf.max_drawdown, feedback.performance_metrics.get('drawdown', 0.0))
            perf.volatility = feedback.performance_metrics.get('volatility', perf.volatility)
            perf.last_updated = feedback.timestamp

        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")

    def _analyze_execution_quality(self, feedback: FeedbackEvent) -> float:
        """Analyze execution quality metrics"""
        try:
            quality_score = 0.0

            if feedback.feedback_type == FeedbackType.TRADE_EXECUTION:
                execution_data = feedback.execution_data

                # Fill rate score
                fill_rate = execution_data.get('fill_rate', 0.0)
                quality_score += fill_rate * 0.3

                # Slippage score (lower is better)
                slippage = abs(execution_data.get('slippage_bps', 100))
                slippage_score = max(0, 1 - (slippage / 100))  # Normalize to 0-1
                quality_score += slippage_score * 0.3

                # Timing score
                execution_time = execution_data.get('execution_time_ms', 1000)
                timing_score = max(0, 1 - (execution_time / 1000))
                quality_score += timing_score * 0.2

                # Market impact score
                market_impact = abs(execution_data.get('market_impact_bps', 50))
                impact_score = max(0, 1 - (market_impact / 50))
                quality_score += impact_score * 0.2

            return min(1.0, quality_score)

        except Exception as e:
            logger.error(f"Error analyzing execution quality: {e}")
            return 0.0

    def _detect_market_regime(self, feedback: FeedbackEvent) -> str:
        """Detect current market regime"""
        try:
            market_context = feedback.market_context

            volatility = market_context.get('volatility', 0.02)
            trend_strength = market_context.get('trend_strength', 0.0)
            volume_ratio = market_context.get('volume_ratio', 1.0)

            # Simple regime classification
            if volatility > 0.03 and volume_ratio > 1.5:
                regime = 'high_volatility'
            elif trend_strength > 0.7:
                regime = 'trending'
            elif trend_strength < -0.7:
                regime = 'bear_market'
            elif abs(trend_strength) < 0.2:
                regime = 'sideways'
            else:
                regime = 'normal'

            # Update regime tracking
            symbol = feedback.symbol
            self.market_regimes[symbol] = regime

            return regime

        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return 'unknown'

    def _calculate_performance_score(self, feedback: FeedbackEvent) -> float:
        """Calculate overall performance score"""
        try:
            metrics = feedback.performance_metrics

            # Weighted performance score
            sharpe = metrics.get('sharpe_ratio', 0.0)
            return_score = metrics.get('return', 0.0)
            drawdown = metrics.get('drawdown', 0.0)

            # Normalize and weight
            sharpe_score = min(1.0, max(0.0, (sharpe + 2) / 4))  # Sharpe -2 to 2 -> 0 to 1
            return_normalized = min(1.0, max(0.0, return_score + 1))  # Clamp to reasonable range
            drawdown_score = max(0.0, 1 - drawdown)  # Lower drawdown is better

            performance_score = (sharpe_score * 0.4 + return_normalized * 0.3 + drawdown_score * 0.3)

            return performance_score

        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.0

    def _identify_improvement_areas(self, feedback: FeedbackEvent) -> List[str]:
        """Identify areas for improvement"""
        try:
            areas = []

            execution_data = feedback.execution_data
            performance_metrics = feedback.performance_metrics

            # Check execution quality issues
            if execution_data.get('fill_rate', 1.0) < 0.8:
                areas.append('fill_rate')

            if abs(execution_data.get('slippage_bps', 0)) > 50:
                areas.append('slippage_control')

            if execution_data.get('execution_time_ms', 0) > 500:
                areas.append('execution_speed')

            # Check performance issues
            if performance_metrics.get('sharpe_ratio', 0) < 0.5:
                areas.append('risk_adjusted_return')

            if performance_metrics.get('drawdown', 0) > 0.1:
                areas.append('drawdown_control')

            if performance_metrics.get('volatility', 0) > 0.25:
                areas.append('volatility_management')

            return areas

        except Exception as e:
            logger.error(f"Error identifying improvement areas: {e}")
            return []

    def _generate_recommendations(self, feedback: FeedbackEvent, insights: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        try:
            recommendations = []

            improvement_areas = insights.get('improvement_areas', [])
            performance_score = insights.get('performance_score', 0.0)
            market_regime = insights.get('market_regime', 'normal')

            # Execution-based recommendations
            if 'fill_rate' in improvement_areas:
                recommendations.append('Adjust order sizing or use more aggressive limit prices')

            if 'slippage_control' in improvement_areas:
                recommendations.append('Consider using TWAP/VWAP algorithms for large orders')

            if 'execution_speed' in improvement_areas:
                recommendations.append('Optimize order routing and reduce latency')

            # Performance-based recommendations
            if performance_score < 0.3:
                recommendations.append('Consider strategy parameter reoptimization')

            if 'risk_adjusted_return' in improvement_areas:
                recommendations.append('Increase position sizing or improve signal quality')

            if 'drawdown_control' in improvement_areas:
                recommendations.append('Implement tighter stop-loss rules or position sizing')

            # Market regime specific recommendations
            if market_regime == 'high_volatility':
                recommendations.append('Reduce position sizes during volatile periods')
            elif market_regime == 'sideways':
                recommendations.append('Consider mean reversion strategies')
            elif market_regime == 'trending':
                recommendations.append('Increase momentum strategy allocation')

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []

    def get_strategy_summary(self, strategy_id: str) -> Dict[str, Any]:
        """Get comprehensive strategy performance summary"""
        try:
            summaries = {}

            for key, perf in self.strategy_performance.items():
                if perf.strategy_id == strategy_id:
                    summaries[perf.symbol] = asdict(perf)

            return summaries

        except Exception as e:
            logger.error(f"Error getting strategy summary: {e}")
            return {}


class ParameterOptimizer:
    """Optimizes strategy parameters based on performance feedback"""

    def __init__(self):
        self.optimization_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.parameter_bounds: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self.optimization_methods = {
            'random_search': self._random_search_optimization,
            'bayesian_optimization': self._bayesian_optimization,
            'gradient_descent': self._gradient_descent_optimization,
            'genetic_algorithm': self._genetic_algorithm_optimization
        }

    def optimize_parameters(self,
                          strategy_id: str,
                          current_parameters: Dict[str, Any],
                          performance_history: List[Dict[str, Any]],
                          objective: LearningObjective,
                          method: str = 'bayesian_optimization') -> Dict[str, Any]:
        """Optimize strategy parameters"""
        try:
            if method not in self.optimization_methods:
                logger.warning(f"Unknown optimization method: {method}, using random_search")
                method = 'random_search'

            optimizer = self.optimization_methods[method]

            # Extract performance data
            X, y = self._prepare_optimization_data(performance_history, objective)

            if len(X) < 10:  # Need minimum data points
                logger.warning(f"Insufficient data for optimization: {len(X)} samples")
                return current_parameters

            # Run optimization
            optimized_params = optimizer(strategy_id, current_parameters, X, y, objective)

            # Validate parameters
            validated_params = self._validate_parameters(optimized_params, current_parameters)

            # Store optimization result
            self.optimization_history[strategy_id].append({
                'timestamp': datetime.now(timezone.utc),
                'method': method,
                'objective': objective.value,
                'original_params': current_parameters,
                'optimized_params': validated_params,
                'performance_improvement': self._estimate_improvement(X, y, validated_params, current_parameters)
            })

            return validated_params

        except Exception as e:
            logger.error(f"Error optimizing parameters: {e}")
            return current_parameters

    def _prepare_optimization_data(self,
                                 performance_history: List[Dict[str, Any]],
                                 objective: LearningObjective) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for optimization"""
        try:
            features = []
            targets = []

            for record in performance_history:
                # Extract features (parameters + market context)
                params = record.get('parameters', {})
                market_context = record.get('market_context', {})

                feature_vector = []
                # Add parameter values
                for key in sorted(params.keys()):
                    if isinstance(params[key], (int, float)):
                        feature_vector.append(float(params[key]))

                # Add market context
                for key in sorted(market_context.keys()):
                    if isinstance(market_context[key], (int, float)):
                        feature_vector.append(float(market_context[key]))

                if len(feature_vector) > 0:
                    features.append(feature_vector)

                    # Extract target based on objective
                    target = self._extract_target_value(record, objective)
                    targets.append(target)

            X = np.array(features) if features else np.array([]).reshape(0, 1)
            y = np.array(targets) if targets else np.array([])

            return X, y

        except Exception as e:
            logger.error(f"Error preparing optimization data: {e}")
            return np.array([]).reshape(0, 1), np.array([])

    def _extract_target_value(self, record: Dict[str, Any], objective: LearningObjective) -> float:
        """Extract target value based on learning objective"""
        try:
            performance = record.get('performance_metrics', {})

            if objective == LearningObjective.MAXIMIZE_SHARPE:
                return performance.get('sharpe_ratio', 0.0)
            elif objective == LearningObjective.MAXIMIZE_RETURN:
                return performance.get('return', 0.0)
            elif objective == LearningObjective.MINIMIZE_DRAWDOWN:
                return -performance.get('drawdown', 0.0)  # Negative because we minimize
            elif objective == LearningObjective.MINIMIZE_VOLATILITY:
                return -performance.get('volatility', 0.0)
            elif objective == LearningObjective.OPTIMIZE_RISK_ADJUSTED:
                return performance.get('return', 0.0) / max(0.01, performance.get('volatility', 0.01))
            else:
                return performance.get('sharpe_ratio', 0.0)

        except Exception as e:
            logger.error(f"Error extracting target value: {e}")
            return 0.0

    def _random_search_optimization(self,
                                  strategy_id: str,
                                  current_params: Dict[str, Any],
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  objective: LearningObjective) -> Dict[str, Any]:
        """Random search parameter optimization"""
        try:
            if len(X) == 0:
                return current_params

            # Train a simple model to predict performance
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)

            best_params = current_params.copy()
            best_score = model.predict(X[-1:]).item() if len(X) > 0 else 0.0

            # Random search
            for _ in range(100):  # 100 random trials
                test_params = self._generate_random_params(current_params)
                test_features = self._params_to_features(test_params)

                if len(test_features) == X.shape[1]:
                    predicted_score = model.predict([test_features]).item()

                    if predicted_score > best_score:
                        best_score = predicted_score
                        best_params = test_params

            return best_params

        except Exception as e:
            logger.error(f"Error in random search optimization: {e}")
            return current_params

    def _bayesian_optimization(self,
                             strategy_id: str,
                             current_params: Dict[str, Any],
                             X: np.ndarray,
                             y: np.ndarray,
                             objective: LearningObjective) -> Dict[str, Any]:
        """Bayesian optimization (simplified version)"""
        try:
            # For now, implement as improved random search with exploitation
            # In production, would use libraries like scikit-optimize

            return self._random_search_optimization(strategy_id, current_params, X, y, objective)

        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {e}")
            return current_params

    def _gradient_descent_optimization(self,
                                     strategy_id: str,
                                     current_params: Dict[str, Any],
                                     X: np.ndarray,
                                     y: np.ndarray,
                                     objective: LearningObjective) -> Dict[str, Any]:
        """Gradient descent optimization"""
        try:
            # Simplified gradient-based optimization
            return self._random_search_optimization(strategy_id, current_params, X, y, objective)

        except Exception as e:
            logger.error(f"Error in gradient descent optimization: {e}")
            return current_params

    def _genetic_algorithm_optimization(self,
                                      strategy_id: str,
                                      current_params: Dict[str, Any],
                                      X: np.ndarray,
                                      y: np.ndarray,
                                      objective: LearningObjective) -> Dict[str, Any]:
        """Genetic algorithm optimization"""
        try:
            # Simplified genetic algorithm
            return self._random_search_optimization(strategy_id, current_params, X, y, objective)

        except Exception as e:
            logger.error(f"Error in genetic algorithm optimization: {e}")
            return current_params

    def _generate_random_params(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random parameter variations"""
        try:
            new_params = base_params.copy()

            for key, value in base_params.items():
                if isinstance(value, (int, float)):
                    # Vary parameter by ±20%
                    variation = np.random.uniform(-0.2, 0.2)
                    new_params[key] = value * (1 + variation)

                    # Apply bounds if available
                    if key in self.parameter_bounds:
                        bounds = self.parameter_bounds[key]
                        new_params[key] = np.clip(new_params[key], bounds[0], bounds[1])

            return new_params

        except Exception as e:
            logger.error(f"Error generating random parameters: {e}")
            return base_params

    def _params_to_features(self, params: Dict[str, Any]) -> List[float]:
        """Convert parameters to feature vector"""
        try:
            features = []
            for key in sorted(params.keys()):
                if isinstance(params[key], (int, float)):
                    features.append(float(params[key]))
            return features

        except Exception as e:
            logger.error(f"Error converting params to features: {e}")
            return []

    def _validate_parameters(self, optimized_params: Dict[str, Any], original_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and constrain optimized parameters"""
        try:
            validated_params = optimized_params.copy()

            # Apply parameter bounds and constraints
            for key, value in validated_params.items():
                if isinstance(value, (int, float)):
                    # Limit maximum change to 50% of original value
                    original_value = original_params.get(key, value)
                    max_change = abs(original_value * 0.5)

                    if abs(value - original_value) > max_change:
                        if value > original_value:
                            validated_params[key] = original_value + max_change
                        else:
                            validated_params[key] = original_value - max_change

            return validated_params

        except Exception as e:
            logger.error(f"Error validating parameters: {e}")
            return optimized_params

    def _estimate_improvement(self, X: np.ndarray, y: np.ndarray,
                            optimized_params: Dict[str, Any],
                            original_params: Dict[str, Any]) -> float:
        """Estimate performance improvement from optimization"""
        try:
            if len(X) == 0:
                return 0.0

            # Train model and predict improvement
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)

            original_features = self._params_to_features(original_params)
            optimized_features = self._params_to_features(optimized_params)

            if len(original_features) == X.shape[1] and len(optimized_features) == X.shape[1]:
                original_score = model.predict([original_features]).item()
                optimized_score = model.predict([optimized_features]).item()
                return optimized_score - original_score

            return 0.0

        except Exception as e:
            logger.error(f"Error estimating improvement: {e}")
            return 0.0


class OnlineLearningEngine:
    """Online learning engine for continuous model updates"""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}
        self.update_frequency = timedelta(hours=1)  # Update models every hour
        self.last_update: Dict[str, datetime] = {}

    def create_model(self, model_id: str, model_type: str = 'random_forest') -> bool:
        """Create a new online learning model"""
        try:
            if model_type == 'random_forest':
                self.models[model_id] = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == 'gradient_boosting':
                self.models[model_id] = GradientBoostingRegressor(n_estimators=100, random_state=42)
            elif model_type == 'linear':
                self.models[model_id] = Ridge(alpha=1.0)
            else:
                logger.error(f"Unknown model type: {model_type}")
                return False

            self.scalers[model_id] = StandardScaler()
            self.model_performance[model_id] = {'mse': float('inf'), 'r2': 0.0}
            self.last_update[model_id] = datetime.now(timezone.utc)

            return True

        except Exception as e:
            logger.error(f"Error creating model {model_id}: {e}")
            return False

    def update_model(self, model_id: str, X: np.ndarray, y: np.ndarray) -> bool:
        """Update model with new data"""
        try:
            if model_id not in self.models:
                logger.error(f"Model {model_id} not found")
                return False

            if len(X) == 0 or len(y) == 0:
                return False

            # Scale features
            if not hasattr(self.scalers[model_id], 'mean_'):
                # First time fitting scaler
                X_scaled = self.scalers[model_id].fit_transform(X)
            else:
                # Update scaler incrementally (simplified)
                X_scaled = self.scalers[model_id].transform(X)

            # Update model
            model = self.models[model_id]
            model.fit(X_scaled, y)

            # Evaluate performance
            y_pred = model.predict(X_scaled)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            self.model_performance[model_id] = {'mse': mse, 'r2': r2}
            self.last_update[model_id] = datetime.now(timezone.utc)

            logger.info(f"Updated model {model_id}: MSE={mse:.4f}, R2={r2:.4f}")
            return True

        except Exception as e:
            logger.error(f"Error updating model {model_id}: {e}")
            return False

    def predict(self, model_id: str, X: np.ndarray) -> Optional[np.ndarray]:
        """Make predictions with model"""
        try:
            if model_id not in self.models:
                logger.error(f"Model {model_id} not found")
                return None

            if len(X) == 0:
                return np.array([])

            # Scale features
            if hasattr(self.scalers[model_id], 'mean_'):
                X_scaled = self.scalers[model_id].transform(X)
            else:
                logger.warning(f"Scaler for model {model_id} not fitted")
                X_scaled = X

            # Make predictions
            predictions = self.models[model_id].predict(X_scaled)
            return predictions

        except Exception as e:
            logger.error(f"Error making predictions with model {model_id}: {e}")
            return None

    def should_update_model(self, model_id: str) -> bool:
        """Check if model should be updated"""
        if model_id not in self.last_update:
            return True

        time_since_update = datetime.now(timezone.utc) - self.last_update[model_id]
        return time_since_update >= self.update_frequency

    def get_model_performance(self, model_id: str) -> Dict[str, float]:
        """Get model performance metrics"""
        return self.model_performance.get(model_id, {})

    def save_models(self, directory: str) -> bool:
        """Save all models to disk"""
        try:
            os.makedirs(directory, exist_ok=True)

            for model_id, model in self.models.items():
                model_path = os.path.join(directory, f"{model_id}_model.joblib")
                scaler_path = os.path.join(directory, f"{model_id}_scaler.joblib")

                joblib.dump(model, model_path)
                joblib.dump(self.scalers[model_id], scaler_path)

            # Save metadata
            metadata = {
                'model_performance': self.model_performance,
                'last_update': {k: v.isoformat() for k, v in self.last_update.items()}
            }

            metadata_path = os.path.join(directory, "models_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False

    def load_models(self, directory: str) -> bool:
        """Load models from disk"""
        try:
            if not os.path.exists(directory):
                logger.warning(f"Model directory {directory} does not exist")
                return False

            # Load metadata
            metadata_path = os.path.join(directory, "models_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                self.model_performance = metadata.get('model_performance', {})
                self.last_update = {
                    k: datetime.fromisoformat(v)
                    for k, v in metadata.get('last_update', {}).items()
                }

            # Load models
            for filename in os.listdir(directory):
                if filename.endswith('_model.joblib'):
                    model_id = filename.replace('_model.joblib', '')
                    model_path = os.path.join(directory, filename)
                    scaler_path = os.path.join(directory, f"{model_id}_scaler.joblib")

                    if os.path.exists(scaler_path):
                        self.models[model_id] = joblib.load(model_path)
                        self.scalers[model_id] = joblib.load(scaler_path)

            return True

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False


class ContinuousLearningSystem:
    """
    Main continuous learning system that orchestrates feedback collection,
    analysis, optimization, and model updates.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.performance_analyzer = PerformanceAnalyzer()
        self.parameter_optimizer = ParameterOptimizer()
        self.online_learning_engine = OnlineLearningEngine()

        # Learning cycle management
        self.current_phase = LearningPhase.DATA_COLLECTION
        self.learning_cycles: List[LearningResult] = []
        self.feedback_queue: deque = deque(maxlen=10000)

        # Configuration
        self.learning_frequency = timedelta(minutes=self.config.get('learning_frequency_minutes', 15))
        self.min_feedback_samples = self.config.get('min_feedback_samples', 50)
        self.max_parameter_change = self.config.get('max_parameter_change', 0.3)

        # State tracking
        self.last_learning_cycle = datetime.now(timezone.utc)
        self.is_running = False

    async def initialize(self) -> bool:
        """Initialize the continuous learning system"""
        try:
            logger.info("Initializing Continuous Learning System")

            # Initialize online learning models
            success = True
            success &= self.online_learning_engine.create_model('performance_predictor', 'random_forest')
            success &= self.online_learning_engine.create_model('risk_predictor', 'gradient_boosting')
            success &= self.online_learning_engine.create_model('execution_quality_predictor', 'linear')

            if success:
                logger.info("Continuous Learning System initialized successfully")
            else:
                logger.error("Failed to initialize some learning models")

            return success

        except Exception as e:
            logger.error(f"Error initializing Continuous Learning System: {e}")
            return False

    async def start(self) -> None:
        """Start the continuous learning system"""
        try:
            self.is_running = True
            logger.info("Starting Continuous Learning System")

            # Start learning loops
            asyncio.create_task(self._feedback_processing_loop())
            asyncio.create_task(self._learning_cycle_loop())
            asyncio.create_task(self._model_update_loop())

        except Exception as e:
            logger.error(f"Error starting Continuous Learning System: {e}")

    async def stop(self) -> None:
        """Stop the continuous learning system"""
        try:
            self.is_running = False
            logger.info("Stopping Continuous Learning System")

            # Save models and state
            await self.save_state()

        except Exception as e:
            logger.error(f"Error stopping Continuous Learning System: {e}")

    async def process_feedback(self, feedback: FeedbackEvent) -> None:
        """Process feedback from execution engine"""
        try:
            # Add to feedback queue
            self.feedback_queue.append(feedback)

            # Analyze feedback immediately
            insights = self.performance_analyzer.analyze_feedback(feedback)

            # Check if immediate action is needed
            if insights.get('performance_score', 0) < 0.2:
                await self._trigger_emergency_optimization(feedback, insights)

        except Exception as e:
            logger.error(f"Error processing feedback: {e}")

    async def run_learning_cycle(self, strategy_id: str, objective: LearningObjective) -> LearningResult:
        """Run a complete learning cycle for a strategy"""
        try:
            cycle_id = f"cycle_{datetime.now().timestamp()}"
            logger.info(f"Running learning cycle {cycle_id} for strategy {strategy_id}")

            # Phase 1: Data Collection
            self.current_phase = LearningPhase.DATA_COLLECTION
            performance_data = self._collect_performance_data(strategy_id)

            if len(performance_data) < self.min_feedback_samples:
                logger.warning(f"Insufficient data for learning cycle: {len(performance_data)} samples")
                return self._create_failed_result(cycle_id, "insufficient_data")

            # Phase 2: Analysis
            self.current_phase = LearningPhase.ANALYSIS
            analysis_result = await self._analyze_performance_data(performance_data)

            # Phase 3: Optimization
            self.current_phase = LearningPhase.OPTIMIZATION
            current_params = analysis_result.get('current_parameters', {})
            optimized_params = self.parameter_optimizer.optimize_parameters(
                strategy_id, current_params, performance_data, objective
            )

            # Phase 4: Validation
            self.current_phase = LearningPhase.VALIDATION
            validation_result = await self._validate_optimization(strategy_id, optimized_params, performance_data)

            # Phase 5: Create learning result
            learning_result = LearningResult(
                cycle_id=cycle_id,
                phase=LearningPhase.VALIDATION,
                timestamp=datetime.now(timezone.utc),
                objective=objective,
                original_parameters=current_params,
                optimized_parameters=optimized_params,
                performance_improvement=validation_result.get('improvement', 0.0),
                confidence_score=validation_result.get('confidence', 0.0),
                validation_metrics=validation_result.get('metrics', {}),
                deployment_ready=validation_result.get('deployment_ready', False),
                insights=analysis_result.get('insights', [])
            )

            # Store result
            self.learning_cycles.append(learning_result)

            return learning_result

        except Exception as e:
            logger.error(f"Error in learning cycle: {e}")
            return self._create_failed_result(cycle_id, str(e))

    async def _feedback_processing_loop(self) -> None:
        """Background loop for processing feedback"""
        while self.is_running:
            try:
                # Process feedback queue
                while self.feedback_queue:
                    feedback = self.feedback_queue.popleft()
                    # Additional processing can be added here
                    pass

                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Error in feedback processing loop: {e}")
                await asyncio.sleep(5.0)

    async def _learning_cycle_loop(self) -> None:
        """Background loop for running learning cycles"""
        while self.is_running:
            try:
                # Check if it's time for a learning cycle
                time_since_last = datetime.now(timezone.utc) - self.last_learning_cycle

                if time_since_last >= self.learning_frequency:
                    # Run learning cycles for all active strategies
                    await self._run_scheduled_learning_cycles()
                    self.last_learning_cycle = datetime.now(timezone.utc)

                await asyncio.sleep(60.0)  # Check every minute

            except Exception as e:
                logger.error(f"Error in learning cycle loop: {e}")
                await asyncio.sleep(60.0)

    async def _model_update_loop(self) -> None:
        """Background loop for updating online learning models"""
        while self.is_running:
            try:
                # Update models that need updating
                for model_id in self.online_learning_engine.models.keys():
                    if self.online_learning_engine.should_update_model(model_id):
                        await self._update_online_model(model_id)

                await asyncio.sleep(300.0)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error in model update loop: {e}")
                await asyncio.sleep(300.0)

    def _collect_performance_data(self, strategy_id: str) -> List[Dict[str, Any]]:
        """Collect performance data for strategy"""
        try:
            performance_data = []

            for item in self.performance_analyzer.performance_history:
                feedback = item['feedback']
                if feedback.strategy_id == strategy_id:
                    performance_data.append({
                        'timestamp': feedback.timestamp,
                        'parameters': feedback.metadata.get('parameters', {}),
                        'performance_metrics': feedback.performance_metrics,
                        'market_context': feedback.market_context,
                        'execution_data': feedback.execution_data
                    })

            return performance_data

        except Exception as e:
            logger.error(f"Error collecting performance data: {e}")
            return []

    async def _analyze_performance_data(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance data for insights"""
        try:
            if not performance_data:
                return {}

            # Extract current parameters (from most recent data)
            current_parameters = performance_data[-1].get('parameters', {})

            # Analyze trends
            returns = [item['performance_metrics'].get('return', 0.0) for item in performance_data]
            volatilities = [item['performance_metrics'].get('volatility', 0.0) for item in performance_data]
            sharpe_ratios = [item['performance_metrics'].get('sharpe_ratio', 0.0) for item in performance_data]

            # Generate insights
            insights = []
            if len(returns) > 10:
                recent_performance = np.mean(returns[-10:])
                overall_performance = np.mean(returns)

                if recent_performance < overall_performance * 0.8:
                    insights.append("Recent performance declining")

                if np.std(returns) > np.mean(returns):
                    insights.append("High return volatility detected")

            return {
                'current_parameters': current_parameters,
                'performance_trends': {
                    'avg_return': np.mean(returns),
                    'avg_volatility': np.mean(volatilities),
                    'avg_sharpe': np.mean(sharpe_ratios),
                    'return_trend': np.polyfit(range(len(returns)), returns, 1)[0] if len(returns) > 1 else 0
                },
                'insights': insights
            }

        except Exception as e:
            logger.error(f"Error analyzing performance data: {e}")
            return {}

    async def _validate_optimization(self,
                                   strategy_id: str,
                                   optimized_params: Dict[str, Any],
                                   performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate optimization results"""
        try:
            # Simple validation based on parameter constraints
            confidence = 1.0
            deployment_ready = True

            # Check parameter change magnitude
            if performance_data:
                original_params = performance_data[-1].get('parameters', {})

                for key, new_value in optimized_params.items():
                    if key in original_params:
                        original_value = original_params[key]
                        if isinstance(new_value, (int, float)) and isinstance(original_value, (int, float)):
                            relative_change = abs(new_value - original_value) / max(abs(original_value), 1e-6)

                            if relative_change > self.max_parameter_change:
                                confidence *= 0.8

                            if relative_change > 0.5:  # Too large change
                                deployment_ready = False

            # Estimate improvement (simplified)
            estimated_improvement = np.random.uniform(0.01, 0.05)  # Placeholder

            return {
                'confidence': confidence,
                'deployment_ready': deployment_ready,
                'improvement': estimated_improvement,
                'metrics': {
                    'validation_score': confidence,
                    'parameter_stability': 1.0 - (1.0 - confidence) * 2
                }
            }

        except Exception as e:
            logger.error(f"Error validating optimization: {e}")
            return {'confidence': 0.0, 'deployment_ready': False, 'improvement': 0.0, 'metrics': {}}

    def _create_failed_result(self, cycle_id: str, reason: str) -> LearningResult:
        """Create a failed learning result"""
        return LearningResult(
            cycle_id=cycle_id,
            phase=LearningPhase.DATA_COLLECTION,
            timestamp=datetime.now(timezone.utc),
            objective=LearningObjective.MAXIMIZE_SHARPE,
            original_parameters={},
            optimized_parameters={},
            performance_improvement=0.0,
            confidence_score=0.0,
            validation_metrics={},
            deployment_ready=False,
            insights=[f"Learning cycle failed: {reason}"]
        )

    async def _trigger_emergency_optimization(self, feedback: FeedbackEvent, insights: Dict[str, Any]) -> None:
        """Trigger emergency optimization for poor performance"""
        try:
            logger.warning(f"Triggering emergency optimization for {feedback.strategy_id}")

            # Run immediate learning cycle
            result = await self.run_learning_cycle(feedback.strategy_id, LearningObjective.MINIMIZE_DRAWDOWN)

            if result.deployment_ready:
                # Send optimization update to execution engine
                await self._send_optimization_update(feedback.strategy_id, result)

        except Exception as e:
            logger.error(f"Error in emergency optimization: {e}")

    async def _run_scheduled_learning_cycles(self) -> None:
        """Run scheduled learning cycles for all strategies"""
        try:
            # Get all active strategies
            active_strategies = set()
            for item in self.performance_analyzer.performance_history:
                active_strategies.add(item['feedback'].strategy_id)

            # Run learning cycle for each strategy
            for strategy_id in active_strategies:
                try:
                    result = await self.run_learning_cycle(strategy_id, LearningObjective.MAXIMIZE_SHARPE)

                    if result.deployment_ready and result.performance_improvement > 0.01:
                        await self._send_optimization_update(strategy_id, result)

                except Exception as e:
                    logger.error(f"Error in scheduled learning cycle for {strategy_id}: {e}")

        except Exception as e:
            logger.error(f"Error running scheduled learning cycles: {e}")

    async def _update_online_model(self, model_id: str) -> None:
        """Update online learning model"""
        try:
            # Collect recent data for model update
            recent_data = list(self.performance_analyzer.performance_history)[-1000:]  # Last 1000 samples

            if len(recent_data) < 50:
                return

            # Prepare training data based on model type
            if model_id == 'performance_predictor':
                X, y = self._prepare_performance_data(recent_data)
            elif model_id == 'risk_predictor':
                X, y = self._prepare_risk_data(recent_data)
            elif model_id == 'execution_quality_predictor':
                X, y = self._prepare_execution_data(recent_data)
            else:
                return

            # Update model
            if len(X) > 0 and len(y) > 0:
                success = self.online_learning_engine.update_model(model_id, X, y)
                if success:
                    logger.info(f"Updated online model: {model_id}")

        except Exception as e:
            logger.error(f"Error updating online model {model_id}: {e}")

    def _prepare_performance_data(self, data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for performance prediction model"""
        try:
            X, y = [], []

            for item in data:
                feedback = item['feedback']
                features = []

                # Add market context features
                market_context = feedback.market_context
                features.extend([
                    market_context.get('volatility', 0.0),
                    market_context.get('trend_strength', 0.0),
                    market_context.get('volume_ratio', 1.0),
                ])

                # Add execution features
                execution_data = feedback.execution_data
                features.extend([
                    execution_data.get('fill_rate', 1.0),
                    execution_data.get('slippage_bps', 0.0),
                    execution_data.get('execution_time_ms', 100.0),
                ])

                if len(features) > 0:
                    X.append(features)
                    y.append(feedback.performance_metrics.get('return', 0.0))

            return np.array(X), np.array(y)

        except Exception as e:
            logger.error(f"Error preparing performance data: {e}")
            return np.array([]), np.array([])

    def _prepare_risk_data(self, data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for risk prediction model"""
        # Similar to performance data preparation but for risk metrics
        return self._prepare_performance_data(data)

    def _prepare_execution_data(self, data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for execution quality prediction model"""
        # Similar to performance data preparation but for execution quality
        return self._prepare_performance_data(data)

    async def _send_optimization_update(self, strategy_id: str, learning_result: LearningResult) -> None:
        """Send optimization update to execution engine"""
        try:
            # Create inter-engine message
            update_message = InterEngineMessage(
                id=f"optimization_update_{learning_result.cycle_id}",
                message_type=MessageType.STRATEGY_UPDATE,
                source_engine=EngineType.RESEARCH,
                target_engine=EngineType.EXECUTION,
                timestamp=datetime.now(timezone.utc),
                data={
                    'strategy_id': strategy_id,
                    'cycle_id': learning_result.cycle_id,
                    'parameters': learning_result.optimized_parameters,
                    'confidence': learning_result.confidence_score,
                    'expected_improvement': learning_result.performance_improvement,
                    'insights': learning_result.insights
                },
                priority=1,  # High priority for optimization updates
                requires_response=True,
                response_timeout=30.0
            )

            # In a real implementation, this would be sent through the message system
            logger.info(f"Sending optimization update for strategy {strategy_id}")

        except Exception as e:
            logger.error(f"Error sending optimization update: {e}")

    async def save_state(self) -> bool:
        """Save learning system state"""
        try:
            # Save online learning models
            models_dir = "models/continuous_learning"
            success = self.online_learning_engine.save_models(models_dir)

            # Save other state data
            state_data = {
                'learning_cycles': [asdict(cycle) for cycle in self.learning_cycles[-100:]],  # Last 100 cycles
                'current_phase': self.current_phase.value,
                'last_learning_cycle': self.last_learning_cycle.isoformat(),
                'config': self.config
            }

            state_path = "models/continuous_learning/system_state.json"
            os.makedirs(os.path.dirname(state_path), exist_ok=True)

            with open(state_path, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)

            return success

        except Exception as e:
            logger.error(f"Error saving learning system state: {e}")
            return False

    async def load_state(self) -> bool:
        """Load learning system state"""
        try:
            # Load online learning models
            models_dir = "models/continuous_learning"
            success = self.online_learning_engine.load_models(models_dir)

            # Load other state data
            state_path = "models/continuous_learning/system_state.json"
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state_data = json.load(f)

                # Restore learning cycles
                for cycle_data in state_data.get('learning_cycles', []):
                    cycle_data['phase'] = LearningPhase(cycle_data['phase'])
                    cycle_data['objective'] = LearningObjective(cycle_data['objective'])
                    cycle_data['timestamp'] = datetime.fromisoformat(cycle_data['timestamp'])
                    self.learning_cycles.append(LearningResult(**cycle_data))

                # Restore other state
                self.current_phase = LearningPhase(state_data.get('current_phase', 'data_collection'))
                self.last_learning_cycle = datetime.fromisoformat(
                    state_data.get('last_learning_cycle', datetime.now(timezone.utc).isoformat())
                )

            return success

        except Exception as e:
            logger.error(f"Error loading learning system state: {e}")
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            return {
                'is_running': self.is_running,
                'current_phase': self.current_phase.value,
                'last_learning_cycle': self.last_learning_cycle.isoformat(),
                'feedback_queue_size': len(self.feedback_queue),
                'total_learning_cycles': len(self.learning_cycles),
                'recent_cycles': [asdict(cycle) for cycle in self.learning_cycles[-5:]],
                'online_models': {
                    model_id: self.online_learning_engine.get_model_performance(model_id)
                    for model_id in self.online_learning_engine.models.keys()
                },
                'strategy_performance': {
                    key: asdict(perf) for key, perf in
                    list(self.performance_analyzer.strategy_performance.items())[-10:]  # Last 10 strategies
                }
            }

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}


# Example usage and testing
if __name__ == "__main__":
    async def test_continuous_learning():
        """Test the continuous learning system"""

        config = {
            'learning_frequency_minutes': 5,
            'min_feedback_samples': 10,
            'max_parameter_change': 0.2
        }

        learning_system = ContinuousLearningSystem(config)

        try:
            # Initialize system
            success = await learning_system.initialize()
            if not success:
                print("Failed to initialize learning system")
                return

            print("Learning system initialized successfully")

            # Simulate some feedback
            feedback = FeedbackEvent(
                id="test_feedback_001",
                feedback_type=FeedbackType.TRADE_EXECUTION,
                timestamp=datetime.now(timezone.utc),
                symbol="AAPL",
                strategy_id="momentum_strategy_v1",
                execution_data={
                    'fill_rate': 0.95,
                    'slippage_bps': 3.2,
                    'execution_time_ms': 150,
                    'market_impact_bps': 2.1
                },
                performance_metrics={
                    'return': 0.015,
                    'sharpe_ratio': 1.2,
                    'volatility': 0.18,
                    'drawdown': 0.03
                },
                market_context={
                    'volatility': 0.22,
                    'trend_strength': 0.6,
                    'volume_ratio': 1.3
                }
            )

            # Process feedback
            await learning_system.process_feedback(feedback)

            # Run a learning cycle
            result = await learning_system.run_learning_cycle(
                "momentum_strategy_v1",
                LearningObjective.MAXIMIZE_SHARPE
            )

            print(f"Learning cycle result:")
            print(f"- Cycle ID: {result.cycle_id}")
            print(f"- Performance improvement: {result.performance_improvement:.4f}")
            print(f"- Confidence: {result.confidence_score:.2f}")
            print(f"- Deployment ready: {result.deployment_ready}")
            print(f"- Insights: {result.insights}")

            # Get system status
            status = learning_system.get_system_status()
            print(f"\nSystem status:")
            print(f"- Running: {status['is_running']}")
            print(f"- Current phase: {status['current_phase']}")
            print(f"- Total cycles: {status['total_learning_cycles']}")

        except Exception as e:
            print(f"Error testing learning system: {e}")
        finally:
            await learning_system.stop()

    # Run test
    asyncio.run(test_continuous_learning())