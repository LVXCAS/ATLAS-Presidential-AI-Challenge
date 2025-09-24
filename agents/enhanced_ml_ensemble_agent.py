"""
Enhanced ML Ensemble Agent
Combines ML4T, Finance, and original ML capabilities for superior predictions
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import our ML agents
try:
    from agents.ml4t_agent import ml4t_agent, ML4TPrediction
    from agents.advanced_ml_finance_agent import advanced_ml_finance_agent, MLPrediction
    from agents.advanced_ml_engine import advanced_ml_engine
    ML_AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"ML agents not available: {e}")
    ML_AGENTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnsemblePrediction:
    """Enhanced ensemble prediction"""
    symbol: str
    final_prediction: str  # 'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'
    confidence: float
    ensemble_components: Dict[str, Any]
    consensus_strength: float
    risk_adjusted_confidence: float
    feature_attribution: Dict[str, float]
    model_weights: Dict[str, float]
    timestamp: datetime

class EnhancedMLEnsembleAgent:
    """
    Enhanced ML Ensemble that combines:
    - ML4T advanced factor models and ensemble methods
    - Finance repository ML models (LSTM, etc.)
    - Original advanced ML engine
    - Custom ensemble weighting based on recent performance
    - Risk-adjusted confidence scoring
    - Feature attribution analysis
    """

    def __init__(self):
        self.name = "Enhanced ML Ensemble Agent"
        self.agent_weights = {
            'ml4t': 0.40,  # Highest weight for sophisticated ML4T models
            'finance_ml': 0.35,  # Finance repository ML models
            'original_ml': 0.25   # Original ML engine
        }
        self.performance_history = {}
        self.adaptive_weights = {}

        logger.info("Enhanced ML Ensemble Agent initialized")

    def _calculate_consensus_strength(self, predictions: Dict) -> float:
        """Calculate consensus strength across models"""
        try:
            directions = []
            confidences = []

            for agent, pred in predictions.items():
                if pred:
                    # Normalize predictions to numeric scale
                    if hasattr(pred, 'prediction'):
                        if pred.prediction in ['UP', 'STRONG_BUY', 'BUY']:
                            directions.append(1)
                        elif pred.prediction in ['DOWN', 'STRONG_SELL', 'SELL']:
                            directions.append(-1)
                        else:
                            directions.append(0)

                        confidences.append(pred.confidence)

            if not directions:
                return 0.0

            # Calculate consensus
            avg_direction = np.mean(directions)
            direction_std = np.std(directions) if len(directions) > 1 else 0
            avg_confidence = np.mean(confidences)

            # Strong consensus when directions align and confidence is high
            consensus = avg_confidence * (1 - direction_std / 2) if direction_std <= 2 else 0

            return max(0, min(1, consensus))

        except Exception as e:
            logger.error(f"Consensus calculation error: {e}")
            return 0.0

    def _adjust_weights_by_performance(self, symbol: str) -> Dict[str, float]:
        """Adjust ensemble weights based on recent performance"""
        try:
            if symbol not in self.performance_history:
                return self.agent_weights.copy()

            history = self.performance_history[symbol]
            if len(history) < 5:  # Need minimum history
                return self.agent_weights.copy()

            # Calculate recent accuracy for each agent
            recent_performance = {}

            for agent in self.agent_weights.keys():
                if agent in history:
                    recent_accuracy = np.mean([h['accuracy'] for h in history[agent][-10:]])
                    recent_performance[agent] = recent_accuracy
                else:
                    recent_performance[agent] = 0.5  # Default

            # Normalize to weights
            total_perf = sum(recent_performance.values())
            if total_perf > 0:
                adapted_weights = {
                    agent: perf / total_perf
                    for agent, perf in recent_performance.items()
                }
            else:
                adapted_weights = self.agent_weights.copy()

            # Smooth with original weights (30% adaptation, 70% original)
            final_weights = {}
            for agent in self.agent_weights:
                final_weights[agent] = (
                    0.7 * self.agent_weights[agent] +
                    0.3 * adapted_weights.get(agent, self.agent_weights[agent])
                )

            return final_weights

        except Exception as e:
            logger.error(f"Weight adjustment error: {e}")
            return self.agent_weights.copy()

    def _calculate_risk_adjusted_confidence(self, base_confidence: float,
                                          predictions: Dict, symbol: str) -> float:
        """Calculate risk-adjusted confidence"""
        try:
            # Start with base confidence
            risk_adjusted = base_confidence

            # Penalty for high disagreement
            directions = []
            for pred in predictions.values():
                if pred and hasattr(pred, 'prediction'):
                    if pred.prediction in ['UP', 'STRONG_BUY', 'BUY']:
                        directions.append(1)
                    elif pred.prediction in ['DOWN', 'STRONG_SELL', 'SELL']:
                        directions.append(-1)
                    else:
                        directions.append(0)

            if len(directions) > 1:
                disagreement = np.std(directions)
                risk_adjusted *= (1 - disagreement / 4)  # Penalty for disagreement

            # Bonus for high individual confidences
            individual_confidences = [
                pred.confidence for pred in predictions.values()
                if pred and hasattr(pred, 'confidence')
            ]

            if individual_confidences:
                avg_individual_conf = np.mean(individual_confidences)
                if avg_individual_conf > 0.8:
                    risk_adjusted *= 1.1  # 10% bonus for high confidence
                elif avg_individual_conf < 0.5:
                    risk_adjusted *= 0.9  # 10% penalty for low confidence

            # Historical performance adjustment
            if symbol in self.performance_history:
                recent_accuracy = np.mean([
                    h.get('ensemble_accuracy', 0.5)
                    for h in self.performance_history[symbol][-5:]
                ])
                if recent_accuracy > 0.6:
                    risk_adjusted *= (1 + (recent_accuracy - 0.5))
                else:
                    risk_adjusted *= recent_accuracy / 0.5

            return max(0.1, min(0.95, risk_adjusted))

        except Exception as e:
            logger.error(f"Risk adjustment error: {e}")
            return base_confidence

    async def generate_ensemble_prediction(self, symbol: str, horizon: int = 1) -> Optional[EnsemblePrediction]:
        """Generate comprehensive ensemble prediction"""
        try:
            if not ML_AGENTS_AVAILABLE:
                return None

            # Get predictions from all ML agents
            predictions = {}

            # 1. ML4T prediction
            try:
                ml4t_pred = await ml4t_agent.predict_ml4t(symbol, horizon)
                predictions['ml4t'] = ml4t_pred
            except Exception as e:
                logger.warning(f"ML4T prediction failed for {symbol}: {e}")
                predictions['ml4t'] = None

            # 2. Finance ML prediction
            try:
                finance_pred = await advanced_ml_finance_agent.predict_price_movement(symbol, f'{horizon}d')
                predictions['finance_ml'] = finance_pred
            except Exception as e:
                logger.warning(f"Finance ML prediction failed for {symbol}: {e}")
                predictions['finance_ml'] = None

            # 3. Original ML prediction
            try:
                # Get market data for original ML
                import yfinance as yf
                stock = yf.Ticker(symbol)
                df = stock.history(period='6mo')

                if not df.empty:
                    original_pred = await advanced_ml_engine.predict_trade_success(
                        symbol, 'long_call', 0.7
                    )
                    # Convert to our format
                    if original_pred:
                        pred_direction = 'UP' if original_pred[0] > 0.5 else 'DOWN'
                        predictions['original_ml'] = type('Prediction', (), {
                            'prediction': pred_direction,
                            'confidence': abs(original_pred[0] - 0.5) * 2,
                            'probability': original_pred[0]
                        })()
                    else:
                        predictions['original_ml'] = None
                else:
                    predictions['original_ml'] = None
            except Exception as e:
                logger.warning(f"Original ML prediction failed for {symbol}: {e}")
                predictions['original_ml'] = None

            # Filter out None predictions
            valid_predictions = {k: v for k, v in predictions.items() if v is not None}

            if not valid_predictions:
                return None

            # Calculate adaptive weights
            adaptive_weights = self._adjust_weights_by_performance(symbol)

            # Calculate ensemble prediction
            weighted_scores = []
            ensemble_components = {}

            for agent, prediction in valid_predictions.items():
                weight = adaptive_weights.get(agent, 0.33)

                # Convert prediction to numeric score
                if hasattr(prediction, 'prediction'):
                    if prediction.prediction in ['STRONG_BUY', 'UP']:
                        score = 1.0
                    elif prediction.prediction in ['BUY']:
                        score = 0.7
                    elif prediction.prediction in ['STRONG_SELL', 'DOWN']:
                        score = -1.0
                    elif prediction.prediction in ['SELL']:
                        score = -0.7
                    else:
                        score = 0.0
                else:
                    score = 0.0

                # Weight by confidence
                if hasattr(prediction, 'confidence'):
                    confidence_weighted_score = score * prediction.confidence * weight
                else:
                    confidence_weighted_score = score * 0.5 * weight

                weighted_scores.append(confidence_weighted_score)

                ensemble_components[agent] = {
                    'prediction': prediction.prediction if hasattr(prediction, 'prediction') else 'UNKNOWN',
                    'confidence': prediction.confidence if hasattr(prediction, 'confidence') else 0.5,
                    'weight': weight,
                    'score': score
                }

            # Calculate final prediction
            if weighted_scores:
                final_score = sum(weighted_scores) / sum(adaptive_weights.values())
                base_confidence = np.mean([
                    p.confidence for p in valid_predictions.values()
                    if hasattr(p, 'confidence')
                ]) if valid_predictions else 0.5

                # Determine final prediction
                if final_score > 0.6:
                    final_prediction = 'STRONG_BUY'
                elif final_score > 0.2:
                    final_prediction = 'BUY'
                elif final_score < -0.6:
                    final_prediction = 'STRONG_SELL'
                elif final_score < -0.2:
                    final_prediction = 'SELL'
                else:
                    final_prediction = 'HOLD'

                # Calculate metrics
                consensus_strength = self._calculate_consensus_strength(valid_predictions)
                risk_adjusted_confidence = self._calculate_risk_adjusted_confidence(
                    base_confidence, valid_predictions, symbol
                )

                return EnsemblePrediction(
                    symbol=symbol,
                    final_prediction=final_prediction,
                    confidence=base_confidence,
                    ensemble_components=ensemble_components,
                    consensus_strength=consensus_strength,
                    risk_adjusted_confidence=risk_adjusted_confidence,
                    feature_attribution={},  # Would implement feature importance
                    model_weights=adaptive_weights,
                    timestamp=datetime.now()
                )

            return None

        except Exception as e:
            logger.error(f"Ensemble prediction error for {symbol}: {e}")
            return None

    async def update_performance(self, symbol: str, prediction: EnsemblePrediction,
                               actual_outcome: float) -> None:
        """Update performance tracking for adaptive weighting"""
        try:
            if symbol not in self.performance_history:
                self.performance_history[symbol] = []

            # Calculate accuracy
            predicted_direction = 1 if prediction.final_prediction in ['BUY', 'STRONG_BUY'] else -1
            actual_direction = 1 if actual_outcome > 0 else -1
            correct = predicted_direction == actual_direction

            # Update history
            performance_record = {
                'timestamp': prediction.timestamp,
                'predicted': prediction.final_prediction,
                'actual_return': actual_outcome,
                'correct': correct,
                'confidence': prediction.confidence,
                'risk_adjusted_confidence': prediction.risk_adjusted_confidence,
                'consensus_strength': prediction.consensus_strength
            }

            self.performance_history[symbol].append(performance_record)

            # Keep only recent history (last 50 predictions)
            if len(self.performance_history[symbol]) > 50:
                self.performance_history[symbol] = self.performance_history[symbol][-50:]

            # Update individual agent performance tracking
            for agent, component in prediction.ensemble_components.items():
                if agent not in self.performance_history:
                    self.performance_history[agent] = []

                agent_correct = (
                    (component['prediction'] in ['BUY', 'STRONG_BUY'] and actual_outcome > 0) or
                    (component['prediction'] in ['SELL', 'STRONG_SELL'] and actual_outcome < 0)
                )

                self.performance_history[agent].append({
                    'timestamp': prediction.timestamp,
                    'accuracy': 1.0 if agent_correct else 0.0,
                    'confidence': component['confidence']
                })

                # Keep recent history
                if len(self.performance_history[agent]) > 50:
                    self.performance_history[agent] = self.performance_history[agent][-50:]

        except Exception as e:
            logger.error(f"Performance update error for {symbol}: {e}")

    async def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            summary = {}

            for agent in ['ml4t', 'finance_ml', 'original_ml']:
                if agent in self.performance_history and self.performance_history[agent]:
                    history = self.performance_history[agent]

                    summary[agent] = {
                        'total_predictions': len(history),
                        'accuracy': np.mean([h['accuracy'] for h in history]),
                        'recent_accuracy': np.mean([h['accuracy'] for h in history[-10:]]) if len(history) >= 10 else 0,
                        'avg_confidence': np.mean([h['confidence'] for h in history]),
                        'current_weight': self.agent_weights.get(agent, 0)
                    }

            return summary

        except Exception as e:
            logger.error(f"Performance summary error: {e}")
            return {}

# Create singleton instance
enhanced_ml_ensemble_agent = EnhancedMLEnsembleAgent()