#!/usr/bin/env python3
"""
AI STRATEGY ENHANCER
Lightweight AI layer that enhances existing trading strategies
Uses ML to score opportunities and track learning

Integration: Forex + Options + Futures
Models: RandomForest scoring, confidence boosting, meta-learning
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import pickle
from pathlib import Path

# Try to import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[WARNING] Sklearn not available. Run: pip install scikit-learn")

@dataclass
class AIEnhancedOpportunity:
    """Opportunity enhanced with AI scoring"""
    symbol: str
    strategy: str
    direction: str
    base_score: float
    ai_score: float
    final_score: float
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: List[str]
    metadata: Dict

class AIStrategyEnhancer:
    """
    AI layer that enhances existing strategies

    Features:
    - ML-based opportunity scoring
    - Confidence boosting based on patterns
    - Meta-learning from outcomes
    - Lightweight (doesn't require GPU training)
    """

    def __init__(self, learning_file: str = "ai_learning_data.pkl"):
        self.learning_file = learning_file
        self.ml_available = ML_AVAILABLE

        # Learning data
        self.opportunity_history = []
        self.outcome_history = []
        self.pattern_weights = {
            'trend_alignment': 0.25,
            'momentum_strength': 0.20,
            'volatility_factor': 0.15,
            'volume_confirmation': 0.15,
            'risk_reward_ratio': 0.15,
            'market_regime': 0.10
        }

        # Performance tracking
        self.performance_by_strategy = {}
        self.performance_by_symbol = {}
        self.win_rates = []

        # ML models (if available)
        self.opportunity_scorer = None
        self.confidence_predictor = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None

        # Load previous learning
        self._load_learning_data()

        print("[AI ENHANCER] Initialized")
        print(f"[AI ENHANCER] ML Available: {self.ml_available}")
        print(f"[AI ENHANCER] Historical opportunities: {len(self.opportunity_history)}")

    def _load_learning_data(self):
        """Load previous learning data"""
        try:
            if Path(self.learning_file).exists():
                with open(self.learning_file, 'rb') as f:
                    data = pickle.load(f)
                    self.opportunity_history = data.get('opportunities', [])
                    self.outcome_history = data.get('outcomes', [])
                    self.pattern_weights = data.get('weights', self.pattern_weights)
                    self.performance_by_strategy = data.get('strategy_performance', {})
                    self.performance_by_symbol = data.get('symbol_performance', {})

                    # Train models if enough data
                    if len(self.opportunity_history) >= 20 and self.ml_available:
                        self._train_ml_models()

                print(f"[AI ENHANCER] Loaded {len(self.opportunity_history)} historical opportunities")
        except Exception as e:
            print(f"[AI ENHANCER] Could not load learning data: {e}")

    def _train_ml_models(self):
        """Train ML models on historical data"""
        if not self.ml_available or len(self.outcome_history) < 20:
            return

        try:
            # Prepare training data
            X = []
            y_success = []
            y_return = []

            for opp, outcome in zip(self.opportunity_history[-200:], self.outcome_history[-200:]):
                features = self._extract_features(opp)
                X.append(features)
                y_success.append(1 if outcome.get('success', False) else 0)
                y_return.append(outcome.get('return', 0))

            if len(X) >= 20:
                X = np.array(X)

                # Train opportunity scorer (predicts success)
                self.opportunity_scorer = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42
                )
                self.opportunity_scorer.fit(X, y_success)

                # Train confidence predictor (predicts return)
                self.confidence_predictor = GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=5,
                    random_state=42
                )
                self.confidence_predictor.fit(X, y_return)

                # Fit scaler
                self.scaler.fit(X)

                print(f"[AI ENHANCER] ML models trained on {len(X)} examples")

        except Exception as e:
            print(f"[AI ENHANCER] Model training error: {e}")

    def _extract_features(self, opportunity: Dict) -> List[float]:
        """Extract features from opportunity for ML"""
        features = [
            opportunity.get('base_score', 0),
            opportunity.get('confidence', 0),
            opportunity.get('risk_reward', 1.5),
            opportunity.get('trend_strength', 0),
            opportunity.get('momentum', 0),
            opportunity.get('volatility', 0),
            opportunity.get('volume_ratio', 1.0),
            1 if opportunity.get('direction') == 'LONG' else 0,
            1 if opportunity.get('strategy') == 'EMA_CROSSOVER' else 0,
            1 if opportunity.get('strategy') == 'BULL_PUT_SPREAD' else 0,
        ]
        return features

    def enhance_forex_opportunity(self, opportunity: Dict, market_data: pd.DataFrame) -> AIEnhancedOpportunity:
        """
        Enhance forex opportunity with AI scoring

        Args:
            opportunity: From EMACrossoverOptimized
            market_data: Recent price data
        """

        # Extract base information
        symbol = opportunity['symbol']
        strategy = opportunity['strategy']
        direction = opportunity['direction']
        base_score = opportunity['score']

        # Calculate AI enhancements
        ai_features = self._calculate_ai_features(opportunity, market_data, asset_type='forex')
        ai_score = self._calculate_ai_score(opportunity, ai_features)
        confidence = self._calculate_confidence(opportunity, ai_features)

        # Combine scores
        final_score = self._combine_scores(base_score, ai_score, confidence)

        # Generate reasoning
        reasoning = self._generate_reasoning(opportunity, ai_features, 'forex')

        # Create enhanced opportunity
        enhanced = AIEnhancedOpportunity(
            symbol=symbol,
            strategy=strategy,
            direction=direction,
            base_score=base_score,
            ai_score=ai_score,
            final_score=final_score,
            confidence=confidence,
            entry_price=opportunity['entry_price'],
            stop_loss=opportunity['stop_loss'],
            take_profit=opportunity['take_profit'],
            reasoning=reasoning,
            metadata={
                'asset_type': 'forex',
                'indicators': opportunity['indicators'],
                'risk_reward': opportunity['risk_reward'],
                'ai_features': ai_features
            }
        )

        # Record for learning
        self._record_opportunity(enhanced)

        return enhanced

    def enhance_options_opportunity(self, opportunity: Dict, market_data: pd.DataFrame) -> AIEnhancedOpportunity:
        """
        Enhance options opportunity with AI scoring

        Args:
            opportunity: From BullPutSpreadEngine or other options strategies
            market_data: Recent price data
        """

        # Extract base information
        symbol = opportunity.get('symbol', 'UNKNOWN')
        strategy = opportunity.get('strategy', 'BULL_PUT_SPREAD')
        direction = 'NEUTRAL'  # Bull put spreads are neutral/bullish
        base_score = opportunity.get('score', 5.0)

        # Calculate AI enhancements
        ai_features = self._calculate_ai_features(opportunity, market_data, asset_type='options')
        ai_score = self._calculate_ai_score(opportunity, ai_features)
        confidence = self._calculate_confidence(opportunity, ai_features)

        # Combine scores
        final_score = self._combine_scores(base_score, ai_score, confidence)

        # Generate reasoning
        reasoning = self._generate_reasoning(opportunity, ai_features, 'options')

        # Create enhanced opportunity
        enhanced = AIEnhancedOpportunity(
            symbol=symbol,
            strategy=strategy,
            direction=direction,
            base_score=base_score,
            ai_score=ai_score,
            final_score=final_score,
            confidence=confidence,
            entry_price=opportunity.get('price', 0),
            stop_loss=0,  # Options have defined risk
            take_profit=0,
            reasoning=reasoning,
            metadata={
                'asset_type': 'options',
                'momentum': opportunity.get('momentum', 0),
                'regime': opportunity.get('regime', 'NEUTRAL'),
                'ai_features': ai_features
            }
        )

        # Record for learning
        self._record_opportunity(enhanced)

        return enhanced

    def _calculate_ai_features(self, opportunity: Dict, market_data: pd.DataFrame, asset_type: str) -> Dict:
        """Calculate AI-specific features"""

        if market_data is None or market_data.empty:
            return self._default_features()

        try:
            # Price features
            closes = market_data['close'].values
            returns = np.diff(closes) / closes[:-1]

            # Trend strength
            if len(closes) >= 20:
                sma_20 = closes[-20:].mean()
                trend_strength = (closes[-1] - sma_20) / sma_20
            else:
                trend_strength = 0

            # Momentum
            if len(closes) >= 10:
                momentum = (closes[-1] - closes[-10]) / closes[-10]
            else:
                momentum = 0

            # Volatility (std of returns)
            volatility = np.std(returns) if len(returns) > 1 else 0

            # Volume ratio (if available)
            if 'volume' in market_data.columns:
                volumes = market_data['volume'].values
                volume_ratio = volumes[-1] / volumes.mean() if len(volumes) > 0 and volumes.mean() > 0 else 1.0
            else:
                volume_ratio = 1.0

            # Trend alignment (price vs trend)
            if asset_type == 'forex':
                indicators = opportunity.get('indicators', {})
                price = opportunity.get('entry_price', closes[-1] if len(closes) > 0 else 0)
                trend_ema = indicators.get('ema_trend', closes[-1] if len(closes) > 0 else 0)
                trend_alignment = 1.0 if price > trend_ema else -1.0
            else:
                trend_alignment = 1.0 if momentum > 0 else -1.0

            # Risk/reward
            risk_reward = opportunity.get('risk_reward', 1.5)

            return {
                'trend_strength': trend_strength,
                'momentum': momentum,
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'trend_alignment': trend_alignment,
                'risk_reward': risk_reward
            }

        except Exception as e:
            print(f"[AI ENHANCER] Feature calculation error: {e}")
            return self._default_features()

    def _default_features(self) -> Dict:
        """Default features when calculation fails"""
        return {
            'trend_strength': 0,
            'momentum': 0,
            'volatility': 0.02,
            'volume_ratio': 1.0,
            'trend_alignment': 0,
            'risk_reward': 1.5
        }

    def _calculate_ai_score(self, opportunity: Dict, ai_features: Dict) -> float:
        """Calculate AI score using weighted features"""

        # If ML models available, use them
        if self.opportunity_scorer and self.ml_available:
            try:
                features = self._extract_features({**opportunity, **ai_features})
                features_scaled = self.scaler.transform([features])

                # Get probability of success
                success_prob = self.opportunity_scorer.predict_proba(features_scaled)[0][1]

                # Get predicted return
                predicted_return = self.confidence_predictor.predict(features_scaled)[0]

                # Combine
                ai_score = (success_prob * 0.6 + min(1.0, max(0, predicted_return)) * 0.4) * 10
                return float(ai_score)

            except Exception as e:
                print(f"[AI ENHANCER] ML scoring error: {e}")

        # Fallback: weighted scoring
        score = 5.0

        # Trend strength bonus
        if abs(ai_features['trend_strength']) > 0.02:
            score += min(2.0, abs(ai_features['trend_strength']) * 100)

        # Momentum bonus
        if abs(ai_features['momentum']) > 0.01:
            score += min(1.5, abs(ai_features['momentum']) * 100)

        # Volatility adjustment (prefer moderate volatility)
        vol = ai_features['volatility']
        if 0.01 < vol < 0.05:  # Sweet spot
            score += 1.0
        elif vol > 0.10:  # Too volatile
            score -= 1.0

        # Volume confirmation
        if ai_features['volume_ratio'] > 1.2:
            score += 0.5

        # Risk/reward bonus
        if ai_features['risk_reward'] >= 2.0:
            score += 1.0
        elif ai_features['risk_reward'] < 1.5:
            score -= 0.5

        # Trend alignment
        direction = opportunity.get('direction', 'LONG')
        if direction == 'LONG' and ai_features['trend_alignment'] > 0:
            score += 1.0
        elif direction == 'SHORT' and ai_features['trend_alignment'] < 0:
            score += 1.0

        return max(0, min(10, score))

    def _calculate_confidence(self, opportunity: Dict, ai_features: Dict) -> float:
        """Calculate confidence score (0-1)"""

        # Start with base confidence
        base_conf = opportunity.get('confidence', 0.5)

        # Adjust based on historical performance
        strategy = opportunity.get('strategy', 'UNKNOWN')
        symbol = opportunity.get('symbol', 'UNKNOWN')

        strategy_perf = self.performance_by_strategy.get(strategy, {})
        symbol_perf = self.performance_by_symbol.get(symbol, {})

        strategy_win_rate = strategy_perf.get('win_rate', 0.5)
        symbol_win_rate = symbol_perf.get('win_rate', 0.5)

        # Combine
        confidence = (base_conf * 0.4 + strategy_win_rate * 0.3 + symbol_win_rate * 0.3)

        # Boost for strong features
        if ai_features['risk_reward'] >= 2.0:
            confidence += 0.05
        if ai_features['volume_ratio'] > 1.5:
            confidence += 0.05
        if abs(ai_features['momentum']) > 0.03:
            confidence += 0.05

        return max(0.1, min(1.0, confidence))

    def _combine_scores(self, base_score: float, ai_score: float, confidence: float) -> float:
        """Combine base strategy score with AI score"""

        # Weighted combination
        # Base score: 60% (trust the original strategy)
        # AI score: 30% (AI enhancement)
        # Confidence: 10% (historical performance)

        final = (base_score * 0.6) + (ai_score * 0.3) + (confidence * 10 * 0.1)

        return max(0, min(10, final))

    def _generate_reasoning(self, opportunity: Dict, ai_features: Dict, asset_type: str) -> List[str]:
        """Generate human-readable reasoning"""

        reasoning = []

        # Base strategy
        strategy = opportunity.get('strategy', 'UNKNOWN')
        reasoning.append(f"Strategy: {strategy}")

        # AI enhancements
        if ai_features['trend_strength'] > 0.02:
            reasoning.append(f"Strong uptrend (trend strength: {ai_features['trend_strength']:.1%})")
        elif ai_features['trend_strength'] < -0.02:
            reasoning.append(f"Strong downtrend (trend strength: {ai_features['trend_strength']:.1%})")

        if ai_features['momentum'] > 0.02:
            reasoning.append(f"Positive momentum ({ai_features['momentum']:.1%})")
        elif ai_features['momentum'] < -0.02:
            reasoning.append(f"Negative momentum ({ai_features['momentum']:.1%})")

        if ai_features['volume_ratio'] > 1.5:
            reasoning.append(f"High volume confirmation ({ai_features['volume_ratio']:.1f}x average)")

        if ai_features['risk_reward'] >= 2.0:
            reasoning.append(f"Excellent R/R ratio ({ai_features['risk_reward']:.1f}:1)")

        # Historical performance
        strategy_perf = self.performance_by_strategy.get(strategy, {})
        if strategy_perf:
            win_rate = strategy_perf.get('win_rate', 0)
            if win_rate > 0.65:
                reasoning.append(f"Strategy historically strong ({win_rate:.0%} win rate)")

        return reasoning

    def _record_opportunity(self, enhanced: AIEnhancedOpportunity):
        """Record opportunity for learning"""

        record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': enhanced.symbol,
            'strategy': enhanced.strategy,
            'direction': enhanced.direction,
            'base_score': enhanced.base_score,
            'ai_score': enhanced.ai_score,
            'final_score': enhanced.final_score,
            'confidence': enhanced.confidence,
            'entry_price': enhanced.entry_price,
            'stop_loss': enhanced.stop_loss,
            'take_profit': enhanced.take_profit,
            **enhanced.metadata.get('ai_features', {})
        }

        self.opportunity_history.append(record)

        # Keep last 1000
        if len(self.opportunity_history) > 1000:
            self.opportunity_history = self.opportunity_history[-1000:]

    def record_outcome(self, symbol: str, strategy: str, success: bool, return_pct: float):
        """
        Record trade outcome for learning

        Args:
            symbol: Traded symbol
            strategy: Strategy used
            success: True if profitable
            return_pct: Return percentage (e.g., 0.05 for 5%)
        """

        outcome = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'strategy': strategy,
            'success': success,
            'return': return_pct
        }

        self.outcome_history.append(outcome)

        # Update performance tracking
        if strategy not in self.performance_by_strategy:
            self.performance_by_strategy[strategy] = {'wins': 0, 'losses': 0, 'win_rate': 0.5}

        if symbol not in self.performance_by_symbol:
            self.performance_by_symbol[symbol] = {'wins': 0, 'losses': 0, 'win_rate': 0.5}

        if success:
            self.performance_by_strategy[strategy]['wins'] += 1
            self.performance_by_symbol[symbol]['wins'] += 1
        else:
            self.performance_by_strategy[strategy]['losses'] += 1
            self.performance_by_symbol[symbol]['losses'] += 1

        # Update win rates
        strat = self.performance_by_strategy[strategy]
        total = strat['wins'] + strat['losses']
        strat['win_rate'] = strat['wins'] / total if total > 0 else 0.5

        sym = self.performance_by_symbol[symbol]
        total = sym['wins'] + sym['losses']
        sym['win_rate'] = sym['wins'] / total if total > 0 else 0.5

        # Re-train models if enough new data
        if len(self.outcome_history) % 10 == 0 and len(self.outcome_history) >= 20:
            self._train_ml_models()

        # Save periodically
        if len(self.outcome_history) % 5 == 0:
            self.save_learning_data()

    def save_learning_data(self):
        """Save learning data to disk"""
        try:
            data = {
                'opportunities': self.opportunity_history[-1000:],
                'outcomes': self.outcome_history[-1000:],
                'weights': self.pattern_weights,
                'strategy_performance': self.performance_by_strategy,
                'symbol_performance': self.performance_by_symbol,
                'timestamp': datetime.now().isoformat()
            }

            with open(self.learning_file, 'wb') as f:
                pickle.dump(data, f)

            # Also save JSON for readability
            json_file = self.learning_file.replace('.pkl', '.json')
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            print(f"[AI ENHANCER] Saved learning data ({len(self.outcome_history)} outcomes)")

        except Exception as e:
            print(f"[AI ENHANCER] Save error: {e}")

    def get_performance_summary(self) -> Dict:
        """Get performance summary"""

        if not self.outcome_history:
            return {'total_outcomes': 0}

        recent = self.outcome_history[-50:] if len(self.outcome_history) >= 50 else self.outcome_history

        wins = len([o for o in recent if o['success']])
        losses = len([o for o in recent if not o['success']])
        win_rate = wins / len(recent) if recent else 0

        avg_return = np.mean([o['return'] for o in recent]) if recent else 0

        return {
            'total_outcomes': len(self.outcome_history),
            'recent_trades': len(recent),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'strategy_performance': self.performance_by_strategy,
            'top_symbols': sorted(
                self.performance_by_symbol.items(),
                key=lambda x: x[1].get('win_rate', 0),
                reverse=True
            )[:10]
        }


def demo():
    """Demo the AI enhancer"""

    print("\n" + "="*70)
    print("AI STRATEGY ENHANCER DEMO")
    print("="*70)

    # Initialize
    enhancer = AIStrategyEnhancer()

    # Simulate forex opportunity
    forex_opp = {
        'symbol': 'EUR_USD',
        'strategy': 'EMA_CROSSOVER_OPTIMIZED',
        'direction': 'LONG',
        'score': 9.0,
        'confidence': 0.75,
        'entry_price': 1.1650,
        'stop_loss': 1.1620,
        'take_profit': 1.1695,
        'risk_reward': 1.5,
        'indicators': {
            'ema_fast': 1.1655,
            'ema_slow': 1.1640,
            'ema_trend': 1.1600,
            'rsi': 58.0,
            'atr': 0.0015
        }
    }

    # Create mock market data
    dates = pd.date_range(start='2025-01-01', periods=100, freq='H')
    prices = np.linspace(1.1500, 1.1650, 100) + np.random.normal(0, 0.001, 100)
    market_data = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'volume': np.random.randint(100000, 200000, 100)
    })

    # Enhance
    enhanced = enhancer.enhance_forex_opportunity(forex_opp, market_data)

    print("\n[ENHANCED OPPORTUNITY]")
    print(f"  Symbol: {enhanced.symbol}")
    print(f"  Strategy: {enhanced.strategy}")
    print(f"  Direction: {enhanced.direction}")
    print(f"  Base Score: {enhanced.base_score:.2f}")
    print(f"  AI Score: {enhanced.ai_score:.2f}")
    print(f"  Final Score: {enhanced.final_score:.2f}")
    print(f"  Confidence: {enhanced.confidence:.1%}")
    print(f"  Entry: {enhanced.entry_price:.5f}")
    print(f"  Stop: {enhanced.stop_loss:.5f}")
    print(f"  Target: {enhanced.take_profit:.5f}")
    print(f"\n  AI Reasoning:")
    for reason in enhanced.reasoning:
        print(f"    - {reason}")

    # Simulate outcome
    print("\n[SIMULATING OUTCOME]")
    enhancer.record_outcome('EUR_USD', 'EMA_CROSSOVER_OPTIMIZED', True, 0.025)

    # Get performance
    perf = enhancer.get_performance_summary()
    print(f"\n[PERFORMANCE SUMMARY]")
    print(f"  Total outcomes: {perf['total_outcomes']}")
    print(f"  Win rate: {perf['win_rate']:.1%}")
    print(f"  Avg return: {perf['avg_return']:.1%}")

    print("\n" + "="*70)
    print("AI Enhancer ready for production")
    print("="*70)


if __name__ == "__main__":
    demo()
