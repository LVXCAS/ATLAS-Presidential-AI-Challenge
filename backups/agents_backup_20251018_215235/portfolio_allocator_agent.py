"""
Portfolio Allocator Agent - LangGraph-based signal fusion and portfolio management

This agent implements sophisticated signal fusion from multiple trading strategies,
conflict resolution, explainability engine, and regime-based strategy weighting.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from langgraph.graph import StateGraph, END

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class SignalType(Enum):
    """Types of trading signals"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    OPTIONS_VOLATILITY = "options_volatility"
    SHORT_SELLING = "short_selling"
    LONG_TERM_CORE = "long_term_core"
    SENTIMENT = "sentiment"
    ARBITRAGE = "arbitrage"


@dataclass
class Reason:
    """Explainable reason for a trading decision"""
    rank: int
    factor: str
    contribution: float
    explanation: str
    confidence: float
    supporting_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal:
    """Base signal structure from individual agents"""
    symbol: str
    signal_type: SignalType
    value: float  # Normalized to [-1, 1] range
    confidence: float  # [0, 1] range
    timestamp: datetime
    agent_name: str
    model_version: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedSignal:
    """Final fused signal with explainability"""
    symbol: str
    signal_type: str
    value: float
    confidence: float
    top_3_reasons: List[Reason]
    timestamp: datetime
    model_version: str
    contributing_agents: List[str]
    conflict_resolution: Optional[str] = None
    fibonacci_levels: Optional[Dict[str, float]] = None
    cross_market_arbitrage: Optional[Dict[str, Any]] = None


@dataclass
class PortfolioState:
    """LangGraph state for portfolio allocation"""
    raw_signals: Dict[str, List[Signal]] = field(default_factory=dict)
    normalized_signals: Dict[str, Signal] = field(default_factory=dict)
    weighted_signals: Dict[str, Signal] = field(default_factory=dict)
    resolved_signals: Dict[str, Signal] = field(default_factory=dict)
    fused_signals: Dict[str, FusedSignal] = field(default_factory=dict)
    market_regime: Optional[MarketRegime] = None
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class ExplainabilityEngine:
    """Engine for generating explainable AI outputs"""
    
    def __init__(self):
        self.factor_importance_weights = {
            'technical_confluence': 0.3,
            'sentiment_alignment': 0.25,
            'fibonacci_confluence': 0.2,
            'volume_confirmation': 0.15,
            'risk_reward_ratio': 0.1
        }
    
    def generate_top_3_reasons(self, signal: Signal, **context) -> List[Reason]:
        """Generate top 3 reasons for any trading decision"""
        reasons = []
        
        # Analyze signal components by importance
        components = self._analyze_signal_components(signal, context)
        
        # Rank by contribution to final decision
        ranked_components = sorted(components, key=lambda x: x['importance'], reverse=True)
        
        # Generate human-readable explanations
        for i, component in enumerate(ranked_components[:3]):
            reason = Reason(
                rank=i+1,
                factor=component['name'],
                contribution=component['importance'],
                explanation=self._generate_explanation(component),
                confidence=component['confidence'],
                supporting_data=component['data']
            )
            reasons.append(reason)
        
        return reasons
    
    def _analyze_signal_components(self, signal: Signal, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze individual components of a signal"""
        components = []
        
        # Technical confluence analysis
        if 'technical_indicators' in signal.metadata:
            tech_data = signal.metadata['technical_indicators']
            confluence_score = self._calculate_technical_confluence(tech_data)
            components.append({
                'name': 'technical_confluence',
                'importance': confluence_score * self.factor_importance_weights['technical_confluence'],
                'confidence': confluence_score,
                'data': tech_data
            })
        
        # Sentiment alignment analysis
        if 'sentiment_score' in signal.metadata:
            sentiment_data = signal.metadata['sentiment_score']
            alignment_score = self._calculate_sentiment_alignment(signal.value, sentiment_data)
            components.append({
                'name': 'sentiment_alignment',
                'importance': alignment_score * self.factor_importance_weights['sentiment_alignment'],
                'confidence': alignment_score,
                'data': {'sentiment_score': sentiment_data, 'signal_direction': signal.value}
            })
        
        # Fibonacci confluence analysis
        if 'fibonacci_levels' in signal.metadata:
            fib_data = signal.metadata['fibonacci_levels']
            fib_confluence = self._calculate_fibonacci_confluence(fib_data, context.get('current_price', 0))
            components.append({
                'name': 'fibonacci_confluence',
                'importance': fib_confluence * self.factor_importance_weights['fibonacci_confluence'],
                'confidence': fib_confluence,
                'data': fib_data
            })
        
        # Volume confirmation analysis
        if 'volume_data' in signal.metadata:
            volume_data = signal.metadata['volume_data']
            volume_confirmation = self._calculate_volume_confirmation(volume_data)
            components.append({
                'name': 'volume_confirmation',
                'importance': volume_confirmation * self.factor_importance_weights['volume_confirmation'],
                'confidence': volume_confirmation,
                'data': volume_data
            })
        
        # Risk-reward ratio analysis
        if 'risk_reward' in signal.metadata:
            rr_data = signal.metadata['risk_reward']
            rr_score = min(rr_data.get('ratio', 1.0) / 3.0, 1.0)  # Normalize to [0,1]
            components.append({
                'name': 'risk_reward_ratio',
                'importance': rr_score * self.factor_importance_weights['risk_reward_ratio'],
                'confidence': rr_score,
                'data': rr_data
            })
        
        return components
    
    def _calculate_technical_confluence(self, tech_data: Dict[str, Any]) -> float:
        """Calculate technical indicator confluence score"""
        indicators = tech_data.get('indicators', {})
        if not indicators:
            return 0.0
        
        # Count aligned indicators
        aligned_count = 0
        total_count = 0
        
        for indicator, value in indicators.items():
            if isinstance(value, dict) and 'signal' in value:
                total_count += 1
                if abs(value['signal']) > 0.5:  # Strong signal threshold
                    aligned_count += 1
        
        return aligned_count / max(total_count, 1)
    
    def _calculate_sentiment_alignment(self, signal_value: float, sentiment_score: float) -> float:
        """Calculate sentiment alignment with signal direction"""
        if signal_value == 0 or sentiment_score == 0:
            return 0.0
        
        # Check if sentiment and signal are aligned
        alignment = np.sign(signal_value) == np.sign(sentiment_score)
        strength = min(abs(signal_value), abs(sentiment_score))
        
        return strength if alignment else 0.0
    
    def _calculate_fibonacci_confluence(self, fib_data: Dict[str, Any], current_price: float) -> float:
        """Calculate Fibonacci level confluence"""
        if not fib_data or current_price == 0:
            return 0.0
        
        fib_levels = fib_data.get('levels', {})
        if not fib_levels:
            return 0.0
        
        # Find closest Fibonacci level
        min_distance = float('inf')
        for level_name, level_price in fib_levels.items():
            distance = abs(current_price - level_price) / current_price
            min_distance = min(min_distance, distance)
        
        # Convert distance to confluence score (closer = higher score)
        confluence_score = max(0, 1 - (min_distance * 20))  # 5% distance = 0 score
        return confluence_score
    
    def _calculate_volume_confirmation(self, volume_data: Dict[str, Any]) -> float:
        """Calculate volume confirmation score"""
        current_volume = volume_data.get('current_volume', 0)
        avg_volume = volume_data.get('average_volume', 1)
        
        if avg_volume == 0:
            return 0.0
        
        volume_ratio = current_volume / avg_volume
        # Higher volume = higher confirmation, capped at 2x average
        return min(volume_ratio / 2.0, 1.0)
    
    def _generate_explanation(self, component: Dict[str, Any]) -> str:
        """Generate human-readable explanation for a component"""
        name = component['name']
        importance = component['importance']
        data = component['data']
        
        if name == 'technical_confluence':
            indicators = data.get('indicators', {})
            aligned_indicators = [k for k, v in indicators.items() 
                                if isinstance(v, dict) and abs(v.get('signal', 0)) > 0.5]
            return f"Multiple technical indicators ({', '.join(aligned_indicators)}) are aligned, " \
                   f"providing {importance:.1%} confidence in the signal direction."
        
        elif name == 'sentiment_alignment':
            sentiment = data.get('sentiment_score', 0)
            direction = "bullish" if data.get('signal_direction', 0) > 0 else "bearish"
            return f"Market sentiment ({sentiment:.2f}) strongly aligns with {direction} signal, " \
                   f"contributing {importance:.1%} to decision confidence."
        
        elif name == 'fibonacci_confluence':
            return f"Current price is near key Fibonacci levels, providing {importance:.1%} " \
                   f"technical confluence for the trade setup."
        
        elif name == 'volume_confirmation':
            volume_ratio = data.get('current_volume', 0) / max(data.get('average_volume', 1), 1)
            return f"Trading volume is {volume_ratio:.1f}x average, confirming {importance:.1%} " \
                   f"of the signal strength."
        
        elif name == 'risk_reward_ratio':
            ratio = data.get('ratio', 1.0)
            return f"Risk-reward ratio of {ratio:.1f}:1 provides {importance:.1%} " \
                   f"favorable trade economics."
        
        return f"{name} contributes {importance:.1%} to the trading decision."


class ConflictResolver:
    """Resolves conflicts between contradictory signals"""
    
    def __init__(self):
        self.resolution_strategies = {
            'weighted_average': self._weighted_average_resolution,
            'confidence_based': self._confidence_based_resolution,
            'expert_override': self._expert_override_resolution,
            'regime_based': self._regime_based_resolution
        }
    
    def detect_conflicts(self, signals: Dict[str, Signal]) -> List[Dict[str, Any]]:
        """Detect conflicting signals for the same symbol"""
        conflicts = []
        symbol_signals = {}
        
        # Group signals by symbol
        for signal_id, signal in signals.items():
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = []
            symbol_signals[signal.symbol].append((signal_id, signal))
        
        # Check for conflicts within each symbol
        for symbol, signal_list in symbol_signals.items():
            if len(signal_list) < 2:
                continue
            
            # Check for opposing signals
            for i, (id1, signal1) in enumerate(signal_list):
                for j, (id2, signal2) in enumerate(signal_list[i+1:], i+1):
                    if self._are_conflicting(signal1, signal2):
                        conflicts.append({
                            'symbol': symbol,
                            'signal_ids': [id1, id2],
                            'signals': [signal1, signal2],
                            'conflict_type': self._classify_conflict(signal1, signal2)
                        })
        
        return conflicts
    
    def resolve_conflict(self, conflict: Dict[str, Any]) -> Signal:
        """Resolve a specific conflict using appropriate strategy"""
        conflict_type = conflict['conflict_type']
        signals = conflict['signals']
        
        # Choose resolution strategy based on conflict type
        if conflict_type == 'directional_opposite':
            strategy = 'confidence_based'
        elif conflict_type == 'magnitude_difference':
            strategy = 'weighted_average'
        elif conflict_type == 'timeframe_mismatch':
            strategy = 'regime_based'
        else:
            strategy = 'weighted_average'
        
        resolver = self.resolution_strategies[strategy]
        resolved_signal = resolver(signals)
        resolved_signal.metadata['conflict_resolution'] = strategy
        
        return resolved_signal
    
    def _are_conflicting(self, signal1: Signal, signal2: Signal) -> bool:
        """Check if two signals are conflicting"""
        # Opposite directions with high confidence
        if (np.sign(signal1.value) != np.sign(signal2.value) and 
            signal1.confidence > 0.7 and signal2.confidence > 0.7):
            return True
        
        # Same direction but very different magnitudes
        if (np.sign(signal1.value) == np.sign(signal2.value) and
            abs(signal1.value - signal2.value) > 0.5):
            return True
        
        return False
    
    def _classify_conflict(self, signal1: Signal, signal2: Signal) -> str:
        """Classify the type of conflict"""
        if np.sign(signal1.value) != np.sign(signal2.value):
            return 'directional_opposite'
        elif abs(signal1.value - signal2.value) > 0.5:
            return 'magnitude_difference'
        elif signal1.signal_type != signal2.signal_type:
            return 'strategy_mismatch'
        else:
            return 'timeframe_mismatch'
    
    def _weighted_average_resolution(self, signals: List[Signal]) -> Signal:
        """Resolve conflict using confidence-weighted average"""
        total_weight = sum(s.confidence for s in signals)
        if total_weight == 0:
            return signals[0]  # Fallback to first signal
        
        weighted_value = sum(s.value * s.confidence for s in signals) / total_weight
        avg_confidence = total_weight / len(signals)
        
        # Create resolved signal
        resolved = Signal(
            symbol=signals[0].symbol,
            signal_type=signals[0].signal_type,
            value=weighted_value,
            confidence=avg_confidence,
            timestamp=datetime.now(timezone.utc),
            agent_name='portfolio_allocator',
            model_version='1.0.0',
            metadata={
                'resolution_method': 'weighted_average',
                'original_signals': len(signals)
            }
        )
        
        return resolved
    
    def _confidence_based_resolution(self, signals: List[Signal]) -> Signal:
        """Resolve conflict by selecting highest confidence signal"""
        best_signal = max(signals, key=lambda s: s.confidence)
        
        # Create resolved signal based on best
        resolved = Signal(
            symbol=best_signal.symbol,
            signal_type=best_signal.signal_type,
            value=best_signal.value,
            confidence=best_signal.confidence * 0.9,  # Slight confidence penalty for conflict
            timestamp=datetime.now(timezone.utc),
            agent_name='portfolio_allocator',
            model_version='1.0.0',
            metadata={
                'resolution_method': 'confidence_based',
                'selected_agent': best_signal.agent_name,
                'original_signals': len(signals)
            }
        )
        
        return resolved
    
    def _expert_override_resolution(self, signals: List[Signal]) -> Signal:
        """Resolve conflict using expert system rules"""
        # Priority order for different signal types
        priority_order = [
            SignalType.SENTIMENT,  # News-driven signals get highest priority
            SignalType.MOMENTUM,
            SignalType.OPTIONS_VOLATILITY,
            SignalType.MEAN_REVERSION,
            SignalType.SHORT_SELLING,
            SignalType.LONG_TERM_CORE
        ]
        
        # Find highest priority signal
        for signal_type in priority_order:
            for signal in signals:
                if signal.signal_type == signal_type:
                    resolved = Signal(
                        symbol=signal.symbol,
                        signal_type=signal.signal_type,
                        value=signal.value,
                        confidence=signal.confidence * 0.95,  # Small penalty for override
                        timestamp=datetime.now(timezone.utc),
                        agent_name='portfolio_allocator',
                        model_version='1.0.0',
                        metadata={
                            'resolution_method': 'expert_override',
                            'priority_reason': f'{signal_type.value}_priority',
                            'original_signals': len(signals)
                        }
                    )
                    return resolved
        
        # Fallback to first signal
        return signals[0]
    
    def _regime_based_resolution(self, signals: List[Signal]) -> Signal:
        """Resolve conflict based on current market regime"""
        # This would integrate with regime detection
        # For now, use weighted average as fallback
        return self._weighted_average_resolution(signals)


class RegimeDetector:
    """Detects current market regime for strategy weighting"""
    
    def __init__(self):
        self.regime_weights = {
            MarketRegime.TRENDING_UP: {
                SignalType.MOMENTUM: 0.4,
                SignalType.MEAN_REVERSION: 0.1,
                SignalType.OPTIONS_VOLATILITY: 0.2,
                SignalType.SHORT_SELLING: 0.05,
                SignalType.LONG_TERM_CORE: 0.25
            },
            MarketRegime.TRENDING_DOWN: {
                SignalType.MOMENTUM: 0.3,
                SignalType.MEAN_REVERSION: 0.15,
                SignalType.OPTIONS_VOLATILITY: 0.25,
                SignalType.SHORT_SELLING: 0.3,
                SignalType.LONG_TERM_CORE: 0.0
            },
            MarketRegime.MEAN_REVERTING: {
                SignalType.MOMENTUM: 0.1,
                SignalType.MEAN_REVERSION: 0.5,
                SignalType.OPTIONS_VOLATILITY: 0.2,
                SignalType.SHORT_SELLING: 0.1,
                SignalType.LONG_TERM_CORE: 0.1
            },
            MarketRegime.HIGH_VOLATILITY: {
                SignalType.MOMENTUM: 0.2,
                SignalType.MEAN_REVERSION: 0.2,
                SignalType.OPTIONS_VOLATILITY: 0.4,
                SignalType.SHORT_SELLING: 0.15,
                SignalType.LONG_TERM_CORE: 0.05
            },
            MarketRegime.LOW_VOLATILITY: {
                SignalType.MOMENTUM: 0.25,
                SignalType.MEAN_REVERSION: 0.25,
                SignalType.OPTIONS_VOLATILITY: 0.1,
                SignalType.SHORT_SELLING: 0.1,
                SignalType.LONG_TERM_CORE: 0.3
            }
        }
    
    def detect_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """Detect current market regime from market data"""
        # Simplified regime detection - in production this would be more sophisticated
        volatility = market_data.get('volatility', 0.2)
        trend_strength = market_data.get('trend_strength', 0.0)
        volume_ratio = market_data.get('volume_ratio', 1.0)
        
        # High volatility regime
        if volatility > 0.3:
            return MarketRegime.HIGH_VOLATILITY
        
        # Low volatility regime
        if volatility < 0.1:
            return MarketRegime.LOW_VOLATILITY
        
        # Trending regimes
        if abs(trend_strength) > 0.6:
            return MarketRegime.TRENDING_UP if trend_strength > 0 else MarketRegime.TRENDING_DOWN
        
        # Mean reverting regime (default)
        return MarketRegime.MEAN_REVERTING
    
    def get_regime_weights(self, regime: MarketRegime) -> Dict[SignalType, float]:
        """Get strategy weights for current regime"""
        return self.regime_weights.get(regime, self.regime_weights[MarketRegime.MEAN_REVERTING])


class PortfolioAllocatorAgent:
    """Main Portfolio Allocator Agent using LangGraph"""
    
    def __init__(self):
        self.explainability_engine = ExplainabilityEngine()
        self.conflict_resolver = ConflictResolver()
        self.regime_detector = RegimeDetector()
        self.model_version = "1.0.0"
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for signal fusion"""
        from typing import TypedDict
        
        # Define state as TypedDict for LangGraph
        class WorkflowState(TypedDict):
            raw_signals: Dict[str, List[Signal]]
            normalized_signals: Dict[str, Signal]
            weighted_signals: Dict[str, Signal]
            resolved_signals: Dict[str, Signal]
            fused_signals: Dict[str, FusedSignal]
            market_regime: Optional[MarketRegime]
            conflicts: List[Dict[str, Any]]
            errors: List[str]
            market_data: Dict[str, Any]
        
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each step
        workflow.add_node("normalize_signals", self._normalize_signals_workflow)
        workflow.add_node("detect_regime", self._detect_regime_workflow)
        workflow.add_node("apply_weights", self._apply_regime_weights_workflow)
        workflow.add_node("resolve_conflicts", self._resolve_conflicts_workflow)
        workflow.add_node("fuse_signals", self._fuse_signals_workflow)
        workflow.add_node("generate_explanations", self._generate_explanations_workflow)
        
        # Define workflow edges
        workflow.add_edge("normalize_signals", "detect_regime")
        workflow.add_edge("detect_regime", "apply_weights")
        workflow.add_edge("apply_weights", "resolve_conflicts")
        workflow.add_edge("resolve_conflicts", "fuse_signals")
        workflow.add_edge("fuse_signals", "generate_explanations")
        workflow.add_edge("generate_explanations", END)
        
        # Set entry point
        workflow.set_entry_point("normalize_signals")
        
        return workflow.compile()

    async def get_rebalancing_recommendations(self, current_portfolio: List[str], target_allocation: str = 'moderate_growth') -> Dict[str, Any]:
        """
        Get portfolio rebalancing recommendations.

        Args:
            current_portfolio: List of current symbols in portfolio
            target_allocation: Target allocation strategy

        Returns:
            Dictionary with rebalancing recommendations
        """
        try:
            # Simple rebalancing logic based on portfolio size and allocation
            recommendations = {
                'should_rebalance': False,
                'recommendation': 'Portfolio is balanced',
                'confidence': 0.7,
                'target_allocation': target_allocation
            }

            portfolio_size = len(current_portfolio)

            # Check if rebalancing is needed based on portfolio composition
            if target_allocation == 'moderate_growth':
                if portfolio_size > 8:
                    recommendations.update({
                        'should_rebalance': True,
                        'recommendation': 'Portfolio too diversified - consolidate top performers',
                        'confidence': 0.85
                    })
                elif portfolio_size < 3:
                    recommendations.update({
                        'should_rebalance': True,
                        'recommendation': 'Portfolio under-diversified - add positions',
                        'confidence': 0.8
                    })

            return recommendations

        except Exception as e:
            logger.error(f"Rebalancing recommendation error: {e}")
            return {
                'should_rebalance': False,
                'recommendation': 'Error in analysis',
                'confidence': 0.5
            }

    async def process_signals(self, raw_signals: Dict[str, List[Signal]], 
                            market_data: Dict[str, Any]) -> Dict[str, FusedSignal]:
        """Process raw signals through the complete fusion pipeline"""
        try:
            # Initialize state
            initial_state = {
                'raw_signals': raw_signals,
                'normalized_signals': {},
                'weighted_signals': {},
                'resolved_signals': {},
                'fused_signals': {},
                'market_regime': None,
                'conflicts': [],
                'errors': [],
                'market_data': market_data
            }
            
            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            return final_state.get('fused_signals', {})
            
        except Exception as e:
            logger.error(f"Error processing signals: {str(e)}")
            return {}
    
    def _normalize_signals(self, state: PortfolioState) -> PortfolioState:
        """Normalize signals from different agents to common scale"""
        normalized = {}
        
        for symbol, signal_list in state.raw_signals.items():
            for signal in signal_list:
                # Ensure signal value is in [-1, 1] range
                normalized_value = np.clip(signal.value, -1.0, 1.0)
                
                # Adjust confidence based on agent historical performance
                # (In production, this would use actual performance data)
                performance_multiplier = self._get_agent_performance_multiplier(signal.agent_name)
                adjusted_confidence = min(signal.confidence * performance_multiplier, 1.0)
                
                # Create normalized signal
                normalized_signal = Signal(
                    symbol=signal.symbol,
                    signal_type=signal.signal_type,
                    value=normalized_value,
                    confidence=adjusted_confidence,
                    timestamp=signal.timestamp,
                    agent_name=signal.agent_name,
                    model_version=signal.model_version,
                    metadata=signal.metadata.copy()
                )
                
                signal_key = f"{symbol}_{signal.agent_name}_{signal.signal_type.value}"
                normalized[signal_key] = normalized_signal
        
        state.normalized_signals = normalized
        return state
    
    def _detect_regime(self, state: PortfolioState) -> PortfolioState:
        """Detect current market regime"""
        market_data = getattr(state, 'market_data', {})
        regime = self.regime_detector.detect_regime(market_data)
        state.market_regime = regime
        
        logger.info(f"Detected market regime: {regime.value}")
        return state
    
    def _apply_regime_weights(self, state: PortfolioState) -> PortfolioState:
        """Apply regime-based strategy weights"""
        if not state.market_regime:
            state.market_regime = MarketRegime.MEAN_REVERTING
        
        regime_weights = self.regime_detector.get_regime_weights(state.market_regime)
        weighted_signals = {}
        
        for signal_key, signal in state.normalized_signals.items():
            weight = regime_weights.get(signal.signal_type, 1.0)
            
            weighted_signal = Signal(
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                value=signal.value * weight,
                confidence=signal.confidence,
                timestamp=signal.timestamp,
                agent_name=signal.agent_name,
                model_version=signal.model_version,
                metadata={
                    **signal.metadata,
                    'regime_weight': weight,
                    'regime': state.market_regime.value
                }
            )
            
            weighted_signals[signal_key] = weighted_signal
        
        state.weighted_signals = weighted_signals
        return state
    
    def _resolve_conflicts(self, state: PortfolioState) -> PortfolioState:
        """Resolve conflicting signals"""
        conflicts = self.conflict_resolver.detect_conflicts(state.weighted_signals)
        state.conflicts = conflicts
        
        resolved_signals = state.weighted_signals.copy()
        
        for conflict in conflicts:
            try:
                resolved_signal = self.conflict_resolver.resolve_conflict(conflict)
                
                # Remove conflicting signals and add resolved one
                for signal_id in conflict['signal_ids']:
                    if signal_id in resolved_signals:
                        del resolved_signals[signal_id]
                
                # Add resolved signal
                resolved_key = f"{resolved_signal.symbol}_resolved_{len(state.conflicts)}"
                resolved_signals[resolved_key] = resolved_signal
                
                logger.info(f"Resolved conflict for {conflict['symbol']}: {conflict['conflict_type']}")
                
            except Exception as e:
                logger.error(f"Error resolving conflict: {str(e)}")
                state.errors.append(f"Conflict resolution error: {str(e)}")
        
        state.resolved_signals = resolved_signals
        return state
    
    def _fuse_signals(self, state: PortfolioState) -> PortfolioState:
        """Fuse resolved signals into final trading signals"""
        fused_signals = {}
        
        # Group signals by symbol
        symbol_signals = {}
        for signal_key, signal in state.resolved_signals.items():
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = []
            symbol_signals[signal.symbol].append(signal)
        
        # Fuse signals for each symbol
        for symbol, signals in symbol_signals.items():
            if not signals:
                continue
            
            # Calculate weighted average of all signals for this symbol
            total_weight = sum(s.confidence for s in signals)
            if total_weight == 0:
                continue
            
            fused_value = sum(s.value * s.confidence for s in signals) / total_weight
            fused_confidence = total_weight / len(signals)
            
            # Create fused signal
            fused_signal = FusedSignal(
                symbol=symbol,
                signal_type="fused",
                value=fused_value,
                confidence=fused_confidence,
                top_3_reasons=[],  # Will be filled in next step
                timestamp=datetime.now(timezone.utc),
                model_version=self.model_version,
                contributing_agents=[s.agent_name for s in signals]
            )
            
            fused_signals[symbol] = fused_signal
        
        state.fused_signals = fused_signals
        return state
    
    def _generate_explanations(self, state: PortfolioState) -> PortfolioState:
        """Generate explainable output for all fused signals"""
        for symbol, fused_signal in state.fused_signals.items():
            # Find contributing signals for this symbol
            contributing_signals = [
                s for s in state.resolved_signals.values() 
                if s.symbol == symbol
            ]
            
            if not contributing_signals:
                continue
            
            # Generate explanations based on strongest contributing signal
            strongest_signal = max(contributing_signals, key=lambda s: s.confidence)
            
            # Generate top 3 reasons
            top_3_reasons = self.explainability_engine.generate_top_3_reasons(
                strongest_signal,
                market_regime=state.market_regime,
                contributing_signals=contributing_signals,
                current_price=100.0  # Would be actual price in production
            )
            
            # Update fused signal with explanations
            fused_signal.top_3_reasons = top_3_reasons
            
            # Add conflict resolution info if applicable
            if any(c['symbol'] == symbol for c in state.conflicts):
                conflict = next(c for c in state.conflicts if c['symbol'] == symbol)
                fused_signal.conflict_resolution = conflict['conflict_type']
        
        return state
    
    def _get_agent_performance_multiplier(self, agent_name: str) -> float:
        """Get performance multiplier for agent (simplified)"""
        # In production, this would use actual historical performance data
        performance_multipliers = {
            'momentum_agent': 1.1,
            'mean_reversion_agent': 1.0,
            'options_volatility_agent': 0.95,
            'sentiment_agent': 1.05,
            'short_selling_agent': 0.9,
            'long_term_core_agent': 1.0
        }
        
        return performance_multipliers.get(agent_name, 1.0)
    
    # Workflow-compatible methods that work with TypedDict state
    def _normalize_signals_workflow(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize signals - workflow compatible version"""
        portfolio_state = PortfolioState(raw_signals=state['raw_signals'])
        normalized_state = self._normalize_signals(portfolio_state)
        state['normalized_signals'] = normalized_state.normalized_signals
        return state
    
    def _detect_regime_workflow(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect regime - workflow compatible version"""
        portfolio_state = PortfolioState()
        portfolio_state.market_data = state['market_data']
        regime_state = self._detect_regime(portfolio_state)
        state['market_regime'] = regime_state.market_regime
        return state
    
    def _apply_regime_weights_workflow(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply regime weights - workflow compatible version"""
        portfolio_state = PortfolioState()
        portfolio_state.market_regime = state['market_regime']
        portfolio_state.normalized_signals = state['normalized_signals']
        weighted_state = self._apply_regime_weights(portfolio_state)
        state['weighted_signals'] = weighted_state.weighted_signals
        return state
    
    def _resolve_conflicts_workflow(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflicts - workflow compatible version"""
        portfolio_state = PortfolioState()
        portfolio_state.weighted_signals = state['weighted_signals']
        resolved_state = self._resolve_conflicts(portfolio_state)
        state['resolved_signals'] = resolved_state.resolved_signals
        state['conflicts'] = resolved_state.conflicts
        state['errors'] = resolved_state.errors
        return state
    
    def _fuse_signals_workflow(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse signals - workflow compatible version"""
        portfolio_state = PortfolioState()
        portfolio_state.resolved_signals = state['resolved_signals']
        fused_state = self._fuse_signals(portfolio_state)
        state['fused_signals'] = fused_state.fused_signals
        return state
    
    def _generate_explanations_workflow(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanations - workflow compatible version"""
        portfolio_state = PortfolioState()
        portfolio_state.fused_signals = state['fused_signals']
        portfolio_state.resolved_signals = state['resolved_signals']
        portfolio_state.conflicts = state['conflicts']
        portfolio_state.market_regime = state['market_regime']
        
        explained_state = self._generate_explanations(portfolio_state)
        state['fused_signals'] = explained_state.fused_signals
        return state


# Example usage and testing functions
async def test_portfolio_allocator():
    """Test the Portfolio Allocator Agent"""
    allocator = PortfolioAllocatorAgent()
    
    # Create sample signals
    sample_signals = {
        "AAPL": [
            Signal(
                symbol="AAPL",
                signal_type=SignalType.MOMENTUM,
                value=0.7,
                confidence=0.8,
                timestamp=datetime.now(timezone.utc),
                agent_name="momentum_agent",
                model_version="1.0.0",
                metadata={
                    'technical_indicators': {
                        'indicators': {
                            'ema_crossover': {'signal': 0.8},
                            'rsi_breakout': {'signal': 0.6},
                            'macd': {'signal': 0.7}
                        }
                    },
                    'sentiment_score': 0.6,
                    'volume_data': {
                        'current_volume': 2000000,
                        'average_volume': 1500000
                    }
                }
            ),
            Signal(
                symbol="AAPL",
                signal_type=SignalType.MEAN_REVERSION,
                value=-0.3,
                confidence=0.6,
                timestamp=datetime.now(timezone.utc),
                agent_name="mean_reversion_agent",
                model_version="1.0.0",
                metadata={
                    'bollinger_bands': {'signal': -0.4},
                    'z_score': -1.2
                }
            )
        ]
    }
    
    # Sample market data
    market_data = {
        'volatility': 0.25,
        'trend_strength': 0.4,
        'volume_ratio': 1.3
    }
    
    # Process signals
    fused_signals = await allocator.process_signals(sample_signals, market_data)
    
    # Print results
    for symbol, signal in fused_signals.items():
        print(f"\n=== {symbol} ===")
        print(f"Signal: {signal.value:.3f} (confidence: {signal.confidence:.3f})")
        print(f"Contributing agents: {', '.join(signal.contributing_agents)}")
        print("Top 3 reasons:")
        for reason in signal.top_3_reasons:
            print(f"  {reason.rank}. {reason.factor}: {reason.explanation}")


if __name__ == "__main__":
    asyncio.run(test_portfolio_allocator())

# Create singleton instance
portfolio_allocator_agent = PortfolioAllocatorAgent()