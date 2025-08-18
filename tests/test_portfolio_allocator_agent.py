"""
Tests for Portfolio Allocator Agent

This module tests the signal fusion, conflict resolution, explainability engine,
and regime-based strategy weighting functionality.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import numpy as np

from agents.portfolio_allocator_agent import (
    PortfolioAllocatorAgent,
    ExplainabilityEngine,
    ConflictResolver,
    RegimeDetector,
    Signal,
    FusedSignal,
    SignalType,
    MarketRegime,
    PortfolioState,
    Reason
)


class TestExplainabilityEngine:
    """Test the explainability engine"""
    
    def setup_method(self):
        self.engine = ExplainabilityEngine()
    
    def test_generate_top_3_reasons_basic(self):
        """Test basic reason generation"""
        signal = Signal(
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
                        'rsi_breakout': {'signal': 0.6}
                    }
                },
                'sentiment_score': 0.6,
                'volume_data': {
                    'current_volume': 2000000,
                    'average_volume': 1500000
                }
            }
        )
        
        reasons = self.engine.generate_top_3_reasons(signal, current_price=150.0)
        
        assert len(reasons) <= 3
        assert all(isinstance(r, Reason) for r in reasons)
        assert all(r.rank in [1, 2, 3] for r in reasons)
        assert all(0 <= r.confidence <= 1 for r in reasons)
        assert all(r.explanation for r in reasons)
    
    def test_technical_confluence_calculation(self):
        """Test technical confluence scoring"""
        tech_data = {
            'indicators': {
                'ema_crossover': {'signal': 0.8},
                'rsi_breakout': {'signal': 0.7},
                'macd': {'signal': 0.6},
                'bollinger': {'signal': 0.2}  # Weak signal
            }
        }
        
        score = self.engine._calculate_technical_confluence(tech_data)
        
        # 3 out of 4 indicators are strong (>0.5)
        assert score == 0.75
    
    def test_sentiment_alignment_calculation(self):
        """Test sentiment alignment scoring"""
        # Aligned case
        aligned_score = self.engine._calculate_sentiment_alignment(0.7, 0.6)
        assert aligned_score == 0.6  # Min of the two values
        
        # Misaligned case
        misaligned_score = self.engine._calculate_sentiment_alignment(0.7, -0.6)
        assert misaligned_score == 0.0
        
        # Zero cases
        zero_score = self.engine._calculate_sentiment_alignment(0.0, 0.6)
        assert zero_score == 0.0
    
    def test_fibonacci_confluence_calculation(self):
        """Test Fibonacci confluence scoring"""
        fib_data = {
            'levels': {
                'fib_382': 148.0,
                'fib_618': 152.0,
                'fib_500': 150.0
            }
        }
        
        # Price near Fibonacci level
        near_score = self.engine._calculate_fibonacci_confluence(fib_data, 150.5)
        assert near_score > 0.8  # Should be high confluence
        
        # Price far from Fibonacci levels
        far_score = self.engine._calculate_fibonacci_confluence(fib_data, 200.0)
        assert far_score < 0.2  # Should be low confluence
    
    def test_volume_confirmation_calculation(self):
        """Test volume confirmation scoring"""
        volume_data = {
            'current_volume': 2000000,
            'average_volume': 1000000
        }
        
        score = self.engine._calculate_volume_confirmation(volume_data)
        assert score == 1.0  # 2x volume = max score
        
        # Low volume case
        low_volume_data = {
            'current_volume': 500000,
            'average_volume': 1000000
        }
        
        low_score = self.engine._calculate_volume_confirmation(low_volume_data)
        assert low_score == 0.25  # 0.5x volume


class TestConflictResolver:
    """Test the conflict resolution system"""
    
    def setup_method(self):
        self.resolver = ConflictResolver()
    
    def test_detect_conflicts_opposing_signals(self):
        """Test detection of opposing signals"""
        signals = {
            'signal1': Signal(
                symbol="AAPL",
                signal_type=SignalType.MOMENTUM,
                value=0.8,
                confidence=0.9,
                timestamp=datetime.now(timezone.utc),
                agent_name="momentum_agent",
                model_version="1.0.0"
            ),
            'signal2': Signal(
                symbol="AAPL",
                signal_type=SignalType.MEAN_REVERSION,
                value=-0.7,
                confidence=0.8,
                timestamp=datetime.now(timezone.utc),
                agent_name="mean_reversion_agent",
                model_version="1.0.0"
            )
        }
        
        conflicts = self.resolver.detect_conflicts(signals)
        
        assert len(conflicts) == 1
        assert conflicts[0]['symbol'] == "AAPL"
        assert conflicts[0]['conflict_type'] == 'directional_opposite'
    
    def test_detect_conflicts_magnitude_difference(self):
        """Test detection of magnitude differences"""
        signals = {
            'signal1': Signal(
                symbol="AAPL",
                signal_type=SignalType.MOMENTUM,
                value=0.9,
                confidence=0.8,
                timestamp=datetime.now(timezone.utc),
                agent_name="momentum_agent",
                model_version="1.0.0"
            ),
            'signal2': Signal(
                symbol="AAPL",
                signal_type=SignalType.MOMENTUM,
                value=0.2,
                confidence=0.7,
                timestamp=datetime.now(timezone.utc),
                agent_name="momentum_agent_2",
                model_version="1.0.0"
            )
        }
        
        conflicts = self.resolver.detect_conflicts(signals)
        
        assert len(conflicts) == 1
        assert conflicts[0]['conflict_type'] == 'magnitude_difference'
    
    def test_weighted_average_resolution(self):
        """Test weighted average conflict resolution"""
        signals = [
            Signal(
                symbol="AAPL",
                signal_type=SignalType.MOMENTUM,
                value=0.8,
                confidence=0.9,
                timestamp=datetime.now(timezone.utc),
                agent_name="agent1",
                model_version="1.0.0"
            ),
            Signal(
                symbol="AAPL",
                signal_type=SignalType.MOMENTUM,
                value=0.4,
                confidence=0.6,
                timestamp=datetime.now(timezone.utc),
                agent_name="agent2",
                model_version="1.0.0"
            )
        ]
        
        resolved = self.resolver._weighted_average_resolution(signals)
        
        # Expected: (0.8 * 0.9 + 0.4 * 0.6) / (0.9 + 0.6) = 0.64
        expected_value = (0.8 * 0.9 + 0.4 * 0.6) / (0.9 + 0.6)
        assert abs(resolved.value - expected_value) < 0.01
        assert resolved.confidence == (0.9 + 0.6) / 2  # Average confidence
    
    def test_confidence_based_resolution(self):
        """Test confidence-based conflict resolution"""
        signals = [
            Signal(
                symbol="AAPL",
                signal_type=SignalType.MOMENTUM,
                value=0.8,
                confidence=0.9,
                timestamp=datetime.now(timezone.utc),
                agent_name="high_confidence_agent",
                model_version="1.0.0"
            ),
            Signal(
                symbol="AAPL",
                signal_type=SignalType.MOMENTUM,
                value=0.4,
                confidence=0.6,
                timestamp=datetime.now(timezone.utc),
                agent_name="low_confidence_agent",
                model_version="1.0.0"
            )
        ]
        
        resolved = self.resolver._confidence_based_resolution(signals)
        
        # Should select the high confidence signal
        assert resolved.value == 0.8
        assert resolved.confidence == 0.9 * 0.9  # With penalty
        assert resolved.metadata['selected_agent'] == 'high_confidence_agent'
    
    def test_expert_override_resolution(self):
        """Test expert system override resolution"""
        signals = [
            Signal(
                symbol="AAPL",
                signal_type=SignalType.LONG_TERM_CORE,
                value=0.8,
                confidence=0.9,
                timestamp=datetime.now(timezone.utc),
                agent_name="long_term_agent",
                model_version="1.0.0"
            ),
            Signal(
                symbol="AAPL",
                signal_type=SignalType.SENTIMENT,
                value=0.4,
                confidence=0.6,
                timestamp=datetime.now(timezone.utc),
                agent_name="sentiment_agent",
                model_version="1.0.0"
            )
        ]
        
        resolved = self.resolver._expert_override_resolution(signals)
        
        # Sentiment should win due to higher priority
        assert resolved.value == 0.4
        assert resolved.metadata['priority_reason'] == 'sentiment_priority'


class TestRegimeDetector:
    """Test the market regime detection system"""
    
    def setup_method(self):
        self.detector = RegimeDetector()
    
    def test_high_volatility_detection(self):
        """Test high volatility regime detection"""
        market_data = {
            'volatility': 0.35,
            'trend_strength': 0.2,
            'volume_ratio': 1.5
        }
        
        regime = self.detector.detect_regime(market_data)
        assert regime == MarketRegime.HIGH_VOLATILITY
    
    def test_low_volatility_detection(self):
        """Test low volatility regime detection"""
        market_data = {
            'volatility': 0.08,
            'trend_strength': 0.1,
            'volume_ratio': 0.9
        }
        
        regime = self.detector.detect_regime(market_data)
        assert regime == MarketRegime.LOW_VOLATILITY
    
    def test_trending_up_detection(self):
        """Test trending up regime detection"""
        market_data = {
            'volatility': 0.2,
            'trend_strength': 0.7,
            'volume_ratio': 1.2
        }
        
        regime = self.detector.detect_regime(market_data)
        assert regime == MarketRegime.TRENDING_UP
    
    def test_trending_down_detection(self):
        """Test trending down regime detection"""
        market_data = {
            'volatility': 0.2,
            'trend_strength': -0.8,
            'volume_ratio': 1.1
        }
        
        regime = self.detector.detect_regime(market_data)
        assert regime == MarketRegime.TRENDING_DOWN
    
    def test_mean_reverting_detection(self):
        """Test mean reverting regime detection (default)"""
        market_data = {
            'volatility': 0.15,
            'trend_strength': 0.3,
            'volume_ratio': 1.0
        }
        
        regime = self.detector.detect_regime(market_data)
        assert regime == MarketRegime.MEAN_REVERTING
    
    def test_regime_weights(self):
        """Test regime-based strategy weights"""
        weights = self.detector.get_regime_weights(MarketRegime.TRENDING_UP)
        
        assert SignalType.MOMENTUM in weights
        assert SignalType.MEAN_REVERSION in weights
        assert weights[SignalType.MOMENTUM] > weights[SignalType.MEAN_REVERSION]
        
        # Weights should sum to approximately 1.0
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01


class TestPortfolioAllocatorAgent:
    """Test the main Portfolio Allocator Agent"""
    
    def setup_method(self):
        self.agent = PortfolioAllocatorAgent()
    
    @pytest.mark.asyncio
    async def test_process_signals_basic(self):
        """Test basic signal processing"""
        raw_signals = {
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
                                'ema_crossover': {'signal': 0.8}
                            }
                        }
                    }
                )
            ]
        }
        
        market_data = {
            'volatility': 0.2,
            'trend_strength': 0.5,
            'volume_ratio': 1.2
        }
        
        fused_signals = await self.agent.process_signals(raw_signals, market_data)
        
        assert "AAPL" in fused_signals
        assert isinstance(fused_signals["AAPL"], FusedSignal)
        assert len(fused_signals["AAPL"].top_3_reasons) <= 3
        assert fused_signals["AAPL"].contributing_agents == ["momentum_agent"]
    
    @pytest.mark.asyncio
    async def test_process_signals_with_conflicts(self):
        """Test signal processing with conflicts"""
        raw_signals = {
            "AAPL": [
                Signal(
                    symbol="AAPL",
                    signal_type=SignalType.MOMENTUM,
                    value=0.8,
                    confidence=0.9,
                    timestamp=datetime.now(timezone.utc),
                    agent_name="momentum_agent",
                    model_version="1.0.0"
                ),
                Signal(
                    symbol="AAPL",
                    signal_type=SignalType.MEAN_REVERSION,
                    value=-0.7,
                    confidence=0.8,
                    timestamp=datetime.now(timezone.utc),
                    agent_name="mean_reversion_agent",
                    model_version="1.0.0"
                )
            ]
        }
        
        market_data = {
            'volatility': 0.2,
            'trend_strength': 0.3,
            'volume_ratio': 1.0
        }
        
        fused_signals = await self.agent.process_signals(raw_signals, market_data)
        
        assert "AAPL" in fused_signals
        # Should have conflict resolution info
        assert fused_signals["AAPL"].conflict_resolution is not None
    
    def test_normalize_signals(self):
        """Test signal normalization"""
        state = PortfolioState()
        state.raw_signals = {
            "AAPL": [
                Signal(
                    symbol="AAPL",
                    signal_type=SignalType.MOMENTUM,
                    value=1.5,  # Out of range
                    confidence=1.2,  # Out of range
                    timestamp=datetime.now(timezone.utc),
                    agent_name="momentum_agent",
                    model_version="1.0.0"
                )
            ]
        }
        
        normalized_state = self.agent._normalize_signals(state)
        
        signal_key = list(normalized_state.normalized_signals.keys())[0]
        normalized_signal = normalized_state.normalized_signals[signal_key]
        
        # Value should be clipped to [-1, 1]
        assert -1.0 <= normalized_signal.value <= 1.0
        # Confidence should be adjusted but not exceed 1.0
        assert 0.0 <= normalized_signal.confidence <= 1.0
    
    def test_detect_regime(self):
        """Test regime detection in workflow"""
        state = PortfolioState()
        state.market_data = {
            'volatility': 0.35,
            'trend_strength': 0.2,
            'volume_ratio': 1.5
        }
        
        regime_state = self.agent._detect_regime(state)
        
        assert regime_state.market_regime == MarketRegime.HIGH_VOLATILITY
    
    def test_apply_regime_weights(self):
        """Test regime-based weight application"""
        state = PortfolioState()
        state.market_regime = MarketRegime.TRENDING_UP
        state.normalized_signals = {
            "AAPL_momentum": Signal(
                symbol="AAPL",
                signal_type=SignalType.MOMENTUM,
                value=0.8,
                confidence=0.9,
                timestamp=datetime.now(timezone.utc),
                agent_name="momentum_agent",
                model_version="1.0.0"
            ),
            "AAPL_mean_reversion": Signal(
                symbol="AAPL",
                signal_type=SignalType.MEAN_REVERSION,
                value=0.6,
                confidence=0.7,
                timestamp=datetime.now(timezone.utc),
                agent_name="mean_reversion_agent",
                model_version="1.0.0"
            )
        }
        
        weighted_state = self.agent._apply_regime_weights(state)
        
        momentum_signal = weighted_state.weighted_signals["AAPL_momentum"]
        mean_reversion_signal = weighted_state.weighted_signals["AAPL_mean_reversion"]
        
        # In trending up regime, momentum should get higher weight
        assert momentum_signal.metadata['regime_weight'] > mean_reversion_signal.metadata['regime_weight']
        assert momentum_signal.metadata['regime'] == 'trending_up'
    
    def test_get_agent_performance_multiplier(self):
        """Test agent performance multiplier"""
        # Test known agents
        momentum_multiplier = self.agent._get_agent_performance_multiplier('momentum_agent')
        assert momentum_multiplier == 1.1
        
        # Test unknown agent (should default to 1.0)
        unknown_multiplier = self.agent._get_agent_performance_multiplier('unknown_agent')
        assert unknown_multiplier == 1.0
    
    @pytest.mark.asyncio
    async def test_empty_signals(self):
        """Test handling of empty signals"""
        raw_signals = {}
        market_data = {'volatility': 0.2}
        
        fused_signals = await self.agent.process_signals(raw_signals, market_data)
        
        assert fused_signals == {}
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in signal processing"""
        # Create invalid signal that might cause errors
        raw_signals = {
            "AAPL": [
                Signal(
                    symbol="AAPL",
                    signal_type=SignalType.MOMENTUM,
                    value=float('nan'),  # Invalid value
                    confidence=0.8,
                    timestamp=datetime.now(timezone.utc),
                    agent_name="momentum_agent",
                    model_version="1.0.0"
                )
            ]
        }
        
        market_data = {'volatility': 0.2}
        
        # Should not raise exception, should return empty dict
        fused_signals = await self.agent.process_signals(raw_signals, market_data)
        
        # Should handle gracefully
        assert isinstance(fused_signals, dict)


class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete signal fusion workflow"""
        agent = PortfolioAllocatorAgent()
        
        # Create comprehensive test signals
        raw_signals = {
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
                        'fibonacci_levels': {
                            'levels': {
                                'fib_382': 148.0,
                                'fib_618': 152.0
                            }
                        },
                        'volume_data': {
                            'current_volume': 2000000,
                            'average_volume': 1500000
                        },
                        'risk_reward': {
                            'ratio': 2.5
                        }
                    }
                ),
                Signal(
                    symbol="AAPL",
                    signal_type=SignalType.SENTIMENT,
                    value=0.5,
                    confidence=0.7,
                    timestamp=datetime.now(timezone.utc),
                    agent_name="sentiment_agent",
                    model_version="1.0.0",
                    metadata={
                        'sentiment_score': 0.5,
                        'news_count': 15,
                        'social_sentiment': 0.6
                    }
                )
            ],
            "MSFT": [
                Signal(
                    symbol="MSFT",
                    signal_type=SignalType.MEAN_REVERSION,
                    value=-0.4,
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
        
        market_data = {
            'volatility': 0.25,
            'trend_strength': 0.4,
            'volume_ratio': 1.3
        }
        
        # Process signals
        fused_signals = await agent.process_signals(raw_signals, market_data)
        
        # Verify results
        assert len(fused_signals) == 2
        assert "AAPL" in fused_signals
        assert "MSFT" in fused_signals
        
        # Check AAPL signal (should be positive due to momentum + sentiment)
        aapl_signal = fused_signals["AAPL"]
        assert aapl_signal.value > 0
        assert len(aapl_signal.top_3_reasons) <= 3
        assert len(aapl_signal.contributing_agents) == 2
        assert "momentum_agent" in aapl_signal.contributing_agents
        assert "sentiment_agent" in aapl_signal.contributing_agents
        
        # Check MSFT signal (should be negative due to mean reversion)
        msft_signal = fused_signals["MSFT"]
        assert msft_signal.value < 0
        assert len(msft_signal.contributing_agents) == 1
        assert "mean_reversion_agent" in msft_signal.contributing_agents
        
        # Verify explainability
        for signal in fused_signals.values():
            for reason in signal.top_3_reasons:
                assert isinstance(reason, Reason)
                assert reason.explanation
                assert 0 <= reason.confidence <= 1
                assert reason.rank in [1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])