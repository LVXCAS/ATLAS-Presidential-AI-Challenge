"""
Portfolio Allocator Agent Validation Script

This script validates the Portfolio Allocator Agent implementation against
the requirements and acceptance criteria.
"""

import asyncio
import sys
import os
from datetime import datetime, timezone
from typing import Dict, List, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.portfolio_allocator_agent import (
    PortfolioAllocatorAgent,
    ExplainabilityEngine,
    ConflictResolver,
    RegimeDetector,
    Signal,
    FusedSignal,
    SignalType,
    MarketRegime,
    Reason
)


class ValidationResult:
    """Validation result container"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.errors = []
        self.warnings = []
    
    def add_success(self, test_name: str):
        """Add successful test"""
        self.tests_passed += 1
        print(f"[OK] {test_name}")
    
    def add_failure(self, test_name: str, error: str):
        """Add failed test"""
        self.tests_failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"[X] {test_name}: {error}")
    
    def add_warning(self, test_name: str, warning: str):
        """Add warning"""
        self.warnings.append(f"{test_name}: {warning}")
        print(f"[WARN]  {test_name}: {warning}")
    
    def print_summary(self):
        """Print validation summary"""
        total_tests = self.tests_passed + self.tests_failed
        success_rate = (self.tests_passed / max(total_tests, 1)) * 100
        
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Warnings: {len(self.warnings)}")
        
        if self.errors:
            print(f"\n[X] FAILURES:")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"\n[WARN]  WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        return self.tests_failed == 0


async def validate_signal_fusion(result: ValidationResult):
    """Validate signal fusion functionality"""
    print("\n[INFO] Validating Signal Fusion...")
    
    try:
        agent = PortfolioAllocatorAgent()
        
        # Test basic signal fusion
        test_signals = {
            "TEST": [
                Signal(
                    symbol="TEST",
                    signal_type=SignalType.MOMENTUM,
                    value=0.7,
                    confidence=0.8,
                    timestamp=datetime.now(timezone.utc),
                    agent_name="momentum_agent",
                    model_version="1.0.0"
                ),
                Signal(
                    symbol="TEST",
                    signal_type=SignalType.SENTIMENT,
                    value=0.5,
                    confidence=0.6,
                    timestamp=datetime.now(timezone.utc),
                    agent_name="sentiment_agent",
                    model_version="1.0.0"
                )
            ]
        }
        
        market_data = {'volatility': 0.2, 'trend_strength': 0.3}
        fused_signals = await agent.process_signals(test_signals, market_data)
        
        # Validate fusion output
        if "TEST" not in fused_signals:
            result.add_failure("Signal Fusion", "No fused signal generated")
            return
        
        fused_signal = fused_signals["TEST"]
        
        # Check signal structure
        if not isinstance(fused_signal, FusedSignal):
            result.add_failure("Signal Structure", "Invalid fused signal type")
            return
        
        # Check required fields
        required_fields = ['symbol', 'signal_type', 'value', 'confidence', 'top_3_reasons', 
                          'timestamp', 'model_version', 'contributing_agents']
        
        for field in required_fields:
            if not hasattr(fused_signal, field):
                result.add_failure("Signal Fields", f"Missing field: {field}")
                return
        
        # Check value range
        if not (-1.0 <= fused_signal.value <= 1.0):
            result.add_failure("Signal Range", f"Signal value {fused_signal.value} out of range [-1, 1]")
            return
        
        # Check confidence range
        if not (0.0 <= fused_signal.confidence <= 1.0):
            result.add_failure("Confidence Range", f"Confidence {fused_signal.confidence} out of range [0, 1]")
            return
        
        # Check contributing agents
        if len(fused_signal.contributing_agents) != 2:
            result.add_failure("Contributing Agents", f"Expected 2 agents, got {len(fused_signal.contributing_agents)}")
            return
        
        result.add_success("Signal Fusion - Basic Functionality")
        
    except Exception as e:
        result.add_failure("Signal Fusion", f"Exception: {str(e)}")


async def validate_explainability_engine(result: ValidationResult):
    """Validate explainability engine"""
    print("\n[AI] Validating Explainability Engine...")
    
    try:
        engine = ExplainabilityEngine()
        
        # Create signal with rich metadata
        signal = Signal(
            symbol="TEST",
            signal_type=SignalType.MOMENTUM,
            value=0.8,
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
            agent_name="test_agent",
            model_version="1.0.0",
            metadata={
                'technical_indicators': {
                    'indicators': {
                        'ema_crossover': {'signal': 0.8},
                        'rsi_breakout': {'signal': 0.7}
                    }
                },
                'sentiment_score': 0.6,
                'volume_data': {
                    'current_volume': 2000000,
                    'average_volume': 1500000
                },
                'fibonacci_levels': {
                    'levels': {
                        'fib_618': 150.0
                    }
                },
                'risk_reward': {
                    'ratio': 2.5
                }
            }
        )
        
        reasons = engine.generate_top_3_reasons(signal, current_price=150.0)
        
        # Validate reasons structure
        if not isinstance(reasons, list):
            result.add_failure("Explainability Structure", "Reasons not returned as list")
            return
        
        if len(reasons) > 3:
            result.add_failure("Explainability Count", f"Too many reasons: {len(reasons)}")
            return
        
        # Validate each reason
        for i, reason in enumerate(reasons):
            if not isinstance(reason, Reason):
                result.add_failure("Reason Type", f"Reason {i} is not Reason type")
                return
            
            if not (1 <= reason.rank <= 3):
                result.add_failure("Reason Rank", f"Invalid rank: {reason.rank}")
                return
            
            if not (0 <= reason.confidence <= 1):
                result.add_failure("Reason Confidence", f"Invalid confidence: {reason.confidence}")
                return
            
            if not reason.explanation:
                result.add_failure("Reason Explanation", f"Empty explanation for reason {i}")
                return
        
        result.add_success("Explainability Engine - Reason Generation")
        
        # Test individual calculation methods
        tech_score = engine._calculate_technical_confluence({
            'indicators': {
                'ema': {'signal': 0.8},
                'rsi': {'signal': 0.6}
            }
        })
        
        if not (0 <= tech_score <= 1):
            result.add_failure("Technical Confluence", f"Invalid score: {tech_score}")
            return
        
        result.add_success("Explainability Engine - Technical Confluence")
        
        # Test sentiment alignment
        alignment_score = engine._calculate_sentiment_alignment(0.7, 0.6)
        if alignment_score != 0.6:  # Should be min of the two
            result.add_failure("Sentiment Alignment", f"Expected 0.6, got {alignment_score}")
            return
        
        result.add_success("Explainability Engine - Sentiment Alignment")
        
    except Exception as e:
        result.add_failure("Explainability Engine", f"Exception: {str(e)}")


async def validate_conflict_resolution(result: ValidationResult):
    """Validate conflict resolution"""
    print("\n[INFO]️ Validating Conflict Resolution...")
    
    try:
        resolver = ConflictResolver()
        
        # Create conflicting signals
        conflicting_signals = {
            'signal1': Signal(
                symbol="TEST",
                signal_type=SignalType.MOMENTUM,
                value=0.8,
                confidence=0.9,
                timestamp=datetime.now(timezone.utc),
                agent_name="momentum_agent",
                model_version="1.0.0"
            ),
            'signal2': Signal(
                symbol="TEST",
                signal_type=SignalType.MEAN_REVERSION,
                value=-0.7,
                confidence=0.8,
                timestamp=datetime.now(timezone.utc),
                agent_name="mean_reversion_agent",
                model_version="1.0.0"
            )
        }
        
        # Test conflict detection
        conflicts = resolver.detect_conflicts(conflicting_signals)
        
        if len(conflicts) != 1:
            result.add_failure("Conflict Detection", f"Expected 1 conflict, found {len(conflicts)}")
            return
        
        conflict = conflicts[0]
        if conflict['symbol'] != "TEST":
            result.add_failure("Conflict Symbol", f"Wrong symbol: {conflict['symbol']}")
            return
        
        if conflict['conflict_type'] != 'directional_opposite':
            result.add_failure("Conflict Type", f"Wrong type: {conflict['conflict_type']}")
            return
        
        result.add_success("Conflict Resolution - Detection")
        
        # Test conflict resolution
        resolved_signal = resolver.resolve_conflict(conflict)
        
        if not isinstance(resolved_signal, Signal):
            result.add_failure("Conflict Resolution", "Invalid resolved signal type")
            return
        
        if not hasattr(resolved_signal, 'metadata') or 'conflict_resolution' not in resolved_signal.metadata:
            result.add_failure("Conflict Metadata", "Missing conflict resolution metadata")
            return
        
        result.add_success("Conflict Resolution - Resolution")
        
        # Test different resolution strategies
        weighted_avg = resolver._weighted_average_resolution([conflicting_signals['signal1'], conflicting_signals['signal2']])
        expected_value = (0.8 * 0.9 + (-0.7) * 0.8) / (0.9 + 0.8)
        
        if abs(weighted_avg.value - expected_value) > 0.01:
            result.add_failure("Weighted Average", f"Expected {expected_value:.3f}, got {weighted_avg.value:.3f}")
            return
        
        result.add_success("Conflict Resolution - Weighted Average")
        
        # Test confidence-based resolution
        confidence_based = resolver._confidence_based_resolution([conflicting_signals['signal1'], conflicting_signals['signal2']])
        
        if confidence_based.value != 0.8:  # Should select higher confidence signal
            result.add_failure("Confidence Based", f"Expected 0.8, got {confidence_based.value}")
            return
        
        result.add_success("Conflict Resolution - Confidence Based")
        
    except Exception as e:
        result.add_failure("Conflict Resolution", f"Exception: {str(e)}")


async def validate_regime_detection(result: ValidationResult):
    """Validate regime detection and weighting"""
    print("\n[INFO] Validating Regime Detection...")
    
    try:
        detector = RegimeDetector()
        
        # Test high volatility detection
        high_vol_data = {'volatility': 0.35, 'trend_strength': 0.2, 'volume_ratio': 1.5}
        regime = detector.detect_regime(high_vol_data)
        
        if regime != MarketRegime.HIGH_VOLATILITY:
            result.add_failure("High Volatility Detection", f"Expected HIGH_VOLATILITY, got {regime}")
            return
        
        result.add_success("Regime Detection - High Volatility")
        
        # Test trending up detection
        trending_up_data = {'volatility': 0.2, 'trend_strength': 0.7, 'volume_ratio': 1.2}
        regime = detector.detect_regime(trending_up_data)
        
        if regime != MarketRegime.TRENDING_UP:
            result.add_failure("Trending Up Detection", f"Expected TRENDING_UP, got {regime}")
            return
        
        result.add_success("Regime Detection - Trending Up")
        
        # Test regime weights
        weights = detector.get_regime_weights(MarketRegime.TRENDING_UP)
        
        if SignalType.MOMENTUM not in weights:
            result.add_failure("Regime Weights", "Missing momentum weight")
            return
        
        if weights[SignalType.MOMENTUM] <= weights[SignalType.MEAN_REVERSION]:
            result.add_failure("Regime Weight Logic", "Momentum should have higher weight in trending market")
            return
        
        # Check weights sum approximately to 1.0
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            result.add_failure("Weight Sum", f"Weights sum to {total_weight}, expected ~1.0")
            return
        
        result.add_success("Regime Detection - Weights")
        
    except Exception as e:
        result.add_failure("Regime Detection", f"Exception: {str(e)}")


async def validate_langgraph_workflow(result: ValidationResult):
    """Validate LangGraph workflow integration"""
    print("\n[INFO] Validating LangGraph Workflow...")
    
    try:
        agent = PortfolioAllocatorAgent()
        
        # Check workflow exists
        if not hasattr(agent, 'workflow'):
            result.add_failure("Workflow Existence", "No workflow attribute found")
            return
        
        # Test workflow with simple signals
        test_signals = {
            "WORKFLOW_TEST": [
                Signal(
                    symbol="WORKFLOW_TEST",
                    signal_type=SignalType.MOMENTUM,
                    value=0.6,
                    confidence=0.7,
                    timestamp=datetime.now(timezone.utc),
                    agent_name="test_agent",
                    model_version="1.0.0"
                )
            ]
        }
        
        market_data = {'volatility': 0.2, 'trend_strength': 0.3}
        
        # This should execute the full workflow
        fused_signals = await agent.process_signals(test_signals, market_data)
        
        if "WORKFLOW_TEST" not in fused_signals:
            result.add_failure("Workflow Execution", "Workflow did not produce expected output")
            return
        
        result.add_success("LangGraph Workflow - Execution")
        
        # Test individual workflow steps
        from agents.portfolio_allocator_agent import PortfolioState
        
        state = PortfolioState(raw_signals=test_signals)
        state.market_data = market_data
        
        # Test normalization step
        normalized_state = agent._normalize_signals(state)
        if not normalized_state.normalized_signals:
            result.add_failure("Workflow Step", "Normalization step failed")
            return
        
        result.add_success("LangGraph Workflow - Normalization Step")
        
        # Test regime detection step
        regime_state = agent._detect_regime(normalized_state)
        if not regime_state.market_regime:
            result.add_failure("Workflow Step", "Regime detection step failed")
            return
        
        result.add_success("LangGraph Workflow - Regime Detection Step")
        
    except Exception as e:
        result.add_failure("LangGraph Workflow", f"Exception: {str(e)}")


async def validate_performance_requirements(result: ValidationResult):
    """Validate performance requirements"""
    print("\n[FAST] Validating Performance Requirements...")
    
    try:
        agent = PortfolioAllocatorAgent()
        
        # Test with multiple symbols and signals
        large_signal_set = {}
        
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "CRM", "ORCL"]
        
        for symbol in symbols:
            large_signal_set[symbol] = [
                Signal(
                    symbol=symbol,
                    signal_type=SignalType.MOMENTUM,
                    value=0.5,
                    confidence=0.7,
                    timestamp=datetime.now(timezone.utc),
                    agent_name="momentum_agent",
                    model_version="1.0.0"
                ),
                Signal(
                    symbol=symbol,
                    signal_type=SignalType.MEAN_REVERSION,
                    value=-0.3,
                    confidence=0.6,
                    timestamp=datetime.now(timezone.utc),
                    agent_name="mean_reversion_agent",
                    model_version="1.0.0"
                )
            ]
        
        market_data = {'volatility': 0.2, 'trend_strength': 0.3}
        
        # Measure processing time
        import time
        start_time = time.time()
        
        fused_signals = await agent.process_signals(large_signal_set, market_data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 10 symbols with 2 signals each in reasonable time
        if processing_time > 5.0:  # 5 seconds threshold
            result.add_warning("Performance", f"Processing took {processing_time:.2f}s for 20 signals")
        else:
            result.add_success("Performance - Processing Time")
        
        # Check all symbols were processed
        if len(fused_signals) != len(symbols):
            result.add_failure("Performance", f"Expected {len(symbols)} signals, got {len(fused_signals)}")
            return
        
        result.add_success("Performance - Signal Count")
        
    except Exception as e:
        result.add_failure("Performance", f"Exception: {str(e)}")


async def validate_error_handling(result: ValidationResult):
    """Validate error handling"""
    print("\n[INFO]️ Validating Error Handling...")
    
    try:
        agent = PortfolioAllocatorAgent()
        
        # Test with empty signals
        empty_signals = {}
        market_data = {'volatility': 0.2}
        
        fused_signals = await agent.process_signals(empty_signals, market_data)
        
        if fused_signals != {}:
            result.add_failure("Empty Signals", "Should return empty dict for empty input")
            return
        
        result.add_success("Error Handling - Empty Signals")
        
        # Test with invalid signal values
        invalid_signals = {
            "TEST": [
                Signal(
                    symbol="TEST",
                    signal_type=SignalType.MOMENTUM,
                    value=float('inf'),  # Invalid value
                    confidence=0.8,
                    timestamp=datetime.now(timezone.utc),
                    agent_name="test_agent",
                    model_version="1.0.0"
                )
            ]
        }
        
        # Should handle gracefully without crashing
        try:
            fused_signals = await agent.process_signals(invalid_signals, market_data)
            result.add_success("Error Handling - Invalid Values")
        except Exception:
            result.add_failure("Error Handling", "Should handle invalid values gracefully")
            return
        
        # Test with missing market data
        try:
            fused_signals = await agent.process_signals(invalid_signals, {})
            result.add_success("Error Handling - Missing Market Data")
        except Exception:
            result.add_failure("Error Handling", "Should handle missing market data gracefully")
            return
        
    except Exception as e:
        result.add_failure("Error Handling", f"Exception: {str(e)}")


async def main():
    """Run all validation tests"""
    print("[LAUNCH] Portfolio Allocator Agent Validation")
    print("=" * 80)
    
    result = ValidationResult()
    
    # Run all validation tests
    await validate_signal_fusion(result)
    await validate_explainability_engine(result)
    await validate_conflict_resolution(result)
    await validate_regime_detection(result)
    await validate_langgraph_workflow(result)
    await validate_performance_requirements(result)
    await validate_error_handling(result)
    
    # Print summary
    success = result.print_summary()
    
    if success:
        print("\n[PARTY] All validations passed! Portfolio Allocator Agent is ready for integration.")
    else:
        print("\n[X] Some validations failed. Please review and fix the issues.")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)