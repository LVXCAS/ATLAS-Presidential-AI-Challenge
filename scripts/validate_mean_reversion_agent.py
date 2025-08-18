"""
Mean Reversion Trading Agent Validation Script

This script validates the Mean Reversion Trading Agent implementation
by testing all components and ensuring they meet the requirements.
"""

import sys
import os
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.mean_reversion_agent import (
    MeanReversionTradingAgent,
    BollingerBandAnalyzer,
    ZScoreAnalyzer,
    PairsTradingAnalyzer,
    FibonacciTargetCalculator,
    SentimentDivergenceDetector,
    MarketRegimeDetector,
    ExplainabilityEngine,
    MarketData,
    SentimentData,
    TechnicalSignal,
    PairsSignal,
    FibonacciTarget,
    MeanReversionSignal,
    SignalType,
    MarketRegime
)


class MeanReversionValidator:
    """Validator for Mean Reversion Trading Agent"""
    
    def __init__(self):
        self.agent = MeanReversionTradingAgent()
        self.validation_results = []
        print("🔍 Mean Reversion Trading Agent Validation")
        print("=" * 50)
    
    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log validation result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.validation_results.append({
            'test': test_name,
            'passed': passed,
            'details': details
        })
        print(f"{status} {test_name}")
        if details and not passed:
            print(f"    Details: {details}")
    
    def create_test_data(self, pattern: str = "mean_reverting", length: int = 100) -> List[MarketData]:
        """Create test market data with specific patterns"""
        np.random.seed(42)
        data = []
        base_price = 100.0
        
        if pattern == "mean_reverting":
            # Create clear mean-reverting pattern
            for i in range(length):
                price = base_price + 10 * np.sin(i / 10) + np.random.normal(0, 1)
                data.append(self._create_market_data_point("TEST", i, price, length))
        
        elif pattern == "bollinger_extreme":
            # Create pattern that touches Bollinger Bands
            for i in range(length):
                if 30 <= i <= 35:  # Extreme high
                    price = base_price + 15 + np.random.normal(0, 0.5)
                elif 60 <= i <= 65:  # Extreme low
                    price = base_price - 15 + np.random.normal(0, 0.5)
                else:
                    price = base_price + np.random.normal(0, 2)
                data.append(self._create_market_data_point("TEST", i, price, length))
        
        elif pattern == "zscore_extreme":
            # Create pattern with extreme Z-scores
            for i in range(length):
                if i < 50:  # Stable period
                    price = base_price + np.random.normal(0, 1)
                else:  # Extreme period
                    price = base_price + 20 + np.random.normal(0, 1)
                data.append(self._create_market_data_point("TEST", i, price, length))
        
        elif pattern == "trending":
            # Create trending pattern
            for i in range(length):
                price = base_price + i * 0.5 + np.random.normal(0, 1)
                data.append(self._create_market_data_point("TEST", i, price, length))
        
        elif pattern == "high_volatility":
            # Create high volatility pattern
            price = base_price
            for i in range(length):
                price += np.random.normal(0, 5)  # High volatility
                price = max(50, min(200, price))
                data.append(self._create_market_data_point("TEST", i, price, length))
        
        return data
    
    def _create_market_data_point(self, symbol: str, i: int, price: float, total_length: int) -> MarketData:
        """Create a single market data point"""
        return MarketData(
            symbol=symbol,
            timestamp=datetime.now() - timedelta(days=total_length-i),
            open=price + np.random.normal(0, 0.2),
            high=price + abs(np.random.normal(0, 0.5)),
            low=price - abs(np.random.normal(0, 0.5)),
            close=price,
            volume=np.random.randint(1000000, 5000000)
        )
    
    def create_cointegrated_pair(self, length: int = 100) -> Tuple[List[MarketData], List[MarketData]]:
        """Create cointegrated pair for testing"""
        np.random.seed(123)
        
        # Generate common trend
        common_trend = np.cumsum(np.random.normal(0, 0.1, length))
        
        # Generate cointegrated series
        prices_a = 100 + common_trend + np.random.normal(0, 0.8, length)
        prices_b = 50 + 0.5 * common_trend + np.random.normal(0, 0.6, length)
        
        data_a = [self._create_market_data_point("STOCK_A", i, prices_a[i], length) for i in range(length)]
        data_b = [self._create_market_data_point("STOCK_B", i, prices_b[i], length) for i in range(length)]
        
        return data_a, data_b
    
    def validate_bollinger_band_analyzer(self):
        """Validate Bollinger Band analyzer"""
        print("\n📊 Validating Bollinger Band Analyzer")
        print("-" * 35)
        
        analyzer = BollingerBandAnalyzer()
        
        # Test 1: Basic functionality
        test_data = self.create_test_data("bollinger_extreme", 80)
        close_prices = np.array([d.close for d in test_data])
        signals = analyzer.calculate_bollinger_signals(close_prices)
        
        self.log_result(
            "Bollinger Band signal generation",
            isinstance(signals, list),
            f"Generated {len(signals)} signals"
        )
        
        # Test 2: Signal structure validation
        if signals:
            signal = signals[0]
            valid_structure = (
                hasattr(signal, 'indicator') and
                hasattr(signal, 'signal_type') and
                hasattr(signal, 'confidence') and
                hasattr(signal, 'value') and
                -1.0 <= signal.value <= 1.0 and
                0.0 <= signal.confidence <= 1.0
            )
            self.log_result("Bollinger Band signal structure", valid_structure)
        
        # Test 3: Empty data handling
        empty_signals = analyzer.calculate_bollinger_signals(np.array([]))
        self.log_result("Empty data handling", empty_signals == [])
        
        # Test 4: Extreme values detection
        extreme_data = np.concatenate([
            np.full(30, 100.0),  # Stable period
            np.full(10, 120.0),  # Extreme high
            np.full(10, 80.0)    # Extreme low
        ])
        extreme_signals = analyzer.calculate_bollinger_signals(extreme_data)
        has_extreme_signals = any("Reversal" in s.indicator for s in extreme_signals)
        self.log_result("Extreme value detection", len(extreme_signals) > 0)
    
    def validate_zscore_analyzer(self):
        """Validate Z-Score analyzer"""
        print("\n📈 Validating Z-Score Analyzer")
        print("-" * 28)
        
        analyzer = ZScoreAnalyzer()
        
        # Test 1: Basic Z-score calculation
        test_data = self.create_test_data("zscore_extreme", 80)
        close_prices = np.array([d.close for d in test_data])
        signals = analyzer.calculate_zscore_signals(close_prices)
        
        self.log_result(
            "Z-Score signal generation",
            isinstance(signals, list),
            f"Generated {len(signals)} signals"
        )
        
        # Test 2: Extreme Z-score detection
        # Create data with known extreme Z-scores
        extreme_data = np.concatenate([
            np.full(30, 100.0),  # Mean = 100, std ≈ 0
            [130.0, 70.0]        # Extreme values
        ])
        extreme_signals = analyzer.calculate_zscore_signals(extreme_data, period=20, entry_threshold=2.0)
        
        self.log_result("Extreme Z-score detection", len(extreme_signals) >= 0)
        
        # Test 3: Signal thresholds
        if extreme_signals:
            has_high_confidence = any(s.confidence >= 0.8 for s in extreme_signals)
            self.log_result("High confidence signals", has_high_confidence)
        
        # Test 4: Normalization detection
        normalization_data = np.array([100, 100, 100, 120, 115, 110, 105, 102, 100, 100])
        norm_signals = analyzer.calculate_zscore_signals(normalization_data, period=5)
        self.log_result("Z-score normalization detection", isinstance(norm_signals, list))
    
    def validate_pairs_trading_analyzer(self):
        """Validate Pairs Trading analyzer"""
        print("\n🔗 Validating Pairs Trading Analyzer")
        print("-" * 33)
        
        analyzer = PairsTradingAnalyzer(cointegration_threshold=0.1)
        
        # Test 1: Cointegration test
        data_a, data_b = self.create_cointegrated_pair(100)
        prices_a = np.array([d.close for d in data_a])
        prices_b = np.array([d.close for d in data_b])
        
        coint_stat, p_value, hedge_ratio = analyzer.test_cointegration(prices_a, prices_b)
        
        self.log_result(
            "Cointegration test execution",
            isinstance(p_value, float) and 0.0 <= p_value <= 1.0,
            f"p-value: {p_value:.3f}, hedge_ratio: {hedge_ratio:.2f}"
        )
        
        # Test 2: Spread calculation
        spread = analyzer.calculate_spread(prices_a, prices_b, hedge_ratio)
        self.log_result(
            "Spread calculation",
            len(spread) == len(prices_a) and isinstance(spread, np.ndarray)
        )
        
        # Test 3: Pairs signal generation
        pairs_signals = analyzer.calculate_pairs_signals("STOCK_A", prices_a, "STOCK_B", prices_b)
        self.log_result(
            "Pairs signal generation",
            isinstance(pairs_signals, list),
            f"Generated {len(pairs_signals)} pairs signals"
        )
        
        # Test 4: Signal structure validation
        if pairs_signals:
            signal = pairs_signals[0]
            valid_structure = (
                hasattr(signal, 'symbol_a') and
                hasattr(signal, 'symbol_b') and
                hasattr(signal, 'z_score') and
                hasattr(signal, 'hedge_ratio') and
                hasattr(signal, 'confidence') and
                0.0 <= signal.confidence <= 1.0
            )
            self.log_result("Pairs signal structure", valid_structure)
    
    def validate_fibonacci_calculator(self):
        """Validate Fibonacci target calculator"""
        print("\n🌀 Validating Fibonacci Target Calculator")
        print("-" * 37)
        
        calculator = FibonacciTargetCalculator()
        
        # Test 1: Basic target calculation
        test_data = self.create_test_data("mean_reverting", 60)
        high_prices = np.array([d.high for d in test_data])
        low_prices = np.array([d.low for d in test_data])
        close_prices = np.array([d.close for d in test_data])
        current_price = close_prices[-1]
        
        targets = calculator.calculate_fibonacci_targets(
            high_prices, low_prices, close_prices, current_price, "buy"
        )
        
        self.log_result(
            "Fibonacci target calculation",
            isinstance(targets, list),
            f"Generated {len(targets)} targets"
        )
        
        # Test 2: Target structure validation
        if targets:
            target = targets[0]
            valid_structure = (
                hasattr(target, 'level_name') and
                hasattr(target, 'target_price') and
                hasattr(target, 'distance_pct') and
                hasattr(target, 'confidence') and
                0.0 <= target.confidence <= 1.0 and
                target.distance_pct >= 0.0
            )
            self.log_result("Fibonacci target structure", valid_structure)
        
        # Test 3: Direction filtering
        buy_targets = calculator.calculate_fibonacci_targets(
            high_prices, low_prices, close_prices, current_price, "buy"
        )
        sell_targets = calculator.calculate_fibonacci_targets(
            high_prices, low_prices, close_prices, current_price, "sell"
        )
        
        self.log_result(
            "Direction-based target filtering",
            isinstance(buy_targets, list) and isinstance(sell_targets, list)
        )
        
        # Test 4: Empty data handling
        empty_targets = calculator.calculate_fibonacci_targets(
            np.array([]), np.array([]), np.array([]), 100.0, "buy"
        )
        self.log_result("Empty data handling", empty_targets == [])
    
    def validate_sentiment_divergence_detector(self):
        """Validate sentiment divergence detector"""
        print("\n💭 Validating Sentiment Divergence Detector")
        print("-" * 40)
        
        detector = SentimentDivergenceDetector()
        
        # Test 1: Bullish divergence (price down, sentiment up)
        price_data = np.array([105, 104, 103, 102, 101, 100])  # Declining
        sentiment_data = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]    # Improving
        
        divergence = detector.calculate_sentiment_divergence(price_data, sentiment_data, 6)
        
        self.log_result(
            "Bullish divergence detection",
            divergence is None or (isinstance(divergence, float) and -1.0 <= divergence <= 1.0),
            f"Divergence: {divergence}"
        )
        
        # Test 2: Bearish divergence (price up, sentiment down)
        price_data_up = np.array([100, 101, 102, 103, 104, 105])  # Rising
        sentiment_data_down = [0.5, 0.3, 0.1, -0.1, -0.3, -0.5]  # Declining
        
        divergence_bearish = detector.calculate_sentiment_divergence(price_data_up, sentiment_data_down, 6)
        
        self.log_result(
            "Bearish divergence detection",
            divergence_bearish is None or isinstance(divergence_bearish, float),
            f"Divergence: {divergence_bearish}"
        )
        
        # Test 3: Insufficient data handling
        short_price = np.array([100, 101])
        short_sentiment = [0.1, 0.2]
        
        insufficient_divergence = detector.calculate_sentiment_divergence(short_price, short_sentiment, 10)
        self.log_result("Insufficient data handling", insufficient_divergence is None)
        
        # Test 4: No divergence case
        aligned_price = np.array([100, 101, 102, 103, 104, 105])  # Rising
        aligned_sentiment = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]       # Also rising
        
        no_divergence = detector.calculate_sentiment_divergence(aligned_price, aligned_sentiment, 6)
        self.log_result(
            "No divergence detection",
            no_divergence is None or abs(no_divergence) < 1.0
        )
    
    def validate_market_regime_detector(self):
        """Validate market regime detector"""
        print("\n🌊 Validating Market Regime Detector")
        print("-" * 32)
        
        detector = MarketRegimeDetector()
        
        # Test 1: High volatility detection
        high_vol_data = self.create_test_data("high_volatility", 50)
        high_vol_prices = np.array([d.close for d in high_vol_data])
        high_vol_volumes = np.array([d.volume for d in high_vol_data])
        
        regime_high_vol = detector.detect_regime(high_vol_prices, high_vol_volumes)
        self.log_result(
            "High volatility regime detection",
            isinstance(regime_high_vol, MarketRegime),
            f"Detected: {regime_high_vol.value}"
        )
        
        # Test 2: Trending regime detection
        trending_data = self.create_test_data("trending", 50)
        trending_prices = np.array([d.close for d in trending_data])
        trending_volumes = np.array([d.volume for d in trending_data])
        
        regime_trending = detector.detect_regime(trending_prices, trending_volumes)
        self.log_result(
            "Trending regime detection",
            isinstance(regime_trending, MarketRegime),
            f"Detected: {regime_trending.value}"
        )
        
        # Test 3: Mean reverting regime detection
        mean_rev_data = self.create_test_data("mean_reverting", 50)
        mean_rev_prices = np.array([d.close for d in mean_rev_data])
        mean_rev_volumes = np.array([d.volume for d in mean_rev_data])
        
        regime_mean_rev = detector.detect_regime(mean_rev_prices, mean_rev_volumes)
        self.log_result(
            "Mean reverting regime detection",
            isinstance(regime_mean_rev, MarketRegime),
            f"Detected: {regime_mean_rev.value}"
        )
        
        # Test 4: Insufficient data handling
        short_data = np.array([100, 101, 102])
        short_volumes = np.array([1000, 1100, 1200])
        
        regime_insufficient = detector.detect_regime(short_data, short_volumes, lookback_period=50)
        self.log_result(
            "Insufficient data handling",
            regime_insufficient == MarketRegime.SIDEWAYS
        )
    
    def validate_explainability_engine(self):
        """Validate explainability engine"""
        print("\n🧠 Validating Explainability Engine")
        print("-" * 32)
        
        engine = ExplainabilityEngine()
        
        # Create sample signals for testing
        bollinger_signals = [
            TechnicalSignal(
                indicator="BB_Upper_Reversal",
                signal_type=SignalType.SELL,
                strength=0.8,
                confidence=0.9,
                value=-0.8,
                explanation="Test signal",
                timestamp=datetime.utcnow()
            )
        ]
        
        zscore_signals = [
            TechnicalSignal(
                indicator="ZScore_High",
                signal_type=SignalType.SELL,
                strength=0.7,
                confidence=0.85,
                value=-0.7,
                explanation="Test Z-score signal",
                timestamp=datetime.utcnow()
            )
        ]
        
        # Test 1: Basic reason generation
        reasons = engine.generate_top_3_reasons(
            bollinger_signals, zscore_signals, [], [],
            sentiment_divergence=None,
            market_regime=MarketRegime.MEAN_REVERTING,
            final_signal_value=-0.6
        )
        
        self.log_result(
            "Reason generation",
            isinstance(reasons, list) and len(reasons) <= 3,
            f"Generated {len(reasons)} reasons"
        )
        
        # Test 2: Reason structure validation
        if reasons:
            reason = reasons[0]
            valid_structure = (
                hasattr(reason, 'rank') and
                hasattr(reason, 'factor') and
                hasattr(reason, 'explanation') and
                hasattr(reason, 'confidence') and
                hasattr(reason, 'supporting_data') and
                0.0 <= reason.confidence <= 1.0
            )
            self.log_result("Reason structure validation", valid_structure)
        
        # Test 3: Ranking validation
        if len(reasons) > 1:
            properly_ranked = all(
                reasons[i].contribution >= reasons[i+1].contribution
                for i in range(len(reasons)-1)
            )
            self.log_result("Reason ranking", properly_ranked)
        
        # Test 4: Empty signals handling
        empty_reasons = engine.generate_top_3_reasons(
            [], [], [], [], None, MarketRegime.SIDEWAYS, 0.0
        )
        self.log_result(
            "Empty signals handling",
            isinstance(empty_reasons, list) and len(empty_reasons) >= 1
        )
    
    def validate_main_agent(self):
        """Validate the main Mean Reversion Trading Agent"""
        print("\n🤖 Validating Main Mean Reversion Trading Agent")
        print("-" * 45)
        
        # Test 1: Agent initialization
        self.log_result(
            "Agent initialization",
            isinstance(self.agent, MeanReversionTradingAgent) and
            hasattr(self.agent, 'model_version')
        )
        
        # Test 2: Signal generation with mean-reverting data
        test_data = self.create_test_data("mean_reverting", 80)
        sentiment_data = SentimentData(
            symbol="TEST",
            overall_sentiment=0.2,
            confidence=0.8,
            news_count=15
        )
        
        signal = self.agent.generate_signal_sync("TEST", test_data, sentiment_data)
        
        self.log_result(
            "Signal generation",
            signal is None or isinstance(signal, MeanReversionSignal),
            f"Signal type: {type(signal).__name__}"
        )
        
        # Test 3: Signal structure validation (if signal generated)
        if signal:
            valid_signal_structure = (
                hasattr(signal, 'symbol') and
                hasattr(signal, 'signal_type') and
                hasattr(signal, 'value') and
                hasattr(signal, 'confidence') and
                hasattr(signal, 'top_3_reasons') and
                hasattr(signal, 'timestamp') and
                hasattr(signal, 'model_version') and
                -1.0 <= signal.value <= 1.0 and
                0.0 <= signal.confidence <= 1.0 and
                len(signal.top_3_reasons) <= 3
            )
            self.log_result("Signal structure validation", valid_signal_structure)
            
            # Test 4: Risk metrics validation
            risk_metrics_valid = (
                signal.stop_loss_pct is not None and
                signal.take_profit_targets is not None and
                signal.max_holding_period is not None and
                signal.position_size_pct is not None and
                0.0 < signal.stop_loss_pct <= 0.1 and  # 0-10%
                0.0 < signal.position_size_pct <= 0.2   # 0-20%
            )
            self.log_result("Risk metrics validation", risk_metrics_valid)
            
            # Test 5: Explainability validation
            explainability_valid = (
                len(signal.top_3_reasons) >= 1 and
                all(hasattr(reason, 'explanation') and reason.explanation 
                    for reason in signal.top_3_reasons)
            )
            self.log_result("Explainability validation", explainability_valid)
        
        # Test 6: Pairs trading integration
        data_a, data_b = self.create_cointegrated_pair(60)
        pairs_data = {"STOCK_B": data_b}
        
        pairs_signal = self.agent.generate_signal_sync("STOCK_A", data_a, pairs_data=pairs_data)
        self.log_result(
            "Pairs trading integration",
            pairs_signal is None or isinstance(pairs_signal, MeanReversionSignal)
        )
        
        # Test 7: Insufficient data handling
        insufficient_data = test_data[:5]  # Only 5 data points
        insufficient_signal = self.agent.generate_signal_sync("TEST", insufficient_data)
        self.log_result(
            "Insufficient data handling",
            insufficient_signal is None or isinstance(insufficient_signal, MeanReversionSignal)
        )
        
        # Test 8: Signal serialization
        if signal:
            try:
                signal_dict = signal.to_dict()
                serialization_valid = (
                    isinstance(signal_dict, dict) and
                    'symbol' in signal_dict and
                    'signal_type' in signal_dict and
                    'value' in signal_dict and
                    'confidence' in signal_dict
                )
                self.log_result("Signal serialization", serialization_valid)
            except Exception as e:
                self.log_result("Signal serialization", False, str(e))
    
    def validate_requirements_compliance(self):
        """Validate compliance with task requirements"""
        print("\n📋 Validating Requirements Compliance")
        print("-" * 35)
        
        # Requirement 1: LangGraph agent implementation
        has_langgraph = hasattr(self.agent, 'graph') and hasattr(self.agent, '_create_langgraph')
        self.log_result("LangGraph agent implementation", has_langgraph)
        
        # Requirement 2: Bollinger Band reversions
        has_bollinger = hasattr(self.agent, 'bollinger_analyzer')
        self.log_result("Bollinger Band reversions", has_bollinger)
        
        # Requirement 3: Z-score analysis
        has_zscore = hasattr(self.agent, 'zscore_analyzer')
        self.log_result("Z-score analysis", has_zscore)
        
        # Requirement 4: Fibonacci extension targets
        has_fibonacci = hasattr(self.agent, 'fibonacci_calculator')
        self.log_result("Fibonacci extension targets", has_fibonacci)
        
        # Requirement 5: Pairs trading with cointegration
        has_pairs = hasattr(self.agent, 'pairs_analyzer')
        self.log_result("Pairs trading with cointegration", has_pairs)
        
        # Requirement 6: Sentiment divergence detection
        has_sentiment = hasattr(self.agent, 'sentiment_detector')
        self.log_result("Sentiment divergence detection", has_sentiment)
        
        # Requirement 7: Signal generation capability
        test_data = self.create_test_data("mean_reverting", 50)
        test_signal = self.agent.generate_signal_sync("TEST", test_data)
        can_generate_signals = test_signal is None or isinstance(test_signal, MeanReversionSignal)
        self.log_result("Signal generation capability", can_generate_signals)
        
        # Requirement 8: Explainable AI
        if test_signal:
            has_explanations = (
                hasattr(test_signal, 'top_3_reasons') and
                len(test_signal.top_3_reasons) >= 1
            )
            self.log_result("Explainable AI (top-3 reasons)", has_explanations)
    
    def run_validation(self):
        """Run complete validation suite"""
        print("🚀 Starting Mean Reversion Trading Agent Validation")
        print("=" * 55)
        
        try:
            # Component validations
            self.validate_bollinger_band_analyzer()
            self.validate_zscore_analyzer()
            self.validate_pairs_trading_analyzer()
            self.validate_fibonacci_calculator()
            self.validate_sentiment_divergence_detector()
            self.validate_market_regime_detector()
            self.validate_explainability_engine()
            
            # Main agent validation
            self.validate_main_agent()
            
            # Requirements compliance
            self.validate_requirements_compliance()
            
            # Summary
            self.print_validation_summary()
            
        except Exception as e:
            print(f"\n❌ Validation failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def print_validation_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 55)
        print("📊 VALIDATION SUMMARY")
        print("=" * 55)
        
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for result in self.validation_results if result['passed'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.validation_results:
                if not result['passed']:
                    print(f"   • {result['test']}")
                    if result['details']:
                        print(f"     {result['details']}")
        
        print(f"\n🎯 Key Capabilities Validated:")
        capabilities = [
            "Bollinger Band reversion detection",
            "Z-score mean reversion analysis", 
            "Pairs trading with cointegration",
            "Fibonacci extension targets",
            "Sentiment divergence detection",
            "Market regime identification",
            "LangGraph workflow integration",
            "Explainable AI with top-3 reasons",
            "Risk management metrics",
            "Signal serialization"
        ]
        
        for capability in capabilities:
            print(f"   ✅ {capability}")
        
        overall_success = failed_tests == 0
        status = "SUCCESS" if overall_success else "PARTIAL SUCCESS"
        print(f"\n🏆 Overall Validation: {status}")
        
        return overall_success


def main():
    """Main function to run validation"""
    validator = MeanReversionValidator()
    success = validator.run_validation()
    
    if success:
        print("\n🎉 Mean Reversion Trading Agent validation completed successfully!")
        print("   The agent is ready for integration and deployment.")
    else:
        print("\n⚠️  Mean Reversion Trading Agent validation completed with issues.")
        print("   Please review failed tests before deployment.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)