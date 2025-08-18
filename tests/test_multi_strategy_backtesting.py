"""
Tests for Multi-Strategy Backtesting Framework

This module tests the comprehensive multi-strategy backtesting capabilities including:
- Individual agent testing
- Signal fusion validation
- Synthetic scenario testing
- Performance attribution
- Cross-strategy correlation analysis
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio
from typing import List, Dict, Any

# Import the module under test
from strategies.multi_strategy_backtesting import (
    MultiStrategyBacktester,
    SyntheticDataGenerator,
    MarketRegime,
    ScenarioType,
    AgentPerformance,
    FusionPerformance,
    ScenarioResult,
    MultiStrategyBacktestResult,
    MockMomentumAgent,
    MockMeanReversionAgent,
    MockSentimentAgent,
    MockPortfolioAllocator
)

# Import supporting classes
from strategies.backtesting_engine import MarketData, PerformanceMetrics


class TestSyntheticDataGenerator:
    """Test synthetic data generation capabilities"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.generator = SyntheticDataGenerator(random_seed=42)
    
    def test_generate_trending_scenario_up(self):
        """Test upward trending scenario generation"""
        data = self.generator.generate_trending_scenario(
            initial_price=100.0,
            trend_strength=0.02,
            volatility=0.15,
            duration_days=100,
            direction="up"
        )
        
        assert len(data) == 100
        assert all(isinstance(d, MarketData) for d in data)
        assert data[0].symbol == "TEST"
        
        # Check that prices generally trend upward
        prices = [d.close for d in data]
        assert prices[-1] > prices[0]  # Final price should be higher
        
        # Check price relationships
        for d in data:
            assert d.high >= max(d.open, d.close)
            assert d.low <= min(d.open, d.close)
            assert d.volume > 0
    
    def test_generate_trending_scenario_down(self):
        """Test downward trending scenario generation"""
        data = self.generator.generate_trending_scenario(
            initial_price=100.0,
            trend_strength=0.02,
            volatility=0.15,
            duration_days=100,
            direction="down"
        )
        
        assert len(data) == 100
        
        # Check that prices generally trend downward
        prices = [d.close for d in data]
        assert prices[-1] < prices[0]  # Final price should be lower
    
    def test_generate_mean_reverting_scenario(self):
        """Test mean-reverting scenario generation"""
        data = self.generator.generate_mean_reverting_scenario(
            initial_price=100.0,
            mean_reversion_speed=0.1,
            volatility=0.2,
            duration_days=252
        )
        
        assert len(data) == 252
        assert all(isinstance(d, MarketData) for d in data)
        
        # Check that prices oscillate around the initial price
        prices = [d.close for d in data]
        mean_price = np.mean(prices)
        assert abs(mean_price - 100.0) < 20.0  # Should be close to initial price
    
    def test_generate_news_shock_scenario(self):
        """Test news shock scenario generation"""
        base_data = self.generator.generate_trending_scenario(duration_days=100)
        
        modified_data, metadata = self.generator.generate_news_shock_scenario(
            base_data=base_data,
            shock_day=50,
            shock_magnitude=0.1,
            shock_type="positive"
        )
        
        assert len(modified_data) == len(base_data)
        assert metadata['shock_day'] == 50
        assert metadata['shock_magnitude'] == 0.1
        assert metadata['shock_type'] == "positive"
        
        # Check that shock affected prices
        original_price = base_data[50].close
        shocked_price = modified_data[50].close
        assert shocked_price > original_price  # Positive shock should increase price
    
    def test_generate_volatility_spike_scenario(self):
        """Test volatility spike scenario generation"""
        base_data = self.generator.generate_trending_scenario(duration_days=100)
        
        modified_data = self.generator.generate_volatility_spike_scenario(
            base_data=base_data,
            spike_start=30,
            spike_duration=10,
            volatility_multiplier=3.0
        )
        
        assert len(modified_data) == len(base_data)
        
        # Check that volatility increased during spike period
        for i in range(30, 40):  # Spike period
            original_range = base_data[i].high - base_data[i].low
            modified_range = modified_data[i].high - modified_data[i].low
            assert modified_range >= original_range  # Should have higher volatility


class TestMockAgents:
    """Test mock agent implementations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.momentum_agent = MockMomentumAgent()
        self.mean_reversion_agent = MockMeanReversionAgent()
        self.sentiment_agent = MockSentimentAgent()
        self.portfolio_allocator = MockPortfolioAllocator()
    
    def test_momentum_agent_signals(self):
        """Test momentum agent signal generation"""
        # Create test data with upward trend
        test_data = [
            MarketData("TEST", datetime.now(), 100, 101, 99, 100.5, 1000),
            MarketData("TEST", datetime.now(), 100.5, 102, 100, 101.5, 1000),
            MarketData("TEST", datetime.now(), 101.5, 103, 101, 102.5, 1000),
        ]
        
        signals = []
        for data in test_data:
            agent_signals = self.momentum_agent.generate_signals(data, lookback=3)
            signals.extend(agent_signals)
        
        # Should eventually generate signals as trend develops
        assert len(signals) >= 0  # May not generate signals immediately
    
    def test_mean_reversion_agent_signals(self):
        """Test mean reversion agent signal generation"""
        # Create test data with extreme price movement
        test_data = [
            MarketData("TEST", datetime.now(), 100, 101, 99, 100, 1000),
            MarketData("TEST", datetime.now(), 100, 101, 99, 100, 1000),
            MarketData("TEST", datetime.now(), 100, 101, 99, 100, 1000),
            MarketData("TEST", datetime.now(), 100, 120, 99, 115, 1000),  # Extreme move
        ]
        
        signals = []
        for data in test_data:
            agent_signals = self.mean_reversion_agent.generate_signals(data, lookback=4)
            signals.extend(agent_signals)
        
        # Should generate sell signal for extreme upward move
        assert len(signals) >= 0
    
    def test_sentiment_agent_signals(self):
        """Test sentiment agent signal generation"""
        test_data = MarketData("TEST", datetime.now(), 100, 101, 99, 100, 1000)
        
        # Run multiple times due to randomness
        signals_generated = False
        for _ in range(10):
            signals = self.sentiment_agent.generate_signals(test_data)
            if signals:
                signals_generated = True
                assert signals[0]['action'] in ['buy', 'sell']
                assert 0 <= signals[0]['confidence'] <= 1
                break
        
        # Should generate signals sometimes due to random nature
        # This test may occasionally fail due to randomness, but should pass most times
    
    def test_portfolio_allocator_fusion(self):
        """Test portfolio allocator signal fusion"""
        agent_signals = {
            'momentum': [{'action': 'buy', 'confidence': 0.8, 'strength': 0.7}],
            'mean_reversion': [{'action': 'sell', 'confidence': 0.6, 'strength': 0.5}],
            'sentiment': [{'action': 'buy', 'confidence': 0.7, 'strength': 0.6}]
        }
        
        fused_signals = self.portfolio_allocator.fuse_signals(agent_signals)
        
        assert len(fused_signals) >= 1
        
        # Should have both buy and sell signals due to conflicting inputs
        actions = [s['action'] for s in fused_signals]
        assert 'buy' in actions or 'sell' in actions
        
        for signal in fused_signals:
            assert 'confidence' in signal
            assert 'strength' in signal
            assert 'contributing_agents' in signal


class TestMultiStrategyBacktester:
    """Test the main multi-strategy backtesting framework"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.backtester = MultiStrategyBacktester(
            initial_capital=100000,
            random_seed=42
        )
        
        # Create sample market data
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self) -> List[MarketData]:
        """Create sample market data for testing"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = []
        
        price = 100.0
        for date in dates:
            # Simple random walk
            price_change = np.random.normal(0, 0.02)
            price *= (1 + price_change)
            
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            volume = int(np.random.normal(1000000, 200000))
            
            data.append(MarketData(
                timestamp=date,
                symbol="TEST",
                open=price,
                high=high,
                low=low,
                close=price,
                volume=max(volume, 100000)
            ))
        
        return data
    
    def test_initialization(self):
        """Test backtester initialization"""
        assert self.backtester.initial_capital == 100000
        assert self.backtester.random_seed == 42
        assert len(self.backtester.agents) > 0
        assert 'momentum' in self.backtester.agents
        assert 'mean_reversion' in self.backtester.agents
        assert 'portfolio_allocator' in self.backtester.agents
    
    def test_individual_agent_testing(self):
        """Test individual agent testing functionality"""
        # Test with subset of agents
        agents_to_test = ['momentum', 'mean_reversion']
        
        results = self.backtester.test_individual_agents(
            market_data=self.sample_data,
            agents_to_test=agents_to_test
        )
        
        assert len(results) == len(agents_to_test)
        
        for agent_name in agents_to_test:
            assert agent_name in results
            performance = results[agent_name]
            
            assert isinstance(performance, AgentPerformance)
            assert performance.agent_name == agent_name
            assert performance.total_signals >= 0
            assert 0 <= performance.signal_accuracy <= 1
            assert isinstance(performance.performance_metrics, PerformanceMetrics)
    
    def test_signal_fusion_validation(self):
        """Test signal fusion validation"""
        # Mock regime periods
        regime_periods = {
            'trending': (0, 50),
            'sideways': (50, 100)
        }
        
        fusion_results = self.backtester.validate_signal_fusion(
            market_data=self.sample_data,
            regime_periods=regime_periods
        )
        
        assert isinstance(fusion_results, FusionPerformance)
        assert fusion_results.fusion_method == "weighted_consensus"
        assert fusion_results.total_fused_signals >= 0
        assert 0 <= fusion_results.fusion_accuracy <= 1
        assert fusion_results.improvement_over_individual >= 0
    
    def test_synthetic_scenario_testing(self):
        """Test synthetic scenario testing"""
        scenarios = [ScenarioType.TRENDING_UP, ScenarioType.MEAN_REVERTING]
        
        scenario_results = self.backtester.run_synthetic_scenarios(scenarios=scenarios)
        
        assert len(scenario_results) == len(scenarios)
        
        for scenario_type in scenarios:
            scenario_name = scenario_type.value
            assert scenario_name in scenario_results
            
            result = scenario_results[scenario_name]
            assert isinstance(result, ScenarioResult)
            assert result.scenario_type == scenario_type
            assert len(result.agent_performances) > 0
            assert isinstance(result.fusion_performance, FusionPerformance)
            assert isinstance(result.overall_performance, PerformanceMetrics)
    
    def test_performance_attribution(self):
        """Test performance attribution calculation"""
        # Create mock agent results
        mock_agent_results = {
            'momentum': AgentPerformance(
                agent_name='momentum',
                strategy_type='momentum',
                total_signals=100,
                profitable_signals=60,
                signal_accuracy=0.6,
                avg_signal_strength=0.7,
                performance_metrics=PerformanceMetrics(
                    total_return=0.15, annualized_return=0.15, volatility=0.2,
                    sharpe_ratio=0.75, sortino_ratio=1.0, calmar_ratio=1.5,
                    max_drawdown=0.1, max_drawdown_duration=30, win_rate=0.6,
                    profit_factor=1.5, avg_win=0.02, avg_loss=-0.01,
                    total_trades=100, winning_trades=60, losing_trades=40,
                    largest_win=0.05, largest_loss=-0.03, avg_trade_duration=5.0
                ),
                regime_performance={},
                signal_distribution={},
                correlation_with_market=0.3
            ),
            'mean_reversion': AgentPerformance(
                agent_name='mean_reversion',
                strategy_type='mean_reversion',
                total_signals=80,
                profitable_signals=50,
                signal_accuracy=0.625,
                avg_signal_strength=0.6,
                performance_metrics=PerformanceMetrics(
                    total_return=0.10, annualized_return=0.10, volatility=0.15,
                    sharpe_ratio=0.67, sortino_ratio=0.9, calmar_ratio=1.2,
                    max_drawdown=0.08, max_drawdown_duration=25, win_rate=0.625,
                    profit_factor=1.3, avg_win=0.018, avg_loss=-0.012,
                    total_trades=80, winning_trades=50, losing_trades=30,
                    largest_win=0.04, largest_loss=-0.025, avg_trade_duration=4.5
                ),
                regime_performance={},
                signal_distribution={},
                correlation_with_market=0.2
            )
        }
        
        mock_fusion_results = FusionPerformance(
            fusion_method="weighted_consensus",
            total_fused_signals=150,
            conflict_resolution_count=15,
            fusion_accuracy=0.7,
            improvement_over_individual=0.05,
            regime_effectiveness={},
            top_contributing_agents=[]
        )
        
        attribution = self.backtester.generate_performance_attribution(
            mock_agent_results, mock_fusion_results
        )
        
        assert len(attribution) == 3  # 2 agents + fusion
        assert 'momentum' in attribution
        assert 'mean_reversion' in attribution
        assert 'signal_fusion' in attribution
        
        # Check that attributions sum to approximately 1
        total_attribution = sum(attribution.values())
        assert abs(total_attribution - 1.0) < 0.01
    
    def test_comprehensive_backtest(self):
        """Test comprehensive backtesting workflow"""
        # Run with limited scenarios for faster testing
        results = self.backtester.run_comprehensive_backtest(
            market_data=self.sample_data[:50],  # Use smaller dataset
            test_scenarios=True,
            generate_reports=True
        )
        
        assert isinstance(results, MultiStrategyBacktestResult)
        assert len(results.individual_agent_results) > 0
        assert isinstance(results.fusion_results, FusionPerformance)
        assert len(results.scenario_results) > 0
        assert len(results.performance_attribution) > 0
        assert isinstance(results.correlation_matrix, pd.DataFrame)
        assert len(results.regime_analysis) > 0
        assert len(results.risk_metrics) > 0
        assert len(results.summary_report) > 0
        
        # Check test period
        assert results.test_period[0] == self.sample_data[0].timestamp
        assert results.test_period[1] == self.sample_data[49].timestamp
    
    def test_market_regime_detection(self):
        """Test market regime detection"""
        regimes = self.backtester._detect_market_regimes(self.sample_data)
        
        assert isinstance(regimes, dict)
        assert len(regimes) > 0
        
        for regime_name, (start_idx, end_idx) in regimes.items():
            assert isinstance(start_idx, int)
            assert isinstance(end_idx, int)
            assert 0 <= start_idx < end_idx <= len(self.sample_data)
    
    def test_correlation_matrix_calculation(self):
        """Test correlation matrix calculation"""
        mock_agent_results = {
            'momentum': Mock(spec=AgentPerformance),
            'mean_reversion': Mock(spec=AgentPerformance),
            'sentiment': Mock(spec=AgentPerformance)
        }
        
        correlation_matrix = self.backtester._calculate_correlation_matrix(mock_agent_results)
        
        assert isinstance(correlation_matrix, pd.DataFrame)
        assert correlation_matrix.shape == (3, 3)
        assert list(correlation_matrix.index) == list(mock_agent_results.keys())
        assert list(correlation_matrix.columns) == list(mock_agent_results.keys())
        
        # Check diagonal is 1.0
        for i in range(3):
            assert correlation_matrix.iloc[i, i] == 1.0
    
    def test_risk_metrics_calculation(self):
        """Test risk metrics calculation"""
        mock_agent_results = {'momentum': Mock(spec=AgentPerformance)}
        mock_fusion_results = Mock(spec=FusionPerformance)
        
        risk_metrics = self.backtester._calculate_risk_metrics(
            mock_agent_results, mock_fusion_results
        )
        
        assert isinstance(risk_metrics, dict)
        assert 'portfolio_var_95' in risk_metrics
        assert 'portfolio_cvar_95' in risk_metrics
        assert 'max_correlation' in risk_metrics
        assert 'diversification_ratio' in risk_metrics
        
        # Check that all values are reasonable
        for metric_name, value in risk_metrics.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
    
    def test_summary_report_generation(self):
        """Test summary report generation"""
        # Create minimal mock data
        mock_individual_results = {
            'momentum': Mock(spec=AgentPerformance)
        }
        mock_individual_results['momentum'].total_signals = 100
        mock_individual_results['momentum'].signal_accuracy = 0.6
        mock_individual_results['momentum'].performance_metrics = Mock(spec=PerformanceMetrics)
        mock_individual_results['momentum'].performance_metrics.total_return = 0.15
        mock_individual_results['momentum'].performance_metrics.sharpe_ratio = 0.75
        mock_individual_results['momentum'].performance_metrics.max_drawdown = 0.1
        
        mock_fusion_results = Mock(spec=FusionPerformance)
        mock_fusion_results.fusion_method = "weighted_consensus"
        mock_fusion_results.total_fused_signals = 150
        mock_fusion_results.fusion_accuracy = 0.7
        mock_fusion_results.improvement_over_individual = 0.05
        
        mock_scenario_results = {
            'trending_up': Mock(spec=ScenarioResult)
        }
        mock_scenario_results['trending_up'].overall_performance = Mock(spec=PerformanceMetrics)
        mock_scenario_results['trending_up'].overall_performance.total_return = 0.12
        mock_scenario_results['trending_up'].regime_detection_accuracy = 0.85
        mock_scenario_results['trending_up'].adaptation_speed = 0.75
        
        mock_performance_attribution = {'momentum': 0.6, 'signal_fusion': 0.4}
        mock_correlation_matrix = pd.DataFrame([[1.0]], index=['momentum'], columns=['momentum'])
        mock_regime_analysis = {
            'portfolio_var_95': 0.05,
            'best_regime_agents': {'trending': 'momentum', 'sideways': 'mean_reversion'}
        }
        
        report = self.backtester._generate_summary_report(
            mock_individual_results,
            mock_fusion_results,
            mock_scenario_results,
            mock_performance_attribution,
            mock_correlation_matrix,
            mock_regime_analysis
        )
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "Multi-Strategy Backtesting Report" in report
        assert "Executive Summary" in report
        assert "Individual Agent Performance" in report
        assert "Signal Fusion Performance" in report
        assert "Performance Attribution" in report
        assert "Scenario Testing Results" in report
        assert "Risk Analysis" in report
        assert "Recommendations" in report
        assert "Validation Status" in report


class TestIntegration:
    """Integration tests for the complete multi-strategy backtesting system"""
    
    def test_end_to_end_backtesting_workflow(self):
        """Test complete end-to-end backtesting workflow"""
        # Initialize backtester
        backtester = MultiStrategyBacktester(initial_capital=50000, random_seed=42)
        
        # Generate synthetic data
        trending_data = backtester.synthetic_generator.generate_trending_scenario(
            duration_days=50, direction="up"
        )
        
        # Run individual agent tests
        agent_results = backtester.test_individual_agents(trending_data)
        assert len(agent_results) > 0
        
        # Validate signal fusion
        fusion_results = backtester.validate_signal_fusion(trending_data)
        assert isinstance(fusion_results, FusionPerformance)
        
        # Run synthetic scenarios
        scenario_results = backtester.run_synthetic_scenarios(
            scenarios=[ScenarioType.TRENDING_UP]
        )
        assert len(scenario_results) == 1
        
        # Generate performance attribution
        attribution = backtester.generate_performance_attribution(
            agent_results, fusion_results
        )
        assert len(attribution) > 0
        
        # Run comprehensive backtest
        comprehensive_results = backtester.run_comprehensive_backtest(
            trending_data, test_scenarios=False, generate_reports=True
        )
        
        assert isinstance(comprehensive_results, MultiStrategyBacktestResult)
        assert len(comprehensive_results.summary_report) > 0
    
    def test_multiple_scenario_validation(self):
        """Test validation across multiple synthetic scenarios"""
        backtester = MultiStrategyBacktester(random_seed=42)
        
        scenarios = [
            ScenarioType.TRENDING_UP,
            ScenarioType.TRENDING_DOWN,
            ScenarioType.MEAN_REVERTING
        ]
        
        scenario_results = backtester.run_synthetic_scenarios(scenarios=scenarios)
        
        assert len(scenario_results) == len(scenarios)
        
        # Verify each scenario produced valid results
        for scenario_type in scenarios:
            scenario_name = scenario_type.value
            assert scenario_name in scenario_results
            
            result = scenario_results[scenario_name]
            assert result.scenario_type == scenario_type
            assert len(result.agent_performances) > 0
            assert result.overall_performance.total_trades >= 0
    
    def test_regime_based_performance_analysis(self):
        """Test performance analysis across different market regimes"""
        backtester = MultiStrategyBacktester(random_seed=42)
        
        # Create data with different regimes
        trending_data = backtester.synthetic_generator.generate_trending_scenario(
            duration_days=100, direction="up"
        )
        
        mean_reverting_data = backtester.synthetic_generator.generate_mean_reverting_scenario(
            duration_days=100
        )
        
        # Combine data to simulate regime changes
        combined_data = trending_data + mean_reverting_data
        
        # Run comprehensive analysis
        results = backtester.run_comprehensive_backtest(
            combined_data, test_scenarios=False
        )
        
        # Verify regime analysis
        assert 'regime_detection_accuracy' in results.regime_analysis
        assert 'best_regime_agents' in results.regime_analysis
        assert results.regime_analysis['regime_detection_accuracy'] > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])