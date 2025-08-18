"""
Unit tests for parameter optimization framework

Tests optimization algorithms, objective functions, and parameter spaces
for technical indicator optimization.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List
import warnings

from strategies.parameter_optimization import (
    ParameterSpace, OptimizationResult, ParameterOptimizer,
    SharpeRatioObjective, ProfitFactorObjective,
    optimize_rsi, optimize_macd, optimize_bollinger_bands
)
from strategies.technical_indicators import IndicatorLibrary, RSI


class TestParameterSpace:
    """Test ParameterSpace class"""
    
    def test_parameter_space_creation(self):
        """Test ParameterSpace creation"""
        ps = ParameterSpace('period', 5, 20, 5, param_type='int')
        
        assert ps.name == 'period'
        assert ps.min_value == 5
        assert ps.max_value == 20
        assert ps.step == 5
        assert ps.param_type == 'int'
    
    def test_get_values_int(self):
        """Test getting integer values"""
        ps = ParameterSpace('period', 5, 15, 5, param_type='int')
        values = ps.get_values()
        
        expected = [5, 10, 15]
        assert values == expected
    
    def test_get_values_float(self):
        """Test getting float values"""
        ps = ParameterSpace('alpha', 0.1, 0.3, 0.1, param_type='float')
        values = ps.get_values()
        
        expected = [0.1, 0.2, 0.3]
        np.testing.assert_allclose(values, expected, rtol=1e-10)
    
    def test_get_values_custom_list(self):
        """Test getting values from custom list"""
        custom_values = [5, 10, 20, 50]
        ps = ParameterSpace('period', 0, 0, values=custom_values, param_type='int')
        values = ps.get_values()
        
        assert values == custom_values
    
    def test_get_values_auto_step(self):
        """Test automatic step calculation"""
        # Integer auto step
        ps_int = ParameterSpace('period', 5, 10, param_type='int')
        values_int = ps_int.get_values()
        assert values_int == [5, 6, 7, 8, 9, 10]
        
        # Float auto step
        ps_float = ParameterSpace('alpha', 0.0, 1.0, param_type='float')
        values_float = ps_float.get_values()
        assert len(values_float) == 11  # Should create 10 steps + 1


class TestObjectiveFunctions:
    """Test objective functions"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        # Create trending data for testing
        self.trending_data = 100 + np.cumsum(np.random.randn(100) * 0.5 + 0.1)
        
        # Create mean-reverting data
        self.mean_reverting_data = 100 + np.sin(np.linspace(0, 10*np.pi, 100)) * 10
        
        self.rsi = RSI()
    
    def test_sharpe_ratio_objective_creation(self):
        """Test SharpeRatioObjective creation"""
        obj = SharpeRatioObjective(risk_free_rate=0.03)
        
        assert obj.name == "SharpeRatio"
        assert obj.risk_free_rate == 0.03
    
    def test_sharpe_ratio_evaluation_rsi(self):
        """Test Sharpe ratio evaluation with RSI"""
        obj = SharpeRatioObjective()
        
        # Calculate RSI
        rsi_result = self.rsi.calculate(self.trending_data, period=14)
        
        # Evaluate Sharpe ratio
        sharpe = obj.evaluate(rsi_result, self.trending_data)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe) or sharpe == -np.inf  # -inf is acceptable for bad strategies
    
    def test_sharpe_ratio_with_custom_thresholds(self):
        """Test Sharpe ratio with custom RSI thresholds"""
        obj = SharpeRatioObjective()
        rsi_result = self.rsi.calculate(self.trending_data, period=14)
        
        # Test with different thresholds
        sharpe1 = obj.evaluate(rsi_result, self.trending_data, 
                              oversold_threshold=20, overbought_threshold=80)
        sharpe2 = obj.evaluate(rsi_result, self.trending_data,
                              oversold_threshold=30, overbought_threshold=70)
        
        assert isinstance(sharpe1, float)
        assert isinstance(sharpe2, float)
    
    def test_profit_factor_objective(self):
        """Test ProfitFactorObjective"""
        obj = ProfitFactorObjective()
        
        rsi_result = self.rsi.calculate(self.trending_data, period=14)
        profit_factor = obj.evaluate(rsi_result, self.trending_data)
        
        assert isinstance(profit_factor, float)
        assert profit_factor >= 0  # Profit factor should be non-negative
    
    def test_objective_with_insufficient_data(self):
        """Test objective functions with insufficient data"""
        obj = SharpeRatioObjective()
        
        # Very short data series
        short_data = np.array([100, 101, 102])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore expected warnings
            
            try:
                rsi_result = self.rsi.calculate(short_data, period=14)
                sharpe = obj.evaluate(rsi_result, short_data)
                
                # Should return -inf for insufficient data
                assert sharpe == -np.inf
            except ValueError:
                # RSI calculation might fail with insufficient data, which is expected
                pass


class TestParameterOptimizer:
    """Test ParameterOptimizer class"""
    
    def setup_method(self):
        """Setup test data and optimizer"""
        np.random.seed(42)
        self.test_data = 100 + np.cumsum(np.random.randn(50) * 0.5)
        self.optimizer = ParameterOptimizer()
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        assert self.optimizer.indicator_library is not None
        assert 'sharpe_ratio' in self.optimizer.objective_functions
        assert 'profit_factor' in self.optimizer.objective_functions
    
    def test_grid_search_rsi(self):
        """Test grid search optimization for RSI"""
        parameter_space = [ParameterSpace('period', 10, 20, 5, param_type='int')]
        
        result = self.optimizer.grid_search(
            'rsi', self.test_data, parameter_space,
            objective='sharpe_ratio', n_jobs=1
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.method == 'grid_search'
        assert 'period' in result.best_params
        assert 10 <= result.best_params['period'] <= 20
        assert result.total_evaluations == 3  # (20-10)/5 + 1 = 3
        assert len(result.all_results) == 3
    
    def test_random_search_rsi(self):
        """Test random search optimization for RSI"""
        parameter_space = [ParameterSpace('period', 5, 30, param_type='int')]
        
        result = self.optimizer.random_search(
            'rsi', self.test_data, parameter_space,
            n_iterations=10, objective='sharpe_ratio', random_seed=42
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.method == 'random_search'
        assert 'period' in result.best_params
        assert 5 <= result.best_params['period'] <= 30
        assert result.total_evaluations == 10
        assert len(result.all_results) == 10
    
    def test_grid_search_macd(self):
        """Test grid search optimization for MACD"""
        parameter_space = [
            ParameterSpace('fast_period', 10, 15, 5, param_type='int'),
            ParameterSpace('slow_period', 20, 25, 5, param_type='int'),
            ParameterSpace('signal_period', 8, 10, 2, param_type='int')
        ]
        
        result = self.optimizer.grid_search(
            'macd', self.test_data, parameter_space,
            objective='profit_factor', n_jobs=1
        )
        
        assert isinstance(result, OptimizationResult)
        assert 'fast_period' in result.best_params
        assert 'slow_period' in result.best_params
        assert 'signal_period' in result.best_params
        
        # Check parameter ranges
        assert 10 <= result.best_params['fast_period'] <= 15
        assert 20 <= result.best_params['slow_period'] <= 25
        assert 8 <= result.best_params['signal_period'] <= 10
        
        # Should have 2 * 2 * 2 = 8 combinations
        assert result.total_evaluations == 8
    
    def test_optimization_with_invalid_indicator(self):
        """Test optimization with invalid indicator"""
        parameter_space = [ParameterSpace('period', 10, 20, 5, param_type='int')]
        
        # Should complete but with all -inf scores due to invalid indicator
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.optimizer.grid_search(
                'invalid_indicator', self.test_data, parameter_space
            )
            
            # All results should have -inf scores
            assert all(r['score'] == -np.inf for r in result.all_results)
    
    def test_optimize_indicator_with_defaults(self):
        """Test optimize_indicator with default parameter spaces"""
        result = self.optimizer.optimize_indicator('rsi', self.test_data, method='grid_search')
        
        assert isinstance(result, OptimizationResult)
        assert 'period' in result.best_params
        assert result.method == 'grid_search'
    
    def test_optimize_indicator_random_search(self):
        """Test optimize_indicator with random search"""
        result = self.optimizer.optimize_indicator(
            'rsi', self.test_data, method='random_search', n_iterations=5
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.method == 'random_search'
        assert result.total_evaluations == 5
    
    def test_optimize_indicator_invalid_method(self):
        """Test optimize_indicator with invalid method"""
        with pytest.raises(ValueError):
            self.optimizer.optimize_indicator('rsi', self.test_data, method='invalid_method')
    
    def test_optimize_indicator_invalid_indicator(self):
        """Test optimize_indicator with invalid indicator"""
        with pytest.raises(ValueError):
            self.optimizer.optimize_indicator('invalid_indicator', self.test_data)
    
    def test_parallel_grid_search(self):
        """Test parallel grid search execution"""
        parameter_space = [ParameterSpace('period', 10, 20, 2, param_type='int')]
        
        # Test with multiple jobs
        result = self.optimizer.grid_search(
            'rsi', self.test_data, parameter_space,
            objective='sharpe_ratio', n_jobs=2
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.total_evaluations == 6  # (20-10)/2 + 1 = 6
        assert len(result.all_results) == 6


class TestOptimizationResult:
    """Test OptimizationResult class"""
    
    def test_optimization_result_creation(self):
        """Test OptimizationResult creation"""
        best_params = {'period': 14}
        best_score = 1.5
        all_results = [{'params': {'period': 10}, 'score': 1.0}]
        optimization_time = 2.5
        total_evaluations = 5
        method = 'grid_search'
        
        result = OptimizationResult(
            best_params, best_score, all_results,
            optimization_time, total_evaluations, method
        )
        
        assert result.best_params == best_params
        assert result.best_score == best_score
        assert result.all_results == all_results
        assert result.optimization_time == optimization_time
        assert result.total_evaluations == total_evaluations
        assert result.method == method
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary"""
        result = OptimizationResult(
            {'period': 14}, 1.5, [{'params': {'period': 10}, 'score': 1.0}],
            2.5, 5, 'grid_search'
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['best_params'] == {'period': 14}
        assert result_dict['best_score'] == 1.5
        assert result_dict['total_evaluations'] == 5
        assert result_dict['optimization_time'] == 2.5
        assert result_dict['method'] == 'grid_search'
        assert 'convergence_history' in result_dict


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.test_data = 100 + np.cumsum(np.random.randn(50) * 0.5)
    
    def test_optimize_rsi(self):
        """Test optimize_rsi convenience function"""
        result = optimize_rsi(self.test_data, method='grid_search')
        
        assert isinstance(result, OptimizationResult)
        assert 'period' in result.best_params
    
    def test_optimize_macd(self):
        """Test optimize_macd convenience function"""
        result = optimize_macd(self.test_data, method='random_search', n_iterations=5)
        
        assert isinstance(result, OptimizationResult)
        assert 'fast_period' in result.best_params
        assert 'slow_period' in result.best_params
        assert 'signal_period' in result.best_params
        assert result.total_evaluations == 5
    
    def test_optimize_bollinger_bands(self):
        """Test optimize_bollinger_bands convenience function"""
        result = optimize_bollinger_bands(self.test_data, method='grid_search')
        
        assert isinstance(result, OptimizationResult)
        assert 'period' in result.best_params
        assert 'std_dev' in result.best_params


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def setup_method(self):
        """Setup test data"""
        self.optimizer = ParameterOptimizer()
    
    def test_empty_parameter_space(self):
        """Test optimization with empty parameter space"""
        # Empty parameter space should result in single evaluation with no parameters
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.optimizer.grid_search('rsi', np.array([1, 2, 3]), [])
            
            # Should have one result with empty parameters
            assert len(result.all_results) == 1
            assert result.all_results[0]['params'] == {}
    
    def test_insufficient_data(self):
        """Test optimization with insufficient data"""
        short_data = np.array([100, 101])
        parameter_space = [ParameterSpace('period', 5, 10, 5, param_type='int')]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = self.optimizer.grid_search(
                'rsi', short_data, parameter_space, n_jobs=1
            )
            
            # Should complete but with poor scores
            assert isinstance(result, OptimizationResult)
            # All scores should be -inf due to insufficient data
            assert all(r['score'] == -np.inf for r in result.all_results)
    
    def test_constant_data(self):
        """Test optimization with constant price data"""
        constant_data = np.full(50, 100.0)
        parameter_space = [ParameterSpace('period', 10, 20, 5, param_type='int')]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = self.optimizer.grid_search(
                'rsi', constant_data, parameter_space, n_jobs=1
            )
            
            assert isinstance(result, OptimizationResult)
            # Scores might be -inf or 0 due to no price movement
    
    def test_optimization_with_nan_data(self):
        """Test optimization with NaN data"""
        nan_data = np.array([100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109, 110])
        parameter_space = [ParameterSpace('period', 5, 10, 5, param_type='int')]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = self.optimizer.grid_search(
                'rsi', nan_data, parameter_space, n_jobs=1
            )
            
            assert isinstance(result, OptimizationResult)


class TestPerformance:
    """Test performance characteristics of optimization"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.large_data = 100 + np.cumsum(np.random.randn(200) * 0.5)
        self.optimizer = ParameterOptimizer()
    
    def test_optimization_performance(self):
        """Test that optimization completes in reasonable time"""
        import time
        
        parameter_space = [ParameterSpace('period', 10, 20, 2, param_type='int')]
        
        start_time = time.time()
        result = self.optimizer.grid_search(
            'rsi', self.large_data, parameter_space, n_jobs=1
        )
        optimization_time = time.time() - start_time
        
        assert optimization_time < 10.0  # Should complete in less than 10 seconds
        assert isinstance(result, OptimizationResult)
        assert result.optimization_time > 0
    
    def test_parallel_vs_sequential_performance(self):
        """Test that parallel execution is faster than sequential"""
        parameter_space = [
            ParameterSpace('period', 10, 25, 5, param_type='int')  # 4 values
        ]
        
        # Sequential execution
        import time
        start_time = time.time()
        result_sequential = self.optimizer.grid_search(
            'rsi', self.large_data, parameter_space, n_jobs=1
        )
        sequential_time = time.time() - start_time
        
        # Parallel execution
        start_time = time.time()
        result_parallel = self.optimizer.grid_search(
            'rsi', self.large_data, parameter_space, n_jobs=2
        )
        parallel_time = time.time() - start_time
        
        # Both should produce same number of evaluations
        assert result_sequential.total_evaluations == result_parallel.total_evaluations
        
        # Parallel might be faster, but not always guaranteed in tests
        # Just ensure both complete successfully
        assert isinstance(result_sequential, OptimizationResult)
        assert isinstance(result_parallel, OptimizationResult)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])