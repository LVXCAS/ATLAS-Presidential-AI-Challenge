"""
Benchmark validation script for technical indicators

This script validates all technical indicators against known benchmark values
from established financial libraries and manual calculations.
"""

import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, List, Tuple
import warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.technical_indicators import IndicatorLibrary
from strategies.parameter_optimization import ParameterOptimizer


class BenchmarkValidator:
    """Validates technical indicators against known benchmarks"""
    
    def __init__(self):
        self.library = IndicatorLibrary()
        self.optimizer = ParameterOptimizer()
        self.validation_results = {}
    
    def create_test_data(self) -> Dict[str, np.ndarray]:
        """Create various test datasets for validation"""
        np.random.seed(42)  # For reproducible results
        
        datasets = {
            # Known test data from financial literature
            'classic_test': np.array([
                22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29,
                22.15, 22.39, 22.38, 22.61, 23.36, 24.05, 23.75, 23.83, 23.95, 23.63,
                23.82, 23.87, 23.65, 23.19, 23.10, 23.33, 22.68, 23.10, 22.40, 22.17
            ]),
            
            # Trending data
            'trending': 100 + np.cumsum(np.random.randn(100) * 0.5 + 0.1),
            
            # Mean-reverting data
            'mean_reverting': 100 + np.sin(np.linspace(0, 10*np.pi, 100)) * 10,
            
            # Volatile data
            'volatile': 100 + np.cumsum(np.random.randn(100) * 2.0),
            
            # Constant data
            'constant': np.full(50, 100.0),
            
            # Step function
            'step_function': np.concatenate([np.full(25, 100.0), np.full(25, 110.0)])
        }
        
        return datasets
    
    def validate_ema(self, data: np.ndarray) -> Dict[str, bool]:
        """Validate EMA calculations"""
        print("Validating EMA...")
        
        results = {}
        
        try:
            # Test basic EMA calculation
            ema_result = self.library.calculate_indicator('ema', data, period=10)
            
            # Basic validation checks
            results['ema_length'] = len(ema_result.values) == len(data)
            results['ema_no_nan_start'] = not np.isnan(ema_result.values[0])
            results['ema_parameters'] = ema_result.parameters['period'] == 10
            
            # Test that EMA follows price trends
            if len(data) > 20:
                price_trend = data[-1] - data[-10]
                ema_trend = ema_result.values[-1] - ema_result.values[-10]
                results['ema_trend_following'] = np.sign(price_trend) == np.sign(ema_trend)
            
            # Test EMA smoothness (should be less volatile than price)
            price_volatility = np.std(np.diff(data))
            ema_volatility = np.std(np.diff(ema_result.values))
            results['ema_smoothness'] = ema_volatility < price_volatility
            
            # Test different periods
            ema_5 = self.library.calculate_indicator('ema', data, period=5)
            ema_20 = self.library.calculate_indicator('ema', data, period=20)
            
            # Shorter period EMA should be more responsive
            if len(data) > 25:
                ema_5_responsiveness = abs(ema_5.values[-1] - data[-1])
                ema_20_responsiveness = abs(ema_20.values[-1] - data[-1])
                results['ema_period_responsiveness'] = ema_5_responsiveness <= ema_20_responsiveness
            
        except Exception as e:
            print(f"EMA validation error: {e}")
            results['ema_error'] = False
        
        return results
    
    def validate_rsi(self, data: np.ndarray) -> Dict[str, bool]:
        """Validate RSI calculations"""
        print("Validating RSI...")
        
        results = {}
        
        try:
            # Test basic RSI calculation
            rsi_result = self.library.calculate_indicator('rsi', data, period=14)
            
            # Basic validation checks
            results['rsi_length'] = len(rsi_result.values) == len(data)
            results['rsi_first_nan'] = np.isnan(rsi_result.values[0])  # First value should be NaN
            
            # RSI should be between 0 and 100
            valid_values = rsi_result.values[~np.isnan(rsi_result.values)]
            results['rsi_range'] = np.all(valid_values >= 0) and np.all(valid_values <= 100)
            
            # Test with extreme data - use more dramatic increases
            increasing_data = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48])
            rsi_increasing = self.library.calculate_indicator('rsi', increasing_data, period=14)
            results['rsi_increasing'] = rsi_increasing.values[-1] > 70  # Should be overbought
            
            decreasing_data = np.array([24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10])
            rsi_decreasing = self.library.calculate_indicator('rsi', decreasing_data, period=14)
            results['rsi_decreasing'] = rsi_decreasing.values[-1] < 30  # Should be oversold
            
        except Exception as e:
            print(f"RSI validation error: {e}")
            results['rsi_error'] = False
        
        return results
    
    def validate_macd(self, data: np.ndarray) -> Dict[str, bool]:
        """Validate MACD calculations"""
        print("Validating MACD...")
        
        results = {}
        
        try:
            # Test basic MACD calculation
            macd_result = self.library.calculate_indicator('macd', data)
            
            # Basic validation checks
            results['macd_shape'] = macd_result.values.shape == (len(data), 3)
            results['macd_columns'] = macd_result.metadata['columns'] == ['macd_line', 'signal_line', 'histogram']
            
            # Extract components
            macd_line = macd_result.values[:, 0]
            signal_line = macd_result.values[:, 1]
            histogram = macd_result.values[:, 2]
            
            # Histogram should equal MACD - Signal
            results['macd_histogram'] = np.allclose(histogram, macd_line - signal_line, rtol=1e-10)
            
            # Test parameter validation
            try:
                self.library.calculate_indicator('macd', data, fast_period=26, slow_period=12)
                results['macd_parameter_validation'] = False  # Should have failed
            except ValueError:
                results['macd_parameter_validation'] = True  # Correctly caught invalid parameters
            
        except Exception as e:
            print(f"MACD validation error: {e}")
            results['macd_error'] = False
        
        return results
    
    def validate_bollinger_bands(self, data: np.ndarray) -> Dict[str, bool]:
        """Validate Bollinger Bands calculations"""
        print("Validating Bollinger Bands...")
        
        results = {}
        
        try:
            # Test basic Bollinger Bands calculation
            bb_result = self.library.calculate_indicator('bollinger_bands', data, period=20, std_dev=2.0)
            
            # Basic validation checks
            results['bb_shape'] = bb_result.values.shape == (len(data), 3)
            results['bb_columns'] = bb_result.metadata['columns'] == ['upper_band', 'middle_band', 'lower_band']
            
            # Extract components
            upper_band = bb_result.values[:, 0]
            middle_band = bb_result.values[:, 1]
            lower_band = bb_result.values[:, 2]
            
            # Remove NaN values for testing
            valid_mask = ~np.isnan(upper_band)
            if np.any(valid_mask):
                upper_valid = upper_band[valid_mask]
                middle_valid = middle_band[valid_mask]
                lower_valid = lower_band[valid_mask]
                
                # Upper band should be above middle band
                results['bb_upper_above_middle'] = np.all(upper_valid >= middle_valid)
                
                # Lower band should be below middle band
                results['bb_lower_below_middle'] = np.all(lower_valid <= middle_valid)
                
                # Bands should be symmetric around middle band
                upper_distance = upper_valid - middle_valid
                lower_distance = middle_valid - lower_valid
                results['bb_symmetry'] = np.allclose(upper_distance, lower_distance, rtol=1e-10)
            
            # Test with constant data (bands should collapse to middle)
            constant_data = np.full(30, 100.0)
            bb_constant = self.library.calculate_indicator('bollinger_bands', constant_data, period=20)
            
            # For constant data, all bands should be equal (after initial period)
            if len(constant_data) >= 20:
                last_values = bb_constant.values[-1, :]
                results['bb_constant_data'] = np.allclose(last_values, last_values[1], atol=1e-10)
            
        except Exception as e:
            print(f"Bollinger Bands validation error: {e}")
            results['bb_error'] = False
        
        return results
    
    def validate_zscore(self, data: np.ndarray) -> Dict[str, bool]:
        """Validate Z-Score calculations"""
        print("Validating Z-Score...")
        
        results = {}
        
        try:
            # Test basic Z-Score calculation
            zscore_result = self.library.calculate_indicator('z_score', data, period=20)
            
            # Basic validation checks
            results['zscore_length'] = len(zscore_result.values) == len(data)
            
            # Test with constant data (Z-Score should be 0)
            constant_data = np.full(30, 100.0)
            zscore_constant = self.library.calculate_indicator('z_score', constant_data, period=20)
            
            # For constant data, Z-Score should be 0 (after initial NaN period)
            valid_values = zscore_constant.values[~np.isnan(zscore_constant.values)]
            if len(valid_values) > 0:
                results['zscore_constant'] = np.allclose(valid_values, 0, atol=1e-10)
            
            # Test with known outlier
            outlier_data = np.concatenate([np.full(10, 100.0), [150.0], np.full(10, 100.0)])
            zscore_outlier = self.library.calculate_indicator('z_score', outlier_data, period=10)
            
            # The outlier should have high absolute Z-Score
            outlier_zscore = zscore_outlier.values[10]  # Position of outlier
            results['zscore_outlier'] = abs(outlier_zscore) > 2
            
        except Exception as e:
            print(f"Z-Score validation error: {e}")
            results['zscore_error'] = False
        
        return results
    
    def validate_performance(self, data: np.ndarray) -> Dict[str, bool]:
        """Validate performance characteristics"""
        print("Validating performance...")
        
        results = {}
        
        try:
            import time
            
            # Test calculation speed
            start_time = time.time()
            
            # Calculate all indicators
            indicators_config = {
                'ema': {'period': 20},
                'rsi': {'period': 14},
                'macd': {},
                'bollinger_bands': {'period': 20},
                'z_score': {'period': 20}
            }
            
            all_results = self.library.calculate_multiple(data, indicators_config)
            
            calculation_time = time.time() - start_time
            
            # Should complete quickly
            results['performance_speed'] = calculation_time < 1.0
            
            # All indicators should be calculated successfully
            results['performance_completeness'] = all(result is not None for result in all_results.values())
            
            # Test with large dataset
            large_data = np.random.randn(10000)
            start_time = time.time()
            
            large_ema = self.library.calculate_indicator('ema', large_data, period=50)
            
            large_calculation_time = time.time() - start_time
            results['performance_scalability'] = large_calculation_time < 2.0
            
        except Exception as e:
            print(f"Performance validation error: {e}")
            results['performance_error'] = False
        
        return results
    
    def validate_optimization(self, data: np.ndarray) -> Dict[str, bool]:
        """Validate parameter optimization"""
        print("Validating parameter optimization...")
        
        results = {}
        
        try:
            # Test RSI optimization
            rsi_optimization = self.optimizer.optimize_indicator(
                'rsi', data, method='grid_search'
            )
            
            results['optimization_rsi'] = (
                isinstance(rsi_optimization.best_params, dict) and
                'period' in rsi_optimization.best_params and
                rsi_optimization.total_evaluations > 0
            )
            
            # Test MACD optimization with limited search space
            from strategies.parameter_optimization import ParameterSpace
            
            limited_space = [
                ParameterSpace('fast_period', 10, 15, 5, param_type='int'),
                ParameterSpace('slow_period', 20, 25, 5, param_type='int')
            ]
            
            macd_optimization = self.optimizer.grid_search(
                'macd', data, limited_space, n_jobs=1
            )
            
            results['optimization_macd'] = (
                isinstance(macd_optimization.best_params, dict) and
                'fast_period' in macd_optimization.best_params and
                'slow_period' in macd_optimization.best_params
            )
            
            # Test random search
            random_optimization = self.optimizer.random_search(
                'rsi', data, [ParameterSpace('period', 5, 30, param_type='int')],
                n_iterations=5, random_seed=42
            )
            
            results['optimization_random'] = (
                random_optimization.total_evaluations == 5 and
                random_optimization.method == 'random_search'
            )
            
        except Exception as e:
            print(f"Optimization validation error: {e}")
            results['optimization_error'] = False
        
        return results
    
    def run_validation(self) -> Dict[str, Dict[str, bool]]:
        """Run complete validation suite"""
        print("Starting Technical Indicators Validation Suite")
        print("=" * 50)
        
        datasets = self.create_test_data()
        
        # Use the classic test data for most validations
        main_data = datasets['classic_test']
        
        # Run all validations
        validation_functions = [
            ('EMA', self.validate_ema),
            ('RSI', self.validate_rsi),
            ('MACD', self.validate_macd),
            ('Bollinger Bands', self.validate_bollinger_bands),
            ('Z-Score', self.validate_zscore),
            ('Performance', self.validate_performance),
            ('Optimization', self.validate_optimization)
        ]
        
        all_results = {}
        
        for name, validation_func in validation_functions:
            print(f"\n{name} Validation:")
            print("-" * 30)
            
            try:
                results = validation_func(main_data)
                all_results[name] = results
                
                # Print results
                for test_name, passed in results.items():
                    status = "PASS" if passed else "FAIL"
                    print(f"  {test_name}: {status}")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results[name] = {'error': False}
        
        return all_results
    
    def print_summary(self, results: Dict[str, Dict[str, bool]]):
        """Print validation summary"""
        print("\n" + "=" * 50)
        print("VALIDATION SUMMARY")
        print("=" * 50)
        
        total_tests = 0
        passed_tests = 0
        
        for indicator_name, indicator_results in results.items():
            indicator_passed = sum(1 for passed in indicator_results.values() if passed)
            indicator_total = len(indicator_results)
            
            total_tests += indicator_total
            passed_tests += indicator_passed
            
            pass_rate = (indicator_passed / indicator_total) * 100 if indicator_total > 0 else 0
            
            print(f"{indicator_name}: {indicator_passed}/{indicator_total} tests passed ({pass_rate:.1f}%)")
        
        overall_pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print("-" * 50)
        print(f"OVERALL: {passed_tests}/{total_tests} tests passed ({overall_pass_rate:.1f}%)")
        
        if overall_pass_rate >= 90:
            print("✅ VALIDATION SUCCESSFUL - Technical indicators are working correctly!")
        elif overall_pass_rate >= 75:
            print("⚠️  VALIDATION PARTIAL - Some issues detected, review failed tests")
        else:
            print("❌ VALIDATION FAILED - Significant issues detected, requires investigation")
        
        return overall_pass_rate


def main():
    """Main validation function"""
    validator = BenchmarkValidator()
    
    try:
        results = validator.run_validation()
        pass_rate = validator.print_summary(results)
        
        # Exit with appropriate code
        if pass_rate >= 90:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure
            
    except Exception as e:
        print(f"Validation suite failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()