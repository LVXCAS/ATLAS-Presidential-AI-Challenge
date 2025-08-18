"""
Comprehensive unit tests for technical indicators library

Tests all indicators against known benchmarks and validates
vectorized implementations for performance and accuracy.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List
import warnings

from strategies.technical_indicators import (
    EMA, RSI, MACD, BollingerBands, ZScore,
    IndicatorLibrary, IndicatorResult,
    calculate_ema, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_z_score
)


class TestIndicatorResult:
    """Test IndicatorResult data structure"""
    
    def test_indicator_result_creation(self):
        """Test IndicatorResult creation and methods"""
        values = np.array([1.0, 2.0, 3.0])
        params = {'period': 10}
        metadata = {'test': 'value'}
        name = 'TestIndicator'
        
        result = IndicatorResult(values, params, metadata, name)
        
        assert np.array_equal(result.values, values)
        assert result.parameters == params
        assert result.metadata == metadata
        assert result.name == name
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary"""
        values = np.array([1.0, 2.0, 3.0])
        result = IndicatorResult(values, {'period': 10}, {'test': 'value'}, 'Test')
        
        result_dict = result.to_dict()
        
        assert result_dict['values'] == [1.0, 2.0, 3.0]
        assert result_dict['parameters'] == {'period': 10}
        assert result_dict['metadata'] == {'test': 'value'}
        assert result_dict['name'] == 'Test'


class TestEMA:
    """Test Exponential Moving Average indicator"""
    
    def setup_method(self):
        """Setup test data"""
        self.ema = EMA()
        # Known test data with expected EMA values
        self.test_data = np.array([
            22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29,
            22.15, 22.39, 22.38, 22.61, 23.36, 24.05, 23.75, 23.83, 23.95, 23.63
        ])
        
        # Expected EMA(10) values calculated manually
        self.expected_ema_10 = np.array([
            22.27, 22.25, 22.21, 22.20, 22.19, 22.17, 22.19, 22.26, 22.25, 22.26,
            22.23, 22.27, 22.30, 22.38, 22.61, 22.97, 23.13, 23.28, 23.44, 23.43
        ])
    
    def test_ema_calculation_basic(self):
        """Test basic EMA calculation"""
        result = self.ema.calculate(self.test_data, period=10)
        
        assert isinstance(result, IndicatorResult)
        assert result.name == "EMA"
        assert len(result.values) == len(self.test_data)
        
        # Check parameters
        assert result.parameters['period'] == 10
        assert 'alpha' in result.parameters
    
    def test_ema_values_accuracy(self):
        """Test EMA values against known benchmarks"""
        result = self.ema.calculate(self.test_data, period=10)
        
        # Allow small numerical differences
        np.testing.assert_allclose(result.values, self.expected_ema_10, rtol=1e-2)
    
    def test_ema_custom_alpha(self):
        """Test EMA with custom alpha parameter"""
        alpha = 0.1
        result = self.ema.calculate(self.test_data, period=10, alpha=alpha)
        
        assert result.parameters['alpha'] == alpha
        assert len(result.values) == len(self.test_data)
    
    def test_ema_pandas_series_input(self):
        """Test EMA with pandas Series input"""
        series_data = pd.Series(self.test_data)
        result = self.ema.calculate(series_data, period=10)
        
        assert len(result.values) == len(series_data)
        np.testing.assert_allclose(result.values, self.expected_ema_10, rtol=1e-2)
    
    def test_ema_invalid_parameters(self):
        """Test EMA with invalid parameters"""
        with pytest.raises(ValueError):
            self.ema.calculate(self.test_data, period=10, alpha=1.5)  # Alpha > 1
        
        with pytest.raises(ValueError):
            self.ema.calculate(self.test_data, period=10, alpha=0)  # Alpha = 0
        
        with pytest.raises(ValueError):
            self.ema.calculate(self.test_data[:5], period=10)  # Insufficient data
    
    def test_ema_convenience_function(self):
        """Test convenience function"""
        result = calculate_ema(self.test_data, period=10)
        
        assert isinstance(result, IndicatorResult)
        assert result.name == "EMA"


class TestRSI:
    """Test Relative Strength Index indicator"""
    
    def setup_method(self):
        """Setup test data"""
        self.rsi = RSI()
        # Known test data for RSI calculation
        self.test_data = np.array([
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89, 46.03,
            46.83, 46.69, 46.45, 46.59, 46.3, 46.28, 46.28, 46.00, 46.03, 46.41
        ])
        
        # Expected RSI(14) values - approximate
        self.expected_rsi_range = (30, 80)  # RSI should be in reasonable range
    
    def test_rsi_calculation_basic(self):
        """Test basic RSI calculation"""
        result = self.rsi.calculate(self.test_data, period=14)
        
        assert isinstance(result, IndicatorResult)
        assert result.name == "RSI"
        assert len(result.values) == len(self.test_data)
        
        # Check parameters
        assert result.parameters['period'] == 14
    
    def test_rsi_value_range(self):
        """Test RSI values are in valid range (0-100)"""
        result = self.rsi.calculate(self.test_data, period=14)
        
        # Remove NaN values for testing
        valid_values = result.values[~np.isnan(result.values)]
        
        assert np.all(valid_values >= 0)
        assert np.all(valid_values <= 100)
    
    def test_rsi_first_value_nan(self):
        """Test that first RSI value is NaN (no change to calculate)"""
        result = self.rsi.calculate(self.test_data, period=14)
        
        assert np.isnan(result.values[0])
    
    def test_rsi_extreme_values(self):
        """Test RSI with extreme price movements"""
        # All increasing prices should give high RSI - use more dramatic increases
        increasing_data = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48])
        result = self.rsi.calculate(increasing_data, period=14)
        
        # RSI should be high (close to 100) for consistently increasing prices
        assert result.values[-1] > 80
        
        # All decreasing prices should give low RSI
        decreasing_data = np.array([24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10])
        result = self.rsi.calculate(decreasing_data, period=14)
        
        # RSI should be low (close to 0) for consistently decreasing prices
        assert result.values[-1] < 20
    
    def test_rsi_convenience_function(self):
        """Test convenience function"""
        result = calculate_rsi(self.test_data, period=14)
        
        assert isinstance(result, IndicatorResult)
        assert result.name == "RSI"


class TestMACD:
    """Test MACD indicator"""
    
    def setup_method(self):
        """Setup test data"""
        self.macd = MACD()
        # Generate test data
        np.random.seed(42)
        self.test_data = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    def test_macd_calculation_basic(self):
        """Test basic MACD calculation"""
        result = self.macd.calculate(self.test_data, fast_period=12, slow_period=26, signal_period=9)
        
        assert isinstance(result, IndicatorResult)
        assert result.name == "MACD"
        assert result.values.shape == (len(self.test_data), 3)  # MACD, Signal, Histogram
        
        # Check parameters
        assert result.parameters['fast_period'] == 12
        assert result.parameters['slow_period'] == 26
        assert result.parameters['signal_period'] == 9
        
        # Check metadata
        assert result.metadata['columns'] == ['macd_line', 'signal_line', 'histogram']
    
    def test_macd_components(self):
        """Test MACD components relationship"""
        result = self.macd.calculate(self.test_data)
        
        macd_line = result.values[:, 0]
        signal_line = result.values[:, 1]
        histogram = result.values[:, 2]
        
        # Histogram should equal MACD - Signal (allowing for small numerical errors)
        np.testing.assert_allclose(histogram, macd_line - signal_line, rtol=1e-10)
    
    def test_macd_invalid_periods(self):
        """Test MACD with invalid period parameters"""
        with pytest.raises(ValueError):
            self.macd.calculate(self.test_data, fast_period=26, slow_period=12)  # Fast >= Slow
    
    def test_macd_convenience_function(self):
        """Test convenience function"""
        result = calculate_macd(self.test_data)
        
        assert isinstance(result, IndicatorResult)
        assert result.name == "MACD"


class TestBollingerBands:
    """Test Bollinger Bands indicator"""
    
    def setup_method(self):
        """Setup test data"""
        self.bb = BollingerBands()
        # Generate test data with known statistical properties
        np.random.seed(42)
        self.test_data = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    def test_bollinger_bands_calculation_basic(self):
        """Test basic Bollinger Bands calculation"""
        result = self.bb.calculate(self.test_data, period=20, std_dev=2.0)
        
        assert isinstance(result, IndicatorResult)
        assert result.name == "BollingerBands"
        assert result.values.shape == (len(self.test_data), 3)  # Upper, Middle, Lower
        
        # Check parameters
        assert result.parameters['period'] == 20
        assert result.parameters['std_dev'] == 2.0
        
        # Check metadata
        assert result.metadata['columns'] == ['upper_band', 'middle_band', 'lower_band']
    
    def test_bollinger_bands_relationships(self):
        """Test relationships between Bollinger Bands components"""
        result = self.bb.calculate(self.test_data, period=20, std_dev=2.0)
        
        upper_band = result.values[:, 0]
        middle_band = result.values[:, 1]
        lower_band = result.values[:, 2]
        
        # Remove NaN values
        valid_mask = ~np.isnan(upper_band)
        upper_valid = upper_band[valid_mask]
        middle_valid = middle_band[valid_mask]
        lower_valid = lower_band[valid_mask]
        
        # Upper band should be above middle band
        assert np.all(upper_valid >= middle_valid)
        
        # Lower band should be below middle band
        assert np.all(lower_valid <= middle_valid)
        
        # Bands should be symmetric around middle band (approximately)
        upper_distance = upper_valid - middle_valid
        lower_distance = middle_valid - lower_valid
        np.testing.assert_allclose(upper_distance, lower_distance, rtol=1e-10)
    
    def test_bollinger_bands_invalid_std_dev(self):
        """Test Bollinger Bands with invalid standard deviation"""
        with pytest.raises(ValueError):
            self.bb.calculate(self.test_data, period=20, std_dev=0)  # std_dev <= 0
        
        with pytest.raises(ValueError):
            self.bb.calculate(self.test_data, period=20, std_dev=-1)  # std_dev < 0
    
    def test_bollinger_bands_convenience_function(self):
        """Test convenience function"""
        result = calculate_bollinger_bands(self.test_data, period=20, std_dev=2.0)
        
        assert isinstance(result, IndicatorResult)
        assert result.name == "BollingerBands"


class TestZScore:
    """Test Z-Score indicator"""
    
    def setup_method(self):
        """Setup test data"""
        self.zscore = ZScore()
        # Generate test data with known mean and std
        np.random.seed(42)
        self.test_data = np.random.normal(100, 10, 100)  # Mean=100, Std=10
    
    def test_zscore_calculation_basic(self):
        """Test basic Z-Score calculation"""
        result = self.zscore.calculate(self.test_data, period=20)
        
        assert isinstance(result, IndicatorResult)
        assert result.name == "ZScore"
        assert len(result.values) == len(self.test_data)
        
        # Check parameters
        assert result.parameters['period'] == 20
    
    def test_zscore_properties(self):
        """Test Z-Score statistical properties"""
        # Create data with known properties
        constant_data = np.full(50, 100.0)  # Constant values
        result = self.zscore.calculate(constant_data, period=20)
        
        # Z-Score of constant data should be 0 (after initial NaN period)
        valid_values = result.values[~np.isnan(result.values)]
        np.testing.assert_allclose(valid_values, 0, atol=1e-10)
    
    def test_zscore_extreme_values(self):
        """Test Z-Score with extreme values"""
        # Create data with outliers
        normal_data = np.full(30, 100.0)
        outlier_data = np.array([100, 100, 100, 100, 100, 200, 100, 100, 100, 100])  # One extreme outlier
        
        result = self.zscore.calculate(outlier_data, period=5)
        
        # The outlier should have a high absolute Z-Score
        outlier_zscore = result.values[5]  # Position of outlier
        assert abs(outlier_zscore) > 1.5  # Should be significant
    
    def test_zscore_convenience_function(self):
        """Test convenience function"""
        result = calculate_z_score(self.test_data, period=20)
        
        assert isinstance(result, IndicatorResult)
        assert result.name == "ZScore"


class TestIndicatorLibrary:
    """Test IndicatorLibrary main class"""
    
    def setup_method(self):
        """Setup test data and library"""
        self.library = IndicatorLibrary()
        np.random.seed(42)
        self.test_data = 100 + np.cumsum(np.random.randn(50) * 0.5)
    
    def test_library_initialization(self):
        """Test library initialization"""
        assert 'ema' in self.library.indicators
        assert 'rsi' in self.library.indicators
        assert 'macd' in self.library.indicators
        assert 'bollinger_bands' in self.library.indicators
        assert 'z_score' in self.library.indicators
    
    def test_calculate_indicator_by_name(self):
        """Test calculating indicators by name"""
        # Test EMA
        result = self.library.calculate_indicator('ema', self.test_data, period=10)
        assert result.name == "EMA"
        
        # Test RSI
        result = self.library.calculate_indicator('rsi', self.test_data, period=14)
        assert result.name == "RSI"
        
        # Test MACD
        result = self.library.calculate_indicator('macd', self.test_data)
        assert result.name == "MACD"
        
        # Test Bollinger Bands
        result = self.library.calculate_indicator('bollinger_bands', self.test_data, period=20)
        assert result.name == "BollingerBands"
        
        # Test Z-Score
        result = self.library.calculate_indicator('z_score', self.test_data, period=20)
        assert result.name == "ZScore"
    
    def test_calculate_unknown_indicator(self):
        """Test calculating unknown indicator"""
        with pytest.raises(ValueError):
            self.library.calculate_indicator('unknown_indicator', self.test_data)
    
    def test_calculate_multiple_indicators(self):
        """Test calculating multiple indicators at once"""
        indicators_config = {
            'ema': {'period': 10},
            'rsi': {'period': 14},
            'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        }
        
        results = self.library.calculate_multiple(self.test_data, indicators_config)
        
        assert len(results) == 3
        assert 'ema' in results
        assert 'rsi' in results
        assert 'macd' in results
        
        assert results['ema'].name == "EMA"
        assert results['rsi'].name == "RSI"
        assert results['macd'].name == "MACD"
    
    def test_calculate_multiple_with_errors(self):
        """Test calculating multiple indicators with some errors"""
        indicators_config = {
            'ema': {'period': 10},
            'invalid_indicator': {'period': 14},  # This will fail
            'rsi': {'period': 14}
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = self.library.calculate_multiple(self.test_data, indicators_config)
            
            # Should have warning about failed indicator
            assert len(w) > 0
            assert "Failed to calculate" in str(w[0].message)
        
        # Should still have valid results
        assert results['ema'] is not None
        assert results['rsi'] is not None
        assert results['invalid_indicator'] is None
    
    def test_get_available_indicators(self):
        """Test getting available indicators"""
        available = self.library.get_available_indicators()
        
        expected = ['ema', 'rsi', 'macd', 'bollinger_bands', 'z_score']
        assert set(available) == set(expected)


class TestDataValidation:
    """Test data validation functionality"""
    
    def setup_method(self):
        """Setup test indicators"""
        self.ema = EMA()
    
    def test_validate_numpy_array(self):
        """Test validation with numpy array"""
        data = np.array([1, 2, 3, 4, 5])
        validated = self.ema.validate_data(data, min_periods=3)
        
        assert isinstance(validated, np.ndarray)
        np.testing.assert_array_equal(validated, data)
    
    def test_validate_pandas_series(self):
        """Test validation with pandas Series"""
        data = pd.Series([1, 2, 3, 4, 5])
        validated = self.ema.validate_data(data, min_periods=3)
        
        assert isinstance(validated, np.ndarray)
        np.testing.assert_array_equal(validated, data.values)
    
    def test_validate_list(self):
        """Test validation with Python list"""
        data = [1, 2, 3, 4, 5]
        validated = self.ema.validate_data(data, min_periods=3)
        
        assert isinstance(validated, np.ndarray)
        np.testing.assert_array_equal(validated, np.array(data))
    
    def test_validate_insufficient_data(self):
        """Test validation with insufficient data"""
        data = np.array([1, 2])
        
        with pytest.raises(ValueError):
            self.ema.validate_data(data, min_periods=5)
    
    def test_validate_nan_data(self):
        """Test validation with NaN data"""
        data = np.array([1, 2, np.nan, 4, 5])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validated = self.ema.validate_data(data, min_periods=3)
            
            # Should have warning about NaN values
            assert len(w) > 0
            assert "NaN values" in str(w[0].message)
        
        assert isinstance(validated, np.ndarray)


class TestPerformance:
    """Test performance characteristics of indicators"""
    
    def setup_method(self):
        """Setup large test dataset"""
        np.random.seed(42)
        self.large_data = 100 + np.cumsum(np.random.randn(10000) * 0.5)
        self.library = IndicatorLibrary()
    
    def test_vectorized_performance(self):
        """Test that vectorized implementations are reasonably fast"""
        import time
        
        # Test EMA performance
        start_time = time.time()
        result = self.library.calculate_indicator('ema', self.large_data, period=20)
        ema_time = time.time() - start_time
        
        assert ema_time < 1.0  # Should complete in less than 1 second
        assert len(result.values) == len(self.large_data)
        
        # Test RSI performance
        start_time = time.time()
        result = self.library.calculate_indicator('rsi', self.large_data, period=14)
        rsi_time = time.time() - start_time
        
        assert rsi_time < 1.0  # Should complete in less than 1 second
        assert len(result.values) == len(self.large_data)
    
    def test_memory_efficiency(self):
        """Test memory efficiency of calculations"""
        # Calculate multiple indicators on large dataset
        indicators_config = {
            'ema': {'period': 20},
            'rsi': {'period': 14},
            'macd': {},
            'bollinger_bands': {'period': 20},
            'z_score': {'period': 20}
        }
        
        results = self.library.calculate_multiple(self.large_data, indicators_config)
        
        # All results should be calculated successfully
        for indicator_name, result in results.items():
            assert result is not None
            assert len(result.values) == len(self.large_data) or result.values.shape[0] == len(self.large_data)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])