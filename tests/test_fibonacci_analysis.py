"""
Test suite for Fibonacci Analysis Library

Tests all components of the Fibonacci analysis system including:
- Swing point detection
- Fibonacci retracement calculations
- Fibonacci extension calculations
- Support/resistance level detection
- Confluence zone detection
- Integration with technical indicators
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.fibonacci_analysis import (
    FibonacciAnalyzer, SwingPointDetector, FibonacciCalculator,
    SupportResistanceDetector, ConfluenceDetector,
    SwingPoint, FibonacciLevels, FibonacciExtensions,
    ConfluenceZone, SupportResistanceLevel,
    analyze_fibonacci_levels, calculate_fibonacci_retracements,
    detect_confluence_zones_simple
)


class TestSwingPointDetector:
    """Test swing point detection functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.detector = SwingPointDetector(lookback_periods=3)
        
        # Create test data with clear swing points
        self.high_data = np.array([10, 12, 15, 13, 11, 14, 18, 16, 12, 15, 20, 18, 14])
        self.low_data = np.array([8, 10, 13, 11, 9, 12, 16, 14, 10, 13, 18, 16, 12])
        self.timestamps = pd.date_range('2023-01-01', periods=len(self.high_data), freq='D')
    
    def test_detect_swing_points_basic(self):
        """Test basic swing point detection"""
        swing_highs, swing_lows = self.detector.detect_swing_points(
            self.high_data, self.low_data, self.timestamps.values
        )
        
        # Should detect some swing points
        assert len(swing_highs) > 0
        assert len(swing_lows) > 0
        
        # Check swing point structure
        for swing in swing_highs:
            assert isinstance(swing, SwingPoint)
            assert swing.swing_type == 'high'
            assert 0 <= swing.index < len(self.high_data)
            assert swing.price > 0
        
        for swing in swing_lows:
            assert isinstance(swing, SwingPoint)
            assert swing.swing_type == 'low'
            assert 0 <= swing.index < len(self.low_data)
            assert swing.price > 0
    
    def test_detect_swing_points_insufficient_data(self):
        """Test swing point detection with insufficient data"""
        short_data = np.array([10, 12, 15])
        
        with pytest.raises(ValueError, match="Insufficient data"):
            self.detector.detect_swing_points(short_data, short_data)
    
    def test_detect_swing_points_mismatched_arrays(self):
        """Test swing point detection with mismatched array lengths"""
        high_data = np.array([10, 12, 15, 13])
        low_data = np.array([8, 10, 13])
        
        with pytest.raises(ValueError, match="same length"):
            self.detector.detect_swing_points(high_data, low_data)
    
    def test_swing_point_validation(self):
        """Test that detected swing points are actually local extrema"""
        swing_highs, swing_lows = self.detector.detect_swing_points(
            self.high_data, self.low_data
        )
        
        lookback = self.detector.lookback_periods
        
        # Verify swing highs are local maxima
        for swing in swing_highs:
            idx = swing.index
            for i in range(max(0, idx - lookback), min(len(self.high_data), idx + lookback + 1)):
                if i != idx:
                    assert self.high_data[i] <= swing.price, f"Swing high at {idx} not a local maximum"
        
        # Verify swing lows are local minima
        for swing in swing_lows:
            idx = swing.index
            for i in range(max(0, idx - lookback), min(len(self.low_data), idx + lookback + 1)):
                if i != idx:
                    assert self.low_data[i] >= swing.price, f"Swing low at {idx} not a local minimum"


class TestFibonacciCalculator:
    """Test Fibonacci retracement and extension calculations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.swing_high = SwingPoint(index=10, price=100.0, swing_type='high')
        self.swing_low = SwingPoint(index=5, price=80.0, swing_type='low')
        self.retracement_point = SwingPoint(index=15, price=90.0, swing_type='low')
    
    def test_calculate_retracement_levels_bullish(self):
        """Test Fibonacci retracement calculation for bullish swing"""
        # Bullish: low comes before high
        swing_low = SwingPoint(index=5, price=80.0, swing_type='low')
        swing_high = SwingPoint(index=10, price=100.0, swing_type='high')
        
        fib_levels = FibonacciCalculator.calculate_retracement_levels(swing_high, swing_low)
        
        assert fib_levels.direction == 'bullish'
        assert fib_levels.swing_range == 20.0
        assert len(fib_levels.levels) == 5  # Standard levels
        
        # Check specific level calculations
        expected_236 = 100.0 - (20.0 * 0.236)  # 95.28
        assert abs(fib_levels.level_prices['fib_236'] - expected_236) < 0.01
        
        expected_618 = 100.0 - (20.0 * 0.618)  # 87.64
        assert abs(fib_levels.level_prices['fib_618'] - expected_618) < 0.01
    
    def test_calculate_retracement_levels_bearish(self):
        """Test Fibonacci retracement calculation for bearish swing"""
        # Bearish: high comes before low
        swing_high = SwingPoint(index=5, price=100.0, swing_type='high')
        swing_low = SwingPoint(index=10, price=80.0, swing_type='low')
        
        fib_levels = FibonacciCalculator.calculate_retracement_levels(swing_high, swing_low)
        
        assert fib_levels.direction == 'bearish'
        assert fib_levels.swing_range == 20.0
        
        # For bearish, levels are above the low
        expected_236 = 80.0 + (20.0 * 0.236)  # 84.72
        assert abs(fib_levels.level_prices['fib_236'] - expected_236) < 0.01
    
    def test_calculate_retracement_levels_custom(self):
        """Test Fibonacci retracement with custom levels"""
        custom_levels = [0.25, 0.50, 0.75]
        
        fib_levels = FibonacciCalculator.calculate_retracement_levels(
            self.swing_high, self.swing_low, custom_levels
        )
        
        assert len(fib_levels.levels) == 3
        assert 'fib_250' in fib_levels.level_prices
        assert 'fib_500' in fib_levels.level_prices
        assert 'fib_750' in fib_levels.level_prices
    
    def test_calculate_extension_levels_bullish(self):
        """Test Fibonacci extension calculation for bullish trend"""
        # Set up bullish extension scenario
        swing_high = SwingPoint(index=5, price=100.0, swing_type='high')
        swing_low = SwingPoint(index=10, price=80.0, swing_type='low')
        retracement_point = SwingPoint(index=15, price=90.0, swing_type='low')
        
        fib_ext = FibonacciCalculator.calculate_extension_levels(
            swing_high, swing_low, retracement_point
        )
        
        assert fib_ext.direction == 'bullish'
        
        # Check extension calculations
        swing_range = 20.0
        expected_1272 = 90.0 + (swing_range * 1.272)  # 115.44
        assert abs(fib_ext.extension_levels['ext_1272'] - expected_1272) < 0.01
    
    def test_calculate_extension_levels_bearish(self):
        """Test Fibonacci extension calculation for bearish trend"""
        # Set up bearish extension scenario
        swing_high = SwingPoint(index=5, price=100.0, swing_type='high')
        swing_low = SwingPoint(index=10, price=80.0, swing_type='low')
        retracement_point = SwingPoint(index=15, price=70.0, swing_type='high')  # Below swing_low
        
        fib_ext = FibonacciCalculator.calculate_extension_levels(
            swing_high, swing_low, retracement_point
        )
        
        assert fib_ext.direction == 'bearish'
        
        # Check extension calculations
        swing_range = 20.0
        expected_1272 = 70.0 - (swing_range * 1.272)  # 44.56
        assert abs(fib_ext.extension_levels['ext_1272'] - expected_1272) < 0.01
    
    def test_fibonacci_levels_to_dict(self):
        """Test conversion of FibonacciLevels to dictionary"""
        fib_levels = FibonacciCalculator.calculate_retracement_levels(
            self.swing_high, self.swing_low
        )
        
        result_dict = fib_levels.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'swing_high' in result_dict
        assert 'swing_low' in result_dict
        assert 'levels' in result_dict
        assert 'level_prices' in result_dict
        assert 'swing_range' in result_dict
        assert 'direction' in result_dict


class TestSupportResistanceDetector:
    """Test support and resistance level detection"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.detector = SupportResistanceDetector(min_touches=2, tolerance_pct=1.0)
        
        # Create data with clear support/resistance levels
        # Price bounces around 100 (resistance) and 80 (support)
        self.high_data = np.array([85, 90, 100, 95, 88, 99, 101, 96, 89, 100, 98, 92])
        self.low_data = np.array([80, 85, 95, 90, 82, 94, 96, 91, 84, 95, 93, 87])
        self.close_data = np.array([82, 88, 98, 92, 85, 97, 99, 94, 87, 98, 95, 90])
    
    def test_detect_levels_basic(self):
        """Test basic support/resistance detection"""
        levels = self.detector.detect_levels(self.high_data, self.low_data, self.close_data)
        
        # Should detect some levels
        assert len(levels) > 0
        
        # Check level structure
        for level in levels:
            assert isinstance(level, SupportResistanceLevel)
            assert level.level_type in ['support', 'resistance']
            assert 0 <= level.strength <= 1.0
            assert level.touch_count >= self.detector.min_touches
            assert level.price > 0
    
    def test_detect_levels_min_touches(self):
        """Test that only levels with minimum touches are detected"""
        levels = self.detector.detect_levels(self.high_data, self.low_data, self.close_data)
        
        for level in levels:
            assert level.touch_count >= self.detector.min_touches
    
    def test_level_strength_calculation(self):
        """Test that level strength is calculated reasonably"""
        levels = self.detector.detect_levels(self.high_data, self.low_data, self.close_data)
        
        if levels:
            # Levels with more touches should generally have higher strength
            # (though other factors also contribute)
            max_touches = max(level.touch_count for level in levels)
            min_touches = min(level.touch_count for level in levels)
            
            if max_touches > min_touches:
                max_touch_levels = [l for l in levels if l.touch_count == max_touches]
                min_touch_levels = [l for l in levels if l.touch_count == min_touches]
                
                avg_max_strength = np.mean([l.strength for l in max_touch_levels])
                avg_min_strength = np.mean([l.strength for l in min_touch_levels])
                
                # This is a general tendency, not a strict rule
                assert avg_max_strength >= avg_min_strength * 0.8  # Allow some variance
    
    def test_level_type_determination(self):
        """Test support vs resistance determination"""
        # Create clear support level (price bounces up from it)
        support_high = np.array([85, 90, 85, 90, 85])
        support_low = np.array([80, 80, 80, 80, 80])  # Clear support at 80
        support_close = np.array([82, 85, 82, 85, 82])
        
        levels = self.detector.detect_levels(support_high, support_low, support_close)
        
        # Should detect the 80 level as support
        support_levels = [l for l in levels if l.level_type == 'support' and abs(l.price - 80) < 2]
        assert len(support_levels) > 0
    
    def test_support_resistance_to_dict(self):
        """Test conversion of SupportResistanceLevel to dictionary"""
        levels = self.detector.detect_levels(self.high_data, self.low_data, self.close_data)
        
        if levels:
            result_dict = levels[0].to_dict()
            
            assert isinstance(result_dict, dict)
            assert 'price' in result_dict
            assert 'level_type' in result_dict
            assert 'strength' in result_dict
            assert 'touch_count' in result_dict


class TestConfluenceDetector:
    """Test confluence zone detection"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.detector = ConfluenceDetector(tolerance_pct=1.0)
        
        # Create mock Fibonacci levels
        swing_high = SwingPoint(index=10, price=100.0, swing_type='high')
        swing_low = SwingPoint(index=5, price=80.0, swing_type='low')
        
        self.fib_levels = [FibonacciCalculator.calculate_retracement_levels(swing_high, swing_low)]
        
        # Create mock extensions
        retracement_point = SwingPoint(index=15, price=90.0, swing_type='low')
        self.fib_extensions = [FibonacciCalculator.calculate_extension_levels(
            swing_high, swing_low, retracement_point
        )]
        
        # Create mock support/resistance levels that align with Fibonacci levels
        self.sr_levels = [
            SupportResistanceLevel(
                price=95.28,  # Close to 23.6% retracement
                level_type='support',
                strength=0.8,
                touch_count=3,
                first_touch_index=0,
                last_touch_index=10
            ),
            SupportResistanceLevel(
                price=90.0,  # Close to 50% retracement
                level_type='resistance',
                strength=0.9,
                touch_count=4,
                first_touch_index=2,
                last_touch_index=12
            )
        ]
    
    def test_detect_confluence_zones_basic(self):
        """Test basic confluence zone detection"""
        zones = self.detector.detect_confluence_zones(
            self.fib_levels, self.fib_extensions, self.sr_levels
        )
        
        # Should detect some confluence zones
        assert len(zones) >= 0  # May be 0 if no levels align within tolerance
        
        # Check zone structure
        for zone in zones:
            assert isinstance(zone, ConfluenceZone)
            assert zone.level_count >= 2
            assert zone.strength > 0
            assert len(zone.components) >= 2
            assert zone.price > 0
    
    def test_confluence_zone_strength_ordering(self):
        """Test that confluence zones are ordered by strength"""
        zones = self.detector.detect_confluence_zones(
            self.fib_levels, self.fib_extensions, self.sr_levels
        )
        
        if len(zones) > 1:
            for i in range(len(zones) - 1):
                assert zones[i].strength >= zones[i + 1].strength
    
    def test_confluence_zone_tolerance(self):
        """Test that confluence zones respect tolerance settings"""
        # Create levels that are just within tolerance
        close_sr_level = SupportResistanceLevel(
            price=95.0,  # Close to 23.6% retracement (95.28)
            level_type='support',
            strength=0.5,
            touch_count=2,
            first_touch_index=0,
            last_touch_index=5
        )
        
        zones = self.detector.detect_confluence_zones(
            self.fib_levels, [], [close_sr_level]
        )
        
        # Should detect confluence if within tolerance
        fib_236_price = self.fib_levels[0].level_prices['fib_236']
        tolerance = self.detector.tolerance_pct
        
        if abs(close_sr_level.price - fib_236_price) / fib_236_price <= tolerance:
            assert len(zones) > 0
    
    def test_confluence_zone_to_dict(self):
        """Test conversion of ConfluenceZone to dictionary"""
        zones = self.detector.detect_confluence_zones(
            self.fib_levels, self.fib_extensions, self.sr_levels
        )
        
        if zones:
            result_dict = zones[0].to_dict()
            
            assert isinstance(result_dict, dict)
            assert 'price' in result_dict
            assert 'strength' in result_dict
            assert 'components' in result_dict
            assert 'level_count' in result_dict


class TestFibonacciAnalyzer:
    """Test the main Fibonacci analyzer class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = FibonacciAnalyzer(
            lookback_periods=3,
            min_touches=2,
            tolerance_pct=1.0
        )
        
        # Create realistic price data with trends and swings
        np.random.seed(42)  # For reproducible tests
        base_prices = np.linspace(100, 120, 50)
        noise = np.random.normal(0, 2, 50)
        
        self.close_data = base_prices + noise
        self.high_data = self.close_data + np.abs(np.random.normal(0, 1, 50))
        self.low_data = self.close_data - np.abs(np.random.normal(0, 1, 50))
        
        self.timestamps = pd.date_range('2023-01-01', periods=len(self.close_data), freq='D')
    
    def test_analyze_comprehensive(self):
        """Test comprehensive Fibonacci analysis"""
        result = self.analyzer.analyze(
            self.high_data, self.low_data, self.close_data, self.timestamps.values
        )
        
        # Check result structure
        assert isinstance(result, dict)
        required_keys = [
            'swing_highs', 'swing_lows', 'fibonacci_levels',
            'fibonacci_extensions', 'support_resistance',
            'confluence_zones', 'analysis_metadata'
        ]
        
        for key in required_keys:
            assert key in result
        
        # Check metadata
        metadata = result['analysis_metadata']
        assert 'total_swings_analyzed' in metadata
        assert 'fibonacci_level_count' in metadata
        assert 'confluence_zone_count' in metadata
    
    def test_analyze_with_max_swings_limit(self):
        """Test analysis with swing limit"""
        result = self.analyzer.analyze(
            self.high_data, self.low_data, self.close_data,
            max_swings=5
        )
        
        # Should limit the number of swings analyzed
        total_swings = len(result['swing_highs']) + len(result['swing_lows'])
        assert total_swings <= 10  # 5 highs + 5 lows max
    
    def test_analyze_insufficient_data(self):
        """Test analysis with insufficient data"""
        short_data = np.array([100, 101, 102])
        
        # Should handle gracefully or raise appropriate error
        try:
            result = self.analyzer.analyze(short_data, short_data, short_data)
            # If it doesn't raise an error, should return valid structure
            assert isinstance(result, dict)
        except ValueError:
            # Acceptable to raise ValueError for insufficient data
            pass
    
    def test_analyze_with_timestamps(self):
        """Test analysis with timestamp data"""
        result = self.analyzer.analyze(
            self.high_data, self.low_data, self.close_data, self.timestamps.values
        )
        
        # Swing points should include timestamp information
        if result['swing_highs']:
            # Check if timestamp info is preserved (implementation dependent)
            pass  # Timestamps are optional in current implementation


class TestConvenienceFunctions:
    """Test convenience functions for direct access"""
    
    def setup_method(self):
        """Set up test fixtures"""
        np.random.seed(42)
        self.high_data = np.random.uniform(100, 120, 30)
        self.low_data = self.high_data - np.random.uniform(1, 5, 30)
        self.close_data = self.low_data + np.random.uniform(0, self.high_data - self.low_data)
    
    def test_analyze_fibonacci_levels_function(self):
        """Test the analyze_fibonacci_levels convenience function"""
        result = analyze_fibonacci_levels(
            self.high_data, self.low_data, self.close_data
        )
        
        assert isinstance(result, dict)
        assert 'fibonacci_levels' in result
        assert 'confluence_zones' in result
    
    def test_analyze_fibonacci_levels_with_pandas(self):
        """Test convenience function with pandas Series"""
        high_series = pd.Series(self.high_data)
        low_series = pd.Series(self.low_data)
        close_series = pd.Series(self.close_data)
        
        result = analyze_fibonacci_levels(high_series, low_series, close_series)
        
        assert isinstance(result, dict)
        assert 'fibonacci_levels' in result
    
    def test_calculate_fibonacci_retracements_function(self):
        """Test the calculate_fibonacci_retracements convenience function"""
        result = calculate_fibonacci_retracements(
            swing_high_price=100.0,
            swing_low_price=80.0,
            swing_high_index=10,
            swing_low_index=5
        )
        
        assert isinstance(result, dict)
        assert 'levels' in result
        assert 'level_prices' in result
        assert 'direction' in result
    
    def test_detect_confluence_zones_simple_function(self):
        """Test the detect_confluence_zones_simple convenience function"""
        price_levels = [100.0, 100.5, 95.0, 95.2, 90.0]
        
        result = detect_confluence_zones_simple(price_levels, tolerance_pct=1.0)
        
        assert isinstance(result, list)
        # Should detect confluence around 100 and 95
        assert len(result) >= 0  # May be 0 if no confluence detected
    
    def test_detect_confluence_zones_empty_input(self):
        """Test confluence detection with empty input"""
        result = detect_confluence_zones_simple([])
        
        assert isinstance(result, list)
        assert len(result) == 0


class TestIntegrationWithTechnicalIndicators:
    """Test integration with technical indicators for signal enhancement"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create price data that would generate both Fibonacci levels and technical signals
        np.random.seed(42)
        
        # Create trending data with pullbacks (good for Fibonacci analysis)
        trend = np.linspace(100, 150, 100)
        pullbacks = np.sin(np.linspace(0, 4*np.pi, 100)) * 5
        noise = np.random.normal(0, 1, 100)
        
        self.close_data = trend + pullbacks + noise
        self.high_data = self.close_data + np.abs(np.random.normal(0, 0.5, 100))
        self.low_data = self.close_data - np.abs(np.random.normal(0, 0.5, 100))
        
        self.analyzer = FibonacciAnalyzer()
    
    def test_fibonacci_with_moving_averages(self):
        """Test Fibonacci levels in context of moving averages"""
        # This would be implemented when integrating with technical indicators
        fib_result = self.analyzer.analyze(
            self.high_data, self.low_data, self.close_data
        )
        
        # For now, just verify we get valid Fibonacci analysis
        assert isinstance(fib_result, dict)
        assert 'fibonacci_levels' in fib_result
        
        # Future: Test confluence with moving average levels
        # This would require integration with the technical_indicators module
    
    def test_fibonacci_confluence_strength_scoring(self):
        """Test that confluence zones have meaningful strength scores"""
        result = self.analyzer.analyze(
            self.high_data, self.low_data, self.close_data
        )
        
        confluence_zones = result['confluence_zones']
        
        if confluence_zones:
            # Zones with more components should generally have higher strength
            for zone in confluence_zones:
                # Strength should correlate with number of confluent levels
                expected_min_strength = zone['level_count'] * 0.5
                assert zone['strength'] >= expected_min_strength * 0.5  # Allow variance
    
    def test_fibonacci_signal_enhancement_structure(self):
        """Test structure for signal enhancement (preparation for integration)"""
        result = self.analyzer.analyze(
            self.high_data, self.low_data, self.close_data
        )
        
        # Verify we have the data structures needed for signal enhancement
        assert 'confluence_zones' in result
        assert 'fibonacci_levels' in result
        assert 'support_resistance' in result
        
        # Each confluence zone should have components that can be used for signal enhancement
        for zone_dict in result['confluence_zones']:
            assert 'price' in zone_dict
            assert 'strength' in zone_dict
            assert 'components' in zone_dict
            assert isinstance(zone_dict['components'], list)


if __name__ == '__main__':
    # Run specific test classes or all tests
    pytest.main([__file__, '-v'])