"""
Validation Script for Fibonacci Analysis Library

This script validates the Fibonacci analysis implementation against known
mathematical relationships and trading requirements. It performs comprehensive
testing to ensure accuracy and reliability.

Usage:
    python scripts/validate_fibonacci_analysis.py
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import time

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.fibonacci_analysis import (
    FibonacciAnalyzer, SwingPointDetector, FibonacciCalculator,
    SupportResistanceDetector, ConfluenceDetector,
    SwingPoint, analyze_fibonacci_levels, calculate_fibonacci_retracements
)


class FibonacciValidationSuite:
    """Comprehensive validation suite for Fibonacci analysis"""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
    
    def log_test(self, test_name: str, passed: bool, message: str = "", 
                execution_time: float = 0.0):
        """Log test result"""
        status = "PASS" if passed else "FAIL"
        self.test_results.append({
            'test': test_name,
            'status': status,
            'message': message,
            'execution_time': execution_time
        })
        
        print(f"[{status}] {test_name}")
        if message:
            print(f"      {message}")
        if execution_time > 0:
            print(f"      Execution time: {execution_time:.4f}s")
    
    def validate_fibonacci_mathematics(self) -> bool:
        """Validate mathematical accuracy of Fibonacci calculations"""
        print("\n" + "="*50)
        print("VALIDATING FIBONACCI MATHEMATICS")
        print("="*50)
        
        all_passed = True
        
        # Test 1: Standard Fibonacci retracement levels
        start_time = time.time()
        try:
            swing_high = SwingPoint(index=10, price=100.0, swing_type='high')
            swing_low = SwingPoint(index=5, price=80.0, swing_type='low')
            
            fib_levels = FibonacciCalculator.calculate_retracement_levels(swing_high, swing_low)
            
            # Validate swing range
            expected_range = 20.0
            actual_range = fib_levels.swing_range
            range_correct = abs(actual_range - expected_range) < 0.001
            
            # Validate specific levels
            expected_236 = 100.0 - (20.0 * 0.236)  # 95.28
            actual_236 = fib_levels.level_prices['fib_236']
            level_236_correct = abs(actual_236 - expected_236) < 0.001
            
            expected_618 = 100.0 - (20.0 * 0.618)  # 87.64
            actual_618 = fib_levels.level_prices['fib_618']
            level_618_correct = abs(actual_618 - expected_618) < 0.001
            
            test_passed = range_correct and level_236_correct and level_618_correct
            
            message = f"Range: {actual_range} (expected {expected_range}), "
            message += f"23.6%: {actual_236:.2f} (expected {expected_236:.2f}), "
            message += f"61.8%: {actual_618:.2f} (expected {expected_618:.2f})"
            
            execution_time = time.time() - start_time
            self.log_test("Fibonacci Retracement Mathematics", test_passed, message, execution_time)
            
            if not test_passed:
                all_passed = False
                
        except Exception as e:
            self.log_test("Fibonacci Retracement Mathematics", False, f"Exception: {str(e)}")
            all_passed = False
        
        # Test 2: Fibonacci extension calculations
        start_time = time.time()
        try:
            swing_high = SwingPoint(index=5, price=100.0, swing_type='high')
            swing_low = SwingPoint(index=10, price=80.0, swing_type='low')
            retracement_point = SwingPoint(index=15, price=90.0, swing_type='low')
            
            fib_ext = FibonacciCalculator.calculate_extension_levels(
                swing_high, swing_low, retracement_point
            )
            
            # Validate extension calculations
            swing_range = 20.0
            expected_1272 = 90.0 + (swing_range * 1.272)  # 115.44
            actual_1272 = fib_ext.extension_levels['ext_1272']
            ext_1272_correct = abs(actual_1272 - expected_1272) < 0.001
            
            expected_1618 = 90.0 + (swing_range * 1.618)  # 122.36
            actual_1618 = fib_ext.extension_levels['ext_1618']
            ext_1618_correct = abs(actual_1618 - expected_1618) < 0.001
            
            test_passed = ext_1272_correct and ext_1618_correct
            
            message = f"127.2%: {actual_1272:.2f} (expected {expected_1272:.2f}), "
            message += f"161.8%: {actual_1618:.2f} (expected {expected_1618:.2f})"
            
            execution_time = time.time() - start_time
            self.log_test("Fibonacci Extension Mathematics", test_passed, message, execution_time)
            
            if not test_passed:
                all_passed = False
                
        except Exception as e:
            self.log_test("Fibonacci Extension Mathematics", False, f"Exception: {str(e)}")
            all_passed = False
        
        # Test 3: Direction detection
        start_time = time.time()
        try:
            # Bullish swing (low before high)
            bullish_high = SwingPoint(index=10, price=100.0, swing_type='high')
            bullish_low = SwingPoint(index=5, price=80.0, swing_type='low')
            bullish_fib = FibonacciCalculator.calculate_retracement_levels(bullish_high, bullish_low)
            
            # Bearish swing (high before low)
            bearish_high = SwingPoint(index=5, price=100.0, swing_type='high')
            bearish_low = SwingPoint(index=10, price=80.0, swing_type='low')
            bearish_fib = FibonacciCalculator.calculate_retracement_levels(bearish_high, bearish_low)
            
            direction_correct = (bullish_fib.direction == 'bullish' and 
                               bearish_fib.direction == 'bearish')
            
            message = f"Bullish: {bullish_fib.direction}, Bearish: {bearish_fib.direction}"
            
            execution_time = time.time() - start_time
            self.log_test("Direction Detection", direction_correct, message, execution_time)
            
            if not direction_correct:
                all_passed = False
                
        except Exception as e:
            self.log_test("Direction Detection", False, f"Exception: {str(e)}")
            all_passed = False
        
        return all_passed
    
    def validate_swing_point_detection(self) -> bool:
        """Validate swing point detection accuracy"""
        print("\n" + "="*50)
        print("VALIDATING SWING POINT DETECTION")
        print("="*50)
        
        all_passed = True
        
        # Test 1: Clear swing points
        start_time = time.time()
        try:
            # Create data with obvious swing points
            high_data = np.array([10, 12, 15, 13, 11, 14, 18, 16, 12, 15, 20, 18, 14])
            low_data = np.array([8, 10, 13, 11, 9, 12, 16, 14, 10, 13, 18, 16, 12])
            
            detector = SwingPointDetector(lookback_periods=2)
            swing_highs, swing_lows = detector.detect_swing_points(high_data, low_data)
            
            # Should detect swing high at index 6 (price 18) and swing low at index 8 (price 10)
            high_at_6 = any(swing.index == 6 and abs(swing.price - 18) < 0.001 for swing in swing_highs)
            low_at_8 = any(swing.index == 8 and abs(swing.price - 10) < 0.001 for swing in swing_lows)
            
            test_passed = high_at_6 and low_at_8
            
            message = f"Detected {len(swing_highs)} highs, {len(swing_lows)} lows. "
            message += f"High at 6: {high_at_6}, Low at 8: {low_at_8}"
            
            execution_time = time.time() - start_time
            self.log_test("Clear Swing Point Detection", test_passed, message, execution_time)
            
            if not test_passed:
                all_passed = False
                
        except Exception as e:
            self.log_test("Clear Swing Point Detection", False, f"Exception: {str(e)}")
            all_passed = False
        
        # Test 2: Swing point validation (local extrema)
        start_time = time.time()
        try:
            # Generate random data
            np.random.seed(42)
            data_length = 50
            high_data = np.random.uniform(100, 120, data_length)
            low_data = high_data - np.random.uniform(1, 5, data_length)
            
            detector = SwingPointDetector(lookback_periods=3)
            swing_highs, swing_lows = detector.detect_swing_points(high_data, low_data)
            
            # Validate that all detected swing points are actually local extrema
            all_highs_valid = True
            all_lows_valid = True
            
            for swing in swing_highs:
                idx = swing.index
                lookback = detector.lookback_periods
                
                for i in range(max(0, idx - lookback), min(len(high_data), idx + lookback + 1)):
                    if i != idx and high_data[i] > swing.price:
                        all_highs_valid = False
                        break
            
            for swing in swing_lows:
                idx = swing.index
                lookback = detector.lookback_periods
                
                for i in range(max(0, idx - lookback), min(len(low_data), idx + lookback + 1)):
                    if i != idx and low_data[i] < swing.price:
                        all_lows_valid = False
                        break
            
            test_passed = all_highs_valid and all_lows_valid
            
            message = f"All {len(swing_highs)} highs valid: {all_highs_valid}, "
            message += f"All {len(swing_lows)} lows valid: {all_lows_valid}"
            
            execution_time = time.time() - start_time
            self.log_test("Swing Point Local Extrema Validation", test_passed, message, execution_time)
            
            if not test_passed:
                all_passed = False
                
        except Exception as e:
            self.log_test("Swing Point Local Extrema Validation", False, f"Exception: {str(e)}")
            all_passed = False
        
        return all_passed
    
    def validate_confluence_detection(self) -> bool:
        """Validate confluence zone detection"""
        print("\n" + "="*50)
        print("VALIDATING CONFLUENCE DETECTION")
        print("="*50)
        
        all_passed = True
        
        # Test 1: Simple confluence detection
        start_time = time.time()
        try:
            from strategies.fibonacci_analysis import detect_confluence_zones_simple
            
            # Create price levels that should form confluence zones
            price_levels = [100.0, 100.2, 95.0, 95.3, 90.0]  # Two confluence zones expected
            
            confluence_zones = detect_confluence_zones_simple(price_levels, tolerance_pct=1.0)
            
            # Should detect confluence around 100 and 95
            confluence_around_100 = any(abs(zone['price'] - 100.1) < 1.0 for zone in confluence_zones)
            confluence_around_95 = any(abs(zone['price'] - 95.15) < 1.0 for zone in confluence_zones)
            
            test_passed = confluence_around_100 and confluence_around_95
            
            message = f"Detected {len(confluence_zones)} zones. "
            message += f"Around 100: {confluence_around_100}, Around 95: {confluence_around_95}"
            
            execution_time = time.time() - start_time
            self.log_test("Simple Confluence Detection", test_passed, message, execution_time)
            
            if not test_passed:
                all_passed = False
                
        except Exception as e:
            self.log_test("Simple Confluence Detection", False, f"Exception: {str(e)}")
            all_passed = False
        
        # Test 2: Confluence strength ordering
        start_time = time.time()
        try:
            # Create levels with different confluence strengths
            price_levels = [
                100.0, 100.1, 100.2,  # Strong confluence (3 levels)
                95.0, 95.1,           # Weaker confluence (2 levels)
                90.0                  # No confluence (1 level)
            ]
            
            confluence_zones = detect_confluence_zones_simple(price_levels, tolerance_pct=0.5)
            
            # Zones should be ordered by strength (strongest first)
            strength_ordered = True
            if len(confluence_zones) > 1:
                for i in range(len(confluence_zones) - 1):
                    if confluence_zones[i]['strength'] < confluence_zones[i + 1]['strength']:
                        strength_ordered = False
                        break
            
            # Strongest zone should be around 100 (3 levels)
            strongest_around_100 = (len(confluence_zones) > 0 and 
                                   abs(confluence_zones[0]['price'] - 100.1) < 1.0)
            
            test_passed = strength_ordered and strongest_around_100
            
            message = f"Strength ordered: {strength_ordered}, Strongest around 100: {strongest_around_100}"
            
            execution_time = time.time() - start_time
            self.log_test("Confluence Strength Ordering", test_passed, message, execution_time)
            
            if not test_passed:
                all_passed = False
                
        except Exception as e:
            self.log_test("Confluence Strength Ordering", False, f"Exception: {str(e)}")
            all_passed = False
        
        return all_passed
    
    def validate_performance(self) -> bool:
        """Validate performance requirements"""
        print("\n" + "="*50)
        print("VALIDATING PERFORMANCE")
        print("="*50)
        
        all_passed = True
        
        # Test 1: Large dataset performance
        start_time = time.time()
        try:
            # Generate large dataset
            np.random.seed(42)
            data_size = 1000
            high_data = np.random.uniform(100, 120, data_size)
            low_data = high_data - np.random.uniform(1, 5, data_size)
            close_data = low_data + np.random.uniform(0, high_data - low_data)
            
            # Perform full analysis
            result = analyze_fibonacci_levels(high_data, low_data, close_data)
            
            execution_time = time.time() - start_time
            
            # Performance requirement: should complete within 5 seconds for 1000 data points
            performance_acceptable = execution_time < 5.0
            
            message = f"Processed {data_size} data points in {execution_time:.3f}s"
            
            self.log_test("Large Dataset Performance", performance_acceptable, message, execution_time)
            self.performance_metrics['large_dataset'] = execution_time
            
            if not performance_acceptable:
                all_passed = False
                
        except Exception as e:
            self.log_test("Large Dataset Performance", False, f"Exception: {str(e)}")
            all_passed = False
        
        # Test 2: Memory efficiency
        start_time = time.time()
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Perform multiple analyses
            for _ in range(10):
                np.random.seed(42)
                high_data = np.random.uniform(100, 120, 500)
                low_data = high_data - np.random.uniform(1, 5, 500)
                close_data = low_data + np.random.uniform(0, high_data - low_data)
                
                result = analyze_fibonacci_levels(high_data, low_data, close_data)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            execution_time = time.time() - start_time
            
            # Memory requirement: should not increase by more than 100MB
            memory_acceptable = memory_increase < 100
            
            message = f"Memory increase: {memory_increase:.1f}MB over 10 analyses"
            
            self.log_test("Memory Efficiency", memory_acceptable, message, execution_time)
            self.performance_metrics['memory_efficiency'] = memory_increase
            
            if not memory_acceptable:
                all_passed = False
                
        except ImportError:
            self.log_test("Memory Efficiency", True, "psutil not available, skipping memory test")
        except Exception as e:
            self.log_test("Memory Efficiency", False, f"Exception: {str(e)}")
            all_passed = False
        
        return all_passed
    
    def validate_edge_cases(self) -> bool:
        """Validate handling of edge cases"""
        print("\n" + "="*50)
        print("VALIDATING EDGE CASES")
        print("="*50)
        
        all_passed = True
        
        # Test 1: Insufficient data
        start_time = time.time()
        try:
            short_data = np.array([100, 101, 102])
            
            # Should handle gracefully
            try:
                result = analyze_fibonacci_levels(short_data, short_data, short_data)
                handled_gracefully = True
            except ValueError:
                handled_gracefully = True  # Acceptable to raise ValueError
            except Exception:
                handled_gracefully = False
            
            execution_time = time.time() - start_time
            
            message = "Handled insufficient data appropriately"
            
            self.log_test("Insufficient Data Handling", handled_gracefully, message, execution_time)
            
            if not handled_gracefully:
                all_passed = False
                
        except Exception as e:
            self.log_test("Insufficient Data Handling", False, f"Exception: {str(e)}")
            all_passed = False
        
        # Test 2: Flat data (no swings)
        start_time = time.time()
        try:
            flat_data = np.full(50, 100.0)  # All same price
            
            result = analyze_fibonacci_levels(flat_data, flat_data, flat_data)
            
            # Should return valid structure with no swings detected
            valid_structure = (isinstance(result, dict) and 
                             'swing_highs' in result and 
                             'swing_lows' in result)
            
            no_swings_detected = (len(result['swing_highs']) == 0 and 
                                len(result['swing_lows']) == 0)
            
            test_passed = valid_structure and no_swings_detected
            
            execution_time = time.time() - start_time
            
            message = f"Valid structure: {valid_structure}, No swings: {no_swings_detected}"
            
            self.log_test("Flat Data Handling", test_passed, message, execution_time)
            
            if not test_passed:
                all_passed = False
                
        except Exception as e:
            self.log_test("Flat Data Handling", False, f"Exception: {str(e)}")
            all_passed = False
        
        # Test 3: NaN values
        start_time = time.time()
        try:
            # Create longer data with NaN values
            base_data = np.linspace(100, 110, 20)
            data_with_nan = base_data.copy()
            data_with_nan[5] = np.nan  # Insert NaN in middle
            data_with_nan[15] = np.nan  # Insert another NaN
            
            # Should handle NaN values appropriately
            try:
                result = analyze_fibonacci_levels(data_with_nan, data_with_nan, data_with_nan)
                handled_nan = True
            except Exception:
                handled_nan = False
            
            execution_time = time.time() - start_time
            
            message = "Handled NaN values appropriately"
            
            self.log_test("NaN Value Handling", handled_nan, message, execution_time)
            
            if not handled_nan:
                all_passed = False
                
        except Exception as e:
            self.log_test("NaN Value Handling", False, f"Exception: {str(e)}")
            all_passed = False
        
        return all_passed
    
    def validate_integration_readiness(self) -> bool:
        """Validate readiness for integration with trading strategies"""
        print("\n" + "="*50)
        print("VALIDATING INTEGRATION READINESS")
        print("="*50)
        
        all_passed = True
        
        # Test 1: Output format consistency
        start_time = time.time()
        try:
            # Generate sample data
            np.random.seed(42)
            high_data = np.random.uniform(100, 120, 100)
            low_data = high_data - np.random.uniform(1, 5, 100)
            close_data = low_data + np.random.uniform(0, high_data - low_data)
            
            result = analyze_fibonacci_levels(high_data, low_data, close_data)
            
            # Check required output structure
            required_keys = [
                'swing_highs', 'swing_lows', 'fibonacci_levels',
                'fibonacci_extensions', 'support_resistance',
                'confluence_zones', 'analysis_metadata'
            ]
            
            all_keys_present = all(key in result for key in required_keys)
            
            # Check that confluence zones have required fields
            confluence_valid = True
            for zone in result['confluence_zones']:
                required_zone_keys = ['price', 'strength', 'components', 'level_count']
                if not all(key in zone for key in required_zone_keys):
                    confluence_valid = False
                    break
            
            test_passed = all_keys_present and confluence_valid
            
            execution_time = time.time() - start_time
            
            message = f"All keys present: {all_keys_present}, Confluence valid: {confluence_valid}"
            
            self.log_test("Output Format Consistency", test_passed, message, execution_time)
            
            if not test_passed:
                all_passed = False
                
        except Exception as e:
            self.log_test("Output Format Consistency", False, f"Exception: {str(e)}")
            all_passed = False
        
        # Test 2: Convenience function compatibility
        start_time = time.time()
        try:
            # Test direct retracement calculation
            fib_result = calculate_fibonacci_retracements(
                swing_high_price=120.0,
                swing_low_price=100.0
            )
            
            # Should return dictionary with expected structure
            required_keys = ['levels', 'level_prices', 'direction', 'swing_range']
            keys_present = all(key in fib_result for key in required_keys)
            
            # Test simple confluence detection
            from strategies.fibonacci_analysis import detect_confluence_zones_simple
            confluence_result = detect_confluence_zones_simple([100, 100.5, 95, 95.3])
            
            confluence_is_list = isinstance(confluence_result, list)
            
            test_passed = keys_present and confluence_is_list
            
            execution_time = time.time() - start_time
            
            message = f"Retracement keys: {keys_present}, Confluence list: {confluence_is_list}"
            
            self.log_test("Convenience Function Compatibility", test_passed, message, execution_time)
            
            if not test_passed:
                all_passed = False
                
        except Exception as e:
            self.log_test("Convenience Function Compatibility", False, f"Exception: {str(e)}")
            all_passed = False
        
        return all_passed
    
    def run_all_validations(self) -> bool:
        """Run all validation tests"""
        print("FIBONACCI ANALYSIS VALIDATION SUITE")
        print("LangGraph Trading System")
        print("="*60)
        
        start_time = time.time()
        
        # Run all validation categories
        math_valid = self.validate_fibonacci_mathematics()
        swing_valid = self.validate_swing_point_detection()
        confluence_valid = self.validate_confluence_detection()
        performance_valid = self.validate_performance()
        edge_cases_valid = self.validate_edge_cases()
        integration_valid = self.validate_integration_readiness()
        
        total_time = time.time() - start_time
        
        # Summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        all_passed = (math_valid and swing_valid and confluence_valid and 
                     performance_valid and edge_cases_valid and integration_valid)
        
        categories = [
            ("Fibonacci Mathematics", math_valid),
            ("Swing Point Detection", swing_valid),
            ("Confluence Detection", confluence_valid),
            ("Performance", performance_valid),
            ("Edge Cases", edge_cases_valid),
            ("Integration Readiness", integration_valid)
        ]
        
        for category, passed in categories:
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {category}")
        
        # Test statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for test in self.test_results if test['status'] == 'PASS')
        failed_tests = total_tests - passed_tests
        
        print(f"\nTest Results: {passed_tests}/{total_tests} passed ({failed_tests} failed)")
        print(f"Total execution time: {total_time:.3f}s")
        
        if self.performance_metrics:
            print(f"\nPerformance Metrics:")
            for metric, value in self.performance_metrics.items():
                if 'time' in metric or 'dataset' in metric:
                    print(f"  {metric}: {value:.3f}s")
                else:
                    print(f"  {metric}: {value:.1f}MB")
        
        # Overall result
        print(f"\nOVERALL VALIDATION: {'PASS' if all_passed else 'FAIL'}")
        
        if all_passed:
            print("\n✓ Fibonacci Analysis Library is ready for production use")
            print("✓ All mathematical calculations are accurate")
            print("✓ Performance requirements are met")
            print("✓ Edge cases are handled properly")
            print("✓ Integration interfaces are ready")
        else:
            print("\n✗ Fibonacci Analysis Library has validation failures")
            print("✗ Review failed tests before production deployment")
        
        return all_passed


def main():
    """Run the validation suite"""
    validator = FibonacciValidationSuite()
    success = validator.run_all_validations()
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())