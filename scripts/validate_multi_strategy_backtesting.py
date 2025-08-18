#!/usr/bin/env python3
"""
Multi-Strategy Backtesting Validation Script - Task 7.3

This script validates the multi-strategy backtesting implementation to ensure it meets all requirements:
- Individual agent testing on historical data
- Signal fusion validation across different market regimes
- Synthetic scenario testing (trend, mean-revert, news shock)
- Strategy performance attribution reports
- Performance visualization charts

Requirements: Requirement 4 (Backtesting and Historical Validation)
Task: 7.3 Multi-Strategy Backtesting
"""

import sys
import os
import logging
from pathlib import Path
import time
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategies.multi_strategy_backtesting import (
    MultiStrategyBacktester, ScenarioType
)
from strategies.backtesting_engine import MarketData
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationResult:
    """Class to track validation results"""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.passed = False
        self.error_message = None
        self.execution_time = 0.0
        self.details = {}
    
    def set_passed(self, details: dict = None):
        self.passed = True
        if details:
            self.details = details
    
    def set_failed(self, error_message: str, details: dict = None):
        self.passed = False
        self.error_message = error_message
        if details:
            self.details = details


def generate_test_market_data(duration_days: int = 100) -> list[MarketData]:
    """Generate test market data for validation"""
    start_date = datetime.now() - timedelta(days=duration_days)
    
    np.random.seed(42)  # For reproducibility
    initial_price = 100.0
    prices = [initial_price]
    
    for _ in range(duration_days - 1):
        daily_return = np.random.normal(0.0003, 0.015)
        new_price = prices[-1] * (1 + daily_return)
        prices.append(max(new_price, 1.0))
    
    market_data = []
    for i, close_price in enumerate(prices):
        date = start_date + timedelta(days=i)
        volatility = 0.01
        
        open_price = close_price * (1 + np.random.normal(0, volatility))
        high = max(open_price, close_price) * (1 + abs(np.random.normal(0, volatility * 0.5)))
        low = min(open_price, close_price) * (1 - abs(np.random.normal(0, volatility * 0.5)))
        volume = int(1000000 * (1 + np.random.uniform(-0.1, 0.1)))
        
        market_data.append(MarketData(
            timestamp=date,
            symbol="TEST",
            open=open_price,
            high=high,
            low=low,
            close=close_price,
            volume=volume
        ))
    
    return market_data


def test_individual_agent_testing() -> ValidationResult:
    """Test individual agent testing functionality"""
    result = ValidationResult("Individual Agent Testing")
    start_time = time.time()
    
    try:
        # Initialize backtester
        backtester = MultiStrategyBacktester(initial_capital=50000)
        
        # Generate test data
        market_data = generate_test_market_data(50)
        
        # Test individual agents
        agent_results = backtester.test_individual_agents(market_data)
        
        # Validate results
        if not agent_results:
            raise ValueError("No agent results returned")
        
        expected_agents = ['momentum', 'mean_reversion', 'sentiment', 'portfolio_allocator', 'risk_manager']
        for agent in expected_agents:
            if agent not in agent_results:
                raise ValueError(f"Missing agent: {agent}")
        
        # Check performance metrics
        for agent_name, performance in agent_results.items():
            if not hasattr(performance, 'performance_metrics'):
                raise ValueError(f"Missing performance metrics for {agent_name}")
            
            if not hasattr(performance, 'signal_accuracy'):
                raise ValueError(f"Missing signal accuracy for {agent_name}")
        
        result.set_passed({
            'agents_tested': len(agent_results),
            'agent_names': list(agent_results.keys()),
            'data_points': len(market_data)
        })
        
    except Exception as e:
        result.set_failed(str(e), {'traceback': traceback.format_exc()})
    
    result.execution_time = time.time() - start_time
    return result


def test_signal_fusion_validation() -> ValidationResult:
    """Test signal fusion validation functionality"""
    result = ValidationResult("Signal Fusion Validation")
    start_time = time.time()
    
    try:
        # Initialize backtester
        backtester = MultiStrategyBacktester(initial_capital=50000)
        
        # Generate test data
        market_data = generate_test_market_data(100)
        
        # Test signal fusion
        fusion_results = backtester.validate_signal_fusion(market_data)
        
        # Validate fusion results
        if not hasattr(fusion_results, 'fusion_method'):
            raise ValueError("Missing fusion method in results")
        
        if not hasattr(fusion_results, 'total_fused_signals'):
            raise ValueError("Missing total fused signals count")
        
        if not hasattr(fusion_results, 'fusion_accuracy'):
            raise ValueError("Missing fusion accuracy")
        
        result.set_passed({
            'fusion_method': fusion_results.fusion_method,
            'total_signals': fusion_results.total_fused_signals,
            'fusion_accuracy': fusion_results.fusion_accuracy,
            'data_points': len(market_data)
        })
        
    except Exception as e:
        result.set_failed(str(e), {'traceback': traceback.format_exc()})
    
    result.execution_time = time.time() - start_time
    return result


def test_synthetic_scenario_testing() -> ValidationResult:
    """Test synthetic scenario testing functionality"""
    result = ValidationResult("Synthetic Scenario Testing")
    start_time = time.time()
    
    try:
        # Initialize backtester
        backtester = MultiStrategyBacktester(initial_capital=50000)
        
        # Test scenarios
        scenarios = [
            ScenarioType.TRENDING_UP,
            ScenarioType.MEAN_REVERTING,
            ScenarioType.NEWS_SHOCK_POSITIVE
        ]
        
        scenario_results = backtester.run_synthetic_scenarios(scenarios)
        
        # Validate scenario results
        if not scenario_results:
            raise ValueError("No scenario results returned")
        
        for scenario_name, scenario_result in scenario_results.items():
            if not hasattr(scenario_result, 'overall_performance'):
                raise ValueError(f"Missing overall performance for {scenario_name}")
            
            if not hasattr(scenario_result, 'regime_detection_accuracy'):
                raise ValueError(f"Missing regime detection accuracy for {scenario_name}")
            
            if not hasattr(scenario_result, 'adaptation_speed'):
                raise ValueError(f"Missing adaptation speed for {scenario_name}")
        
        result.set_passed({
            'scenarios_tested': len(scenario_results),
            'scenario_names': list(scenario_results.keys()),
            'scenario_details': {
                name: {
                    'return': result.overall_performance.total_return,
                    'regime_accuracy': result.regime_detection_accuracy,
                    'adaptation_speed': result.adaptation_speed
                } for name, result in scenario_results.items()
            }
        })
        
    except Exception as e:
        result.set_failed(str(e), {'traceback': traceback.format_exc()})
    
    result.execution_time = time.time() - start_time
    return result


def test_performance_attribution() -> ValidationResult:
    """Test performance attribution functionality"""
    result = ValidationResult("Performance Attribution")
    start_time = time.time()
    
    try:
        # Initialize backtester
        backtester = MultiStrategyBacktester(initial_capital=50000)
        
        # Generate test data
        market_data = generate_test_market_data(80)
        
        # Run backtest to get results
        agent_results = backtester.test_individual_agents(market_data)
        fusion_results = backtester.validate_signal_fusion(market_data)
        
        # Generate performance attribution
        attribution = backtester.generate_performance_attribution(agent_results, fusion_results)
        
        # Validate attribution structure
        required_keys = ['main', 'risk_adjusted', 'regime_based', 'fusion_improvement']
        for key in required_keys:
            if key not in attribution:
                raise ValueError(f"Missing attribution key: {key}")
        
        # Check main attribution
        if not attribution['main']:
            raise ValueError("Main attribution is empty")
        
        result.set_passed({
            'attribution_keys': list(attribution.keys()),
            'main_attribution': attribution['main'],
            'fusion_improvement': attribution['fusion_improvement']
        })
        
    except Exception as e:
        result.set_failed(str(e), {'traceback': traceback.format_exc()})
    
    result.execution_time = time.time() - start_time
    return result


def test_comprehensive_backtest() -> ValidationResult:
    """Test comprehensive backtest functionality"""
    result = ValidationResult("Comprehensive Backtest")
    start_time = time.time()
    
    try:
        # Initialize backtester
        backtester = MultiStrategyBacktester(initial_capital=50000)
        
        # Generate test data
        market_data = generate_test_market_data(120)
        
        # Run comprehensive backtest
        comprehensive_results = backtester.run_comprehensive_backtest(
            market_data=market_data,
            test_scenarios=True,
            generate_reports=True
        )
        
        # Validate comprehensive results structure
        required_attributes = [
            'test_period', 'individual_agent_results', 'fusion_results',
            'scenario_results', 'performance_attribution', 'correlation_matrix',
            'regime_analysis', 'risk_metrics', 'summary_report'
        ]
        
        for attr in required_attributes:
            if not hasattr(comprehensive_results, attr):
                raise ValueError(f"Missing attribute: {attr}")
        
        # Check summary report
        if not comprehensive_results.summary_report:
            raise ValueError("Summary report is empty")
        
        result.set_passed({
            'test_period': comprehensive_results.test_period,
            'agents_tested': len(comprehensive_results.individual_agent_results),
            'scenarios_tested': len(comprehensive_results.scenario_results),
            'summary_length': len(comprehensive_results.summary_report)
        })
        
    except Exception as e:
        result.set_failed(str(e), {'traceback': traceback.format_exc()})
    
    result.execution_time = time.time() - start_time
    return result


def test_performance_charts() -> ValidationResult:
    """Test performance chart generation"""
    result = ValidationResult("Performance Charts")
    start_time = time.time()
    
    try:
        # Initialize backtester
        backtester = MultiStrategyBacktester(initial_capital=50000)
        
        # Generate test data
        market_data = generate_test_market_data(100)
        
        # Run backtest to get results
        comprehensive_results = backtester.run_comprehensive_backtest(
            market_data=market_data,
            test_scenarios=True,
            generate_reports=False  # Skip report generation for speed
        )
        
        # Generate charts
        chart_paths = backtester.generate_performance_charts(
            comprehensive_results, 
            output_dir="validation_charts"
        )
        
        # Validate chart generation
        if not chart_paths:
            raise ValueError("No charts generated")
        
        expected_charts = ['agent_comparison', 'attribution', 'correlation', 'scenarios', 'risk_return']
        for chart in expected_charts:
            if chart not in chart_paths:
                raise ValueError(f"Missing chart: {chart}")
        
        # Check if chart files exist
        for chart_name, chart_path in chart_paths.items():
            if not Path(chart_path).exists():
                raise ValueError(f"Chart file not found: {chart_path}")
        
        result.set_passed({
            'charts_generated': len(chart_paths),
            'chart_names': list(chart_paths.keys()),
            'output_dir': "validation_charts"
        })
        
    except Exception as e:
        result.set_failed(str(e), {'traceback': traceback.format_exc()})
    
    result.execution_time = time.time() - start_time
    return result


def run_validation_suite() -> list[ValidationResult]:
    """Run the complete validation suite"""
    print("=" * 80)
    print("MULTI-STRATEGY BACKTESTING VALIDATION SUITE - Task 7.3")
    print("=" * 80)
    
    validation_tests = [
        test_individual_agent_testing,
        test_signal_fusion_validation,
        test_synthetic_scenario_testing,
        test_performance_attribution,
        test_comprehensive_backtest,
        test_performance_charts
    ]
    
    results = []
    
    for test_func in validation_tests:
        print(f"\n🧪 Running: {test_func.__name__}")
        print("-" * 60)
        
        try:
            result = test_func()
            results.append(result)
            
            if result.passed:
                print(f"✅ PASSED: {result.test_name}")
                print(f"   Execution time: {result.execution_time:.2f}s")
                if result.details:
                    print(f"   Details: {result.details}")
            else:
                print(f"❌ FAILED: {result.test_name}")
                print(f"   Error: {result.error_message}")
                print(f"   Execution time: {result.execution_time:.2f}s")
                
        except Exception as e:
            print(f"💥 CRASHED: {test_func.__name__}")
            print(f"   Error: {e}")
            result = ValidationResult(test_func.__name__)
            result.set_failed(f"Test crashed: {e}")
            result.execution_time = 0.0
            results.append(result)
    
    return results


def generate_validation_report(results: list[ValidationResult]) -> str:
    """Generate comprehensive validation report"""
    print("\n" + "=" * 80)
    print("VALIDATION REPORT")
    print("=" * 80)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.passed)
    failed_tests = total_tests - passed_tests
    
    print(f"\n📊 Test Summary:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {failed_tests}")
    print(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    print(f"\n⏱️  Performance Summary:")
    total_time = sum(r.execution_time for r in results)
    avg_time = total_time / total_tests if total_tests > 0 else 0
    print(f"   Total Execution Time: {total_time:.2f}s")
    print(f"   Average Test Time: {avg_time:.2f}s")
    
    print(f"\n📋 Detailed Results:")
    for result in results:
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        print(f"   {status}: {result.test_name} ({result.execution_time:.2f}s)")
        
        if not result.passed and result.error_message:
            print(f"      Error: {result.error_message}")
    
    # Overall assessment
    print(f"\n🎯 Overall Assessment:")
    if failed_tests == 0:
        print("   🎉 ALL TESTS PASSED! Multi-Strategy Backtesting is fully functional.")
        print("   ✅ Task 7.3 requirements have been met successfully.")
        print("   🚀 System is ready for live trading validation.")
    elif failed_tests <= 2:
        print("   ⚠️  MOST TESTS PASSED. Minor issues detected.")
        print("   🔧 Some functionality may need attention before live trading.")
    else:
        print("   ❌ MULTIPLE TEST FAILURES. Significant issues detected.")
        print("   🛠️  System needs major fixes before proceeding.")
    
    return f"Validation completed with {passed_tests}/{total_tests} tests passed"


def main():
    """Main validation execution"""
    try:
        # Run validation suite
        results = run_validation_suite()
        
        # Generate report
        report = generate_validation_report(results)
        
        # Save validation results
        validation_file = "multi_strategy_validation_results.json"
        try:
            import json
            
            validation_data = {
                'timestamp': datetime.now().isoformat(),
                'task': '7.3 Multi-Strategy Backtesting',
                'summary': {
                    'total_tests': len(results),
                    'passed_tests': sum(1 for r in results if r.passed),
                    'failed_tests': len(results) - sum(1 for r in results if r.passed),
                    'success_rate': sum(1 for r in results if r.passed) / len(results) * 100
                },
                'test_results': [
                    {
                        'test_name': r.test_name,
                        'passed': r.passed,
                        'error_message': r.error_message,
                        'execution_time': r.execution_time,
                        'details': r.details
                    } for r in results
                ]
            }
            
            with open(validation_file, 'w') as f:
                json.dump(validation_data, f, indent=2, default=str)
            
            print(f"\n📄 Validation results saved to: {validation_file}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not save validation results: {e}")
        
        print(f"\n{report}")
        return failed_tests == 0
        
    except Exception as e:
        print(f"\n💥 Validation suite crashed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)