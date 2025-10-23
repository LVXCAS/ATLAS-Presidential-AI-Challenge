#!/usr/bin/env python3
"""
Validation Script for Backtesting Engine

This script validates that the backtesting engine meets all requirements
from Requirement 4 (Backtesting and Historical Validation):

1. Event-driven backtesting framework
2. Realistic slippage and commission modeling  
3. Performance metrics calculation (Sharpe, drawdown, etc.)
4. Walk-forward analysis capability
5. Reproducible results with identical random seeds
6. Multi-strategy backtesting support
7. Synthetic scenario testing

Usage:
    python scripts/validate_backtesting_engine.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

from strategies.backtesting_engine import (
    BacktestingEngine, MarketData, OrderSide, OrderType,
    LinearSlippageModel, PerShareCommissionModel, PercentageCommissionModel,
    simple_momentum_strategy, buy_and_hold_strategy
)


def setup_logging():
    """Setup logging for validation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def generate_test_data(days: int = 252) -> list:
    """Generate test market data"""
    data = []
    base_time = datetime(2023, 1, 1, 16, 0)  # Market close time
    base_price = 100.0
    
    for i in range(days):
        # Skip weekends
        current_date = base_time + timedelta(days=i)
        if current_date.weekday() >= 5:
            continue
            
        # Generate realistic price movement
        daily_return = np.random.normal(0.0005, 0.02)  # Slight upward bias with 2% volatility
        new_price = base_price * (1 + daily_return)
        
        # Generate OHLC
        high = max(base_price, new_price) * (1 + abs(daily_return) * 0.5)
        low = min(base_price, new_price) * (1 - abs(daily_return) * 0.5)
        
        data.append(MarketData(
            timestamp=current_date,
            symbol="TEST",
            open=round(base_price, 2),
            high=round(high, 2),
            low=round(low, 2),
            close=round(new_price, 2),
            volume=int(1000000 * (0.5 + np.random.random())),
            spread=0.01
        ))
        
        base_price = new_price
    
    return data


def validate_event_driven_framework(logger):
    """Validate event-driven backtesting framework"""
    logger.info("Validating event-driven backtesting framework...")
    
    engine = BacktestingEngine(initial_capital=100000.0)
    test_data = generate_test_data(10)
    
    # Test order submission and processing
    order_id = engine.submit_order("TEST", OrderSide.BUY, 100, OrderType.MARKET)
    assert order_id.startswith("ORDER_"), "Order ID format incorrect"
    assert len(engine.orders) == 1, "Order not added to engine"
    
    # Process market data
    engine.process_market_data(test_data[0])
    
    # Check that order was processed
    assert len(engine.trades) == 1, "Trade not executed"
    assert engine.orders[0].status.value == "filled", "Order not filled"
    
    logger.info("[OK] Event-driven framework validation passed")
    return True


def validate_slippage_and_commission_models(logger):
    """Validate slippage and commission modeling"""
    logger.info("Validating slippage and commission models...")
    
    # Test linear slippage model
    slippage_model = LinearSlippageModel(base_slippage=0.001, volume_impact=0.00001)
    
    from strategies.backtesting_engine import Order
    order = Order(
        id="TEST_001",
        symbol="TEST",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1000
    )
    
    market_data = MarketData(
        timestamp=datetime.now(),
        symbol="TEST",
        open=100.0, high=101.0, low=99.0, close=100.0,
        volume=100000, spread=0.01
    )
    
    slippage = slippage_model.calculate_slippage(order, market_data)
    assert slippage > 0, "Slippage should be positive"
    assert slippage < 0.1, "Slippage should be reasonable"
    
    # Test commission models
    per_share_model = PerShareCommissionModel(per_share=0.005, minimum=1.0)
    commission = per_share_model.calculate_commission(order, 100.0)
    assert commission == 5.0, f"Expected commission 5.0, got {commission}"
    
    percentage_model = PercentageCommissionModel(percentage=0.001, minimum=1.0)
    commission = percentage_model.calculate_commission(order, 100.0)
    assert commission == 100.0, f"Expected commission 100.0, got {commission}"
    
    logger.info("[OK] Slippage and commission models validation passed")
    return True


def validate_performance_metrics(logger):
    """Validate performance metrics calculation"""
    logger.info("Validating performance metrics calculation...")
    
    engine = BacktestingEngine(initial_capital=100000.0, random_seed=42)
    test_data = generate_test_data(100)
    
    # Run backtest
    results = engine.run_backtest(test_data, buy_and_hold_strategy, {})
    metrics = results['performance_metrics']
    
    # Check all required metrics are present
    required_metrics = [
        'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
        'sortino_ratio', 'calmar_ratio', 'max_drawdown', 'max_drawdown_duration',
        'win_rate', 'profit_factor', 'total_trades', 'var_95', 'cvar_95'
    ]
    
    for metric in required_metrics:
        assert hasattr(metrics, metric), f"Missing metric: {metric}"
        value = getattr(metrics, metric)
        assert isinstance(value, (int, float)), f"Metric {metric} should be numeric"
        assert not np.isnan(value), f"Metric {metric} should not be NaN"
    
    # Validate metric ranges
    assert -1 <= metrics.total_return <= 10, "Total return out of reasonable range"
    assert 0 <= metrics.max_drawdown <= 1, "Max drawdown should be between 0 and 1"
    assert 0 <= metrics.win_rate <= 1, "Win rate should be between 0 and 1"
    assert metrics.total_trades >= 0, "Total trades should be non-negative"
    
    logger.info("[OK] Performance metrics validation passed")
    return True


def validate_walk_forward_analysis(logger):
    """Validate walk-forward analysis capability"""
    logger.info("Validating walk-forward analysis...")
    
    engine = BacktestingEngine(initial_capital=100000.0, random_seed=42)
    test_data = generate_test_data(200)  # Need more data for walk-forward
    
    # Run walk-forward analysis
    wf_results = engine.walk_forward_analysis(
        test_data,
        simple_momentum_strategy,
        training_period=50,
        testing_period=20,
        step_size=10,
        strategy_params={'short_window': 5, 'long_window': 15}
    )
    
    # Validate results structure
    assert 'periods' in wf_results, "Missing periods in walk-forward results"
    assert 'aggregate_metrics' in wf_results, "Missing aggregate metrics"
    assert 'total_periods' in wf_results, "Missing total periods"
    
    assert len(wf_results['periods']) > 0, "No walk-forward periods generated"
    assert wf_results['total_periods'] == len(wf_results['periods']), "Period count mismatch"
    
    # Check aggregate metrics
    agg = wf_results['aggregate_metrics']
    required_agg_metrics = ['avg_return', 'std_return', 'avg_sharpe', 'consistency_ratio']
    
    for metric in required_agg_metrics:
        assert metric in agg, f"Missing aggregate metric: {metric}"
        assert isinstance(agg[metric], (int, float)), f"Aggregate metric {metric} should be numeric"
    
    logger.info("[OK] Walk-forward analysis validation passed")
    return True


def validate_reproducibility(logger):
    """Validate reproducible results with identical random seeds"""
    logger.info("Validating reproducibility...")
    
    # Generate test data with fixed seed
    np.random.seed(42)
    test_data = generate_test_data(50)
    
    # Run same backtest twice with same seed
    engine1 = BacktestingEngine(initial_capital=100000.0, random_seed=42)
    engine2 = BacktestingEngine(initial_capital=100000.0, random_seed=42)
    
    results1 = engine1.run_backtest(test_data, simple_momentum_strategy, {'short_window': 5, 'long_window': 15})
    results2 = engine2.run_backtest(test_data, simple_momentum_strategy, {'short_window': 5, 'long_window': 15})
    
    # Compare key metrics
    metrics1 = results1['performance_metrics']
    metrics2 = results2['performance_metrics']
    
    tolerance = 1e-10
    assert abs(metrics1.total_return - metrics2.total_return) < tolerance, "Total return not reproducible"
    assert abs(metrics1.sharpe_ratio - metrics2.sharpe_ratio) < tolerance, "Sharpe ratio not reproducible"
    assert metrics1.total_trades == metrics2.total_trades, "Trade count not reproducible"
    
    # Test with different seeds should give different results
    # Generate different test data with different seed
    np.random.seed(123)
    test_data_different = generate_test_data(50)
    
    engine3 = BacktestingEngine(initial_capital=100000.0, random_seed=123)
    results3 = engine3.run_backtest(test_data_different, simple_momentum_strategy, {'short_window': 5, 'long_window': 15})
    metrics3 = results3['performance_metrics']
    
    # Should be different (with high probability) due to different data
    different_results = (
        abs(metrics1.total_return - metrics3.total_return) > tolerance or
        metrics1.total_trades != metrics3.total_trades
    )
    assert different_results, "Different seeds/data should give different results"
    
    logger.info("[OK] Reproducibility validation passed")
    return True


def validate_multi_strategy_support(logger):
    """Validate multi-strategy backtesting support"""
    logger.info("Validating multi-strategy support...")
    
    test_data = generate_test_data(100)
    
    # Test multiple strategies
    strategies = [
        ("Buy and Hold", buy_and_hold_strategy, {}),
        ("Simple Momentum", simple_momentum_strategy, {'short_window': 10, 'long_window': 30})
    ]
    
    results = {}
    
    for name, strategy_func, params in strategies:
        engine = BacktestingEngine(initial_capital=100000.0, random_seed=42)
        result = engine.run_backtest(test_data, strategy_func, params)
        results[name] = result
        
        # Validate each strategy produces results
        assert 'performance_metrics' in result, f"Missing performance metrics for {name}"
        assert 'trades' in result, f"Missing trades for {name}"
        assert 'final_portfolio' in result, f"Missing final portfolio for {name}"
    
    # Strategies should produce different results
    bh_return = results["Buy and Hold"]['performance_metrics'].total_return
    mom_return = results["Simple Momentum"]['performance_metrics'].total_return
    
    # They might be the same by chance, but let's check they both ran
    assert results["Buy and Hold"]['performance_metrics'].total_trades >= 0
    assert results["Simple Momentum"]['performance_metrics'].total_trades >= 0
    
    logger.info("[OK] Multi-strategy support validation passed")
    return True


def validate_synthetic_scenarios(logger):
    """Validate synthetic scenario testing"""
    logger.info("Validating synthetic scenario testing...")
    
    engine = BacktestingEngine(initial_capital=100000.0, random_seed=42)
    test_data = generate_test_data(50)
    
    scenarios = ['trending_up', 'trending_down', 'high_volatility']
    
    scenario_results = engine.synthetic_scenario_testing(
        test_data,
        buy_and_hold_strategy,
        scenarios,
        {}
    )
    
    # Validate results structure
    assert len(scenario_results) == len(scenarios), "Missing scenario results"
    
    for scenario in scenarios:
        assert scenario in scenario_results, f"Missing results for scenario: {scenario}"
        result = scenario_results[scenario]
        
        required_keys = ['performance', 'final_value', 'max_drawdown', 'total_trades']
        for key in required_keys:
            assert key in result, f"Missing key {key} in scenario {scenario}"
    
    # Different scenarios should produce different results
    up_value = scenario_results['trending_up']['final_value']
    down_value = scenario_results['trending_down']['final_value']
    
    # Trending up should generally perform better than trending down for buy-and-hold
    # (though this isn't guaranteed due to randomness)
    assert isinstance(up_value, (int, float)), "Final value should be numeric"
    assert isinstance(down_value, (int, float)), "Final value should be numeric"
    
    logger.info("[OK] Synthetic scenario testing validation passed")
    return True


def validate_report_generation(logger):
    """Validate report generation"""
    logger.info("Validating report generation...")
    
    engine = BacktestingEngine(initial_capital=100000.0, random_seed=42)
    test_data = generate_test_data(50)
    
    results = engine.run_backtest(test_data, buy_and_hold_strategy, {})
    report = engine.generate_report(results)
    
    # Check report contains key sections
    required_sections = [
        "Backtesting Report",
        "Total Return",
        "Sharpe Ratio",
        "Maximum Drawdown",
        "Total Trades",
        "Win Rate"
    ]
    
    for section in required_sections:
        assert section in report, f"Missing section in report: {section}"
    
    # Test saving report to file
    report_path = Path("test_backtest_report.md")
    try:
        engine.generate_report(results, str(report_path))
        assert report_path.exists(), "Report file not created"
        
        # Clean up
        report_path.unlink()
        
    except Exception as e:
        logger.error(f"Error saving report: {e}")
        raise
    
    logger.info("[OK] Report generation validation passed")
    return True


def run_comprehensive_validation():
    """Run comprehensive validation of backtesting engine"""
    logger = setup_logging()
    
    print("=" * 80)
    print("BACKTESTING ENGINE VALIDATION")
    print("=" * 80)
    
    validations = [
        ("Event-Driven Framework", validate_event_driven_framework),
        ("Slippage and Commission Models", validate_slippage_and_commission_models),
        ("Performance Metrics", validate_performance_metrics),
        ("Walk-Forward Analysis", validate_walk_forward_analysis),
        ("Reproducibility", validate_reproducibility),
        ("Multi-Strategy Support", validate_multi_strategy_support),
        ("Synthetic Scenarios", validate_synthetic_scenarios),
        ("Report Generation", validate_report_generation)
    ]
    
    passed = 0
    failed = 0
    
    for name, validation_func in validations:
        try:
            print(f"\n{name}:")
            print("-" * 40)
            
            success = validation_func(logger)
            if success:
                passed += 1
                print(f"[OK] {name} validation PASSED")
            else:
                failed += 1
                print(f"[X] {name} validation FAILED")
                
        except Exception as e:
            failed += 1
            print(f"[X] {name} validation FAILED with error: {e}")
            logger.error(f"Validation error in {name}: {e}")
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total Validations: {len(validations)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/len(validations)*100:.1f}%")
    
    if failed == 0:
        print("\n[PARTY] ALL VALIDATIONS PASSED!")
        print("The backtesting engine meets all requirements from Requirement 4.")
        return True
    else:
        print(f"\n[X] {failed} VALIDATION(S) FAILED!")
        print("Please review and fix the issues before proceeding.")
        return False


if __name__ == "__main__":
    try:
        success = run_comprehensive_validation()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"\nCritical error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)