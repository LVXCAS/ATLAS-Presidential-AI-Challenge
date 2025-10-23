#!/usr/bin/env python3
"""
Multi-Strategy Backtesting Demo - Task 7.3 Implementation

This demo showcases the comprehensive multi-strategy backtesting framework including:
- Individual agent testing on historical data
- Signal fusion validation across different market regimes
- Synthetic scenario testing (trend, mean-revert, news shock, etc.)
- Strategy performance attribution reports
- Performance visualization charts

Requirements: Requirement 4 (Backtesting and Historical Validation)
Task: 7.3 Multi-Strategy Backtesting
"""

import sys
import os
import logging
from pathlib import Path

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


def generate_sample_market_data(
    symbol: str = "AAPL",
    start_date: datetime = None,
    duration_days: int = 252
) -> list[MarketData]:
    """Generate realistic sample market data for testing"""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=duration_days)
    
    # Generate realistic price movements
    np.random.seed(42)  # For reproducibility
    
    # Start with realistic price
    initial_price = 150.0
    prices = [initial_price]
    
    # Generate daily returns with realistic characteristics
    for _ in range(duration_days - 1):
        # Daily return with slight positive drift and realistic volatility
        daily_return = np.random.normal(0.0005, 0.02)  # 0.05% daily drift, 2% volatility
        
        # Add some mean reversion
        if len(prices) > 20:
            long_term_avg = np.mean(prices[-20:])
            mean_reversion = 0.1 * (long_term_avg - prices[-1]) / prices[-1]
            daily_return += mean_reversion
        
        # Ensure price doesn't go negative
        new_price = prices[-1] * (1 + daily_return)
        if new_price < 1.0:
            new_price = prices[-1] * 0.99  # Max 1% drop
        
        prices.append(new_price)
    
    # Generate OHLCV data
    market_data = []
    for i, close_price in enumerate(prices):
        date = start_date + timedelta(days=i)
        
        # Generate realistic OHLC from close price
        volatility = 0.015  # 1.5% intraday volatility
        
        # Random intraday movement
        intraday_move = np.random.normal(0, volatility)
        open_price = close_price * (1 + intraday_move)
        
        # High and low based on open and close
        high = max(open_price, close_price) * (1 + abs(np.random.normal(0, volatility * 0.5)))
        low = min(open_price, close_price) * (1 - abs(np.random.normal(0, volatility * 0.5)))
        
        # Volume with some correlation to price movement
        base_volume = 1000000
        price_change = abs(close_price - open_price) / open_price
        volume = int(base_volume * (1 + price_change * 10 + np.random.uniform(-0.2, 0.2)))
        
        market_data.append(MarketData(
            timestamp=date,
            symbol=symbol,
            open=open_price,
            high=high,
            low=low,
            close=close_price,
            volume=volume,
            vwap=(high + low + close_price) / 3
        ))
    
    return market_data


def run_basic_backtest_demo():
    """Run basic multi-strategy backtest demo"""
    print("=" * 80)
    print("MULTI-STRATEGY BACKTESTING DEMO - Task 7.3")
    print("=" * 80)
    
    # Initialize backtester
    print("\n1. Initializing Multi-Strategy Backtester...")
    backtester = MultiStrategyBacktester(
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
        random_seed=42
    )
    
    # Generate sample market data
    print("\n2. Generating sample market data...")
    market_data = generate_sample_market_data(
        symbol="AAPL",
        duration_days=252  # 1 year of trading data
    )
    print(f"   Generated {len(market_data)} data points")
    print(f"   Date range: {market_data[0].timestamp.date()} to {market_data[-1].timestamp.date()}")
    
    # Run comprehensive backtest
    print("\n3. Running comprehensive multi-strategy backtest...")
    results = backtester.run_comprehensive_backtest(
        market_data=market_data,
        test_scenarios=True,
        generate_reports=True
    )
    
    # Display results summary
    print("\n4. Backtest Results Summary:")
    print(f"   Test Period: {results.test_period[0].date()} to {results.test_period[1].date()}")
    print(f"   Agents Tested: {len(results.individual_agent_results)}")
    print(f"   Scenarios Tested: {len(results.scenario_results)}")
    
    # Individual agent performance
    print("\n5. Individual Agent Performance:")
    for agent_name, performance in results.individual_agent_results.items():
        print(f"   {agent_name.upper()}:")
        print(f"     Total Signals: {performance.total_signals}")
        print(f"     Signal Accuracy: {performance.signal_accuracy:.2%}")
        print(f"     Total Return: {performance.performance_metrics.total_return:.2%}")
        print(f"     Sharpe Ratio: {performance.performance_metrics.sharpe_ratio:.2f}")
        print(f"     Max Drawdown: {performance.performance_metrics.max_drawdown:.2%}")
    
    # Signal fusion results
    print(f"\n6. Signal Fusion Performance:")
    print(f"   Fusion Method: {results.fusion_results.fusion_method}")
    print(f"   Total Fused Signals: {results.fusion_results.total_fused_signals}")
    print(f"   Fusion Accuracy: {results.fusion_results.fusion_accuracy:.2%}")
    print(f"   Improvement over Individual: {results.fusion_results.improvement_over_individual:.2%}")
    
    # Scenario testing results
    print(f"\n7. Synthetic Scenario Testing Results:")
    for scenario_name, result in results.scenario_results.items():
        print(f"   {scenario_name.replace('_', ' ').title()}:")
        print(f"     Overall Return: {result.overall_performance.total_return:.2%}")
        print(f"     Regime Detection Accuracy: {result.regime_detection_accuracy:.2%}")
        print(f"     Adaptation Speed: {result.adaptation_speed:.2%}")
    
    # Performance attribution
    print(f"\n8. Performance Attribution:")
    if 'main' in results.performance_attribution:
        for strategy, contribution in results.performance_attribution['main'].items():
            print(f"   {strategy.title()}: {contribution:.2%}")
    
    return results, backtester


def run_scenario_testing_demo(backtester: MultiStrategyBacktester):
    """Run enhanced scenario testing demo"""
    print("\n" + "=" * 80)
    print("ENHANCED SCENARIO TESTING DEMO")
    print("=" * 80)
    
    # Test specific scenarios with custom parameters
    print("\n1. Testing Custom Market Scenarios...")
    
    custom_scenarios = {
        'trending_up': {
            'trend_strength': 0.03,  # 3% daily trend
            'volatility': 0.12,      # 12% volatility
            'duration_days': 180     # 6 months
        },
        'mean_reverting': {
            'mean_reversion_speed': 0.15,  # Faster mean reversion
            'volatility': 0.25,            # Higher volatility
            'duration_days': 180
        },
        'news_shock_positive': {
            'shock_magnitude': 0.20,  # 20% positive shock
            'shock_day': 60,          # Shock at day 60
            'decay_days': 8           # 8 days to decay
        }
    }
    
    # Run custom scenario tests
    scenario_results = backtester.run_synthetic_scenarios(
        scenarios=[
            ScenarioType.TRENDING_UP,
            ScenarioType.MEAN_REVERTING,
            ScenarioType.NEWS_SHOCK_POSITIVE
        ],
        scenario_params=custom_scenarios
    )
    
    print(f"\n2. Custom Scenario Results:")
    for scenario_name, result in scenario_results.items():
        print(f"   {scenario_name.replace('_', ' ').title()}:")
        print(f"     Overall Return: {result.overall_performance.total_return:.2%}")
        print(f"     Sharpe Ratio: {result.overall_performance.sharpe_ratio:.2f}")
        print(f"     Max Drawdown: {result.overall_performance.max_drawdown:.2%}")
        print(f"     Total Trades: {result.overall_performance.total_trades}")
    
    return scenario_results


def run_performance_analysis_demo(results, backtester: MultiStrategyBacktester):
    """Run performance analysis and visualization demo"""
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS AND VISUALIZATION DEMO")
    print("=" * 80)
    
    # Generate performance charts
    print("\n1. Generating Performance Charts...")
    chart_paths = backtester.generate_performance_charts(
        results, 
        output_dir="demo_charts"
    )
    
    print(f"\n2. Charts Generated:")
    for chart_name, chart_path in chart_paths.items():
        print(f"   {chart_name}: {chart_path}")
    
    # Risk analysis
    print(f"\n3. Risk Analysis:")
    print(f"   Portfolio VaR (95%): {results.risk_metrics.get('portfolio_var_95', 0):.2%}")
    print(f"   Portfolio CVaR (95%): {results.risk_metrics.get('portfolio_cvar_95', 0):.2%}")
    print(f"   Max Correlation: {results.risk_metrics.get('max_correlation', 0):.2f}")
    print(f"   Diversification Ratio: {results.risk_metrics.get('diversification_ratio', 0):.2f}")
    
    # Regime analysis
    print(f"\n4. Market Regime Analysis:")
    regime_analysis = results.regime_analysis
    print(f"   Regime Detection Accuracy: {regime_analysis.get('regime_detection_accuracy', 0):.2%}")
    print(f"   Regime Transitions: {regime_analysis.get('regime_transitions', 0)}")
    print(f"   Best Trending Agent: {regime_analysis.get('best_regime_agents', {}).get('trending', 'N/A')}")
    print(f"   Best Sideways Agent: {regime_analysis.get('best_regime_agents', {}).get('sideways', 'N/A')}")
    
    return chart_paths


def save_demo_results(results, scenario_results, chart_paths):
    """Save demo results to files"""
    print("\n" + "=" * 80)
    print("SAVING DEMO RESULTS")
    print("=" * 80)
    
    # Save main results
    main_results_file = "demo_multi_strategy_results.json"
    try:
        import json
        
        # Convert results to serializable format
        results_dict = {
            'demo_info': {
                'timestamp': datetime.now().isoformat(),
                'description': 'Multi-Strategy Backtesting Demo - Task 7.3',
                'test_period': [str(results.test_period[0]), str(results.test_period[1])]
            },
            'individual_agent_results': {
                name: {
                    'agent_name': perf.agent_name,
                    'strategy_type': perf.strategy_type,
                    'total_signals': perf.total_signals,
                    'profitable_signals': perf.profitable_signals,
                    'signal_accuracy': perf.signal_accuracy,
                    'performance_metrics': {
                        'total_return': perf.performance_metrics.total_return,
                        'annualized_return': perf.performance_metrics.annualized_return,
                        'sharpe_ratio': perf.performance_metrics.sharpe_ratio,
                        'max_drawdown': perf.performance_metrics.max_drawdown,
                        'win_rate': perf.performance_metrics.win_rate
                    }
                } for name, perf in results.individual_agent_results.items()
            },
            'fusion_results': {
                'fusion_method': results.fusion_results.fusion_method,
                'total_fused_signals': results.fusion_results.total_fused_signals,
                'fusion_accuracy': results.fusion_results.fusion_accuracy,
                'improvement_over_individual': results.fusion_results.improvement_over_individual
            },
            'scenario_results': {
                name: {
                    'scenario_type': result.scenario_type.value,
                    'overall_performance': {
                        'total_return': result.overall_performance.total_return,
                        'sharpe_ratio': result.overall_performance.sharpe_ratio,
                        'max_drawdown': result.overall_performance.max_drawdown
                    },
                    'regime_detection_accuracy': result.regime_detection_accuracy,
                    'adaptation_speed': result.adaptation_speed
                } for name, result in results.scenario_results.items()
            },
            'performance_attribution': results.performance_attribution,
            'risk_metrics': results.risk_metrics,
            'chart_paths': chart_paths
        }
        
        with open(main_results_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        print(f"[OK] Main results saved to: {main_results_file}")
        
    except Exception as e:
        print(f"[X] Error saving main results: {e}")
    
    # Save scenario results separately
    scenario_results_file = "demo_scenario_results.json"
    try:
        scenario_dict = {
            'scenario_testing_info': {
                'timestamp': datetime.now().isoformat(),
                'description': 'Custom Scenario Testing Results',
                'scenarios_tested': list(scenario_results.keys())
            },
            'scenario_results': {
                name: {
                    'scenario_type': result.scenario_type.value,
                    'scenario_params': result.scenario_params,
                    'overall_performance': {
                        'total_return': result.overall_performance.total_return,
                        'sharpe_ratio': result.overall_performance.sharpe_ratio,
                        'max_drawdown': result.overall_performance.max_drawdown,
                        'total_trades': result.overall_performance.total_trades
                    },
                    'agent_performances': {
                        agent_name: {
                            'signal_accuracy': perf.signal_accuracy,
                            'total_return': perf.performance_metrics.total_return,
                            'sharpe_ratio': perf.performance_metrics.sharpe_ratio
                        } for agent_name, perf in result.agent_performances.items()
                    }
                } for name, result in scenario_results.items()
            }
        }
        
        with open(scenario_results_file, 'w') as f:
            json.dump(scenario_dict, f, indent=2, default=str)
        
        print(f"[OK] Scenario results saved to: {scenario_results_file}")
        
    except Exception as e:
        print(f"[X] Error saving scenario results: {e}")
    
    # Save summary report
    summary_file = "demo_summary_report.md"
    try:
        with open(summary_file, 'w') as f:
            f.write("# Multi-Strategy Backtesting Demo - Task 7.3\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(results.summary_report)
        
        print(f"[OK] Summary report saved to: {summary_file}")
        
    except Exception as e:
        print(f"[X] Error saving summary report: {e}")


def main():
    """Main demo execution"""
    print("[LAUNCH] MULTI-STRATEGY BACKTESTING DEMO - Task 7.3")
    print("Testing comprehensive multi-strategy backtesting capabilities...")
    
    try:
        # Run basic backtest demo
        results, backtester = run_basic_backtest_demo()
        
        # Run enhanced scenario testing
        scenario_results = run_scenario_testing_demo(backtester)
        
        # Run performance analysis
        chart_paths = run_performance_analysis_demo(results, backtester)
        
        # Save all results
        save_demo_results(results, scenario_results, chart_paths)
        
        print("\n" + "=" * 80)
        print("[PARTY] DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nTask 7.3 - Multi-Strategy Backtesting has been implemented and demonstrated:")
        print("[OK] Individual agent testing on historical data")
        print("[OK] Signal fusion validation across different market regimes")
        print("[OK] Synthetic scenario testing (trend, mean-revert, news shock, etc.)")
        print("[OK] Strategy performance attribution reports")
        print("[OK] Performance visualization charts")
        print("[OK] Comprehensive risk analysis")
        print("[OK] Market regime analysis")
        
        print(f"\n[CHART] Generated {len(chart_paths)} performance charts")
        print(f"[UP] Tested {len(results.individual_agent_results)} trading agents")
        print(f"[INFO] Tested {len(results.scenario_results)} synthetic scenarios")
        print(f"[INFO] Generated comprehensive performance reports")
        
        print("\nThe system is now ready for live trading validation!")
        
    except Exception as e:
        print(f"\n[X] Demo failed with error: {e}")
        logger.error(f"Demo execution failed: {e}", exc_info=True)
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)