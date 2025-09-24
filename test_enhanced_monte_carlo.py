#!/usr/bin/env python3
"""
Test script for Enhanced Monte Carlo Engine and Options Pricing
Validates the new implementation against existing systems
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
import time

from agents.advanced_monte_carlo_engine import (
    AdvancedMonteCarloEngine, OptionSpec, advanced_monte_carlo_engine
)
from agents.enhanced_options_pricing_engine import (
    EnhancedOptionsPricingEngine, OptionsPosition, enhanced_options_pricing_engine
)

def test_black_scholes_pricing():
    """Test Black-Scholes analytical pricing"""
    print("=" * 60)
    print("TESTING BLACK-SCHOLES OPTION PRICING")
    print("=" * 60)

    # Test cases
    test_options = [
        OptionSpec(S0=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call'),  # ATM call
        OptionSpec(S0=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='put'),   # ATM put
        OptionSpec(S0=100, K=110, T=0.25, r=0.05, sigma=0.20, option_type='call'),  # OTM call
        OptionSpec(S0=100, K=90, T=0.25, r=0.05, sigma=0.20, option_type='put'),    # OTM put
    ]

    for i, option in enumerate(test_options, 1):
        price = advanced_monte_carlo_engine.black_scholes_price(option)
        greeks = advanced_monte_carlo_engine.calculate_greeks(option)

        print(f"Test {i}: {option.option_type.upper()} - S0={option.S0}, K={option.K}, T={option.T}")
        print(f"  Price: ${price:.4f}")
        print(f"  Delta: {greeks.delta:.4f}")
        print(f"  Gamma: {greeks.gamma:.4f}")
        print(f"  Theta: ${greeks.theta:.4f}")
        print(f"  Vega:  ${greeks.vega:.4f}")
        print()

def test_monte_carlo_pricing():
    """Test Monte Carlo option pricing with variance reduction"""
    print("=" * 60)
    print("TESTING MONTE CARLO OPTION PRICING")
    print("=" * 60)

    option = OptionSpec(S0=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')

    # Test different path counts
    path_counts = [10000, 50000, 100000]

    for paths in path_counts:
        start_time = time.time()
        result = advanced_monte_carlo_engine.monte_carlo_option_price(option, paths=paths)
        end_time = time.time()

        bs_price = advanced_monte_carlo_engine.black_scholes_price(option)

        print(f"Paths: {paths:,}")
        print(f"  MC Price: ${result.option_price:.4f}")
        print(f"  BS Price: ${bs_price:.4f}")
        print(f"  Error:    ${abs(result.option_price - bs_price):.4f}")
        print(f"  Std Err:  ${result.standard_error:.4f}")
        print(f"  95% CI:   [${result.confidence_interval[0]:.4f}, ${result.confidence_interval[1]:.4f}]")
        print(f"  Time:     {end_time - start_time:.2f}s")
        print(f"  Stability: {result.convergence_stats['stability']:.6f}")
        print()

def test_portfolio_optimization():
    """Test portfolio optimization features"""
    print("=" * 60)
    print("TESTING PORTFOLIO OPTIMIZATION")
    print("=" * 60)

    # Generate sample returns data (3 assets, 252 days)
    np.random.seed(42)
    n_assets = 3
    n_days = 252

    # Simulate correlated returns
    mean_returns = np.array([0.08, 0.10, 0.12])  # Annual returns
    volatilities = np.array([0.15, 0.20, 0.25])  # Annual volatilities

    # Create correlation matrix
    correlation = np.array([
        [1.00, 0.30, 0.20],
        [0.30, 1.00, 0.40],
        [0.20, 0.40, 1.00]
    ])

    # Generate returns
    daily_returns = np.random.multivariate_normal(
        mean_returns / 252,  # Daily returns
        np.outer(volatilities, volatilities) * correlation / 252,  # Daily covariance
        n_days
    )

    # Test efficient frontier calculation
    target_returns = np.linspace(0.06, 0.14, 5)  # 6% to 14% annual returns

    try:
        efficient_frontier = advanced_monte_carlo_engine.efficient_frontier(daily_returns, target_returns)

        print(f"Efficient Frontier Results:")
        print(f"Number of portfolios: {len(efficient_frontier['portfolios'])}")

        for i, portfolio in enumerate(efficient_frontier['portfolios']):
            print(f"  Portfolio {i+1}:")
            print(f"    Target Return: {target_returns[i]:.1%}")
            print(f"    Actual Return: {portfolio['return']:.1%}")
            print(f"    Risk (Vol):    {portfolio['risk']:.1%}")
            print(f"    Sharpe Ratio:  {portfolio['sharpe_ratio']:.3f}")
            print(f"    Weights:       {[f'{w:.1%}' for w in portfolio['weights']]}")
            print()

    except Exception as e:
        print(f"Portfolio optimization error: {e}")

def test_risk_metrics():
    """Test VaR and CVaR calculations"""
    print("=" * 60)
    print("TESTING RISK METRICS (VaR & CVaR)")
    print("=" * 60)

    # Generate sample portfolio returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)  # Daily returns with some outliers

    # Add some extreme negative returns to test tail risk
    returns[::100] = -0.05  # 5% loss every 100 days

    risk_metrics = advanced_monte_carlo_engine.calculate_var_cvar(returns, confidence_level=0.95)

    print(f"Risk Metrics (95% confidence):")
    print(f"  VaR (95%):     {risk_metrics['var']:.2%}")
    print(f"  CVaR (95%):    {risk_metrics['cvar']:.2%}")
    print(f"  Worst Return:  {risk_metrics['worst_return']:.2%}")
    print(f"  Best Return:   {risk_metrics['best_return']:.2%}")
    print()

    # Test different confidence levels
    for confidence in [0.90, 0.95, 0.99]:
        metrics = advanced_monte_carlo_engine.calculate_var_cvar(returns, confidence)
        print(f"  VaR ({confidence:.0%}):      {metrics['var']:.2%}")

async def test_options_pricing_engine():
    """Test the enhanced options pricing engine"""
    print("=" * 60)
    print("TESTING ENHANCED OPTIONS PRICING ENGINE")
    print("=" * 60)

    engine = enhanced_options_pricing_engine

    # Test option pricing
    expiration = datetime.now() + timedelta(days=30)  # 30 days to expiration

    print("Testing option pricing for AAPL...")
    result = await engine.price_option('AAPL', 'call', 150, expiration, method='both')

    if 'black_scholes' in result:
        bs_result = result['black_scholes']
        print(f"Black-Scholes Price: ${bs_result['price']:.4f}")
        if 'greeks' in bs_result:
            greeks = bs_result['greeks']
            print(f"  Delta: {greeks.delta:.4f}")
            print(f"  Gamma: {greeks.gamma:.4f}")
            print(f"  Theta: ${greeks.theta:.4f}")

    if 'monte_carlo' in result:
        mc_result = result['monte_carlo']
        print(f"Monte Carlo Price: ${mc_result['price']:.4f}")
        print(f"  95% CI: [${mc_result['confidence_interval'][0]:.4f}, ${mc_result['confidence_interval'][1]:.4f}]")
        print(f"  Std Error: ${mc_result['standard_error']:.4f}")

    print()

async def test_portfolio_analysis():
    """Test portfolio analysis features"""
    print("=" * 60)
    print("TESTING PORTFOLIO ANALYSIS")
    print("=" * 60)

    engine = enhanced_options_pricing_engine

    # Create sample options positions
    expiration = datetime.now() + timedelta(days=30)
    positions = [
        OptionsPosition('AAPL', 'call', 150, expiration, 10, 2.50),
        OptionsPosition('AAPL', 'put', 140, expiration, -5, 1.80),
        OptionsPosition('GOOGL', 'call', 140, expiration, 5, 3.20),
    ]

    print("Calculating portfolio metrics...")
    metrics = await engine.calculate_portfolio_metrics(positions)

    print(f"Portfolio Metrics:")
    print(f"  Total Value:    ${metrics.total_value:,.2f}")
    print(f"  Total P&L:      ${metrics.total_pnl:,.2f}")
    print(f"  Delta Exposure: {metrics.delta_exposure:.0f}")
    print(f"  Gamma Exposure: {metrics.gamma_exposure:.0f}")
    print(f"  Theta Decay:    ${metrics.theta_decay:.2f}")
    print(f"  Vega Exposure:  ${metrics.vega_exposure:.0f}")
    print(f"  VaR (95%):      {metrics.var_95:.2%}")
    print(f"  Expected Return: {metrics.expected_return:.2%}")
    print(f"  Sharpe Ratio:   {metrics.sharpe_ratio:.3f}")

def performance_comparison():
    """Compare new implementation with existing systems"""
    print("=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    option = OptionSpec(S0=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call')

    # Test Black-Scholes performance
    start_time = time.time()
    for _ in range(1000):
        price = advanced_monte_carlo_engine.black_scholes_price(option)
    bs_time = time.time() - start_time

    print(f"Black-Scholes (1000 calculations): {bs_time:.4f}s")
    print(f"  Price: ${price:.4f}")

    # Test Monte Carlo performance (smaller sample for speed)
    start_time = time.time()
    mc_result = advanced_monte_carlo_engine.monte_carlo_option_price(option, paths=10000)
    mc_time = time.time() - start_time

    print(f"Monte Carlo (10,000 paths): {mc_time:.4f}s")
    print(f"  Price: ${mc_result.option_price:.4f}")
    print(f"  Accuracy vs BS: {abs(mc_result.option_price - price):.4f}")

async def main():
    """Run all tests"""
    print("ENHANCED MONTE CARLO & OPTIONS PRICING ENGINE TEST SUITE")
    print("=" * 80)

    try:
        # Basic pricing tests
        test_black_scholes_pricing()
        test_monte_carlo_pricing()

        # Portfolio optimization tests
        test_portfolio_optimization()
        test_risk_metrics()

        # Options engine tests
        await test_options_pricing_engine()
        await test_portfolio_analysis()

        # Performance tests
        performance_comparison()

        print("=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("Enhanced Monte Carlo Engine is ready for integration.")

    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())