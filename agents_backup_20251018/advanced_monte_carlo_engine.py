#!/usr/bin/env python3
"""
Advanced Monte Carlo Simulation Engine for Options Pricing and Portfolio Optimization
Based on Black-Scholes-Merton framework with modern portfolio theory integration

Features:
- Geometric Brownian Motion simulation with variance reduction
- Black-Scholes analytical pricing with Greeks calculation
- Monte Carlo option pricing with confidence intervals
- Portfolio optimization using Modern Portfolio Theory
- Risk metrics including VaR and CVaR
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from config.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class OptionSpec:
    """Option specification for pricing"""
    S0: float          # Current stock price
    K: float           # Strike price
    T: float           # Time to expiration (years)
    r: float           # Risk-free rate
    sigma: float       # Volatility
    option_type: str   # 'call' or 'put'
    dividend_yield: float = 0.0  # Dividend yield

@dataclass
class SimulationResult:
    """Monte Carlo simulation result"""
    option_price: float
    confidence_interval: Tuple[float, float]
    standard_error: float
    paths_used: int
    convergence_stats: Dict

@dataclass
class Greeks:
    """Option Greeks"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

class AdvancedMonteCarloEngine:
    """Advanced Monte Carlo Engine for Options and Portfolio Analytics"""

    def __init__(self):
        self.random_seed = None
        self.variance_reduction = True
        self.control_variates = True
        self.antithetic_paths = True

    def set_random_seed(self, seed: int):
        """Set random seed for reproducibility"""
        self.random_seed = seed
        np.random.seed(seed)

    def black_scholes_price(self, option: OptionSpec) -> float:
        """
        Calculate Black-Scholes option price analytically

        Formula: C = S0*e^(-q*T)*N(d1) - K*e^(-r*T)*N(d2)
        where d1 = (ln(S0/K) + (r-q+σ²/2)*T) / (σ*√T)
              d2 = d1 - σ*√T
        """
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available, using approximation")
            return self._approximate_option_price(option)

        S, K, T, r, sigma, q = option.S0, option.K, option.T, option.r, option.sigma, option.dividend_yield

        if T <= 0:
            # At expiration
            if option.option_type.lower() == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option.option_type.lower() == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

        return max(price, 0)

    def calculate_greeks(self, option: OptionSpec) -> Greeks:
        """Calculate option Greeks using Black-Scholes formulas"""
        if not SCIPY_AVAILABLE:
            return Greeks(0.5, 0.0, 0.0, 0.0, 0.0)  # Default values

        S, K, T, r, sigma, q = option.S0, option.K, option.T, option.r, option.sigma, option.dividend_yield

        if T <= 0:
            return Greeks(0.0, 0.0, 0.0, 0.0, 0.0)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option.option_type.lower() == 'call':
            # Call Greeks
            delta = np.exp(-q * T) * norm.cdf(d1)
            gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (-S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                    - r * K * np.exp(-r * T) * norm.cdf(d2)
                    + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            # Put Greeks
            delta = -np.exp(-q * T) * norm.cdf(-d1)
            gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (-S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)
                    - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        return Greeks(delta, gamma, theta, vega, rho)

    def generate_price_paths(self, S0: float, r: float, sigma: float, T: float,
                           steps: int, paths: int, dividend_yield: float = 0.0) -> np.ndarray:
        """
        Generate stock price paths using geometric Brownian motion

        dS = (r - q)*S*dt + σ*S*dW
        S(t) = S0 * exp((r - q - σ²/2)*t + σ*√t*Z)
        """
        dt = T / steps

        # Generate random numbers
        if self.antithetic_paths and paths % 2 == 0:
            # Use antithetic variance reduction
            half_paths = paths // 2
            Z = np.random.standard_normal((half_paths, steps))
            Z_anti = -Z
            Z = np.vstack([Z, Z_anti])
        else:
            Z = np.random.standard_normal((paths, steps))

        # Calculate drift
        drift = (r - dividend_yield - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)

        # Generate price paths
        log_returns = drift + diffusion * Z
        log_price_paths = np.log(S0) + np.cumsum(log_returns, axis=1)
        price_paths = np.exp(log_price_paths)

        # Add initial price
        initial_prices = np.full((paths, 1), S0)
        price_paths = np.hstack([initial_prices, price_paths])

        return price_paths

    def monte_carlo_option_price(self, option: OptionSpec, paths: int = 100000,
                               steps: int = 252) -> SimulationResult:
        """
        Price option using Monte Carlo simulation with variance reduction techniques
        """
        logger.info(f"Pricing {option.option_type} option using Monte Carlo with {paths:,} paths")

        # Generate price paths
        price_paths = self.generate_price_paths(
            option.S0, option.r, option.sigma, option.T,
            steps, paths, option.dividend_yield
        )

        # Calculate payoffs at expiration
        final_prices = price_paths[:, -1]

        if option.option_type.lower() == 'call':
            payoffs = np.maximum(final_prices - option.K, 0)
        else:  # put
            payoffs = np.maximum(option.K - final_prices, 0)

        # Apply control variates if enabled
        if self.control_variates:
            payoffs = self._apply_control_variates(payoffs, final_prices, option)

        # Discount payoffs to present value
        option_prices = payoffs * np.exp(-option.r * option.T)

        # Calculate statistics
        mean_price = np.mean(option_prices)
        std_error = np.std(option_prices) / np.sqrt(paths)

        # 95% confidence interval
        confidence_interval = (
            mean_price - 1.96 * std_error,
            mean_price + 1.96 * std_error
        )

        # Convergence statistics
        convergence_stats = self._calculate_convergence_stats(option_prices)

        return SimulationResult(
            option_price=mean_price,
            confidence_interval=confidence_interval,
            standard_error=std_error,
            paths_used=paths,
            convergence_stats=convergence_stats
        )

    def _apply_control_variates(self, payoffs: np.ndarray, final_prices: np.ndarray,
                              option: OptionSpec) -> np.ndarray:
        """Apply control variates variance reduction technique"""
        try:
            # Use Black-Scholes price as control variate
            bs_price = self.black_scholes_price(option)

            if option.option_type.lower() == 'call':
                control_payoffs = np.maximum(final_prices - option.K, 0)
            else:
                control_payoffs = np.maximum(option.K - final_prices, 0)

            control_prices = control_payoffs * np.exp(-option.r * option.T)

            # Calculate optimal control coefficient
            covariance = np.cov(payoffs, control_prices)[0, 1]
            variance = np.var(control_prices)

            if variance > 0:
                beta = covariance / variance
                adjusted_payoffs = payoffs - beta * (control_prices - bs_price)
                return adjusted_payoffs

        except Exception as e:
            logger.warning(f"Control variates failed: {e}")

        return payoffs

    def _calculate_convergence_stats(self, prices: np.ndarray) -> Dict:
        """Calculate convergence statistics for the simulation"""
        n = len(prices)
        running_means = np.cumsum(prices) / np.arange(1, n + 1)

        # Calculate stability metric (variance of last 10% of running means)
        stability_window = int(0.1 * n)
        stability = np.var(running_means[-stability_window:]) if stability_window > 1 else 0

        return {
            'final_mean': running_means[-1],
            'stability': stability,
            'convergence_rate': np.abs(running_means[-1] - running_means[-min(1000, n//2)]),
            'path_efficiency': min(1.0, 10000 / n)  # Efficiency metric
        }

    def _approximate_option_price(self, option: OptionSpec) -> float:
        """Fallback option pricing when SciPy is not available"""
        # Simple approximation using intrinsic value + time value estimate
        intrinsic = max(option.S0 - option.K, 0) if option.option_type.lower() == 'call' else max(option.K - option.S0, 0)
        time_value = option.sigma * option.S0 * np.sqrt(option.T) * 0.4  # Rough approximation
        return intrinsic + time_value

    def efficient_frontier(self, returns: np.ndarray, target_returns: np.ndarray) -> Dict:
        """
        Calculate efficient frontier using Modern Portfolio Theory

        Optimization problem:
        minimize: w^T * Σ * w (portfolio variance)
        subject to: w^T * μ = target_return
                   w^T * 1 = 1 (fully invested)
                   w >= 0 (no short selling)
        """
        n_assets = returns.shape[1]
        mean_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns.T)

        efficient_portfolios = []

        for target_return in target_returns:
            try:
                # Define optimization problem
                def objective(weights):
                    return np.dot(weights.T, np.dot(cov_matrix, weights))

                # Constraints
                constraints = [
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Fully invested
                    {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target_return}  # Target return
                ]

                # Bounds (no short selling)
                bounds = tuple((0, 1) for _ in range(n_assets))

                # Initial guess (equal weights)
                initial_weights = np.array([1/n_assets] * n_assets)

                # Optimize
                result = minimize(objective, initial_weights, method='SLSQP',
                                bounds=bounds, constraints=constraints)

                if result.success:
                    weights = result.x
                    portfolio_return = np.dot(weights, mean_returns)
                    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0

                    efficient_portfolios.append({
                        'weights': weights,
                        'return': portfolio_return,
                        'risk': portfolio_risk,
                        'sharpe_ratio': sharpe_ratio
                    })

            except Exception as e:
                logger.warning(f"Portfolio optimization failed for target return {target_return}: {e}")

        return {
            'portfolios': efficient_portfolios,
            'mean_returns': mean_returns,
            'covariance_matrix': cov_matrix
        }

    def calculate_var_cvar(self, returns: np.ndarray, confidence_level: float = 0.95) -> Dict:
        """
        Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR)
        """
        sorted_returns = np.sort(returns)
        index = int((1 - confidence_level) * len(sorted_returns))

        var = -sorted_returns[index] if index < len(sorted_returns) else 0
        cvar = -np.mean(sorted_returns[:index]) if index > 0 else 0

        return {
            'var': var,
            'cvar': cvar,
            'confidence_level': confidence_level,
            'worst_return': -sorted_returns[0] if len(sorted_returns) > 0 else 0,
            'best_return': -sorted_returns[-1] if len(sorted_returns) > 0 else 0
        }

    def portfolio_monte_carlo(self, weights: np.ndarray, returns_data: np.ndarray,
                            days: int = 252, simulations: int = 10000) -> Dict:
        """
        Run Monte Carlo simulation for portfolio performance
        """
        n_assets = len(weights)
        portfolio_returns = []

        for _ in range(simulations):
            # Sample random returns for each asset
            random_indices = np.random.choice(len(returns_data), size=days, replace=True)
            sampled_returns = returns_data[random_indices]

            # Calculate portfolio returns
            portfolio_daily_returns = np.dot(sampled_returns, weights)
            total_return = np.prod(1 + portfolio_daily_returns) - 1
            portfolio_returns.append(total_return)

        portfolio_returns = np.array(portfolio_returns)

        # Calculate statistics
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)

        # Risk metrics
        risk_metrics = self.calculate_var_cvar(portfolio_returns)

        return {
            'mean_return': mean_return,
            'std_return': std_return,
            'sharpe_ratio': mean_return / std_return if std_return > 0 else 0,
            'risk_metrics': risk_metrics,
            'return_distribution': portfolio_returns,
            'positive_returns_pct': np.sum(portfolio_returns > 0) / len(portfolio_returns)
        }

# Global instance
advanced_monte_carlo_engine = AdvancedMonteCarloEngine()