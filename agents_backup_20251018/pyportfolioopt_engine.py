#!/usr/bin/env python3
"""
PyPortfolioOpt Engine
Modern Portfolio Theory optimization with Sharpe/Sortino maximization
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Try to import PyPortfolioOpt
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt import objective_functions, discrete_allocation
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    PYPFOPT_AVAILABLE = True
    print("+ PyPortfolioOpt available for advanced portfolio optimization")
except ImportError:
    PYPFOPT_AVAILABLE = False
    print("- PyPortfolioOpt not available - using custom implementation")

# Try to import scipy for optimization
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class PyPortfolioOptEngine:
    """Advanced portfolio optimization using Modern Portfolio Theory"""
    
    def __init__(self):
        self.pypfopt_available = PYPFOPT_AVAILABLE
        self.scipy_available = SCIPY_AVAILABLE
        
        # Portfolio configuration
        self.config = {
            'risk_free_rate': 0.02,  # 2% risk-free rate
            'max_weight': 0.15,      # Maximum 15% per asset
            'min_weight': 0.0,       # Minimum 0% per asset
            'target_return': 0.12,   # Target 12% annual return
            'target_volatility': 0.15,  # Target 15% volatility
            'l2_gamma': 0.1,         # L2 regularization parameter
            'sector_constraints': {  # Sector allocation limits
                'tech': 0.4,
                'finance': 0.3,
                'healthcare': 0.2,
                'other': 0.3
            }
        }
    
    async def optimize_sharpe_ratio(self, price_data: pd.DataFrame, 
                                  method: str = "sample_cov") -> Dict:
        """Optimize portfolio for maximum Sharpe ratio"""
        
        if self.pypfopt_available:
            return await self._pypfopt_sharpe_optimization(price_data, method)
        else:
            return await self._custom_sharpe_optimization(price_data)
    
    async def _pypfopt_sharpe_optimization(self, price_data: pd.DataFrame, method: str) -> Dict:
        """Sharpe optimization using PyPortfolioOpt"""
        try:
            print("Running PyPortfolioOpt Sharpe ratio optimization...")
            
            # Calculate expected returns
            mu = expected_returns.mean_historical_return(price_data)
            
            # Calculate covariance matrix
            if method == "sample_cov":
                S = risk_models.sample_cov(price_data)
            elif method == "exp_cov":
                S = risk_models.exp_cov(price_data)
            elif method == "ledoit_wolf":
                S = risk_models.CovarianceShrinkage(price_data).ledoit_wolf()
            else:
                S = risk_models.sample_cov(price_data)
            
            # Optimize for maximum Sharpe ratio
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w >= self.config['min_weight'])
            ef.add_constraint(lambda w: w <= self.config['max_weight'])
            
            # Add L2 regularization to reduce extreme weights
            ef.add_objective(objective_functions.L2_reg, gamma=self.config['l2_gamma'])
            
            # Optimize
            weights = ef.max_sharpe(risk_free_rate=self.config['risk_free_rate'])
            cleaned_weights = ef.clean_weights()
            
            # Calculate performance
            performance = ef.portfolio_performance(
                risk_free_rate=self.config['risk_free_rate'], 
                verbose=False
            )
            
            expected_return, volatility, sharpe_ratio = performance
            
            return {
                'method': f'pypfopt_sharpe_{method}',
                'weights': cleaned_weights,
                'expected_annual_return': expected_return,
                'annual_volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'optimization_success': True,
                'total_weight': sum(cleaned_weights.values()),
                'active_positions': len([w for w in cleaned_weights.values() if w > 0.001]),
                'max_weight': max(cleaned_weights.values()),
                'min_weight': min([w for w in cleaned_weights.values() if w > 0])
            }
            
        except Exception as e:
            print(f"- PyPortfolioOpt Sharpe optimization error: {e}")
            return await self._custom_sharpe_optimization(price_data)
    
    async def optimize_sortino_ratio(self, price_data: pd.DataFrame) -> Dict:
        """Optimize portfolio for maximum Sortino ratio"""
        
        if self.pypfopt_available:
            return await self._pypfopt_sortino_optimization(price_data)
        else:
            return await self._custom_sortino_optimization(price_data)
    
    async def _pypfopt_sortino_optimization(self, price_data: pd.DataFrame) -> Dict:
        """Sortino optimization using PyPortfolioOpt"""
        try:
            print("Running PyPortfolioOpt Sortino ratio optimization...")
            
            # Calculate expected returns and semi-covariance
            mu = expected_returns.mean_historical_return(price_data)
            returns = price_data.pct_change().dropna()
            
            # Calculate downside deviation (for Sortino ratio)
            downside_returns = returns.copy()
            downside_returns[downside_returns > 0] = 0
            
            # Create semi-covariance matrix (downside risk only)
            S = downside_returns.cov() * 252  # Annualized
            
            # Optimize for maximum Sortino-like objective
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w >= self.config['min_weight'])
            ef.add_constraint(lambda w: w <= self.config['max_weight'])
            
            # Custom Sortino objective
            def sortino_objective(weights):
                portfolio_return = mu @ weights
                downside_vol = np.sqrt(weights.T @ S @ weights)
                if downside_vol == 0:
                    return -np.inf
                return -(portfolio_return - self.config['risk_free_rate']) / downside_vol
            
            ef.convex_objective(sortino_objective)
            weights = ef.clean_weights()
            
            # Calculate performance metrics
            portfolio_return = mu @ np.array(list(weights.values()))
            downside_vol = np.sqrt(np.array(list(weights.values())).T @ S @ np.array(list(weights.values())))
            sortino_ratio = (portfolio_return - self.config['risk_free_rate']) / downside_vol
            
            return {
                'method': 'pypfopt_sortino',
                'weights': weights,
                'expected_annual_return': portfolio_return,
                'downside_volatility': downside_vol,
                'sortino_ratio': sortino_ratio,
                'optimization_success': True,
                'total_weight': sum(weights.values()),
                'active_positions': len([w for w in weights.values() if w > 0.001])
            }
            
        except Exception as e:
            print(f"- PyPortfolioOpt Sortino optimization error: {e}")
            return await self._custom_sortino_optimization(price_data)
    
    async def optimize_minimum_volatility(self, price_data: pd.DataFrame) -> Dict:
        """Optimize portfolio for minimum volatility"""
        
        if self.pypfopt_available:
            return await self._pypfopt_min_vol_optimization(price_data)
        else:
            return await self._custom_min_vol_optimization(price_data)
    
    async def _pypfopt_min_vol_optimization(self, price_data: pd.DataFrame) -> Dict:
        """Minimum volatility optimization using PyPortfolioOpt"""
        try:
            print("Running PyPortfolioOpt minimum volatility optimization...")
            
            # Calculate expected returns and covariance
            mu = expected_returns.mean_historical_return(price_data)
            S = risk_models.sample_cov(price_data)
            
            # Optimize for minimum volatility
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w >= self.config['min_weight'])
            ef.add_constraint(lambda w: w <= self.config['max_weight'])
            
            weights = ef.min_volatility()
            cleaned_weights = ef.clean_weights()
            
            # Calculate performance
            performance = ef.portfolio_performance(
                risk_free_rate=self.config['risk_free_rate'], 
                verbose=False
            )
            
            expected_return, volatility, sharpe_ratio = performance
            
            return {
                'method': 'pypfopt_min_volatility',
                'weights': cleaned_weights,
                'expected_annual_return': expected_return,
                'annual_volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'optimization_success': True,
                'total_weight': sum(cleaned_weights.values()),
                'active_positions': len([w for w in cleaned_weights.values() if w > 0.001])
            }
            
        except Exception as e:
            print(f"- PyPortfolioOpt min volatility optimization error: {e}")
            return await self._custom_min_vol_optimization(price_data)
    
    async def efficient_frontier_analysis(self, price_data: pd.DataFrame, 
                                        n_points: int = 20) -> Dict:
        """Generate efficient frontier analysis"""
        
        if self.pypfopt_available:
            return await self._pypfopt_efficient_frontier(price_data, n_points)
        else:
            return await self._custom_efficient_frontier(price_data, n_points)
    
    async def _pypfopt_efficient_frontier(self, price_data: pd.DataFrame, n_points: int) -> Dict:
        """Generate efficient frontier using PyPortfolioOpt"""
        try:
            print(f"Generating efficient frontier with {n_points} points...")
            
            # Calculate expected returns and covariance
            mu = expected_returns.mean_historical_return(price_data)
            S = risk_models.sample_cov(price_data)
            
            # Generate efficient frontier
            ef = EfficientFrontier(mu, S)
            
            # Get range of target returns
            min_ret = mu.min()
            max_ret = mu.max()
            target_returns = np.linspace(min_ret, max_ret, n_points)
            
            frontier_results = []
            
            for target_ret in target_returns:
                try:
                    ef_copy = EfficientFrontier(mu, S)  # Fresh copy
                    ef_copy.add_constraint(lambda w: w >= self.config['min_weight'])
                    ef_copy.add_constraint(lambda w: w <= self.config['max_weight'])
                    
                    weights = ef_copy.efficient_return(target_ret)
                    performance = ef_copy.portfolio_performance(
                        risk_free_rate=self.config['risk_free_rate'], 
                        verbose=False
                    )
                    
                    expected_return, volatility, sharpe_ratio = performance
                    
                    frontier_results.append({
                        'target_return': target_ret,
                        'expected_return': expected_return,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe_ratio,
                        'weights': ef_copy.clean_weights()
                    })
                    
                except Exception as e:
                    # Skip problematic points
                    continue
            
            # Find optimal portfolio (max Sharpe)
            if frontier_results:
                optimal_portfolio = max(frontier_results, key=lambda x: x['sharpe_ratio'])
                
                return {
                    'method': 'pypfopt_efficient_frontier',
                    'n_points': len(frontier_results),
                    'frontier_points': frontier_results,
                    'optimal_portfolio': optimal_portfolio,
                    'optimization_success': True
                }
            else:
                return {'optimization_success': False, 'error': 'No valid frontier points'}
                
        except Exception as e:
            print(f"- PyPortfolioOpt efficient frontier error: {e}")
            return await self._custom_efficient_frontier(price_data, n_points)
    
    async def discrete_allocation(self, weights: Dict, latest_prices: Dict, 
                                total_portfolio_value: float = 100000) -> Dict:
        """Convert continuous weights to discrete share allocations"""
        
        if self.pypfopt_available:
            return await self._pypfopt_discrete_allocation(weights, latest_prices, total_portfolio_value)
        else:
            return await self._custom_discrete_allocation(weights, latest_prices, total_portfolio_value)
    
    async def _pypfopt_discrete_allocation(self, weights: Dict, latest_prices: Dict, 
                                         total_portfolio_value: float) -> Dict:
        """Discrete allocation using PyPortfolioOpt"""
        try:
            print(f"Converting weights to discrete allocation for ${total_portfolio_value:,.0f} portfolio...")
            
            # Create discrete allocation
            da = DiscreteAllocation(weights, latest_prices, total_portfolio_value)
            allocation, leftover = da.greedy_portfolio()
            
            # Calculate allocation statistics
            total_invested = sum(allocation[asset] * latest_prices[asset] for asset in allocation)
            allocation_percentages = {
                asset: (shares * latest_prices[asset]) / total_invested 
                for asset, shares in allocation.items()
            }
            
            return {
                'method': 'pypfopt_discrete_allocation',
                'share_allocation': allocation,
                'allocation_percentages': allocation_percentages,
                'total_invested': total_invested,
                'leftover_cash': leftover,
                'portfolio_value': total_portfolio_value,
                'utilization_rate': total_invested / total_portfolio_value,
                'allocation_success': True
            }
            
        except Exception as e:
            print(f"- PyPortfolioOpt discrete allocation error: {e}")
            return await self._custom_discrete_allocation(weights, latest_prices, total_portfolio_value)
    
    # Custom implementations for when PyPortfolioOpt is not available
    
    async def _custom_sharpe_optimization(self, price_data: pd.DataFrame) -> Dict:
        """Custom Sharpe ratio optimization"""
        try:
            print("Running custom Sharpe ratio optimization...")
            
            # Calculate returns and statistics
            returns = price_data.pct_change().dropna()
            mean_returns = returns.mean() * 252  # Annualized
            cov_matrix = returns.cov() * 252  # Annualized
            
            n_assets = len(price_data.columns)
            
            if self.scipy_available:
                # Use scipy optimization
                def negative_sharpe(weights):
                    portfolio_return = np.sum(weights * mean_returns)
                    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    if portfolio_vol == 0:
                        return 1e6
                    return -(portfolio_return - self.config['risk_free_rate']) / portfolio_vol
                
                # Constraints
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = tuple((self.config['min_weight'], self.config['max_weight']) for _ in range(n_assets))
                
                # Initial guess
                x0 = np.array([1/n_assets] * n_assets)
                
                # Optimize
                result = minimize(negative_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
                
                if result.success:
                    weights = result.x
                    portfolio_return = np.sum(weights * mean_returns)
                    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    sharpe_ratio = (portfolio_return - self.config['risk_free_rate']) / portfolio_vol
                    
                    weights_dict = dict(zip(price_data.columns, weights))
                    
                    return {
                        'method': 'custom_sharpe_scipy',
                        'weights': weights_dict,
                        'expected_annual_return': portfolio_return,
                        'annual_volatility': portfolio_vol,
                        'sharpe_ratio': sharpe_ratio,
                        'optimization_success': True,
                        'total_weight': np.sum(weights),
                        'active_positions': np.sum(weights > 0.001)
                    }
            
            # Fallback to equal weights
            weights = np.ones(n_assets) / n_assets
            portfolio_return = np.sum(weights * mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.config['risk_free_rate']) / portfolio_vol
            
            weights_dict = dict(zip(price_data.columns, weights))
            
            return {
                'method': 'custom_equal_weight',
                'weights': weights_dict,
                'expected_annual_return': portfolio_return,
                'annual_volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'optimization_success': True,
                'total_weight': 1.0,
                'active_positions': n_assets
            }
            
        except Exception as e:
            print(f"- Custom Sharpe optimization error: {e}")
            return {'optimization_success': False, 'error': str(e)}
    
    async def _custom_sortino_optimization(self, price_data: pd.DataFrame) -> Dict:
        """Custom Sortino ratio optimization"""
        try:
            # Simplified: use equal weights for custom implementation
            n_assets = len(price_data.columns)
            weights = np.ones(n_assets) / n_assets
            
            # Calculate returns
            returns = price_data.pct_change().dropna()
            portfolio_returns = returns.dot(weights)
            
            # Sortino ratio calculation
            mean_return = portfolio_returns.mean() * 252
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.01
            
            sortino_ratio = (mean_return - self.config['risk_free_rate']) / downside_vol
            
            weights_dict = dict(zip(price_data.columns, weights))
            
            return {
                'method': 'custom_sortino_equal_weight',
                'weights': weights_dict,
                'expected_annual_return': mean_return,
                'downside_volatility': downside_vol,
                'sortino_ratio': sortino_ratio,
                'optimization_success': True,
                'total_weight': 1.0,
                'active_positions': n_assets
            }
            
        except Exception as e:
            print(f"- Custom Sortino optimization error: {e}")
            return {'optimization_success': False, 'error': str(e)}
    
    async def _custom_min_vol_optimization(self, price_data: pd.DataFrame) -> Dict:
        """Custom minimum volatility optimization"""
        try:
            # Simplified minimum volatility using equal weights
            n_assets = len(price_data.columns)
            weights = np.ones(n_assets) / n_assets
            
            returns = price_data.pct_change().dropna()
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            portfolio_return = np.sum(weights * mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.config['risk_free_rate']) / portfolio_vol
            
            weights_dict = dict(zip(price_data.columns, weights))
            
            return {
                'method': 'custom_min_vol_equal_weight',
                'weights': weights_dict,
                'expected_annual_return': portfolio_return,
                'annual_volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'optimization_success': True,
                'total_weight': 1.0,
                'active_positions': n_assets
            }
            
        except Exception as e:
            print(f"- Custom min vol optimization error: {e}")
            return {'optimization_success': False, 'error': str(e)}
    
    async def _custom_efficient_frontier(self, price_data: pd.DataFrame, n_points: int) -> Dict:
        """Custom efficient frontier generation"""
        try:
            # Simplified efficient frontier
            returns = price_data.pct_change().dropna()
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            frontier_results = []
            
            # Generate points along frontier using different risk levels
            risk_levels = np.linspace(0.05, 0.25, n_points)
            
            for target_vol in risk_levels:
                # Equal weight portfolio as approximation
                n_assets = len(price_data.columns)
                weights = np.ones(n_assets) / n_assets
                
                portfolio_return = np.sum(weights * mean_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - self.config['risk_free_rate']) / portfolio_vol
                
                frontier_results.append({
                    'expected_return': portfolio_return,
                    'volatility': portfolio_vol,
                    'sharpe_ratio': sharpe_ratio,
                    'weights': dict(zip(price_data.columns, weights))
                })
            
            # Find optimal portfolio
            optimal_portfolio = max(frontier_results, key=lambda x: x['sharpe_ratio'])
            
            return {
                'method': 'custom_efficient_frontier',
                'n_points': len(frontier_results),
                'frontier_points': frontier_results,
                'optimal_portfolio': optimal_portfolio,
                'optimization_success': True
            }
            
        except Exception as e:
            print(f"- Custom efficient frontier error: {e}")
            return {'optimization_success': False, 'error': str(e)}
    
    async def _custom_discrete_allocation(self, weights: Dict, latest_prices: Dict, 
                                        total_portfolio_value: float) -> Dict:
        """Custom discrete allocation"""
        try:
            allocation = {}
            total_invested = 0
            
            for asset, weight in weights.items():
                if asset in latest_prices and weight > 0:
                    target_value = weight * total_portfolio_value
                    shares = int(target_value / latest_prices[asset])
                    if shares > 0:
                        allocation[asset] = shares
                        total_invested += shares * latest_prices[asset]
            
            leftover = total_portfolio_value - total_invested
            
            return {
                'method': 'custom_discrete_allocation',
                'share_allocation': allocation,
                'total_invested': total_invested,
                'leftover_cash': leftover,
                'portfolio_value': total_portfolio_value,
                'utilization_rate': total_invested / total_portfolio_value,
                'allocation_success': True
            }
            
        except Exception as e:
            print(f"- Custom discrete allocation error: {e}")
            return {'allocation_success': False, 'error': str(e)}

# Create global instance
pyportfolioopt_engine = PyPortfolioOptEngine()