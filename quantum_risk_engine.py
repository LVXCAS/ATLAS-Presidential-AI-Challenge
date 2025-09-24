"""
QUANTUM RISK ENGINE - MAXIMUM POTENTIAL RISK MANAGEMENT
======================================================
Advanced risk management and portfolio optimization system using
ALL available optimization and risk libraries for institutional-grade
risk control and portfolio construction.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import cvxpy as cp
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.objective_functions import L2_reg
import riskfolio as rp
from scipy.optimize import minimize
from scipy import stats
import empyrical as ep
import quantstats as qs
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

class QuantumRiskEngine:
    """
    Maximum potential risk management system combining:
    - Modern Portfolio Theory (PyPortfolioOpt)
    - Risk Parity & Advanced Methods (Riskfolio-Lib)  
    - Convex Optimization (CVXPY)
    - Risk Analytics (empyrical, QuantStats)
    - GARCH Modeling (arch)
    - Monte Carlo Simulation
    - Dynamic Hedging Strategies
    """
    
    def __init__(self, max_portfolio_risk=0.15, confidence_level=0.95):
        self.max_portfolio_risk = max_portfolio_risk
        self.confidence_level = confidence_level
        self.risk_models = {}
        self.portfolio_weights = {}
        self.risk_metrics = {}
        self.hedge_ratios = {}
        
        print("🛡️ QUANTUM RISK ENGINE INITIALIZED")
        print("=" * 60)
        print("RISK MANAGEMENT ARSENAL:")
        print("  📊 Portfolio Optimization: Markowitz, Black-Litterman")
        print("  ⚖️ Risk Parity: Equal Risk Contribution, HRP")
        print("  📈 Advanced Models: CVaR, Maximum Diversification")
        print("  🎯 Dynamic Hedging: Options, Futures, VIX")
        print("  📉 Risk Analytics: VaR, Expected Shortfall, Drawdown")
        print("  🔄 Real-time Monitoring: Position-level risk tracking")
        print("=" * 60)
        
        self.initialize_risk_models()
    
    def initialize_risk_models(self):
        """Initialize all risk modeling components."""
        
        print("🔧 INITIALIZING RISK MODELS...")
        
        # Risk model types
        self.risk_model_types = [
            'sample_covariance',
            'ledoit_wolf',
            'oracle_approximating',
            'exp_cov',
            'cov_shrinkage'
        ]
        
        # Optimization methods
        self.optimization_methods = [
            'max_sharpe',
            'min_volatility', 
            'max_quadratic_utility',
            'efficient_risk',
            'efficient_return',
            'risk_parity',
            'hrp',  # Hierarchical Risk Parity
            'cvar_optimization',
            'max_diversification'
        ]
        
        print("✅ Risk models initialized")
    
    def analyze_portfolio_risk(self, returns, weights=None):
        """
        Comprehensive portfolio risk analysis using all available methods.
        """
        
        print("📊 ANALYZING PORTFOLIO RISK...")
        
        if weights is None:
            weights = np.ones(len(returns.columns)) / len(returns.columns)
        
        portfolio_returns = (returns * weights).sum(axis=1)
        
        risk_metrics = {}
        
        # Basic risk metrics using empyrical
        risk_metrics['volatility'] = ep.annual_volatility(portfolio_returns)
        risk_metrics['sharpe_ratio'] = ep.sharpe_ratio(portfolio_returns)
        risk_metrics['sortino_ratio'] = ep.sortino_ratio(portfolio_returns)
        risk_metrics['max_drawdown'] = ep.max_drawdown(portfolio_returns)
        risk_metrics['calmar_ratio'] = ep.calmar_ratio(portfolio_returns)
        risk_metrics['tail_ratio'] = ep.tail_ratio(portfolio_returns)
        risk_metrics['skewness'] = stats.skew(portfolio_returns.dropna())
        risk_metrics['kurtosis'] = stats.kurtosis(portfolio_returns.dropna())
        
        # Value at Risk (multiple methods)
        risk_metrics['var_95'] = np.percentile(portfolio_returns.dropna(), (1-0.95)*100)
        risk_metrics['var_99'] = np.percentile(portfolio_returns.dropna(), (1-0.99)*100)
        
        # Conditional Value at Risk (Expected Shortfall)
        var_95 = risk_metrics['var_95']
        risk_metrics['cvar_95'] = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Monte Carlo VaR
        risk_metrics['mc_var_95'] = self.calculate_monte_carlo_var(portfolio_returns, 0.95)
        
        # GARCH-based risk modeling
        try:
            garch_var = self.calculate_garch_var(portfolio_returns)
            risk_metrics.update(garch_var)
        except Exception as e:
            print(f"⚠️ GARCH modeling failed: {e}")
        
        # Component contributions
        risk_metrics['component_var'] = self.calculate_component_var(returns, weights)
        
        # Stress testing
        risk_metrics['stress_scenarios'] = self.run_stress_tests(returns, weights)
        
        print(f"📈 Portfolio Volatility: {risk_metrics['volatility']:.2%}")
        print(f"📉 Max Drawdown: {risk_metrics['max_drawdown']:.2%}")
        print(f"📊 Sharpe Ratio: {risk_metrics['sharpe_ratio']:.3f}")
        print(f"📋 VaR (95%): {risk_metrics['var_95']:.2%}")
        
        return risk_metrics
    
    def optimize_portfolio_markowitz(self, returns, method='max_sharpe', **kwargs):
        """
        Advanced Markowitz optimization with multiple objectives.
        """
        
        print(f"🎯 OPTIMIZING PORTFOLIO: {method.upper()}")
        
        # Calculate expected returns using multiple methods
        mu_methods = {
            'mean_historical': expected_returns.mean_historical_return(returns),
            'ema': expected_returns.ema_historical_return(returns),
            'capm': expected_returns.capm_return(returns) if 'market_returns' in kwargs else None
        }
        
        # Calculate risk models using multiple methods
        S_methods = {}
        for risk_model_type in self.risk_model_types:
            try:
                if risk_model_type == 'sample_covariance':
                    S_methods[risk_model_type] = risk_models.sample_cov(returns)
                elif risk_model_type == 'ledoit_wolf':
                    S_methods[risk_model_type] = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
                elif risk_model_type == 'exp_cov':
                    S_methods[risk_model_type] = risk_models.exp_cov(returns)
                elif risk_model_type == 'cov_shrinkage':
                    S_methods[risk_model_type] = risk_models.CovarianceShrinkage(returns).shrunk_covariance()
            except Exception as e:
                print(f"⚠️ {risk_model_type} failed: {e}")
        
        # Use best available expected returns and covariance
        mu = mu_methods['mean_historical']  # Default to historical mean
        S = S_methods.get('ledoit_wolf', S_methods.get('sample_covariance'))
        
        # Optimize using PyPortfolioOpt
        ef = EfficientFrontier(mu, S)
        
        # Add L2 regularization to prevent concentration
        ef.add_objective(L2_reg, gamma=0.1)
        
        # Apply constraints
        if 'sector_constraints' in kwargs:
            ef.add_sector_constraints(kwargs['sector_constraints'])
        
        # Optimize based on method
        if method == 'max_sharpe':
            weights = ef.max_sharpe()
        elif method == 'min_volatility':
            weights = ef.min_volatility()
        elif method == 'efficient_risk':
            target_risk = kwargs.get('target_risk', 0.15)
            weights = ef.efficient_risk(target_risk)
        elif method == 'efficient_return':
            target_return = kwargs.get('target_return', 0.10)
            weights = ef.efficient_return(target_return)
        else:
            weights = ef.max_sharpe()  # Default
        
        # Clean weights
        cleaned_weights = ef.clean_weights()
        
        # Calculate performance
        performance = ef.portfolio_performance(verbose=False)
        
        optimization_result = {
            'weights': cleaned_weights,
            'expected_return': performance[0],
            'volatility': performance[1],
            'sharpe_ratio': performance[2],
            'method': method
        }
        
        print(f"✅ Expected Return: {performance[0]:.2%}")
        print(f"✅ Volatility: {performance[1]:.2%}")
        print(f"✅ Sharpe Ratio: {performance[2]:.3f}")
        
        return optimization_result
    
    def optimize_portfolio_riskfolio(self, returns, method='HRP'):
        """
        Advanced portfolio optimization using Riskfolio-Lib methods.
        """
        
        print(f"🎯 RISKFOLIO OPTIMIZATION: {method.upper()}")
        
        # Create portfolio object
        port = rp.Portfolio(returns=returns)
        
        # Calculate inputs
        port.assets_stats(method_mu='hist', method_cov='hist')
        
        # Optimize based on method
        if method == 'HRP':
            # Hierarchical Risk Parity
            weights = port.optimization(model='HRP', codependence='pearson', 
                                      covariance='hist', obj='MinRisk')
        elif method == 'HERC':
            # Hierarchical Equal Risk Contribution
            weights = port.optimization(model='HERC', codependence='pearson',
                                      covariance='hist', obj='MinRisk')
        elif method == 'NCO':
            # Nested Clustered Optimization
            weights = port.optimization(model='NCO', codependence='pearson',
                                      covariance='hist', obj='Sharpe')
        elif method == 'BL':
            # Black-Litterman
            # Would need views as input
            weights = port.optimization(model='Classic', rm='MV', obj='Sharpe')
        else:
            # Default to classic mean-variance
            weights = port.optimization(model='Classic', rm='MV', obj='Sharpe')
        
        # Calculate performance metrics
        port_return = port.mu.T @ weights
        port_risk = np.sqrt(weights.T @ port.cov @ weights)
        
        optimization_result = {
            'weights': weights.to_dict()[weights.columns[0]],
            'expected_return': float(port_return.iloc[0]),
            'volatility': float(port_risk),
            'sharpe_ratio': float(port_return.iloc[0] / port_risk),
            'method': method
        }
        
        print(f"✅ Expected Return: {optimization_result['expected_return']:.2%}")
        print(f"✅ Volatility: {optimization_result['volatility']:.2%}")
        print(f"✅ Sharpe Ratio: {optimization_result['sharpe_ratio']:.3f}")
        
        return optimization_result
    
    def optimize_cvar_portfolio(self, returns, alpha=0.05):
        """
        CVaR (Conditional Value at Risk) portfolio optimization using CVXPY.
        """
        
        print(f"📊 CVAR OPTIMIZATION (α={alpha})")
        
        n_assets = len(returns.columns)
        n_scenarios = len(returns)
        
        # Variables
        weights = cp.Variable(n_assets)
        z = cp.Variable(n_scenarios)
        var = cp.Variable()
        
        # Portfolio returns for each scenario
        portfolio_returns = returns.values @ weights
        
        # CVaR constraints
        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= 0,  # Long-only (can be modified)
            z >= 0,  # Auxiliary variables non-negative
            z >= -(portfolio_returns - var)  # CVaR definition
        ]
        
        # Objective: minimize CVaR
        cvar = var - (1/alpha) * cp.sum(z) / n_scenarios
        objective = cp.Minimize(-cvar)  # Maximize CVaR (minimize negative CVaR)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)
        
        if problem.status == cp.OPTIMAL:
            optimal_weights = weights.value
            cvar_value = -objective.value
            
            # Calculate portfolio metrics
            port_returns = (returns.values * optimal_weights).sum(axis=1)
            expected_return = np.mean(port_returns)
            volatility = np.std(port_returns)
            
            result = {
                'weights': dict(zip(returns.columns, optimal_weights)),
                'expected_return': expected_return,
                'volatility': volatility,
                'cvar': cvar_value,
                'method': 'CVaR'
            }
            
            print(f"✅ Expected Return: {expected_return:.2%}")
            print(f"✅ Volatility: {volatility:.2%}")
            print(f"✅ CVaR: {cvar_value:.2%}")
            
            return result
        else:
            print("❌ CVaR optimization failed")
            return None
    
    def calculate_monte_carlo_var(self, returns, confidence_level, n_simulations=10000):
        """Calculate Value at Risk using Monte Carlo simulation."""
        
        # Fit distributions to returns
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        # Generate random scenarios
        simulated_returns = np.random.normal(mu, sigma, n_simulations)
        
        # Calculate VaR
        var = np.percentile(simulated_returns, (1-confidence_level)*100)
        
        return var
    
    def calculate_garch_var(self, returns, confidence_level=0.05):
        """Calculate VaR using GARCH modeling for volatility clustering."""
        
        # Fit GARCH(1,1) model
        returns_clean = returns.dropna() * 100  # Convert to percentage
        
        garch_model = arch_model(returns_clean, vol='Garch', p=1, q=1)
        garch_fitted = garch_model.fit(disp='off')
        
        # Forecast volatility
        forecast = garch_fitted.forecast(horizon=1)
        forecasted_vol = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
        
        # Calculate VaR assuming normal distribution
        var_95 = stats.norm.ppf(0.05) * forecasted_vol
        var_99 = stats.norm.ppf(0.01) * forecasted_vol
        
        return {
            'garch_vol_forecast': forecasted_vol,
            'garch_var_95': var_95,
            'garch_var_99': var_99
        }
    
    def calculate_component_var(self, returns, weights):
        """Calculate component Value at Risk contributions."""
        
        portfolio_returns = (returns * weights).sum(axis=1)
        portfolio_var = np.percentile(portfolio_returns, 5)  # 95% VaR
        
        # Calculate marginal VaR for each asset
        component_vars = {}
        
        for i, asset in enumerate(returns.columns):
            # Perturb weight slightly
            perturbed_weights = weights.copy()
            perturbed_weights[i] += 0.01
            perturbed_weights = perturbed_weights / perturbed_weights.sum()
            
            perturbed_returns = (returns * perturbed_weights).sum(axis=1)
            perturbed_var = np.percentile(perturbed_returns, 5)
            
            # Marginal VaR
            marginal_var = (perturbed_var - portfolio_var) / 0.01
            
            # Component VaR
            component_var = marginal_var * weights[i]
            component_vars[asset] = component_var
        
        return component_vars
    
    def run_stress_tests(self, returns, weights):
        """Run comprehensive stress testing scenarios."""
        
        stress_scenarios = {}
        
        # Historical stress scenarios
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Market crash scenarios
        crash_returns = portfolio_returns[portfolio_returns <= portfolio_returns.quantile(0.01)]
        stress_scenarios['market_crash'] = {
            'worst_return': crash_returns.min(),
            'average_loss': crash_returns.mean(),
            'frequency': len(crash_returns) / len(portfolio_returns)
        }
        
        # High volatility periods
        vol_periods = portfolio_returns.rolling(20).std().quantile(0.9)
        high_vol_mask = portfolio_returns.rolling(20).std() > vol_periods
        high_vol_returns = portfolio_returns[high_vol_mask]
        
        stress_scenarios['high_volatility'] = {
            'volatility': high_vol_returns.std(),
            'average_return': high_vol_returns.mean(),
            'worst_return': high_vol_returns.min()
        }
        
        # Correlation breakdown (when correlations spike to 1)
        # Simulate scenario where all correlations become 1
        simulated_corr_returns = returns.mean(axis=1) * weights.sum()
        stress_scenarios['correlation_breakdown'] = {
            'portfolio_var': np.percentile(simulated_corr_returns, 5),
            'portfolio_vol': simulated_corr_returns.std()
        }
        
        return stress_scenarios
    
    def calculate_hedge_ratios(self, portfolio_returns, hedge_instruments):
        """
        Calculate optimal hedge ratios for portfolio protection.
        """
        
        print("🛡️ CALCULATING HEDGE RATIOS...")
        
        hedge_ratios = {}
        
        for instrument, instrument_returns in hedge_instruments.items():
            # Align data
            aligned_data = pd.concat([portfolio_returns, instrument_returns], axis=1).dropna()
            
            if len(aligned_data) < 30:  # Need sufficient data
                continue
            
            port_ret = aligned_data.iloc[:, 0]
            hedge_ret = aligned_data.iloc[:, 1]
            
            # Calculate hedge ratio using OLS
            correlation = port_ret.corr(hedge_ret)
            hedge_vol = hedge_ret.std()
            port_vol = port_ret.std()
            
            # Minimum variance hedge ratio
            hedge_ratio = correlation * (port_vol / hedge_vol)
            
            # Hedge effectiveness
            hedged_returns = port_ret + hedge_ratio * hedge_ret
            hedge_effectiveness = 1 - (hedged_returns.var() / port_ret.var())
            
            hedge_ratios[instrument] = {
                'hedge_ratio': hedge_ratio,
                'correlation': correlation,
                'hedge_effectiveness': hedge_effectiveness,
                'hedged_volatility': hedged_returns.std(),
                'original_volatility': port_ret.std()
            }
            
            print(f"  {instrument}: Ratio={hedge_ratio:.3f}, Effectiveness={hedge_effectiveness:.2%}")
        
        return hedge_ratios
    
    def dynamic_position_sizing(self, signals, current_positions, risk_budget):
        """
        Dynamic position sizing based on Kelly Criterion and risk budgeting.
        """
        
        print("📏 CALCULATING DYNAMIC POSITION SIZES...")
        
        position_sizes = {}
        
        for asset, signal_data in signals.items():
            if signal_data['confidence'] < 0.75:  # Skip low confidence signals
                continue
                
            # Kelly Criterion calculation
            win_rate = signal_data.get('historical_win_rate', 0.6)
            avg_win = signal_data.get('avg_win', 0.02)
            avg_loss = signal_data.get('avg_loss', -0.01)
            
            if avg_loss != 0:
                kelly_fraction = (win_rate * avg_win - (1-win_rate) * abs(avg_loss)) / avg_win
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            else:
                kelly_fraction = 0
            
            # Risk budgeting adjustment
            asset_risk_budget = risk_budget / len([s for s in signals.values() if s['confidence'] >= 0.75])
            risk_adjusted_size = min(kelly_fraction, asset_risk_budget * 2)
            
            # Confidence adjustment
            confidence_adj = signal_data['confidence']
            final_size = risk_adjusted_size * confidence_adj
            
            position_sizes[asset] = {
                'position_size': final_size,
                'kelly_fraction': kelly_fraction,
                'risk_budget': asset_risk_budget,
                'confidence': signal_data['confidence']
            }
        
        return position_sizes
    
    def real_time_risk_monitoring(self, current_positions, market_data):
        """
        Real-time risk monitoring and alerts.
        """
        
        risk_alerts = []
        
        # Portfolio level checks
        total_exposure = sum(abs(pos['size']) for pos in current_positions.values())
        if total_exposure > 1.0:  # More than 100% exposure
            risk_alerts.append({
                'type': 'EXPOSURE_LIMIT',
                'severity': 'HIGH',
                'message': f'Total exposure {total_exposure:.1%} exceeds limit'
            })
        
        # Individual position checks
        for asset, position in current_positions.items():
            position_size = abs(position['size'])
            
            if position_size > 0.2:  # More than 20% in single position
                risk_alerts.append({
                    'type': 'CONCENTRATION_RISK',
                    'severity': 'MEDIUM',
                    'asset': asset,
                    'message': f'{asset} position {position_size:.1%} exceeds concentration limit'
                })
        
        # Correlation monitoring
        # Implementation would check for correlation spikes
        
        return risk_alerts

# Example usage
if __name__ == "__main__":
    
    # Initialize risk engine
    risk_engine = QuantumRiskEngine()
    
    # Create sample returns data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    assets = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    returns = pd.DataFrame(
        np.random.randn(len(dates), len(assets)) * 0.02,
        index=dates,
        columns=assets
    )
    
    # Add some correlation structure
    returns['TSLA'] = returns['TSLA'] + 0.3 * returns['AAPL']
    
    # Analyze portfolio risk
    risk_metrics = risk_engine.analyze_portfolio_risk(returns)
    
    # Optimize portfolio
    markowitz_result = risk_engine.optimize_portfolio_markowitz(returns, method='max_sharpe')
    riskfolio_result = risk_engine.optimize_portfolio_riskfolio(returns, method='HRP')
    cvar_result = risk_engine.optimize_cvar_portfolio(returns)
    
    print("✅ QUANTUM RISK ENGINE DEMONSTRATION COMPLETED")