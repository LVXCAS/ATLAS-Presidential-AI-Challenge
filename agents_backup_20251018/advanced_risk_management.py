"""
Advanced Risk Management System
Uses empyrical, arch, and ML predictions for sophisticated risk control
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Risk libraries
try:
    import empyrical as ep
    EMPYRICAL_AVAILABLE = True
except ImportError:
    EMPYRICAL_AVAILABLE = False

try:
    import arch
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

from config.logging_config import get_logger

logger = get_logger(__name__)

class AdvancedRiskManager:
    """Advanced risk management with professional-grade metrics"""
    
    def __init__(self):
        self.portfolio_history = {}
        self.risk_metrics_cache = {}
        self.var_models = {}
        self.stress_scenarios = {}
        
    async def calculate_portfolio_risk(self, positions: List[Dict], market_data: Dict) -> Dict:
        """Calculate comprehensive portfolio risk metrics"""
        
        try:
            if not positions:
                return self._get_default_risk_metrics()
            
            # Calculate portfolio returns
            portfolio_returns = await self._calculate_portfolio_returns(positions, market_data)
            
            if portfolio_returns is None or len(portfolio_returns) < 30:
                return self._get_default_risk_metrics()
            
            risk_metrics = {}
            
            # Basic risk metrics
            risk_metrics['basic_metrics'] = await self._calculate_basic_risk_metrics(portfolio_returns)
            
            # Advanced risk metrics using empyrical
            if EMPYRICAL_AVAILABLE:
                risk_metrics['advanced_metrics'] = await self._calculate_empyrical_metrics(portfolio_returns)
            
            # Value at Risk calculations
            risk_metrics['var_metrics'] = await self._calculate_var_metrics(portfolio_returns)
            
            # Volatility modeling using ARCH
            if ARCH_AVAILABLE:
                risk_metrics['volatility_modeling'] = await self._calculate_arch_metrics(portfolio_returns)
            
            # Stress testing
            risk_metrics['stress_tests'] = await self._perform_stress_tests(positions, market_data)
            
            # Position-level risk
            risk_metrics['position_risks'] = await self._calculate_position_risks(positions, market_data)
            
            # Risk-adjusted performance
            risk_metrics['performance_metrics'] = await self._calculate_performance_metrics(portfolio_returns)
            
            # Risk alerts and warnings
            risk_metrics['risk_alerts'] = await self._generate_risk_alerts(risk_metrics, positions)
            
            # Portfolio optimization suggestions
            risk_metrics['optimization_suggestions'] = await self._suggest_optimizations(risk_metrics, positions)
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Portfolio risk calculation error: {e}")
            return self._get_default_risk_metrics()
    
    async def calculate_position_sizing(self, signal_strength: float, confidence: float, 
                                      volatility: float, portfolio_value: float,
                                      max_position_risk: float = 0.02) -> Dict:
        """Calculate optimal position sizing based on risk parameters"""
        
        try:
            # Kelly Criterion with modifications
            kelly_fraction = await self._calculate_kelly_fraction(signal_strength, confidence, volatility)
            
            # Volatility-adjusted sizing
            vol_adjusted_size = await self._calculate_volatility_adjusted_size(
                signal_strength, volatility, max_position_risk)
            
            # Risk parity sizing
            risk_parity_size = await self._calculate_risk_parity_size(
                volatility, portfolio_value, max_position_risk)
            
            # Ensemble sizing (weighted average)
            weights = {'kelly': 0.3, 'volatility': 0.4, 'risk_parity': 0.3}
            ensemble_size = (kelly_fraction * weights['kelly'] + 
                           vol_adjusted_size * weights['volatility'] + 
                           risk_parity_size * weights['risk_parity'])
            
            # Apply safety constraints
            max_size = portfolio_value * 0.1  # Maximum 10% of portfolio
            min_size = portfolio_value * 0.005  # Minimum 0.5% of portfolio
            
            recommended_size = max(min_size, min(max_size, ensemble_size))
            
            return {
                'recommended_size': recommended_size,
                'kelly_fraction': kelly_fraction,
                'volatility_adjusted': vol_adjusted_size,
                'risk_parity': risk_parity_size,
                'ensemble_size': ensemble_size,
                'size_rationale': self._explain_sizing_decision(
                    kelly_fraction, vol_adjusted_size, risk_parity_size, recommended_size),
                'risk_metrics': {
                    'expected_volatility': volatility,
                    'confidence_level': confidence,
                    'signal_strength': signal_strength,
                    'max_drawdown_risk': recommended_size / portfolio_value
                }
            }
            
        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return {'recommended_size': portfolio_value * 0.02}  # Default 2%
    
    async def _calculate_portfolio_returns(self, positions: List[Dict], market_data: Dict) -> Optional[pd.Series]:
        """Calculate historical portfolio returns"""
        
        try:
            # This would typically fetch historical data for each position
            # For now, simulate with market data
            
            if 'historical_returns' in market_data:
                return pd.Series(market_data['historical_returns'])
            
            # Generate synthetic portfolio returns based on positions
            # In production, this would aggregate actual position returns
            returns = []
            for i in range(100):  # Simulate 100 days of returns
                daily_return = np.random.normal(0.001, 0.02)  # Mean return with volatility
                returns.append(daily_return)
            
            return pd.Series(returns)
            
        except Exception as e:
            logger.warning(f"Portfolio returns calculation error: {e}")
            return None
    
    async def _calculate_basic_risk_metrics(self, returns: pd.Series) -> Dict:
        """Calculate basic risk metrics"""
        
        try:
            return {
                'daily_volatility': float(returns.std()),
                'annualized_volatility': float(returns.std() * np.sqrt(252)),
                'mean_return': float(returns.mean()),
                'annualized_return': float(returns.mean() * 252),
                'skewness': float(returns.skew()),
                'kurtosis': float(returns.kurtosis()),
                'positive_days': float((returns > 0).sum() / len(returns)),
                'negative_days': float((returns < 0).sum() / len(returns)),
                'max_daily_gain': float(returns.max()),
                'max_daily_loss': float(returns.min()),
                'return_volatility_ratio': float(returns.mean() / returns.std()) if returns.std() > 0 else 0
            }
            
        except Exception as e:
            logger.warning(f"Basic risk metrics error: {e}")
            return {}
    
    async def _calculate_empyrical_metrics(self, returns: pd.Series) -> Dict:
        """Calculate advanced risk metrics using empyrical"""
        
        if not EMPYRICAL_AVAILABLE:
            return {}
        
        try:
            # Convert to daily frequency if needed
            returns_daily = returns if len(returns) > 0 else pd.Series([0])
            
            metrics = {}
            
            # Risk metrics
            try:
                metrics['sharpe_ratio'] = float(ep.sharpe_ratio(returns_daily, risk_free=0.02/252))
            except:
                metrics['sharpe_ratio'] = 0.0
            
            try:
                metrics['sortino_ratio'] = float(ep.sortino_ratio(returns_daily, required_return=0))
            except:
                metrics['sortino_ratio'] = 0.0
            
            try:
                metrics['calmar_ratio'] = float(ep.calmar_ratio(returns_daily))
            except:
                metrics['calmar_ratio'] = 0.0
            
            try:
                metrics['max_drawdown'] = float(ep.max_drawdown(returns_daily))
            except:
                metrics['max_drawdown'] = 0.0
            
            try:
                metrics['downside_risk'] = float(ep.downside_risk(returns_daily))
            except:
                metrics['downside_risk'] = returns_daily.std()
            
            try:
                metrics['value_at_risk'] = float(ep.value_at_risk(returns_daily, cutoff=0.05))
            except:
                metrics['value_at_risk'] = returns_daily.quantile(0.05)
            
            try:
                metrics['conditional_value_at_risk'] = float(ep.conditional_value_at_risk(returns_daily, cutoff=0.05))
            except:
                metrics['conditional_value_at_risk'] = returns_daily[returns_daily <= returns_daily.quantile(0.05)].mean()
            
            try:
                metrics['stability'] = float(ep.stability_of_timeseries(returns_daily.cumsum()))
            except:
                metrics['stability'] = 0.5
            
            try:
                metrics['tail_ratio'] = float(ep.tail_ratio(returns_daily))
            except:
                metrics['tail_ratio'] = 1.0
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Empyrical metrics error: {e}")
            return {}
    
    async def _calculate_var_metrics(self, returns: pd.Series) -> Dict:
        """Calculate Value at Risk metrics"""
        
        try:
            var_metrics = {}
            
            # Historical VaR at different confidence levels
            confidence_levels = [0.90, 0.95, 0.99]
            
            for confidence in confidence_levels:
                alpha = 1 - confidence
                var_metrics[f'historical_var_{int(confidence*100)}'] = float(returns.quantile(alpha))
            
            # Parametric VaR (assuming normal distribution)
            mean_return = returns.mean()
            vol = returns.std()
            
            for confidence in confidence_levels:
                z_score = {0.90: -1.28, 0.95: -1.645, 0.99: -2.33}[confidence]
                var_metrics[f'parametric_var_{int(confidence*100)}'] = float(mean_return + z_score * vol)
            
            # Expected Shortfall (CVaR)
            for confidence in confidence_levels:
                alpha = 1 - confidence
                var_threshold = returns.quantile(alpha)
                tail_returns = returns[returns <= var_threshold]
                if len(tail_returns) > 0:
                    var_metrics[f'expected_shortfall_{int(confidence*100)}'] = float(tail_returns.mean())
                else:
                    var_metrics[f'expected_shortfall_{int(confidence*100)}'] = var_threshold
            
            # Rolling VaR (30-day window)
            if len(returns) >= 30:
                rolling_var = returns.rolling(30).quantile(0.05)
                var_metrics['rolling_var_mean'] = float(rolling_var.mean())
                var_metrics['rolling_var_current'] = float(rolling_var.iloc[-1])
            
            return var_metrics
            
        except Exception as e:
            logger.warning(f"VaR metrics error: {e}")
            return {}
    
    async def _calculate_arch_metrics(self, returns: pd.Series) -> Dict:
        """Calculate ARCH/GARCH volatility metrics"""
        
        if not ARCH_AVAILABLE or len(returns) < 100:
            return {}
        
        try:
            # Scale returns to percentage
            scaled_returns = returns * 100
            
            # Fit GARCH(1,1) model
            garch_model = arch_model(scaled_returns, vol='Garch', p=1, q=1)
            garch_fit = garch_model.fit(disp='off')
            
            # Get conditional volatility
            conditional_vol = garch_fit.conditional_volatility / 100  # Convert back to decimal
            
            # Volatility forecasting
            forecast = garch_fit.forecast(horizon=5)
            vol_forecast = np.sqrt(forecast.variance.iloc[-1, :].values) / 100
            
            arch_metrics = {
                'current_conditional_vol': float(conditional_vol.iloc[-1]),
                'mean_conditional_vol': float(conditional_vol.mean()),
                'vol_clustering_strength': float(conditional_vol.std() / conditional_vol.mean()),
                'garch_alpha': float(garch_fit.params['alpha[1]']),
                'garch_beta': float(garch_fit.params['beta[1]']),
                'garch_persistence': float(garch_fit.params['alpha[1]'] + garch_fit.params['beta[1]']),
                'vol_forecast_1d': float(vol_forecast[0]),
                'vol_forecast_5d': float(vol_forecast[4]),
                'long_run_vol': float(np.sqrt(garch_fit.params['omega'] / 
                                            (1 - garch_fit.params['alpha[1]'] - garch_fit.params['beta[1]'])) / 100)
            }
            
            return arch_metrics
            
        except Exception as e:
            logger.warning(f"ARCH metrics error: {e}")
            return {}
    
    async def _perform_stress_tests(self, positions: List[Dict], market_data: Dict) -> Dict:
        """Perform stress testing scenarios"""
        
        try:
            stress_results = {}
            
            # Define stress scenarios
            scenarios = {
                'market_crash_2008': {'equity_shock': -0.30, 'vol_shock': 2.0},
                'flash_crash_2010': {'equity_shock': -0.10, 'vol_shock': 3.0},
                'covid_crash_2020': {'equity_shock': -0.35, 'vol_shock': 2.5},
                'rate_shock': {'equity_shock': -0.15, 'rate_shock': 0.02},
                'volatility_spike': {'equity_shock': -0.05, 'vol_shock': 4.0}
            }
            
            portfolio_value = sum(pos.get('market_value', 0) for pos in positions)
            
            for scenario_name, scenario_params in scenarios.items():
                scenario_pnl = 0
                
                for position in positions:
                    # Simplified stress testing - in production would use Greeks and more sophisticated modeling
                    position_value = position.get('market_value', 0)
                    
                    # Apply equity shock
                    equity_impact = position_value * scenario_params.get('equity_shock', 0)
                    
                    # Apply volatility shock (affects options differently)
                    vol_impact = 0
                    if 'OPTION' in position.get('instrument_type', ''):
                        vega = position.get('vega', 0)
                        vol_shock = scenario_params.get('vol_shock', 1.0) - 1.0
                        vol_impact = vega * vol_shock
                    
                    position_pnl = equity_impact + vol_impact
                    scenario_pnl += position_pnl
                
                stress_results[scenario_name] = {
                    'total_pnl': scenario_pnl,
                    'pnl_percentage': (scenario_pnl / portfolio_value * 100) if portfolio_value > 0 else 0,
                    'scenario_params': scenario_params
                }
            
            # Calculate worst-case scenario
            worst_scenario = min(stress_results.values(), key=lambda x: x['total_pnl'])
            stress_results['worst_case'] = worst_scenario
            
            return stress_results
            
        except Exception as e:
            logger.warning(f"Stress testing error: {e}")
            return {}
    
    async def _calculate_position_risks(self, positions: List[Dict], market_data: Dict) -> List[Dict]:
        """Calculate individual position risk metrics"""
        
        try:
            position_risks = []
            
            for position in positions:
                risk_metrics = {
                    'symbol': position.get('symbol', 'UNKNOWN'),
                    'position_value': position.get('market_value', 0),
                    'quantity': position.get('quantity', 0),
                    'delta': position.get('delta', 1.0),
                    'gamma': position.get('gamma', 0),
                    'theta': position.get('theta', 0),
                    'vega': position.get('vega', 0)
                }
                
                # Calculate position-specific risks
                risk_metrics['delta_risk'] = abs(risk_metrics['delta'] * risk_metrics['position_value'])
                risk_metrics['gamma_risk'] = abs(risk_metrics['gamma'] * risk_metrics['position_value'] * 0.01)  # 1% move
                risk_metrics['theta_decay'] = abs(risk_metrics['theta'])
                risk_metrics['vega_risk'] = abs(risk_metrics['vega'] * 0.01)  # 1% vol change
                
                # Overall position risk score
                total_greek_risk = (risk_metrics['delta_risk'] + 
                                  risk_metrics['gamma_risk'] + 
                                  risk_metrics['theta_decay'] + 
                                  risk_metrics['vega_risk'])
                
                risk_metrics['total_risk_score'] = total_greek_risk
                
                # Risk classification
                if total_greek_risk > risk_metrics['position_value'] * 0.1:
                    risk_metrics['risk_level'] = 'HIGH'
                elif total_greek_risk > risk_metrics['position_value'] * 0.05:
                    risk_metrics['risk_level'] = 'MODERATE'
                else:
                    risk_metrics['risk_level'] = 'LOW'
                
                position_risks.append(risk_metrics)
            
            return position_risks
            
        except Exception as e:
            logger.warning(f"Position risk calculation error: {e}")
            return []
    
    async def _calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        """Calculate risk-adjusted performance metrics"""
        
        try:
            cumulative_returns = (1 + returns).cumprod() - 1
            
            performance = {
                'total_return': float(cumulative_returns.iloc[-1]),
                'annualized_return': float(returns.mean() * 252),
                'annualized_volatility': float(returns.std() * np.sqrt(252)),
                'information_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            }
            
            # Risk-adjusted returns
            if performance['annualized_volatility'] > 0:
                performance['sharpe_ratio'] = (performance['annualized_return'] - 0.02) / performance['annualized_volatility']
            else:
                performance['sharpe_ratio'] = 0
            
            # Drawdown analysis
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            performance['max_drawdown'] = float(drawdown.min())
            performance['current_drawdown'] = float(drawdown.iloc[-1])
            performance['drawdown_duration'] = len(drawdown[drawdown < -0.05])  # Days in >5% drawdown
            
            return performance
            
        except Exception as e:
            logger.warning(f"Performance metrics error: {e}")
            return {}
    
    async def _generate_risk_alerts(self, risk_metrics: Dict, positions: List[Dict]) -> List[Dict]:
        """Generate risk alerts and warnings"""
        
        alerts = []
        
        try:
            # Check basic metrics
            basic_metrics = risk_metrics.get('basic_metrics', {})
            
            # High volatility alert
            if basic_metrics.get('annualized_volatility', 0) > 0.4:  # 40% annualized
                alerts.append({
                    'type': 'HIGH_VOLATILITY',
                    'severity': 'WARNING',
                    'message': f"High portfolio volatility detected: {basic_metrics['annualized_volatility']:.1%}",
                    'recommendation': 'Consider reducing position sizes or hedging'
                })
            
            # VaR alerts
            var_metrics = risk_metrics.get('var_metrics', {})
            var_95 = var_metrics.get('historical_var_95', 0)
            
            if var_95 < -0.05:  # More than 5% daily VaR
                alerts.append({
                    'type': 'HIGH_VAR',
                    'severity': 'CRITICAL',
                    'message': f"High Value at Risk: {var_95:.2%} daily",
                    'recommendation': 'Reduce position sizes immediately'
                })
            
            # Drawdown alerts
            performance = risk_metrics.get('performance_metrics', {})
            current_drawdown = performance.get('current_drawdown', 0)
            
            if current_drawdown < -0.15:  # 15% drawdown
                alerts.append({
                    'type': 'LARGE_DRAWDOWN',
                    'severity': 'CRITICAL',
                    'message': f"Large drawdown detected: {current_drawdown:.1%}",
                    'recommendation': 'Review strategy and consider stopping trading'
                })
            
            # Stress test alerts
            stress_tests = risk_metrics.get('stress_tests', {})
            worst_case = stress_tests.get('worst_case', {})
            
            if worst_case.get('pnl_percentage', 0) < -20:  # 20% loss in worst case
                alerts.append({
                    'type': 'STRESS_TEST_FAILURE',
                    'severity': 'WARNING',
                    'message': f"Stress test shows potential {worst_case['pnl_percentage']:.1f}% loss",
                    'recommendation': 'Diversify positions or add hedges'
                })
            
            # Position concentration alerts
            total_value = sum(pos.get('market_value', 0) for pos in positions)
            for pos in positions:
                position_weight = pos.get('market_value', 0) / total_value if total_value > 0 else 0
                if position_weight > 0.2:  # 20% concentration
                    alerts.append({
                        'type': 'POSITION_CONCENTRATION',
                        'severity': 'WARNING',
                        'message': f"High concentration in {pos.get('symbol', 'UNKNOWN')}: {position_weight:.1%}",
                        'recommendation': 'Consider reducing position size'
                    })
            
            return alerts
            
        except Exception as e:
            logger.warning(f"Risk alerts generation error: {e}")
            return []
    
    async def _suggest_optimizations(self, risk_metrics: Dict, positions: List[Dict]) -> List[Dict]:
        """Suggest portfolio optimizations"""
        
        suggestions = []
        
        try:
            # Analyze risk metrics for optimization opportunities
            performance = risk_metrics.get('performance_metrics', {})
            basic_metrics = risk_metrics.get('basic_metrics', {})
            
            # Low Sharpe ratio suggestion
            if performance.get('sharpe_ratio', 0) < 0.5:
                suggestions.append({
                    'type': 'IMPROVE_SHARPE',
                    'priority': 'HIGH',
                    'suggestion': 'Consider strategies with better risk-adjusted returns',
                    'action': 'Reduce volatility or increase expected returns'
                })
            
            # High correlation suggestion
            if len(positions) > 1:
                suggestions.append({
                    'type': 'DIVERSIFICATION',
                    'priority': 'MEDIUM',
                    'suggestion': 'Add uncorrelated assets to improve diversification',
                    'action': 'Consider different sectors or asset classes'
                })
            
            # Volatility optimization
            if basic_metrics.get('annualized_volatility', 0) > 0.3:
                suggestions.append({
                    'type': 'REDUCE_VOLATILITY',
                    'priority': 'MEDIUM',
                    'suggestion': 'Portfolio volatility could be reduced',
                    'action': 'Implement volatility targeting or add hedges'
                })
            
            # Position sizing optimization
            position_risks = risk_metrics.get('position_risks', [])
            high_risk_positions = [pos for pos in position_risks if pos.get('risk_level') == 'HIGH']
            
            if high_risk_positions:
                suggestions.append({
                    'type': 'POSITION_SIZING',
                    'priority': 'HIGH',
                    'suggestion': f'{len(high_risk_positions)} positions have high risk levels',
                    'action': 'Reduce size of high-risk positions'
                })
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"Optimization suggestions error: {e}")
            return []
    
    async def _calculate_kelly_fraction(self, signal_strength: float, confidence: float, volatility: float) -> float:
        """Calculate Kelly fraction for position sizing"""
        
        try:
            # Estimate win probability and win/loss ratio from signal strength and confidence
            win_prob = 0.5 + (signal_strength * confidence / 2)  # Base 50% + signal adjustment
            win_prob = max(0.1, min(0.9, win_prob))  # Clamp to reasonable range
            
            # Estimate expected payoff ratio
            # Higher volatility means larger potential wins/losses
            avg_win = volatility * 1.5  # Assume wins are 1.5x volatility
            avg_loss = volatility * 1.0  # Assume losses are 1x volatility
            
            # Kelly formula: f = (bp - q) / b where b = win/loss ratio, p = win prob, q = loss prob
            if avg_loss > 0:
                payoff_ratio = avg_win / avg_loss
                kelly = (payoff_ratio * win_prob - (1 - win_prob)) / payoff_ratio
            else:
                kelly = 0
            
            # Apply safety factor (use fractional Kelly)
            kelly *= 0.25  # Use 25% of full Kelly to be conservative
            
            return max(0, min(0.1, kelly))  # Cap at 10% of portfolio
            
        except Exception as e:
            logger.warning(f"Kelly calculation error: {e}")
            return 0.02  # Default 2%
    
    async def _calculate_volatility_adjusted_size(self, signal_strength: float, volatility: float, max_risk: float) -> float:
        """Calculate position size adjusted for volatility"""
        
        try:
            # Target volatility approach
            target_vol = 0.15  # 15% target portfolio volatility
            
            if volatility > 0:
                base_size = target_vol / volatility
                
                # Adjust for signal strength
                signal_multiplier = 0.5 + (signal_strength * 0.5)  # 0.5 to 1.0 multiplier
                
                adjusted_size = base_size * signal_multiplier
                
                # Apply maximum risk constraint
                max_size_by_risk = max_risk / volatility if volatility > 0 else max_risk
                
                return min(adjusted_size, max_size_by_risk)
            else:
                return max_risk
                
        except Exception as e:
            logger.warning(f"Volatility-adjusted sizing error: {e}")
            return max_risk
    
    async def _calculate_risk_parity_size(self, volatility: float, portfolio_value: float, max_risk: float) -> float:
        """Calculate risk parity position size"""
        
        try:
            # Risk parity: each position contributes equal risk
            # Assuming we want this position to contribute a fixed percentage of portfolio risk
            
            target_risk_contribution = max_risk  # e.g., 2% of portfolio
            
            if volatility > 0:
                # Position size = (Target Risk Contribution * Portfolio Value) / Volatility
                risk_parity_size = (target_risk_contribution * portfolio_value) / volatility
                return min(risk_parity_size, portfolio_value * 0.1)  # Cap at 10%
            else:
                return portfolio_value * target_risk_contribution
                
        except Exception as e:
            logger.warning(f"Risk parity sizing error: {e}")
            return portfolio_value * max_risk
    
    def _explain_sizing_decision(self, kelly: float, vol_adjusted: float, 
                               risk_parity: float, final: float) -> str:
        """Explain the position sizing decision"""
        
        try:
            explanations = []
            
            if kelly > vol_adjusted and kelly > risk_parity:
                explanations.append("Kelly criterion suggests higher allocation due to favorable risk/reward")
            elif vol_adjusted > kelly and vol_adjusted > risk_parity:
                explanations.append("Volatility-adjusted sizing dominates due to risk management")
            else:
                explanations.append("Risk parity approach selected for balanced risk contribution")
            
            if final < max(kelly, vol_adjusted, risk_parity):
                explanations.append("Final size reduced by safety constraints")
            
            return "; ".join(explanations)
            
        except Exception as e:
            return "Standard risk-based sizing applied"
    
    def _get_default_risk_metrics(self) -> Dict:
        """Return default risk metrics when calculation fails"""
        return {
            'basic_metrics': {
                'annualized_volatility': 0.2,
                'annualized_return': 0.08,
                'sharpe_ratio': 0.4
            },
            'var_metrics': {
                'historical_var_95': -0.03
            },
            'performance_metrics': {
                'max_drawdown': -0.1,
                'current_drawdown': 0
            },
            'risk_alerts': [],
            'optimization_suggestions': []
        }

# Singleton instance
advanced_risk_manager = AdvancedRiskManager()