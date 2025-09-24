#!/usr/bin/env python3
"""
Enhanced Options Pricing Engine
Integrates Black-Scholes analytical pricing with Monte Carlo simulation
and advanced portfolio optimization techniques

Features:
- Multiple pricing models (Black-Scholes, Monte Carlo, Binomial)
- Real-time Greeks calculation
- Portfolio optimization with options
- Risk management and scenario analysis
- Integration with live market data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio

from .advanced_monte_carlo_engine import (
    AdvancedMonteCarloEngine, OptionSpec, SimulationResult, Greeks,
    advanced_monte_carlo_engine
)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from config.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class OptionsPosition:
    """Options position with pricing and risk metrics"""
    symbol: str
    option_type: str  # 'call' or 'put'
    strike: float
    expiration: datetime
    quantity: int
    entry_price: float
    current_price: Optional[float] = None
    greeks: Optional[Greeks] = None
    pnl: Optional[float] = None

@dataclass
class PortfolioMetrics:
    """Portfolio-level metrics"""
    total_value: float
    total_pnl: float
    delta_exposure: float
    gamma_exposure: float
    theta_decay: float
    vega_exposure: float
    var_95: float
    expected_return: float
    sharpe_ratio: float

class EnhancedOptionsPricingEngine:
    """Enhanced options pricing with Monte Carlo and portfolio optimization"""

    def __init__(self):
        self.monte_carlo = advanced_monte_carlo_engine
        self.risk_free_rate = 0.05  # Default 5%
        self.dividend_yields = {}   # Symbol -> dividend yield mapping

        # Pricing configuration
        self.mc_paths = 100000      # Monte Carlo paths
        self.mc_steps = 252         # Daily steps for MC
        self.confidence_level = 0.95 # For risk metrics

        # Cache for market data
        self.market_data_cache = {}
        self.cache_expiry = {}

    async def get_market_data(self, symbol: str) -> Dict:
        """Get current market data for the underlying"""
        try:
            # Check cache first
            if symbol in self.market_data_cache:
                cache_time = self.cache_expiry.get(symbol, datetime.min)
                if datetime.now() - cache_time < timedelta(minutes=5):
                    return self.market_data_cache[symbol]

            if not YFINANCE_AVAILABLE:
                logger.warning("yfinance not available, using mock data")
                return self._get_mock_market_data(symbol)

            # Fetch real market data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            info = ticker.info

            if hist.empty:
                return self._get_mock_market_data(symbol)

            current_price = float(hist['Close'].iloc[-1])

            # Calculate implied volatility from recent price movements
            returns = hist['Close'].pct_change().dropna()
            realized_vol = returns.std() * np.sqrt(252)  # Annualized

            market_data = {
                'current_price': current_price,
                'volatility': realized_vol,
                'dividend_yield': info.get('dividendYield', 0.0) or 0.0,
                'volume': hist['Volume'].iloc[-1],
                'bid': current_price * 0.999,  # Approximate bid/ask
                'ask': current_price * 1.001,
                'last_updated': datetime.now()
            }

            # Cache the data
            self.market_data_cache[symbol] = market_data
            self.cache_expiry[symbol] = datetime.now()

            return market_data

        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return self._get_mock_market_data(symbol)

    def _get_mock_market_data(self, symbol: str) -> Dict:
        """Generate mock market data for testing"""
        base_price = 100.0  # Default price
        if symbol == 'AAPL':
            base_price = 150.0
        elif symbol == 'GOOGL':
            base_price = 140.0
        elif symbol == 'TSLA':
            base_price = 250.0

        return {
            'current_price': base_price,
            'volatility': 0.25,
            'dividend_yield': 0.01,
            'volume': 1000000,
            'bid': base_price * 0.999,
            'ask': base_price * 1.001,
            'last_updated': datetime.now()
        }

    async def price_option(self, symbol: str, option_type: str, strike: float,
                          expiration: datetime, method: str = 'black_scholes') -> Dict:
        """
        Price option using specified method

        Methods:
        - 'black_scholes': Analytical Black-Scholes pricing
        - 'monte_carlo': Monte Carlo simulation
        - 'both': Compare both methods
        """
        try:
            # Get market data
            market_data = await self.get_market_data(symbol)
            current_price = market_data['current_price']
            volatility = market_data['volatility']
            dividend_yield = market_data['dividend_yield']

            # Calculate time to expiration
            now = datetime.now()
            time_to_expiry = (expiration - now).total_seconds() / (365.25 * 24 * 3600)

            if time_to_expiry <= 0:
                # Option has expired
                if option_type.lower() == 'call':
                    intrinsic_value = max(current_price - strike, 0)
                else:
                    intrinsic_value = max(strike - current_price, 0)

                return {
                    'price': intrinsic_value,
                    'method': 'intrinsic',
                    'greeks': Greeks(0, 0, 0, 0, 0),
                    'expired': True
                }

            # Create option specification
            option_spec = OptionSpec(
                S0=current_price,
                K=strike,
                T=time_to_expiry,
                r=self.risk_free_rate,
                sigma=volatility,
                option_type=option_type,
                dividend_yield=dividend_yield
            )

            results = {}

            if method in ['black_scholes', 'both']:
                # Black-Scholes pricing
                bs_price = self.monte_carlo.black_scholes_price(option_spec)
                greeks = self.monte_carlo.calculate_greeks(option_spec)

                results['black_scholes'] = {
                    'price': bs_price,
                    'greeks': greeks,
                    'method': 'analytical'
                }

            if method in ['monte_carlo', 'both']:
                # Monte Carlo pricing
                mc_result = self.monte_carlo.monte_carlo_option_price(
                    option_spec, paths=self.mc_paths, steps=self.mc_steps
                )

                results['monte_carlo'] = {
                    'price': mc_result.option_price,
                    'confidence_interval': mc_result.confidence_interval,
                    'standard_error': mc_result.standard_error,
                    'convergence': mc_result.convergence_stats,
                    'method': 'simulation'
                }

            # Return appropriate result
            if method == 'both':
                return results
            elif method == 'monte_carlo':
                return results['monte_carlo']
            else:
                return results['black_scholes']

        except Exception as e:
            logger.error(f"Error pricing option {symbol} {strike} {option_type}: {e}")
            return {'price': 0.0, 'error': str(e)}

    async def get_comprehensive_option_analysis(self, underlying_price: float,
                                             strike_price: float, time_to_expiry_days: int,
                                             volatility: float, option_type: str = 'call',
                                             dividend_yield: float = 0.0) -> Dict:
        """
        Compatibility method for old OPTIONS_BOT code
        Returns data in the format expected by the existing bot
        """
        try:
            # Convert days to years
            time_to_expiry = time_to_expiry_days / 365.25

            # Create option specification directly
            option_spec = OptionSpec(
                S0=underlying_price,
                K=strike_price,
                T=time_to_expiry,
                r=self.risk_free_rate,
                sigma=volatility,
                option_type=option_type,
                dividend_yield=dividend_yield
            )

            # Use Black-Scholes pricing for speed
            theoretical_price = self.monte_carlo.black_scholes_price(option_spec)
            greeks = self.monte_carlo.calculate_greeks(option_spec)

            # Return in the format expected by OPTIONS_BOT
            return {
                'pricing': {
                    'theoretical_price': theoretical_price,
                    'pricing_method': 'black_scholes_enhanced'
                },
                'greeks': {
                    'delta': greeks.delta,
                    'gamma': greeks.gamma,
                    'theta': greeks.theta,
                    'vega': greeks.vega,
                    'rho': greeks.rho
                }
            }

        except Exception as e:
            logger.error(f"Comprehensive option analysis failed: {e}")
            # Return fallback values
            return {
                'pricing': {
                    'theoretical_price': max(underlying_price - strike_price, 0) if option_type.lower() == 'call' else max(strike_price - underlying_price, 0),
                    'pricing_method': 'intrinsic_fallback'
                },
                'greeks': {
                    'delta': 0.5 if option_type.lower() == 'call' else -0.5,
                    'gamma': 0.0,
                    'theta': 0.0,
                    'vega': 0.0,
                    'rho': 0.0
                }
            }

    async def analyze_options_chain(self, symbol: str, expiration: datetime) -> Dict:
        """Analyze entire options chain for given expiration"""
        try:
            market_data = await self.get_market_data(symbol)
            current_price = market_data['current_price']

            # Generate strikes around current price
            strikes = self._generate_strike_prices(current_price)

            calls_analysis = []
            puts_analysis = []

            # Analyze each strike
            for strike in strikes:
                # Price call
                call_result = await self.price_option(symbol, 'call', strike, expiration)
                if 'price' in call_result:
                    calls_analysis.append({
                        'strike': strike,
                        'price': call_result['price'],
                        'greeks': call_result.get('greeks'),
                        'moneyness': current_price / strike
                    })

                # Price put
                put_result = await self.price_option(symbol, 'put', strike, expiration)
                if 'price' in put_result:
                    puts_analysis.append({
                        'strike': strike,
                        'price': put_result['price'],
                        'greeks': put_result.get('greeks'),
                        'moneyness': current_price / strike
                    })

            return {
                'symbol': symbol,
                'expiration': expiration,
                'current_price': current_price,
                'calls': calls_analysis,
                'puts': puts_analysis,
                'market_data': market_data
            }

        except Exception as e:
            logger.error(f"Error analyzing options chain for {symbol}: {e}")
            return {'error': str(e)}

    def _generate_strike_prices(self, current_price: float) -> List[float]:
        """Generate reasonable strike prices around current price"""
        strikes = []

        # Determine strike spacing based on price level
        if current_price < 25:
            spacing = 1
        elif current_price < 100:
            spacing = 2.5
        elif current_price < 200:
            spacing = 5
        else:
            spacing = 10

        # Generate strikes from 70% to 130% of current price
        start_strike = int((current_price * 0.7) / spacing) * spacing
        end_strike = int((current_price * 1.3) / spacing) * spacing

        strike = start_strike
        while strike <= end_strike:
            strikes.append(strike)
            strike += spacing

        return strikes

    async def optimize_options_portfolio(self, positions: List[OptionsPosition],
                                       target_metrics: Dict) -> Dict:
        """
        Optimize options portfolio using Modern Portfolio Theory principles
        """
        try:
            # Calculate current portfolio metrics
            current_metrics = await self.calculate_portfolio_metrics(positions)

            # Generate scenario analysis
            scenarios = await self._generate_market_scenarios(positions)

            # Optimize portfolio weights
            optimization_result = self._optimize_portfolio_allocation(
                positions, scenarios, target_metrics
            )

            return {
                'current_metrics': current_metrics,
                'scenarios': scenarios,
                'optimization': optimization_result,
                'recommendations': self._generate_portfolio_recommendations(
                    current_metrics, optimization_result
                )
            }

        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            return {'error': str(e)}

    async def calculate_portfolio_metrics(self, positions: List[OptionsPosition]) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        try:
            total_value = 0.0
            total_pnl = 0.0
            delta_exposure = 0.0
            gamma_exposure = 0.0
            theta_decay = 0.0
            vega_exposure = 0.0

            # Update positions with current pricing
            for position in positions:
                # Get current option price
                pricing_result = await self.price_option(
                    position.symbol, position.option_type,
                    position.strike, position.expiration
                )

                if 'price' in pricing_result:
                    position.current_price = pricing_result['price']
                    position.greeks = pricing_result.get('greeks')

                    # Calculate position value and P&L
                    position_value = position.current_price * position.quantity * 100  # Options multiplier
                    position.pnl = position_value - (position.entry_price * position.quantity * 100)

                    total_value += position_value
                    total_pnl += position.pnl

                    # Aggregate Greeks
                    if position.greeks:
                        position_size = position.quantity * 100  # Options multiplier
                        delta_exposure += position.greeks.delta * position_size
                        gamma_exposure += position.greeks.gamma * position_size
                        theta_decay += position.greeks.theta * position_size
                        vega_exposure += position.greeks.vega * position_size

            # Calculate portfolio-level risk metrics
            portfolio_returns = self._estimate_portfolio_returns(positions)
            risk_metrics = self.monte_carlo.calculate_var_cvar(portfolio_returns, self.confidence_level)

            # Calculate expected return and Sharpe ratio
            expected_return = np.mean(portfolio_returns) if len(portfolio_returns) > 0 else 0.0
            return_volatility = np.std(portfolio_returns) if len(portfolio_returns) > 0 else 0.0
            sharpe_ratio = (expected_return - self.risk_free_rate) / return_volatility if return_volatility > 0 else 0.0

            return PortfolioMetrics(
                total_value=total_value,
                total_pnl=total_pnl,
                delta_exposure=delta_exposure,
                gamma_exposure=gamma_exposure,
                theta_decay=theta_decay,
                vega_exposure=vega_exposure,
                var_95=risk_metrics['var'],
                expected_return=expected_return,
                sharpe_ratio=sharpe_ratio
            )

        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return PortfolioMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

    def _estimate_portfolio_returns(self, positions: List[OptionsPosition]) -> np.ndarray:
        """Estimate portfolio returns using Monte Carlo simulation"""
        # Simplified return estimation - would be enhanced in production
        returns = []
        for _ in range(1000):  # 1000 scenarios
            scenario_return = 0.0
            for position in positions:
                # Simulate random price change
                price_change = np.random.normal(0, 0.02)  # 2% daily volatility
                option_return = price_change * (position.greeks.delta if position.greeks else 0.5)
                scenario_return += option_return * position.quantity
            returns.append(scenario_return)

        return np.array(returns)

    async def _generate_market_scenarios(self, positions: List[OptionsPosition]) -> Dict:
        """Generate market scenarios for risk analysis"""
        scenarios = {
            'bull_market': {'underlying_change': 0.20, 'vol_change': -0.20},
            'bear_market': {'underlying_change': -0.20, 'vol_change': 0.30},
            'high_volatility': {'underlying_change': 0.05, 'vol_change': 0.50},
            'low_volatility': {'underlying_change': 0.02, 'vol_change': -0.40},
            'crash': {'underlying_change': -0.35, 'vol_change': 0.80}
        }

        scenario_results = {}

        for scenario_name, scenario_params in scenarios.items():
            portfolio_pnl = 0.0

            for position in positions:
                # Simulate position P&L under scenario
                if position.greeks:
                    underlying_pnl = position.greeks.delta * scenario_params['underlying_change'] * position.quantity * 100
                    vol_pnl = position.greeks.vega * scenario_params['vol_change'] * position.quantity * 100
                    total_pnl = underlying_pnl + vol_pnl
                    portfolio_pnl += total_pnl

            scenario_results[scenario_name] = {
                'parameters': scenario_params,
                'portfolio_pnl': portfolio_pnl,
                'pnl_percentage': portfolio_pnl / max(sum(p.current_price * p.quantity * 100 for p in positions if p.current_price), 1)
            }

        return scenario_results

    def _optimize_portfolio_allocation(self, positions: List[OptionsPosition],
                                     scenarios: Dict, target_metrics: Dict) -> Dict:
        """Optimize portfolio allocation using scenario analysis"""
        # Simplified optimization - would use advanced techniques in production
        recommendations = []

        # Analyze current risk exposure
        current_delta = sum(p.greeks.delta * p.quantity if p.greeks else 0 for p in positions)
        current_gamma = sum(p.greeks.gamma * p.quantity if p.greeks else 0 for p in positions)
        current_theta = sum(p.greeks.theta * p.quantity if p.greeks else 0 for p in positions)

        # Risk management recommendations
        if abs(current_delta) > target_metrics.get('max_delta', 1000):
            recommendations.append({
                'action': 'hedge_delta',
                'description': f"Current delta exposure ({current_delta:.0f}) exceeds target",
                'suggested_hedge': -current_delta * 0.5
            })

        if current_theta < target_metrics.get('min_theta', -100):
            recommendations.append({
                'action': 'reduce_theta',
                'description': f"High theta decay ({current_theta:.2f}) - consider closing short-term positions",
                'urgency': 'high'
            })

        return {
            'current_exposures': {
                'delta': current_delta,
                'gamma': current_gamma,
                'theta': current_theta
            },
            'recommendations': recommendations,
            'optimal_allocation': None  # Would be calculated with proper optimization
        }

    def _generate_portfolio_recommendations(self, current_metrics: PortfolioMetrics,
                                          optimization_result: Dict) -> List[Dict]:
        """Generate actionable portfolio recommendations"""
        recommendations = []

        # Risk-based recommendations
        if current_metrics.var_95 > 0.05:  # 5% VaR threshold
            recommendations.append({
                'type': 'risk_management',
                'priority': 'high',
                'description': f"Portfolio VaR ({current_metrics.var_95:.1%}) exceeds 5% threshold",
                'action': 'Consider reducing position sizes or adding hedges'
            })

        # Return-based recommendations
        if current_metrics.sharpe_ratio < 1.0:
            recommendations.append({
                'type': 'performance',
                'priority': 'medium',
                'description': f"Sharpe ratio ({current_metrics.sharpe_ratio:.2f}) below 1.0",
                'action': 'Focus on higher probability trades or reduce risk'
            })

        # Greek-based recommendations
        if abs(current_metrics.delta_exposure) > 5000:
            recommendations.append({
                'type': 'greek_management',
                'priority': 'medium',
                'description': f"High delta exposure ({current_metrics.delta_exposure:.0f})",
                'action': 'Consider delta hedging with underlying stock'
            })

        return recommendations

# Global instance
enhanced_options_pricing_engine = EnhancedOptionsPricingEngine()