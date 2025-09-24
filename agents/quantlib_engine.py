#!/usr/bin/env python3
"""
QuantLib Options Pricing Engine
Professional options pricing, Greeks calculation, and volatility surface modeling
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Try to import QuantLib
try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
    print("+ QuantLib available for professional options pricing")
except ImportError:
    QUANTLIB_AVAILABLE = False
    print("- QuantLib not available - using custom Black-Scholes implementation")

# Additional dependencies
try:
    from scipy.stats import norm
    from scipy.optimize import minimize_scalar
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("- SciPy not available - limited functionality")

class QuantLibEngine:
    """Professional options pricing and risk management using QuantLib"""
    
    def __init__(self):
        self.quantlib_available = QUANTLIB_AVAILABLE
        self.scipy_available = SCIPY_AVAILABLE
        
        # Set up QuantLib environment if available
        if QUANTLIB_AVAILABLE:
            # Set evaluation date
            self.today = ql.Date.todaysDate()
            ql.Settings.instance().evaluationDate = self.today
            
            # Common market data setup
            self.day_count = ql.Actual365Fixed()
            self.calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
            
        # Risk-free rate curve (default 2%)
        self.risk_free_rate = 0.02
        
        # Option pricing cache
        self.pricing_cache = {}
        
        print(f"+ QuantLib Engine initialized (QuantLib: {QUANTLIB_AVAILABLE}, SciPy: {SCIPY_AVAILABLE})")
    
    async def price_option(self, option_type: str, underlying_price: float, 
                          strike_price: float, time_to_expiry: float, 
                          volatility: float, risk_free_rate: float = None,
                          dividend_yield: float = 0.0) -> Dict:
        """Price an option using Black-Scholes model"""
        
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        if self.quantlib_available:
            return await self._quantlib_price_option(
                option_type, underlying_price, strike_price, 
                time_to_expiry, volatility, risk_free_rate, dividend_yield
            )
        else:
            return await self._custom_price_option(
                option_type, underlying_price, strike_price,
                time_to_expiry, volatility, risk_free_rate, dividend_yield
            )
    
    async def _quantlib_price_option(self, option_type: str, S: float, K: float, 
                                   T: float, vol: float, r: float, q: float) -> Dict:
        """Price option using QuantLib"""
        try:
            # Set up option parameters
            option_type_ql = ql.Option.Call if option_type.upper() == 'CALL' else ql.Option.Put
            
            # Create dates
            maturity_date = self.today + int(T * 365)
            
            # Create option
            exercise = ql.EuropeanExercise(maturity_date)
            payoff = ql.PlainVanillaPayoff(option_type_ql, K)
            option = ql.VanillaOption(payoff, exercise)
            
            # Market data
            underlying = ql.SimpleQuote(S)
            volatility_ql = ql.BlackConstantVol(self.today, self.calendar, vol, self.day_count)
            risk_free_curve = ql.FlatForward(self.today, r, self.day_count)
            dividend_curve = ql.FlatForward(self.today, q, self.day_count)
            
            # Create handles
            underlying_handle = ql.QuoteHandle(underlying)
            volatility_handle = ql.BlackVolTermStructureHandle(volatility_ql)
            risk_free_handle = ql.YieldTermStructureHandle(risk_free_curve)
            dividend_handle = ql.YieldTermStructureHandle(dividend_curve)
            
            # Black-Scholes process
            bs_process = ql.BlackScholesMertonProcess(
                underlying_handle, dividend_handle, risk_free_handle, volatility_handle
            )
            
            # Pricing engine
            engine = ql.AnalyticEuropeanEngine(bs_process)
            option.setPricingEngine(engine)
            
            # Calculate price and Greeks
            price = option.NPV()
            delta = option.delta()
            gamma = option.gamma()
            theta = option.theta()
            vega = option.vega()
            rho = option.rho()
            
            return {
                'option_type': option_type.upper(),
                'underlying_price': S,
                'strike_price': K,
                'time_to_expiry': T,
                'volatility': vol,
                'risk_free_rate': r,
                'dividend_yield': q,
                'option_price': price,
                'delta': delta,
                'gamma': gamma,
                'theta': theta / 365,  # Daily theta
                'vega': vega / 100,    # 1% vega
                'rho': rho / 100,      # 1% rho
                'intrinsic_value': max(0, S - K if option_type.upper() == 'CALL' else K - S),
                'time_value': price - max(0, S - K if option_type.upper() == 'CALL' else K - S),
                'moneyness': S / K,
                'engine': 'quantlib'
            }
            
        except Exception as e:
            print(f"- QuantLib pricing error: {e}")
            return await self._custom_price_option(option_type, S, K, T, vol, r, q)
    
    async def _custom_price_option(self, option_type: str, S: float, K: float,
                                 T: float, vol: float, r: float, q: float) -> Dict:
        """Custom Black-Scholes implementation"""
        try:
            if not self.scipy_available:
                return {'error': 'SciPy required for custom implementation'}
            
            # Black-Scholes formula
            d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
            d2 = d1 - vol * np.sqrt(T)
            
            if option_type.upper() == 'CALL':
                price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                delta = np.exp(-q * T) * norm.cdf(d1)
            else:  # PUT
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
                delta = -np.exp(-q * T) * norm.cdf(-d1)
            
            # Greeks calculations
            gamma = np.exp(-q * T) * norm.pdf(d1) / (S * vol * np.sqrt(T))
            theta_call = (-S * np.exp(-q * T) * norm.pdf(d1) * vol / (2 * np.sqrt(T)) 
                         - r * K * np.exp(-r * T) * norm.cdf(d2)
                         + q * S * np.exp(-q * T) * norm.cdf(d1))
            theta_put = (-S * np.exp(-q * T) * norm.pdf(d1) * vol / (2 * np.sqrt(T)) 
                        + r * K * np.exp(-r * T) * norm.cdf(-d2)
                        - q * S * np.exp(-q * T) * norm.cdf(-d1))
            theta = theta_call if option_type.upper() == 'CALL' else theta_put
            
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
            
            if option_type.upper() == 'CALL':
                rho = K * T * np.exp(-r * T) * norm.cdf(d2)
            else:
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
            
            return {
                'option_type': option_type.upper(),
                'underlying_price': S,
                'strike_price': K,
                'time_to_expiry': T,
                'volatility': vol,
                'risk_free_rate': r,
                'dividend_yield': q,
                'option_price': price,
                'delta': delta,
                'gamma': gamma,
                'theta': theta / 365,  # Daily theta
                'vega': vega / 100,    # 1% vega
                'rho': rho / 100,      # 1% rho
                'intrinsic_value': max(0, S - K if option_type.upper() == 'CALL' else K - S),
                'time_value': price - max(0, S - K if option_type.upper() == 'CALL' else K - S),
                'moneyness': S / K,
                'engine': 'custom_black_scholes'
            }
            
        except Exception as e:
            print(f"- Custom Black-Scholes error: {e}")
            return {}
    
    async def calculate_implied_volatility(self, option_type: str, market_price: float,
                                         underlying_price: float, strike_price: float,
                                         time_to_expiry: float, risk_free_rate: float = None,
                                         dividend_yield: float = 0.0) -> float:
        """Calculate implied volatility from market price"""
        
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        if self.quantlib_available:
            return await self._quantlib_implied_vol(
                option_type, market_price, underlying_price, strike_price,
                time_to_expiry, risk_free_rate, dividend_yield
            )
        elif self.scipy_available:
            return await self._custom_implied_vol(
                option_type, market_price, underlying_price, strike_price,
                time_to_expiry, risk_free_rate, dividend_yield
            )
        else:
            print("- Implied volatility calculation requires QuantLib or SciPy")
            return 0.0
    
    async def _quantlib_implied_vol(self, option_type: str, market_price: float,
                                  S: float, K: float, T: float, r: float, q: float) -> float:
        """Calculate implied volatility using QuantLib"""
        try:
            # Set up option
            option_type_ql = ql.Option.Call if option_type.upper() == 'CALL' else ql.Option.Put
            maturity_date = self.today + int(T * 365)
            exercise = ql.EuropeanExercise(maturity_date)
            payoff = ql.PlainVanillaPayoff(option_type_ql, K)
            
            # Market data
            underlying = ql.SimpleQuote(S)
            risk_free_curve = ql.FlatForward(self.today, r, self.day_count)
            dividend_curve = ql.FlatForward(self.today, q, self.day_count)
            
            # Calculate implied volatility
            implied_vol = ql.blackImpliedVol(
                market_price, S, K, T, r, q, option_type_ql
            )
            
            return implied_vol
            
        except Exception as e:
            print(f"- QuantLib implied vol error: {e}")
            return await self._custom_implied_vol(option_type, market_price, S, K, T, r, q)
    
    async def _custom_implied_vol(self, option_type: str, market_price: float,
                                S: float, K: float, T: float, r: float, q: float) -> float:
        """Calculate implied volatility using numerical optimization"""
        try:
            async def objective(vol):
                result = await self._custom_price_option(option_type, S, K, T, vol, r, q)
                return abs(result.get('option_price', 0) - market_price)
            
            # Use minimize_scalar to find optimal volatility
            result = minimize_scalar(lambda vol: asyncio.run(objective(vol)), 
                                   bounds=(0.01, 5.0), method='bounded')
            
            return result.x if result.success else 0.2  # Default 20% vol
            
        except Exception as e:
            print(f"- Custom implied vol error: {e}")
            return 0.2
    
    async def build_volatility_surface(self, underlying_price: float, 
                                     strikes: List[float], 
                                     expiries: List[float],
                                     market_data: Dict) -> pd.DataFrame:
        """Build volatility surface from market data"""
        try:
            vol_surface_data = []
            
            for expiry in expiries:
                for strike in strikes:
                    # Get market price for this strike/expiry combination
                    key = f"{strike}_{expiry}"
                    if key in market_data:
                        market_price = market_data[key]['price']
                        option_type = market_data[key].get('type', 'CALL')
                        
                        # Calculate implied volatility
                        implied_vol = await self.calculate_implied_volatility(
                            option_type, market_price, underlying_price, 
                            strike, expiry
                        )
                        
                        vol_surface_data.append({
                            'strike': strike,
                            'expiry': expiry,
                            'implied_volatility': implied_vol,
                            'moneyness': underlying_price / strike,
                            'time_to_expiry': expiry
                        })
            
            if vol_surface_data:
                df = pd.DataFrame(vol_surface_data)
                
                # Create pivot table for surface visualization
                surface = df.pivot(index='strike', columns='expiry', values='implied_volatility')
                
                return surface
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"- Volatility surface error: {e}")
            return pd.DataFrame()
    
    async def calculate_option_strategies(self, strategy_name: str, 
                                        underlying_price: float,
                                        legs: List[Dict]) -> Dict:
        """Calculate complex option strategy payoffs and Greeks"""
        try:
            total_cost = 0
            total_delta = 0
            total_gamma = 0
            total_theta = 0
            total_vega = 0
            total_rho = 0
            
            strategy_legs = []
            
            for leg in legs:
                option_data = await self.price_option(
                    leg['option_type'],
                    underlying_price,
                    leg['strike'],
                    leg['time_to_expiry'],
                    leg['volatility'],
                    leg.get('risk_free_rate'),
                    leg.get('dividend_yield', 0.0)
                )
                
                position_size = leg.get('position_size', 1)  # +1 for long, -1 for short
                
                if option_data:
                    leg_cost = option_data['option_price'] * position_size
                    leg_delta = option_data['delta'] * position_size
                    leg_gamma = option_data['gamma'] * position_size
                    leg_theta = option_data['theta'] * position_size
                    leg_vega = option_data['vega'] * position_size
                    leg_rho = option_data['rho'] * position_size
                    
                    total_cost += leg_cost
                    total_delta += leg_delta
                    total_gamma += leg_gamma
                    total_theta += leg_theta
                    total_vega += leg_vega
                    total_rho += leg_rho
                    
                    strategy_legs.append({
                        'leg_number': len(strategy_legs) + 1,
                        'option_type': leg['option_type'],
                        'strike': leg['strike'],
                        'position_size': position_size,
                        'position_type': 'LONG' if position_size > 0 else 'SHORT',
                        'option_price': option_data['option_price'],
                        'leg_cost': leg_cost,
                        'leg_delta': leg_delta,
                        'leg_gamma': leg_gamma,
                        'leg_theta': leg_theta,
                        'leg_vega': leg_vega,
                        'leg_rho': leg_rho
                    })
            
            # Calculate strategy payoff at expiration
            payoff_range = np.linspace(underlying_price * 0.5, underlying_price * 1.5, 100)
            payoffs = []
            
            for spot_price in payoff_range:
                strategy_payoff = -total_cost  # Start with initial cost
                
                for i, leg in enumerate(legs):
                    option_type = leg['option_type']
                    strike = leg['strike']
                    position_size = leg.get('position_size', 1)
                    
                    if option_type.upper() == 'CALL':
                        intrinsic = max(0, spot_price - strike)
                    else:  # PUT
                        intrinsic = max(0, strike - spot_price)
                    
                    strategy_payoff += intrinsic * position_size
                
                payoffs.append(strategy_payoff)
            
            # Find breakeven points
            breakevens = []
            for i in range(len(payoffs) - 1):
                if payoffs[i] * payoffs[i + 1] <= 0:  # Sign change indicates breakeven
                    breakeven_price = payoff_range[i] + (payoff_range[i + 1] - payoff_range[i]) * \
                                     (-payoffs[i] / (payoffs[i + 1] - payoffs[i]))
                    breakevens.append(breakeven_price)
            
            return {
                'strategy_name': strategy_name,
                'total_cost': total_cost,
                'max_profit': max(payoffs) if payoffs else 0,
                'max_loss': min(payoffs) if payoffs else 0,
                'breakeven_points': breakevens,
                'total_delta': total_delta,
                'total_gamma': total_gamma,
                'total_theta': total_theta,
                'total_vega': total_vega,
                'total_rho': total_rho,
                'strategy_legs': strategy_legs,
                'payoff_diagram': {
                    'spot_prices': payoff_range.tolist(),
                    'payoffs': payoffs
                }
            }
            
        except Exception as e:
            print(f"- Option strategy calculation error: {e}")
            return {}
    
    async def calculate_portfolio_var(self, positions: List[Dict], 
                                    confidence_level: float = 0.05,
                                    time_horizon: int = 1) -> Dict:
        """Calculate Value at Risk for options portfolio"""
        try:
            print(f"Calculating portfolio VaR (confidence: {confidence_level}, horizon: {time_horizon} days)")
            
            portfolio_value = 0
            portfolio_delta = 0
            portfolio_gamma = 0
            portfolio_vega = 0
            
            for position in positions:
                option_data = await self.price_option(
                    position['option_type'],
                    position['underlying_price'],
                    position['strike'],
                    position['time_to_expiry'],
                    position['volatility']
                )
                
                if option_data:
                    quantity = position.get('quantity', 1)
                    portfolio_value += option_data['option_price'] * quantity
                    portfolio_delta += option_data['delta'] * quantity
                    portfolio_gamma += option_data['gamma'] * quantity
                    portfolio_vega += option_data['vega'] * quantity
            
            # Simple VaR calculation using delta-normal method
            # In practice, would use Monte Carlo or historical simulation
            underlying_vol = 0.2  # Assume 20% annual volatility
            daily_vol = underlying_vol / np.sqrt(252)
            
            # Calculate VaR components
            delta_var = abs(portfolio_delta) * daily_vol * norm.ppf(confidence_level) * time_horizon**0.5
            vega_var = abs(portfolio_vega) * 0.01  # 1% vol shock
            
            total_var = (delta_var**2 + vega_var**2)**0.5  # Simplified combination
            
            return {
                'portfolio_value': portfolio_value,
                'portfolio_delta': portfolio_delta,
                'portfolio_gamma': portfolio_gamma,
                'portfolio_vega': portfolio_vega,
                'confidence_level': confidence_level,
                'time_horizon': time_horizon,
                'delta_var': delta_var,
                'vega_var': vega_var,
                'total_var': total_var,
                'var_percentage': total_var / portfolio_value if portfolio_value > 0 else 0
            }
            
        except Exception as e:
            print(f"- Portfolio VaR calculation error: {e}")
            return {}

# Create global instance
quantlib_engine = QuantLibEngine()