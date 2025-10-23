"""
QuantLib Integration for Advanced Options Pricing
Provides accurate options pricing, Greeks, and volatility calculations
"""

import sys
import os
from pathlib import Path

# Add project root to Python path to ensure local config is imported
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import numpy as np
import yfinance as yf

try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False
    print("QuantLib not available, falling back to basic pricing")

from config.logging_config import get_logger

logger = get_logger(__name__)

class QuantLibPricer:
    """Advanced options pricing using QuantLib"""
    
    def __init__(self):
        self.calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        self.day_count = ql.Actual365Fixed()
        
        # Cache for market data
        self.risk_free_rate_cache = {}
        self.dividend_yield_cache = {}
        self.volatility_cache = {}
        self.last_update = {}
        
    def _get_risk_free_rate(self) -> float:
        """Get current risk-free rate (10-year Treasury)"""
        cache_key = "risk_free_rate"
        now = datetime.now()
        
        # Use cached rate if less than 1 hour old
        if (cache_key in self.risk_free_rate_cache and 
            cache_key in self.last_update and
            (now - self.last_update[cache_key]).seconds < 3600):
            return self.risk_free_rate_cache[cache_key]
        
        try:
            # Get 10-year Treasury rate
            treasury = yf.Ticker("^TNX")
            hist = treasury.history(period="1d")
            
            if not hist.empty:
                rate = float(hist['Close'].iloc[-1]) / 100  # Convert percentage to decimal
                self.risk_free_rate_cache[cache_key] = rate
                self.last_update[cache_key] = now
                return rate
        except Exception as e:
            logger.warning(f"Error getting risk-free rate: {e}")
        
        # Default to 5% if unable to fetch
        default_rate = 0.05
        self.risk_free_rate_cache[cache_key] = default_rate
        self.last_update[cache_key] = now
        return default_rate
    
    def _get_dividend_yield(self, symbol: str) -> float:
        """Get dividend yield for a stock"""
        cache_key = f"{symbol}_dividend"
        now = datetime.now()
        
        # Use cached yield if less than 24 hours old
        if (cache_key in self.dividend_yield_cache and 
            cache_key in self.last_update and
            (now - self.last_update[cache_key]).seconds < 86400):
            return self.dividend_yield_cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            dividend_yield = 0.0
            if 'dividendYield' in info and info['dividendYield']:
                dividend_yield = float(info['dividendYield'])
            elif 'trailingAnnualDividendYield' in info and info['trailingAnnualDividendYield']:
                dividend_yield = float(info['trailingAnnualDividendYield'])
            
            self.dividend_yield_cache[cache_key] = dividend_yield
            self.last_update[cache_key] = now
            return dividend_yield
            
        except Exception as e:
            logger.warning(f"Error getting dividend yield for {symbol}: {e}")
            
        # Default to 0% if unable to fetch
        self.dividend_yield_cache[cache_key] = 0.0
        self.last_update[cache_key] = now
        return 0.0
    
    def _calculate_historical_volatility(self, symbol: str, days: int = 30) -> float:
        """Calculate historical volatility"""
        cache_key = f"{symbol}_vol_{days}"
        now = datetime.now()
        
        # Use cached volatility if less than 1 hour old
        if (cache_key in self.volatility_cache and 
            cache_key in self.last_update and
            (now - self.last_update[cache_key]).seconds < 3600):
            return self.volatility_cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days + 10}d")  # Get extra days for calculation
            
            if len(hist) >= days:
                # Calculate daily returns
                returns = hist['Close'].pct_change().dropna()
                
                # Annualize volatility
                volatility = float(returns.std() * np.sqrt(252))  # 252 trading days per year
                
                # Ensure reasonable bounds
                volatility = max(0.05, min(2.0, volatility))  # Between 5% and 200%
                
                self.volatility_cache[cache_key] = volatility
                self.last_update[cache_key] = now
                return volatility
                
        except Exception as e:
            logger.warning(f"Error calculating volatility for {symbol}: {e}")
        
        # Default volatility based on asset class
        if symbol in ['SPY', 'QQQ', 'IWM', 'DIA']:
            default_vol = 0.20  # 20% for ETFs
        elif symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
            default_vol = 0.25  # 25% for large tech
        else:
            default_vol = 0.30  # 30% for other stocks
            
        self.volatility_cache[cache_key] = default_vol
        self.last_update[cache_key] = now
        return default_vol
    
    def price_european_option(self, 
                            option_type: str,
                            underlying_price: float,
                            strike: float,
                            expiry_date: datetime,
                            symbol: str = "UNKNOWN",
                            volatility: Optional[float] = None) -> Dict[str, float]:
        """
        Price European option using Black-Scholes with QuantLib
        
        Returns dict with: price, delta, gamma, theta, vega, rho
        """
        
        if not QUANTLIB_AVAILABLE:
            return self._fallback_pricing(option_type, underlying_price, strike, expiry_date, symbol, volatility)
        
        try:
            # Set up QuantLib calculation date
            today = ql.Date.todaysDate()
            ql.Settings.instance().evaluationDate = today
            
            # Calculate time to expiry
            expiry_ql = ql.Date(expiry_date.day, expiry_date.month, expiry_date.year)
            
            if expiry_ql <= today:
                # Option has expired
                intrinsic = max(0, underlying_price - strike) if option_type.lower() == 'call' else max(0, strike - underlying_price)
                return {
                    'price': intrinsic,
                    'delta': 1.0 if intrinsic > 0 else 0.0,
                    'gamma': 0.0,
                    'theta': 0.0,
                    'vega': 0.0,
                    'rho': 0.0,
                    'intrinsic_value': intrinsic,
                    'time_value': 0.0
                }
            
            # Market data
            risk_free_rate = self._get_risk_free_rate()
            dividend_yield = self._get_dividend_yield(symbol)
            
            if volatility is None:
                volatility = self._calculate_historical_volatility(symbol)
            
            # Create QuantLib objects
            underlying_handle = ql.QuoteHandle(ql.SimpleQuote(underlying_price))
            flat_ts = ql.YieldTermStructureHandle(
                ql.FlatForward(today, risk_free_rate, self.day_count)
            )
            dividend_handle = ql.YieldTermStructureHandle(
                ql.FlatForward(today, dividend_yield, self.day_count)
            )
            flat_vol_ts = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(today, self.calendar, volatility, self.day_count)
            )
            
            # Black-Scholes process
            bs_process = ql.BlackScholesMertonProcess(
                underlying_handle, dividend_handle, flat_ts, flat_vol_ts
            )
            
            # Option setup
            exercise = ql.EuropeanExercise(expiry_ql)
            payoff_type = ql.Option.Call if option_type.lower() == 'call' else ql.Option.Put
            payoff = ql.PlainVanillaPayoff(payoff_type, strike)
            
            option = ql.VanillaOption(payoff, exercise)
            engine = ql.AnalyticEuropeanEngine(bs_process)
            option.setPricingEngine(engine)
            
            # Calculate price and Greeks
            price = option.NPV()
            delta = option.delta()
            gamma = option.gamma()
            theta = option.theta() / 365.0  # Convert to per-day
            vega = option.vega() / 100.0    # Convert to per 1% vol change
            rho = option.rho() / 100.0      # Convert to per 1% rate change
            
            # Calculate intrinsic and time value
            intrinsic = max(0, underlying_price - strike) if option_type.lower() == 'call' else max(0, strike - underlying_price)
            time_value = max(0, price - intrinsic)
            
            return {
                'price': float(price),
                'delta': float(delta),
                'gamma': float(gamma),
                'theta': float(theta),
                'vega': float(vega),
                'rho': float(rho),
                'intrinsic_value': float(intrinsic),
                'time_value': float(time_value),
                'volatility_used': volatility,
                'risk_free_rate': risk_free_rate,
                'dividend_yield': dividend_yield
            }
            
        except Exception as e:
            logger.error(f"QuantLib pricing error: {e}")
            return self._fallback_pricing(option_type, underlying_price, strike, expiry_date, symbol, volatility)
    
    def _fallback_pricing(self, option_type: str, underlying_price: float, strike: float, 
                         expiry_date: datetime, symbol: str, volatility: Optional[float]) -> Dict[str, float]:
        """Fallback Black-Scholes pricing without QuantLib"""
        
        try:
            from scipy.stats import norm
            
            # Time to expiry in years
            time_to_expiry = max(1/365, (expiry_date - datetime.now()).days / 365.0)
            
            # Market parameters
            r = self._get_risk_free_rate()
            q = self._get_dividend_yield(symbol)
            
            if volatility is None:
                volatility = self._calculate_historical_volatility(symbol)
            
            S = underlying_price
            K = strike
            T = time_to_expiry
            sigma = volatility
            
            # Black-Scholes calculations
            d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type.lower() == 'call':
                price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
                delta = np.exp(-q*T)*norm.cdf(d1)
            else:  # put
                price = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
                delta = -np.exp(-q*T)*norm.cdf(-d1)
            
            # Greeks
            gamma = np.exp(-q*T)*norm.pdf(d1)/(S*sigma*np.sqrt(T))
            theta = (-S*norm.pdf(d1)*sigma*np.exp(-q*T)/(2*np.sqrt(T)) 
                    - r*K*np.exp(-r*T)*norm.cdf(d2 if option_type.lower() == 'call' else -d2)
                    + q*S*np.exp(-q*T)*norm.cdf(d1 if option_type.lower() == 'call' else -d1)) / 365
            vega = S*np.exp(-q*T)*norm.pdf(d1)*np.sqrt(T) / 100
            rho = (K*T*np.exp(-r*T)*norm.cdf(d2 if option_type.lower() == 'call' else -d2)) / 100
            
            # Intrinsic and time value
            intrinsic = max(0, S - K) if option_type.lower() == 'call' else max(0, K - S)
            time_value = max(0, price - intrinsic)
            
            return {
                'price': float(max(0, price)),
                'delta': float(delta),
                'gamma': float(gamma),
                'theta': float(theta),
                'vega': float(vega),
                'rho': float(rho),
                'intrinsic_value': float(intrinsic),
                'time_value': float(time_value),
                'volatility_used': volatility,
                'risk_free_rate': r,
                'dividend_yield': q
            }
            
        except Exception as e:
            logger.error(f"Fallback pricing error: {e}")
            # Ultra-simple fallback
            intrinsic = max(0, underlying_price - strike) if option_type.lower() == 'call' else max(0, strike - underlying_price)
            return {
                'price': intrinsic + 0.50,  # Add small time value
                'delta': 0.5,
                'gamma': 0.1,
                'theta': -0.01,
                'vega': 0.1,
                'rho': 0.05,
                'intrinsic_value': intrinsic,
                'time_value': 0.50,
                'volatility_used': 0.25,
                'risk_free_rate': 0.05,
                'dividend_yield': 0.0
            }
    
    def calculate_implied_volatility(self, 
                                   option_price: float,
                                   option_type: str,
                                   underlying_price: float,
                                   strike: float,
                                   expiry_date: datetime,
                                   symbol: str = "UNKNOWN") -> float:
        """Calculate implied volatility from market price"""
        
        if not QUANTLIB_AVAILABLE:
            return self._calculate_historical_volatility(symbol)
        
        try:
            # Set up QuantLib
            today = ql.Date.todaysDate()
            ql.Settings.instance().evaluationDate = today
            
            expiry_ql = ql.Date(expiry_date.day, expiry_date.month, expiry_date.year)
            
            if expiry_ql <= today:
                return 0.0  # Expired option
            
            # Market data
            risk_free_rate = self._get_risk_free_rate()
            dividend_yield = self._get_dividend_yield(symbol)
            
            underlying_handle = ql.QuoteHandle(ql.SimpleQuote(underlying_price))
            flat_ts = ql.YieldTermStructureHandle(
                ql.FlatForward(today, risk_free_rate, self.day_count)
            )
            dividend_handle = ql.YieldTermStructureHandle(
                ql.FlatForward(today, dividend_yield, self.day_count)
            )
            
            # Option setup
            exercise = ql.EuropeanExercise(expiry_ql)
            payoff_type = ql.Option.Call if option_type.lower() == 'call' else ql.Option.Put
            payoff = ql.PlainVanillaPayoff(payoff_type, strike)
            
            # Use QuantLib's implied volatility calculation
            implied_vol = ql.blackImpliedVol(
                option_price,
                underlying_price,
                strike,
                (expiry_ql - today) / 365.0,
                risk_free_rate,
                dividend_yield,
                payoff_type
            )
            
            return float(implied_vol)
            
        except Exception as e:
            logger.warning(f"Implied volatility calculation error: {e}")
            return self._calculate_historical_volatility(symbol)
    
    def get_pricing_summary(self, option_type: str, underlying_price: float, strike: float,
                          expiry_date: datetime, symbol: str, market_price: Optional[float] = None) -> Dict:
        """Get comprehensive options pricing analysis"""
        
        # Get theoretical pricing
        theoretical = self.price_european_option(option_type, underlying_price, strike, expiry_date, symbol)
        
        result = {
            'symbol': symbol,
            'option_type': option_type,
            'underlying_price': underlying_price,
            'strike': strike,
            'expiry_date': expiry_date.strftime('%Y-%m-%d'),
            'days_to_expiry': (expiry_date - datetime.now()).days,
            'theoretical_price': theoretical['price'],
            'greeks': {
                'delta': theoretical['delta'],
                'gamma': theoretical['gamma'],
                'theta': theoretical['theta'],
                'vega': theoretical['vega'],
                'rho': theoretical['rho']
            },
            'intrinsic_value': theoretical['intrinsic_value'],
            'time_value': theoretical['time_value'],
            'volatility_used': theoretical['volatility_used'],
            'risk_free_rate': theoretical['risk_free_rate'],
            'dividend_yield': theoretical['dividend_yield']
        }
        
        # Add market comparison if price provided
        if market_price:
            result['market_price'] = market_price
            result['price_difference'] = market_price - theoretical['price']
            result['price_difference_pct'] = (result['price_difference'] / theoretical['price']) * 100
            result['implied_volatility'] = self.calculate_implied_volatility(
                market_price, option_type, underlying_price, strike, expiry_date, symbol
            )
            
            # Valuation assessment
            if abs(result['price_difference_pct']) < 5:
                result['valuation'] = 'Fair Value'
            elif result['price_difference'] > 0:
                result['valuation'] = 'Overvalued'
            else:
                result['valuation'] = 'Undervalued'
        
        return result

# Global instance
quantlib_pricer = QuantLibPricer() if QUANTLIB_AVAILABLE else None