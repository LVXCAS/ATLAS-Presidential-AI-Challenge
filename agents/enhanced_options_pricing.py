"""
Enhanced Options Pricing Agent
Professional options pricing using py_vollib and mibian
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math

try:
    import py_vollib.black_scholes as bs
    import py_vollib.black_scholes.greeks as greeks
    PY_VOLLIB_AVAILABLE = True
except ImportError:
    PY_VOLLIB_AVAILABLE = False

try:
    import mibian
    MIBIAN_AVAILABLE = True
except ImportError:
    MIBIAN_AVAILABLE = False

from config.logging_config import get_logger

logger = get_logger(__name__)

class EnhancedOptionsPricing:
    """Professional options pricing with advanced Greeks and IV analysis"""
    
    def __init__(self):
        self.risk_free_rate = 0.05  # 5% default risk-free rate
        self.pricing_cache = {}
        self.cache_expiry = {}
        self.cache_duration = 60  # 1 minute cache
        
    def update_risk_free_rate(self, rate: float):
        """Update the risk-free rate"""
        self.risk_free_rate = rate
        logger.info(f"Updated risk-free rate to {rate:.2%}")
    
    async def get_comprehensive_option_analysis(self, 
                                              underlying_price: float,
                                              strike_price: float,
                                              time_to_expiry_days: int,
                                              volatility: float,
                                              option_type: str = 'call',
                                              dividend_yield: float = 0.0) -> Dict:
        """Get comprehensive options analysis including pricing and Greeks"""
        
        cache_key = f"{underlying_price}_{strike_price}_{time_to_expiry_days}_{volatility}_{option_type}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.pricing_cache[cache_key]
        
        try:
            # Convert inputs
            time_to_expiry = max(1/365, time_to_expiry_days / 365.0)  # Convert to years
            volatility_decimal = volatility / 100.0 if volatility > 1 else volatility
            
            analysis = {
                'inputs': {
                    'underlying_price': underlying_price,
                    'strike_price': strike_price,
                    'time_to_expiry_days': time_to_expiry_days,
                    'time_to_expiry_years': time_to_expiry,
                    'volatility': volatility,
                    'risk_free_rate': self.risk_free_rate,
                    'dividend_yield': dividend_yield,
                    'option_type': option_type.lower()
                },
                'pricing': {},
                'greeks': {},
                'risk_analysis': {},
                'volatility_analysis': {},
                'profitability_analysis': {}
            }
            
            # Get pricing using multiple methods
            analysis['pricing'] = self._calculate_option_pricing(
                underlying_price, strike_price, time_to_expiry, 
                volatility_decimal, option_type, dividend_yield
            )
            
            # Calculate Greeks
            analysis['greeks'] = self._calculate_greeks(
                underlying_price, strike_price, time_to_expiry,
                volatility_decimal, option_type, dividend_yield
            )
            
            # Risk analysis
            analysis['risk_analysis'] = self._analyze_option_risks(
                underlying_price, strike_price, time_to_expiry,
                volatility_decimal, option_type, analysis['pricing']
            )
            
            # Volatility analysis
            analysis['volatility_analysis'] = self._analyze_volatility_impact(
                underlying_price, strike_price, time_to_expiry,
                volatility_decimal, option_type
            )
            
            # Profitability analysis
            analysis['profitability_analysis'] = self._analyze_profitability_scenarios(
                underlying_price, strike_price, time_to_expiry,
                volatility_decimal, option_type, analysis['pricing']['theoretical_price']
            )
            
            # Cache result
            self.pricing_cache[cache_key] = analysis
            self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Options analysis error: {e}")
            return self._get_default_analysis(underlying_price, strike_price, option_type)
    
    def _calculate_option_pricing(self, S: float, K: float, T: float, 
                                sigma: float, option_type: str, q: float = 0.0) -> Dict:
        """Calculate option price using multiple methods"""
        pricing = {
            'theoretical_price': 0.0,
            'intrinsic_value': 0.0,
            'time_value': 0.0,
            'moneyness': 'ATM',
            'pricing_method': 'fallback'
        }
        
        try:
            # Calculate intrinsic value
            if option_type.lower() == 'call':
                pricing['intrinsic_value'] = max(0, S - K)
            else:  # put
                pricing['intrinsic_value'] = max(0, K - S)
            
            # Determine moneyness
            moneyness_ratio = S / K
            if moneyness_ratio > 1.05:
                pricing['moneyness'] = 'ITM' if option_type.lower() == 'call' else 'OTM'
            elif moneyness_ratio < 0.95:
                pricing['moneyness'] = 'OTM' if option_type.lower() == 'call' else 'ITM'
            else:
                pricing['moneyness'] = 'ATM'
            
            # Try py_vollib first (most accurate)
            if PY_VOLLIB_AVAILABLE:
                try:
                    if option_type.lower() == 'call':
                        theoretical_price = bs.black_scholes('c', S, K, T, self.risk_free_rate, sigma)
                    else:
                        theoretical_price = bs.black_scholes('p', S, K, T, self.risk_free_rate, sigma)
                    
                    pricing['theoretical_price'] = theoretical_price
                    pricing['pricing_method'] = 'py_vollib_black_scholes'
                    
                except Exception as e:
                    logger.warning(f"py_vollib pricing error: {e}")
                    raise
            
            # Try mibian as backup
            elif MIBIAN_AVAILABLE:
                try:
                    bs_calc = mibian.BS([S, K, self.risk_free_rate, T * 365], volatility=sigma * 100)
                    if option_type.lower() == 'call':
                        theoretical_price = bs_calc.callPrice
                    else:
                        theoretical_price = bs_calc.putPrice
                    
                    pricing['theoretical_price'] = theoretical_price
                    pricing['pricing_method'] = 'mibian_black_scholes'
                    
                except Exception as e:
                    logger.warning(f"mibian pricing error: {e}")
                    raise
            
            else:
                # Fallback to manual Black-Scholes
                theoretical_price = self._manual_black_scholes(S, K, T, sigma, self.risk_free_rate, option_type)
                pricing['theoretical_price'] = theoretical_price
                pricing['pricing_method'] = 'manual_black_scholes'
            
            # Calculate time value
            pricing['time_value'] = pricing['theoretical_price'] - pricing['intrinsic_value']
            
        except Exception as e:
            logger.error(f"Pricing calculation error: {e}")
            # Fallback pricing
            pricing['theoretical_price'] = self._manual_black_scholes(S, K, T, sigma, self.risk_free_rate, option_type)
            pricing['time_value'] = pricing['theoretical_price'] - pricing['intrinsic_value']
            pricing['pricing_method'] = 'fallback'
        
        return pricing
    
    def _calculate_greeks(self, S: float, K: float, T: float, 
                         sigma: float, option_type: str, q: float = 0.0) -> Dict:
        """Calculate option Greeks"""
        greeks_dict = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0,
            'calculation_method': 'fallback'
        }
        
        try:
            # Try py_vollib Greeks (most accurate)
            if PY_VOLLIB_AVAILABLE:
                try:
                    flag = 'c' if option_type.lower() == 'call' else 'p'
                    
                    greeks_dict['delta'] = greeks.delta(flag, S, K, T, self.risk_free_rate, sigma)
                    greeks_dict['gamma'] = greeks.gamma(flag, S, K, T, self.risk_free_rate, sigma)
                    greeks_dict['theta'] = greeks.theta(flag, S, K, T, self.risk_free_rate, sigma)
                    greeks_dict['vega'] = greeks.vega(flag, S, K, T, self.risk_free_rate, sigma)
                    greeks_dict['rho'] = greeks.rho(flag, S, K, T, self.risk_free_rate, sigma)
                    greeks_dict['calculation_method'] = 'py_vollib'
                    
                except Exception as e:
                    logger.warning(f"py_vollib Greeks error: {e}")
                    raise
            
            # Try mibian Greeks
            elif MIBIAN_AVAILABLE:
                try:
                    bs_calc = mibian.BS([S, K, self.risk_free_rate, T * 365], volatility=sigma * 100)
                    
                    if option_type.lower() == 'call':
                        greeks_dict['delta'] = bs_calc.callDelta
                        greeks_dict['theta'] = bs_calc.callTheta
                    else:
                        greeks_dict['delta'] = bs_calc.putDelta
                        greeks_dict['theta'] = bs_calc.putTheta
                    
                    greeks_dict['gamma'] = bs_calc.gamma
                    greeks_dict['vega'] = bs_calc.vega
                    greeks_dict['calculation_method'] = 'mibian'
                    
                except Exception as e:
                    logger.warning(f"mibian Greeks error: {e}")
                    raise
            
            else:
                # Manual Greeks calculation
                greeks_dict = self._manual_greeks(S, K, T, sigma, self.risk_free_rate, option_type)
                greeks_dict['calculation_method'] = 'manual'
        
        except Exception as e:
            logger.error(f"Greeks calculation error: {e}")
            # Fallback to manual calculation
            greeks_dict = self._manual_greeks(S, K, T, sigma, self.risk_free_rate, option_type)
            greeks_dict['calculation_method'] = 'fallback'
        
        return greeks_dict
    
    def _analyze_option_risks(self, S: float, K: float, T: float, 
                            sigma: float, option_type: str, pricing: Dict) -> Dict:
        """Analyze option risks"""
        return {
            'max_loss': pricing['theoretical_price'] if option_type.lower() in ['call', 'put'] else float('inf'),
            'max_gain': float('inf') if option_type.lower() == 'call' else K - pricing['theoretical_price'],
            'breakeven': K + pricing['theoretical_price'] if option_type.lower() == 'call' else K - pricing['theoretical_price'],
            'probability_itm': self._calculate_probability_itm(S, K, T, sigma, option_type),
            'time_decay_risk': 'HIGH' if T < 30/365 else 'MODERATE' if T < 60/365 else 'LOW',
            'volatility_risk': 'HIGH' if sigma > 0.4 else 'MODERATE' if sigma > 0.25 else 'LOW'
        }
    
    def _analyze_volatility_impact(self, S: float, K: float, T: float, 
                                 sigma: float, option_type: str) -> Dict:
        """Analyze impact of volatility changes"""
        base_price = self._manual_black_scholes(S, K, T, sigma, self.risk_free_rate, option_type)
        
        vol_scenarios = {
            'vol_crush_20pct': self._manual_black_scholes(S, K, T, sigma * 0.8, self.risk_free_rate, option_type),
            'vol_expansion_20pct': self._manual_black_scholes(S, K, T, sigma * 1.2, self.risk_free_rate, option_type),
            'vol_crush_impact': 0.0,
            'vol_expansion_impact': 0.0
        }
        
        vol_scenarios['vol_crush_impact'] = (vol_scenarios['vol_crush_20pct'] - base_price) / base_price * 100
        vol_scenarios['vol_expansion_impact'] = (vol_scenarios['vol_expansion_20pct'] - base_price) / base_price * 100
        
        return vol_scenarios
    
    def _analyze_profitability_scenarios(self, S: float, K: float, T: float,
                                       sigma: float, option_type: str, entry_price: float) -> Dict:
        """Analyze profit/loss scenarios at different price levels"""
        scenarios = {}
        
        # Price movement scenarios
        price_changes = [-0.10, -0.05, -0.02, 0.00, 0.02, 0.05, 0.10]
        
        for change in price_changes:
            new_price = S * (1 + change)
            option_value = self._manual_black_scholes(new_price, K, T, sigma, self.risk_free_rate, option_type)
            pnl = option_value - entry_price
            pnl_pct = (pnl / entry_price) * 100 if entry_price > 0 else 0
            
            scenarios[f'price_change_{change:+.0%}'] = {
                'new_stock_price': new_price,
                'option_value': option_value,
                'pnl_dollar': pnl,
                'pnl_percent': pnl_pct
            }
        
        return scenarios
    
    def _calculate_probability_itm(self, S: float, K: float, T: float, 
                                 sigma: float, option_type: str) -> float:
        """Calculate probability of finishing in-the-money"""
        try:
            # Using Black-Scholes assumptions
            d1 = (math.log(S/K) + (self.risk_free_rate + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            
            if option_type.lower() == 'call':
                # For calls: probability that S > K
                prob_itm = self._normal_cdf(d1 - sigma * math.sqrt(T))
            else:
                # For puts: probability that S < K  
                prob_itm = self._normal_cdf(-(d1 - sigma * math.sqrt(T)))
            
            return prob_itm
            
        except Exception as e:
            logger.error(f"Probability ITM calculation error: {e}")
            return 0.5  # Default to 50%
    
    def _manual_black_scholes(self, S: float, K: float, T: float, 
                            sigma: float, r: float, option_type: str) -> float:
        """Manual Black-Scholes calculation"""
        try:
            if T <= 0:
                if option_type.lower() == 'call':
                    return max(0, S - K)
                else:
                    return max(0, K - S)
            
            d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            if option_type.lower() == 'call':
                price = S * self._normal_cdf(d1) - K * math.exp(-r * T) * self._normal_cdf(d2)
            else:  # put
                price = K * math.exp(-r * T) * self._normal_cdf(-d2) - S * self._normal_cdf(-d1)
            
            return max(0, price)
            
        except Exception as e:
            logger.error(f"Manual Black-Scholes error: {e}")
            return 1.0  # Fallback price
    
    def _manual_greeks(self, S: float, K: float, T: float, 
                      sigma: float, r: float, option_type: str) -> Dict:
        """Manual Greeks calculation"""
        try:
            if T <= 0:
                return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
            
            d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            # Common calculations
            nd1 = self._normal_cdf(d1)
            nd2 = self._normal_cdf(d2)
            npd1 = self._normal_pdf(d1)
            
            if option_type.lower() == 'call':
                delta = nd1
                theta = -(S * npd1 * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * nd2
                rho = K * T * math.exp(-r * T) * nd2
            else:  # put
                delta = nd1 - 1
                theta = -(S * npd1 * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * (1 - nd2)
                rho = -K * T * math.exp(-r * T) * (1 - nd2)
            
            gamma = npd1 / (S * sigma * math.sqrt(T))
            vega = S * npd1 * math.sqrt(T)
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta / 365,  # Convert to daily
                'vega': vega / 100,    # Convert to 1% vol change
                'rho': rho / 100       # Convert to 1% rate change
            }
            
        except Exception as e:
            logger.error(f"Manual Greeks error: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    def _normal_cdf(self, x: float) -> float:
        """Standard normal cumulative distribution function"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def _normal_pdf(self, x: float) -> float:
        """Standard normal probability density function"""
        return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.pricing_cache:
            return False
        if cache_key not in self.cache_expiry:
            return False
        return datetime.now() < self.cache_expiry[cache_key]
    
    def _get_default_analysis(self, S: float, K: float, option_type: str) -> Dict:
        """Return default analysis when calculation fails"""
        intrinsic = max(0, S - K) if option_type.lower() == 'call' else max(0, K - S)
        
        return {
            'inputs': {'underlying_price': S, 'strike_price': K, 'option_type': option_type},
            'pricing': {'theoretical_price': max(intrinsic, 0.5), 'intrinsic_value': intrinsic},
            'greeks': {'delta': 0.5, 'gamma': 0, 'theta': -0.02, 'vega': 0.1},
            'risk_analysis': {'max_loss': max(intrinsic, 0.5), 'probability_itm': 0.5},
            'volatility_analysis': {},
            'profitability_analysis': {}
        }

# Singleton instance
enhanced_options_pricing = EnhancedOptionsPricing()