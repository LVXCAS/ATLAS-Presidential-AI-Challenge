#!/usr/bin/env python3
"""
OPTIONS PRICING INTEGRATION
Black-Scholes options pricing for your 68.3% avg ROI put patterns
Integrates with Level 4 AI Trading Agent for options analysis
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta
import math

class OptionsPricingIntegration:
    """Options pricing analysis for put success patterns"""

    def __init__(self):
        # Your successful put patterns for analysis
        self.successful_puts = {
            'INTC': {'roi': 0.706, 'pattern': 'earnings_catalyst'},
            'RIVN': {'roi': 0.898, 'pattern': 'volatility_explosion'},
            'SNAP': {'roi': 0.447, 'pattern': 'social_media_volatility'},
            'LYFT': {'roi': 0.683, 'pattern': 'momentum_reversal'}
        }

    def black_scholes_put(self, S, K, T, r, sigma):
        """
        Calculate Black-Scholes put option price
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility
        """
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            put_price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
            return max(put_price, 0)

        except Exception as e:
            print(f"Black-Scholes calculation error: {e}")
            return 0

    def calculate_implied_volatility(self, S, K, T, r, market_price, option_type='put'):
        """
        Calculate implied volatility using Newton-Raphson method
        """
        try:
            # Initial volatility guess
            sigma = 0.3

            for i in range(100):  # Max iterations
                if option_type == 'put':
                    price = self.black_scholes_put(S, K, T, r, sigma)
                    # Vega (sensitivity to volatility)
                    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                    vega = S * norm.pdf(d1) * np.sqrt(T)
                else:
                    # Call option (if needed)
                    price = self.black_scholes_call(S, K, T, r, sigma)
                    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                    vega = S * norm.pdf(d1) * np.sqrt(T)

                price_diff = price - market_price

                if abs(price_diff) < 0.01:  # Convergence
                    return sigma

                if vega != 0:
                    sigma = sigma - price_diff / vega

                # Keep sigma positive
                sigma = max(sigma, 0.01)

            return sigma

        except Exception as e:
            print(f"Implied volatility calculation error: {e}")
            return 0.3  # Default volatility

    def analyze_put_opportunity(self, symbol, current_price, strike_price, days_to_expiration,
                               expected_volatility=None):
        """
        Analyze put option opportunity using your successful patterns
        """
        print(f"\nOPTIONS PRICING ANALYSIS: {symbol}")
        print("=" * 50)

        # Market parameters
        risk_free_rate = 0.05  # 5% risk-free rate
        T = days_to_expiration / 365.0  # Time to expiration in years

        # Use pattern-based volatility if not provided
        if expected_volatility is None:
            if symbol in self.successful_puts:
                pattern = self.successful_puts[symbol]['pattern']
                if pattern == 'volatility_explosion':
                    expected_volatility = 0.8  # High vol like RIVN
                elif pattern == 'earnings_catalyst':
                    expected_volatility = 0.6  # Earnings vol like INTC
                else:
                    expected_volatility = 0.5  # Default high vol
            else:
                expected_volatility = 0.5

        # Calculate theoretical put price
        theoretical_price = self.black_scholes_put(
            current_price, strike_price, T, risk_free_rate, expected_volatility
        )

        # Calculate Greeks
        greeks = self.calculate_greeks(
            current_price, strike_price, T, risk_free_rate, expected_volatility
        )

        # Analysis based on your successful patterns
        pattern_analysis = self.analyze_success_pattern(symbol, current_price, strike_price)

        analysis = {
            'symbol': symbol,
            'current_price': current_price,
            'strike_price': strike_price,
            'days_to_expiration': days_to_expiration,
            'expected_volatility': expected_volatility,
            'theoretical_put_price': theoretical_price,
            'greeks': greeks,
            'pattern_analysis': pattern_analysis,
            'moneyness': strike_price / current_price,
            'time_decay_factor': T
        }

        self.print_analysis(analysis)
        return analysis

    def calculate_greeks(self, S, K, T, r, sigma):
        """Calculate option Greeks for risk management"""
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            # Delta (price sensitivity)
            delta = -norm.cdf(-d1)  # Put delta is negative

            # Gamma (delta sensitivity)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

            # Theta (time decay)
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                    - r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

            # Vega (volatility sensitivity)
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100

            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega
            }

        except Exception as e:
            print(f"Greeks calculation error: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

    def analyze_success_pattern(self, symbol, current_price, strike_price):
        """
        Analyze if current setup matches your successful put patterns
        """
        if symbol in self.successful_puts:
            pattern_info = self.successful_puts[symbol]
            expected_roi = pattern_info['roi']
            pattern_type = pattern_info['pattern']

            similarity_score = self.calculate_pattern_similarity(
                symbol, current_price, strike_price, pattern_type
            )

            return {
                'historical_success': True,
                'historical_roi': expected_roi,
                'pattern_type': pattern_type,
                'similarity_score': similarity_score,
                'recommendation': 'HIGH CONVICTION' if similarity_score > 0.7 else 'MODERATE'
            }
        else:
            # Analyze similarity to known successful patterns
            best_match = self.find_best_pattern_match(symbol, current_price, strike_price)
            return {
                'historical_success': False,
                'best_pattern_match': best_match,
                'recommendation': 'RESEARCH NEEDED'
            }

    def calculate_pattern_similarity(self, symbol, current_price, strike_price, pattern_type):
        """Calculate similarity to your successful patterns"""
        # Simplified pattern similarity scoring
        moneyness = strike_price / current_price

        if pattern_type == 'volatility_explosion':
            # High vol patterns work best slightly OTM
            optimal_moneyness = 0.95
        elif pattern_type == 'earnings_catalyst':
            # Earnings plays work well ATM to slightly ITM
            optimal_moneyness = 1.02
        elif pattern_type == 'momentum_reversal':
            # Momentum reversals work OTM
            optimal_moneyness = 0.92
        else:
            optimal_moneyness = 0.98

        # Calculate similarity based on moneyness
        moneyness_diff = abs(moneyness - optimal_moneyness)
        similarity = max(0, 1 - (moneyness_diff * 5))  # Scale similarity

        return similarity

    def find_best_pattern_match(self, symbol, current_price, strike_price):
        """Find which successful pattern this most resembles"""
        best_match = None
        best_score = 0

        for successful_symbol, pattern_info in self.successful_puts.items():
            score = self.calculate_pattern_similarity(
                symbol, current_price, strike_price, pattern_info['pattern']
            )
            if score > best_score:
                best_score = score
                best_match = {
                    'pattern': pattern_info['pattern'],
                    'reference_symbol': successful_symbol,
                    'reference_roi': pattern_info['roi'],
                    'similarity_score': score
                }

        return best_match

    def print_analysis(self, analysis):
        """Print detailed options analysis"""
        print(f"Current Price: ${analysis['current_price']:.2f}")
        print(f"Strike Price: ${analysis['strike_price']:.2f}")
        print(f"Days to Expiration: {analysis['days_to_expiration']}")
        print(f"Expected Volatility: {analysis['expected_volatility']:.1%}")
        print(f"Theoretical Put Price: ${analysis['theoretical_put_price']:.2f}")
        print(f"Moneyness: {analysis['moneyness']:.3f}")

        print(f"\nGREEKS:")
        greeks = analysis['greeks']
        print(f"Delta: {greeks['delta']:.3f}")
        print(f"Gamma: {greeks['gamma']:.3f}")
        print(f"Theta: ${greeks['theta']:.2f}")
        print(f"Vega: ${greeks['vega']:.2f}")

        print(f"\nPATTERN ANALYSIS:")
        pattern = analysis['pattern_analysis']
        if pattern['historical_success']:
            print(f"Historical ROI: {pattern['historical_roi']:.1%}")
            print(f"Pattern Type: {pattern['pattern_type']}")
            print(f"Similarity Score: {pattern['similarity_score']:.1%}")
            print(f"Recommendation: {pattern['recommendation']}")
        else:
            best_match = pattern['best_pattern_match']
            if best_match:
                print(f"Best Pattern Match: {best_match['pattern']}")
                print(f"Reference: {best_match['reference_symbol']} (+{best_match['reference_roi']:.1%})")
                print(f"Similarity: {best_match['similarity_score']:.1%}")

def main():
    """Test options pricing integration"""
    pricer = OptionsPricingIntegration()

    # Test with your successful put symbols
    test_cases = [
        # symbol, current_price, strike_price, days_to_exp
        ('INTC', 25.50, 24.00, 30),    # Earnings catalyst pattern
        ('RIVN', 12.75, 12.00, 21),   # Volatility explosion pattern
        ('SNAP', 18.20, 17.50, 35),   # Social media volatility
        ('META', 575.00, 560.00, 28), # New opportunity analysis
    ]

    print("OPTIONS PRICING INTEGRATION FOR 68.3% AVG ROI PATTERNS")
    print("=" * 80)

    for symbol, price, strike, days in test_cases:
        analysis = pricer.analyze_put_opportunity(symbol, price, strike, days)

    print("\n" + "=" * 80)
    print("OPTIONS PRICING ANALYSIS COMPLETE")
    print("Ready for Level 4 AI Trading Agent integration!")

if __name__ == "__main__":
    main()