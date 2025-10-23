#!/usr/bin/env python3
"""
Greeks Optimization Module
Filters and selects options based on optimal Greeks for profitability
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class GreeksOptimizer:
    """Optimize option selection based on Greeks"""

    def __init__(self):
        # Optimal ranges for Greeks
        self.optimal_ranges = {
            'delta': {
                'min': 0.40,
                'max': 0.60,
                'ideal': 0.50
            },
            'theta': {
                'max': -0.05,  # Minimize decay
                'acceptable': -0.10
            },
            'vega': {
                'min': 0.10,  # Benefit from IV increase
                'ideal': 0.15
            },
            'gamma': {
                'min': 0.02,  # Some convexity
                'max': 0.10
            }
        }

        # Optimal DTE ranges
        self.dte_ranges = {
            'min': 21,  # Minimum days to expiration
            'max': 45,  # Maximum days to expiration
            'ideal': 30  # Sweet spot
        }

    def calculate_black_scholes_greeks(self, S: float, K: float, T: float, r: float,
                                       sigma: float, option_type: str = 'call') -> Dict:
        """
        Calculate Greeks using Black-Scholes model

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Implied volatility
            option_type: 'call' or 'put'

        Returns:
            Dictionary with Greeks
        """
        try:
            from scipy.stats import norm

            # Prevent division by zero
            if T <= 0:
                return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'valid': False}

            # Calculate d1 and d2
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            # Calculate Greeks
            if option_type.lower() == 'call':
                delta = norm.cdf(d1)
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                        r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            else:  # put
                delta = -norm.cdf(-d1)
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                        r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in IV

            return {
                'delta': float(delta),
                'gamma': float(gamma),
                'theta': float(theta),
                'vega': float(vega),
                'valid': True
            }

        except Exception as e:
            logger.error(f"Greeks calculation error: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'valid': False}

    def estimate_greeks_from_price(self, current_price: float, strike: float,
                                   dte: int, option_type: str = 'call',
                                   iv: float = 0.30) -> Dict:
        """
        Estimate Greeks when we don't have live data
        Uses Black-Scholes approximation
        """
        T = dte / 365.0  # Convert to years
        r = 0.05  # Risk-free rate assumption

        greeks = self.calculate_black_scholes_greeks(
            S=current_price,
            K=strike,
            T=T,
            r=r,
            sigma=iv,
            option_type=option_type
        )

        return greeks

    def score_greeks(self, greeks: Dict, strategy: str = 'long_call') -> Dict:
        """
        Score Greeks based on how close they are to optimal ranges

        Returns:
            {
                'score': float (0-100),
                'delta_score': float,
                'theta_score': float,
                'vega_score': float,
                'gamma_score': float,
                'issues': List[str],
                'approved': bool
            }
        """
        if not greeks.get('valid', False):
            return {
                'score': 0,
                'delta_score': 0,
                'theta_score': 0,
                'vega_score': 0,
                'gamma_score': 0,
                'issues': ['Invalid Greeks'],
                'approved': False
            }

        issues = []

        # Delta score (0-30 points)
        delta = abs(greeks['delta'])  # Use absolute value for puts
        if self.optimal_ranges['delta']['min'] <= delta <= self.optimal_ranges['delta']['max']:
            delta_score = 30
        elif delta < self.optimal_ranges['delta']['min']:
            delta_score = (delta / self.optimal_ranges['delta']['min']) * 20
            issues.append(f"Delta too low ({delta:.2f})")
        else:
            delta_score = max(0, 30 - (delta - self.optimal_ranges['delta']['max']) * 50)
            issues.append(f"Delta too high ({delta:.2f})")

        # Theta score (0-25 points) - prefer less decay
        theta = greeks['theta']
        if theta >= self.optimal_ranges['theta']['max']:
            theta_score = 25
        elif theta >= self.optimal_ranges['theta']['acceptable']:
            theta_score = 15
        else:
            theta_score = max(0, 15 - abs(theta) * 50)
            issues.append(f"High theta decay ({theta:.3f})")

        # Vega score (0-25 points)
        vega = greeks['vega']
        if vega >= self.optimal_ranges['vega']['ideal']:
            vega_score = 25
        elif vega >= self.optimal_ranges['vega']['min']:
            vega_score = 15
        else:
            vega_score = (vega / self.optimal_ranges['vega']['min']) * 10
            issues.append(f"Low vega ({vega:.2f})")

        # Gamma score (0-20 points)
        gamma = greeks['gamma']
        if self.optimal_ranges['gamma']['min'] <= gamma <= self.optimal_ranges['gamma']['max']:
            gamma_score = 20
        else:
            gamma_score = 10

        # Total score
        total_score = delta_score + theta_score + vega_score + gamma_score

        # Approval threshold
        approved = total_score >= 60 and delta_score >= 15

        return {
            'score': float(total_score),
            'delta_score': float(delta_score),
            'theta_score': float(theta_score),
            'vega_score': float(vega_score),
            'gamma_score': float(gamma_score),
            'issues': issues,
            'approved': approved
        }

    def find_optimal_strike(self, symbol: str, current_price: float,
                           strategy: str = 'long_call',
                           dte_target: int = 30) -> Optional[Dict]:
        """
        Find the optimal strike price based on Greeks

        Returns:
            {
                'strike': float,
                'greeks': Dict,
                'score': float,
                'reasoning': str
            }
        """
        # Estimate IV (would normally fetch from market)
        iv = 0.30  # 30% IV assumption

        option_type = 'call' if 'call' in strategy.lower() else 'put'

        # Test multiple strikes
        if option_type == 'call':
            strike_range = np.arange(current_price * 0.95, current_price * 1.10, current_price * 0.01)
        else:
            strike_range = np.arange(current_price * 0.90, current_price * 1.05, current_price * 0.01)

        best_strike = None
        best_score = 0
        best_greeks = None

        for strike in strike_range:
            greeks = self.estimate_greeks_from_price(
                current_price=current_price,
                strike=float(strike),
                dte=dte_target,
                option_type=option_type,
                iv=iv
            )

            score_result = self.score_greeks(greeks, strategy)

            if score_result['score'] > best_score:
                best_score = score_result['score']
                best_strike = strike
                best_greeks = greeks

        if best_strike is None:
            return None

        # Calculate moneyness
        if option_type == 'call':
            moneyness = (best_strike - current_price) / current_price
            position = 'OTM' if moneyness > 0 else 'ITM' if moneyness < -0.02 else 'ATM'
        else:
            moneyness = (current_price - best_strike) / current_price
            position = 'OTM' if moneyness > 0 else 'ITM' if moneyness < -0.02 else 'ATM'

        return {
            'strike': float(best_strike),
            'greeks': best_greeks,
            'score': float(best_score),
            'position': position,
            'moneyness_pct': float(moneyness * 100),
            'reasoning': f"{position} strike with optimal Delta {abs(best_greeks['delta']):.2f}"
        }

    def check_dte_optimal(self, expiration_date: datetime) -> Dict:
        """
        Check if DTE (Days to Expiration) is in optimal range

        Returns:
            {
                'dte': int,
                'optimal': bool,
                'score': float (0-100),
                'reasoning': str
            }
        """
        dte = (expiration_date - datetime.now()).days

        if dte < 0:
            return {
                'dte': dte,
                'optimal': False,
                'score': 0,
                'reasoning': 'Option expired'
            }

        # Score DTE
        if self.dte_ranges['min'] <= dte <= self.dte_ranges['max']:
            # Within optimal range
            if dte == self.dte_ranges['ideal']:
                score = 100
            else:
                distance = abs(dte - self.dte_ranges['ideal'])
                score = max(70, 100 - distance * 2)

            return {
                'dte': dte,
                'optimal': True,
                'score': float(score),
                'reasoning': f'{dte} days - optimal range ({self.dte_ranges["min"]}-{self.dte_ranges["max"]})'
            }

        elif dte < self.dte_ranges['min']:
            score = (dte / self.dte_ranges['min']) * 50
            return {
                'dte': dte,
                'optimal': False,
                'score': float(score),
                'reasoning': f'{dte} days - too soon (high theta decay risk)'
            }

        else:  # dte > max
            excess = dte - self.dte_ranges['max']
            score = max(30, 60 - excess * 2)
            return {
                'dte': dte,
                'optimal': False,
                'score': float(score),
                'reasoning': f'{dte} days - too far out (low gamma, high cost)'
            }

    def approve_option(self, current_price: float, strike: float,
                      expiration_date: datetime, strategy: str = 'long_call',
                      iv: float = 0.30) -> Dict:
        """
        Comprehensive approval check for an option trade

        Returns:
            {
                'approved': bool,
                'greeks_score': float,
                'dte_score': float,
                'total_score': float,
                'greeks': Dict,
                'issues': List[str],
                'recommendation': str
            }
        """
        # Check DTE
        dte_result = self.check_dte_optimal(expiration_date)

        # Calculate Greeks
        option_type = 'call' if 'call' in strategy.lower() else 'put'
        greeks = self.estimate_greeks_from_price(
            current_price=current_price,
            strike=strike,
            dte=dte_result['dte'],
            option_type=option_type,
            iv=iv
        )

        # Score Greeks
        greeks_score_result = self.score_greeks(greeks, strategy)

        # Combined score (60% Greeks, 40% DTE)
        total_score = (greeks_score_result['score'] * 0.6) + (dte_result['score'] * 0.4)

        # Approval decision
        approved = (
            greeks_score_result['approved'] and
            dte_result['optimal'] and
            total_score >= 65
        )

        # Collect all issues
        all_issues = greeks_score_result['issues'].copy()
        if not dte_result['optimal']:
            all_issues.append(dte_result['reasoning'])

        # Generate recommendation
        if approved:
            recommendation = f"APPROVED - Strong Greeks (Delta: {abs(greeks['delta']):.2f}, Theta: {greeks['theta']:.3f}), Optimal DTE: {dte_result['dte']} days"
        else:
            recommendation = f"REJECTED - Issues: {', '.join(all_issues[:2])}"

        return {
            'approved': approved,
            'greeks_score': float(greeks_score_result['score']),
            'dte_score': float(dte_result['score']),
            'total_score': float(total_score),
            'greeks': greeks,
            'dte': dte_result['dte'],
            'issues': all_issues,
            'recommendation': recommendation
        }


# Global instance
_greeks_optimizer = None

def get_greeks_optimizer() -> GreeksOptimizer:
    """Get singleton Greeks optimizer"""
    global _greeks_optimizer
    if _greeks_optimizer is None:
        _greeks_optimizer = GreeksOptimizer()
    return _greeks_optimizer


if __name__ == "__main__":
    # Test
    optimizer = GreeksOptimizer()

    symbol = 'AAPL'
    current_price = 175.0

    print("="*70)
    print(f"GREEKS OPTIMIZATION TEST: {symbol}")
    print("="*70)

    # Find optimal strike
    print(f"\nCurrent Price: ${current_price}")
    print("\nFinding optimal CALL strike...")

    optimal = optimizer.find_optimal_strike(
        symbol=symbol,
        current_price=current_price,
        strategy='long_call',
        dte_target=30
    )

    if optimal:
        print(f"\nOptimal Strike: ${optimal['strike']:.2f}")
        print(f"Position: {optimal['position']} ({optimal['moneyness_pct']:+.1f}%)")
        print(f"Score: {optimal['score']:.1f}/100")
        print(f"Reasoning: {optimal['reasoning']}")
        print(f"\nGreeks:")
        print(f"  Delta: {optimal['greeks']['delta']:.3f}")
        print(f"  Gamma: {optimal['greeks']['gamma']:.3f}")
        print(f"  Theta: {optimal['greeks']['theta']:.3f}")
        print(f"  Vega: {optimal['greeks']['vega']:.3f}")

    # Test approval
    print(f"\n{'='*70}")
    print("APPROVAL TEST")
    print("="*70)

    test_strike = 180.0
    test_expiry = datetime.now() + timedelta(days=30)

    approval = optimizer.approve_option(
        current_price=current_price,
        strike=test_strike,
        expiration_date=test_expiry,
        strategy='long_call'
    )

    print(f"\nTest: ${current_price} stock, ${test_strike} strike, 30 DTE")
    print(f"Approved: {approval['approved']}")
    print(f"Total Score: {approval['total_score']:.1f}/100")
    print(f"  Greeks Score: {approval['greeks_score']:.1f}/100")
    print(f"  DTE Score: {approval['dte_score']:.1f}/100")
    print(f"\nRecommendation: {approval['recommendation']}")
