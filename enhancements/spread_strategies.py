#!/usr/bin/env python3
"""
Spread Strategies Module
Bull/Bear call/put spreads for better risk/reward and win rates
"""

import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SpreadStrategies:
    """Define and analyze spread strategies"""

    def __init__(self):
        # Spread width preferences (% of stock price)
        self.spread_widths = {
            'TIGHT': 0.02,   # 2% width
            'NORMAL': 0.05,  # 5% width
            'WIDE': 0.10     # 10% width
        }

    def design_bull_call_spread(self, current_price: float, confidence: float,
                                 budget: float = 500) -> Dict:
        """
        Design a bull call spread

        Structure:
        - Buy lower strike call (long)
        - Sell higher strike call (short)
        - Net debit (cost to enter)

        Args:
            current_price: Current stock price
            confidence: Trade confidence (affects spread width)
            budget: Maximum budget for the spread

        Returns:
            {
                'long_strike': float,
                'short_strike': float,
                'spread_width': float,
                'est_cost': float,
                'est_max_profit': float,
                'est_max_loss': float,
                'breakeven': float,
                'risk_reward': float,
                'strategy': str
            }
        """
        # Higher confidence = wider spread = more profit potential
        if confidence >= 0.70:
            width_pct = self.spread_widths['WIDE']
        elif confidence >= 0.60:
            width_pct = self.spread_widths['NORMAL']
        else:
            width_pct = self.spread_widths['TIGHT']

        # Long strike: slightly OTM or ATM
        long_strike = round(current_price * 1.00, 0)  # ATM

        # Short strike: above long strike
        spread_width = current_price * width_pct
        short_strike = round(long_strike + spread_width, 0)

        # Estimate costs (rough approximation)
        # Long call costs more, short call credits less
        # Net debit = difference
        intrinsic_long = max(0, current_price - long_strike)
        intrinsic_short = max(0, current_price - short_strike)

        # Estimate option prices (simplified)
        est_long_price = intrinsic_long + (spread_width * 0.6)  # Time value
        est_short_price = intrinsic_short + (spread_width * 0.3)

        est_cost = est_long_price - est_short_price  # Net debit
        est_max_profit = (short_strike - long_strike) - est_cost
        est_max_loss = est_cost
        breakeven = long_strike + est_cost

        risk_reward = est_max_profit / est_max_loss if est_max_loss > 0 else 0

        return {
            'long_strike': float(long_strike),
            'short_strike': float(short_strike),
            'spread_width': float(short_strike - long_strike),
            'est_cost': float(est_cost),
            'est_max_profit': float(est_max_profit),
            'est_max_loss': float(est_max_loss),
            'breakeven': float(breakeven),
            'risk_reward': float(risk_reward),
            'strategy': 'BULL_CALL_SPREAD',
            'description': f'Buy ${long_strike:.0f} call, Sell ${short_strike:.0f} call'
        }

    def design_bear_put_spread(self, current_price: float, confidence: float,
                               budget: float = 500) -> Dict:
        """
        Design a bear put spread

        Structure:
        - Buy higher strike put (long)
        - Sell lower strike put (short)
        - Net debit

        Args:
            current_price: Current stock price
            confidence: Trade confidence
            budget: Maximum budget

        Returns: Same structure as bull call spread
        """
        # Spread width based on confidence
        if confidence >= 0.70:
            width_pct = self.spread_widths['WIDE']
        elif confidence >= 0.60:
            width_pct = self.spread_widths['NORMAL']
        else:
            width_pct = self.spread_widths['TIGHT']

        # Long strike: slightly OTM or ATM
        long_strike = round(current_price * 1.00, 0)  # ATM

        # Short strike: below long strike
        spread_width = current_price * width_pct
        short_strike = round(long_strike - spread_width, 0)

        # Estimate costs
        intrinsic_long = max(0, long_strike - current_price)
        intrinsic_short = max(0, short_strike - current_price)

        est_long_price = intrinsic_long + (spread_width * 0.6)
        est_short_price = intrinsic_short + (spread_width * 0.3)

        est_cost = est_long_price - est_short_price
        est_max_profit = (long_strike - short_strike) - est_cost
        est_max_loss = est_cost
        breakeven = long_strike - est_cost

        risk_reward = est_max_profit / est_max_loss if est_max_loss > 0 else 0

        return {
            'long_strike': float(long_strike),
            'short_strike': float(short_strike),
            'spread_width': float(long_strike - short_strike),
            'est_cost': float(est_cost),
            'est_max_profit': float(est_max_profit),
            'est_max_loss': float(est_max_loss),
            'breakeven': float(breakeven),
            'risk_reward': float(risk_reward),
            'strategy': 'BEAR_PUT_SPREAD',
            'description': f'Buy ${long_strike:.0f} put, Sell ${short_strike:.0f} put'
        }

    def compare_spread_vs_naked(self, current_price: float,
                                direction: str = 'CALL',
                                confidence: float = 0.65) -> Dict:
        """
        Compare spread strategy vs naked option

        Returns:
            {
                'spread': Dict,
                'naked': Dict,
                'recommendation': str,
                'reasoning': str
            }
        """
        # Design spread
        if direction.upper() == 'CALL':
            spread = self.design_bull_call_spread(current_price, confidence)
            naked_strike = current_price * 1.02  # Slightly OTM
            naked_est_cost = current_price * 0.08  # ~8% of stock price
        else:  # PUT
            spread = self.design_bear_put_spread(current_price, confidence)
            naked_strike = current_price * 0.98
            naked_est_cost = current_price * 0.08

        naked = {
            'strike': float(naked_strike),
            'est_cost': float(naked_est_cost),
            'est_max_profit': float('inf'),  # Unlimited for calls
            'est_max_loss': float(naked_est_cost),
            'risk_reward': float('inf'),
            'strategy': f'LONG_{direction}'
        }

        # Compare
        cost_savings = ((naked['est_cost'] - spread['est_cost']) / naked['est_cost']) * 100

        # Spread advantages
        if spread['risk_reward'] >= 1.5:
            recommendation = 'SPREAD'
            reasoning = (
                f"Spread preferred: {cost_savings:.0f}% cheaper, "
                f"R/R {spread['risk_reward']:.2f}, "
                f"Defined risk ${spread['est_max_loss']:.0f}"
            )
        elif confidence >= 0.75:
            recommendation = 'NAKED'
            reasoning = (
                f"High confidence ({confidence:.0%}) favors naked option for unlimited upside"
            )
        else:
            recommendation = 'SPREAD'
            reasoning = (
                f"Spread preferred for defined risk and {cost_savings:.0f}% cost savings"
            )

        return {
            'spread': spread,
            'naked': naked,
            'recommendation': recommendation,
            'reasoning': reasoning,
            'cost_savings_pct': float(cost_savings)
        }

    def calculate_spread_greeks(self, spread_design: Dict, dte: int = 30) -> Dict:
        """
        Estimate Greeks for a spread position

        Spread Greeks = Long Greeks - Short Greeks
        """
        long_strike = spread_design['long_strike']
        short_strike = spread_design['short_strike']

        # Simplified Greek estimates for spreads
        # Delta: Spread delta is difference between long and short deltas
        # Typically 0.3-0.6 for well-designed spreads
        est_delta = 0.45  # Neutral assumption

        # Theta: Spreads have less theta decay than naked options
        # Because short leg generates credit
        est_theta = -0.02  # Much lower than naked option

        # Vega: Spreads less sensitive to IV changes
        est_vega = 0.05  # Much lower than naked

        # Gamma: Lower gamma in spreads
        est_gamma = 0.03

        return {
            'delta': float(est_delta),
            'theta': float(est_theta),
            'vega': float(est_vega),
            'gamma': float(est_gamma),
            'note': 'Estimated - spreads have more predictable Greeks'
        }

    def evaluate_spread_quality(self, spread: Dict) -> Dict:
        """
        Evaluate quality of spread design

        Returns:
            {
                'score': float (0-100),
                'quality': str,
                'strengths': List[str],
                'weaknesses': List[str]
            }
        """
        score = 0
        strengths = []
        weaknesses = []

        # Risk/Reward score (40 points max)
        rr = spread['risk_reward']
        if rr >= 2.0:
            score += 40
            strengths.append(f"Excellent R/R ({rr:.2f})")
        elif rr >= 1.5:
            score += 30
            strengths.append(f"Good R/R ({rr:.2f})")
        elif rr >= 1.0:
            score += 20
            strengths.append(f"Acceptable R/R ({rr:.2f})")
        else:
            score += 10
            weaknesses.append(f"Poor R/R ({rr:.2f})")

        # Cost efficiency (30 points max)
        if spread['est_cost'] < 200:
            score += 30
            strengths.append(f"Low cost (${spread['est_cost']:.0f})")
        elif spread['est_cost'] < 400:
            score += 20
        else:
            score += 10
            weaknesses.append(f"High cost (${spread['est_cost']:.0f})")

        # Profit potential (30 points max)
        if spread['est_max_profit'] >= spread['est_cost'] * 2:
            score += 30
            strengths.append(f"High profit potential (${spread['est_max_profit']:.0f})")
        elif spread['est_max_profit'] >= spread['est_cost']:
            score += 20
        else:
            score += 10
            weaknesses.append(f"Limited profit (${spread['est_max_profit']:.0f})")

        # Quality rating
        if score >= 80:
            quality = 'EXCELLENT'
        elif score >= 60:
            quality = 'GOOD'
        elif score >= 40:
            quality = 'FAIR'
        else:
            quality = 'POOR'

        return {
            'score': float(score),
            'quality': quality,
            'strengths': strengths,
            'weaknesses': weaknesses
        }


# Global instance
_spread_strategies = None

def get_spread_strategies() -> SpreadStrategies:
    """Get singleton spread strategies"""
    global _spread_strategies
    if _spread_strategies is None:
        _spread_strategies = SpreadStrategies()
    return _spread_strategies


if __name__ == "__main__":
    # Test
    spreads = SpreadStrategies()

    current_price = 175.0
    print("="*70)
    print(f"SPREAD STRATEGIES TEST - Stock Price: ${current_price}")
    print("="*70)

    # Test bull call spread
    print("\n[BULL CALL SPREAD - High Confidence]")
    bull_spread = spreads.design_bull_call_spread(current_price, confidence=0.75)
    print(f"Strategy: {bull_spread['description']}")
    print(f"Spread Width: ${bull_spread['spread_width']:.0f}")
    print(f"Est Cost (Max Loss): ${bull_spread['est_cost']:.2f}")
    print(f"Est Max Profit: ${bull_spread['est_max_profit']:.2f}")
    print(f"Risk/Reward: {bull_spread['risk_reward']:.2f}")
    print(f"Breakeven: ${bull_spread['breakeven']:.2f}")

    # Evaluate quality
    quality = spreads.evaluate_spread_quality(bull_spread)
    print(f"\nQuality: {quality['quality']} ({quality['score']:.0f}/100)")
    print(f"Strengths: {', '.join(quality['strengths'])}")
    if quality['weaknesses']:
        print(f"Weaknesses: {', '.join(quality['weaknesses'])}")

    # Test bear put spread
    print(f"\n{'='*70}")
    print("[BEAR PUT SPREAD - Medium Confidence]")
    bear_spread = spreads.design_bear_put_spread(current_price, confidence=0.60)
    print(f"Strategy: {bear_spread['description']}")
    print(f"Spread Width: ${bear_spread['spread_width']:.0f}")
    print(f"Est Cost (Max Loss): ${bear_spread['est_cost']:.2f}")
    print(f"Est Max Profit: ${bear_spread['est_max_profit']:.2f}")
    print(f"Risk/Reward: {bear_spread['risk_reward']:.2f}")
    print(f"Breakeven: ${bear_spread['breakeven']:.2f}")

    # Compare spread vs naked
    print(f"\n{'='*70}")
    print("[SPREAD VS NAKED COMPARISON]")
    comparison = spreads.compare_spread_vs_naked(current_price, 'CALL', confidence=0.65)
    print(f"\nRecommendation: {comparison['recommendation']}")
    print(f"Reasoning: {comparison['reasoning']}")
    print(f"Cost Savings: {comparison['cost_savings_pct']:.0f}%")

    print(f"\nSpread: ${comparison['spread']['est_cost']:.0f} cost, "
          f"${comparison['spread']['est_max_profit']:.0f} max profit")
    print(f"Naked: ${comparison['naked']['est_cost']:.0f} cost, "
          f"unlimited max profit")
