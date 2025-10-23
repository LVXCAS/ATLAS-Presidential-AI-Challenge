#!/usr/bin/env python3
"""
Liquidity Filtering
Ensures options have sufficient liquidity to trade efficiently
"""

import yfinance as yf
import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class LiquidityFilter:
    """Filter trades based on liquidity criteria"""

    def __init__(self):
        # Minimum liquidity requirements
        self.min_requirements = {
            'option_volume': 100,        # Min daily option volume
            'option_open_interest': 500,  # Min open interest
            'stock_volume': 1_000_000,    # Min daily stock volume
            'bid_ask_spread_pct': 0.10    # Max 10% bid-ask spread
        }

        # Scoring weights
        self.weights = {
            'volume': 0.30,
            'open_interest': 0.30,
            'bid_ask': 0.25,
            'stock_volume': 0.15
        }

    def check_stock_liquidity(self, symbol: str) -> Dict:
        """
        Check if underlying stock is liquid enough

        Returns:
            {
                'liquid': bool,
                'volume': float,
                'avg_volume': float,
                'score': float (0-100),
                'reason': str
            }
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5d', interval='1d')

            if hist.empty:
                return {
                    'liquid': False,
                    'volume': 0,
                    'avg_volume': 0,
                    'score': 0,
                    'reason': 'No volume data available'
                }

            avg_volume = hist['Volume'].mean()
            latest_volume = hist['Volume'].iloc[-1]

            # Check if meets minimum
            if avg_volume < self.min_requirements['stock_volume']:
                return {
                    'liquid': False,
                    'volume': float(latest_volume),
                    'avg_volume': float(avg_volume),
                    'score': (avg_volume / self.min_requirements['stock_volume']) * 100,
                    'reason': f"Low stock volume ({avg_volume/1e6:.1f}M vs {self.min_requirements['stock_volume']/1e6:.0f}M min)"
                }

            # Score based on volume relative to minimum
            score = min(100, (avg_volume / self.min_requirements['stock_volume']) * 50)

            return {
                'liquid': True,
                'volume': float(latest_volume),
                'avg_volume': float(avg_volume),
                'score': float(score),
                'reason': f"Good stock liquidity ({avg_volume/1e6:.1f}M avg volume)"
            }

        except Exception as e:
            logger.error(f"Stock liquidity check error for {symbol}: {e}")
            return {
                'liquid': False,
                'volume': 0,
                'avg_volume': 0,
                'score': 0,
                'reason': f'Error checking liquidity: {e}'
            }

    def estimate_option_liquidity(self, symbol: str, strike: float,
                                  expiration_date: str,
                                  option_type: str = 'call') -> Dict:
        """
        Estimate option liquidity (simplified - real implementation would use options chain API)

        For now, returns estimates based on stock liquidity and moneyness

        Returns:
            {
                'liquid': bool,
                'est_volume': int,
                'est_open_interest': int,
                'est_bid_ask_pct': float,
                'score': float (0-100),
                'reason': str
            }
        """
        # Get stock liquidity first
        stock_liq = self.check_stock_liquidity(symbol)

        if not stock_liq['liquid']:
            return {
                'liquid': False,
                'est_volume': 0,
                'est_open_interest': 0,
                'est_bid_ask_pct': 0.50,
                'score': 0,
                'reason': 'Stock not liquid enough for options'
            }

        # Estimate option liquidity based on stock volume
        # Popular stocks (>5M volume) typically have liquid options
        stock_vol = stock_liq['avg_volume']

        if stock_vol > 10_000_000:
            # Very liquid stock - options should be liquid
            est_volume = 500
            est_oi = 2000
            est_spread = 0.03
            liquid = True
            reason = "High stock volume - liquid options expected"

        elif stock_vol > 5_000_000:
            est_volume = 300
            est_oi = 1000
            est_spread = 0.05
            liquid = True
            reason = "Good stock volume - options should be tradeable"

        elif stock_vol > 2_000_000:
            est_volume = 150
            est_oi = 600
            est_spread = 0.08
            liquid = True
            reason = "Moderate stock volume - check specific contracts"

        else:
            # Lower volume - may have illiquid options
            est_volume = 50
            est_oi = 200
            est_spread = 0.15
            liquid = False
            reason = "Low stock volume - options may be illiquid"

        # Calculate score
        vol_score = min(100, (est_volume / self.min_requirements['option_volume']) * 100)
        oi_score = min(100, (est_oi / self.min_requirements['option_open_interest']) * 100)
        spread_score = max(0, 100 - (est_spread / self.min_requirements['bid_ask_spread_pct']) * 100)

        total_score = (
            vol_score * self.weights['volume'] +
            oi_score * self.weights['open_interest'] +
            spread_score * self.weights['bid_ask'] +
            stock_liq['score'] * self.weights['stock_volume']
        )

        return {
            'liquid': liquid and total_score >= 60,
            'est_volume': int(est_volume),
            'est_open_interest': int(est_oi),
            'est_bid_ask_pct': float(est_spread),
            'score': float(total_score),
            'reason': reason
        }

    def approve_for_trading(self, symbol: str, strike: float = None,
                           expiration_date: str = None,
                           option_type: str = 'call') -> Dict:
        """
        Comprehensive liquidity approval check

        Returns:
            {
                'approved': bool,
                'stock_liquidity': Dict,
                'option_liquidity': Dict,
                'overall_score': float,
                'warnings': List[str],
                'recommendation': str
            }
        """
        warnings = []

        # Check stock liquidity
        stock_liq = self.check_stock_liquidity(symbol)

        # Estimate option liquidity if strike provided
        if strike is not None and expiration_date is not None:
            option_liq = self.estimate_option_liquidity(
                symbol, strike, expiration_date, option_type
            )
        else:
            # Just use stock liquidity
            option_liq = {
                'liquid': stock_liq['liquid'],
                'score': stock_liq['score']
            }

        # Calculate overall score
        overall_score = (stock_liq['score'] * 0.4 + option_liq['score'] * 0.6)

        # Generate warnings
        if not stock_liq['liquid']:
            warnings.append(stock_liq['reason'])

        if 'est_bid_ask_pct' in option_liq and option_liq['est_bid_ask_pct'] > 0.08:
            warnings.append(f"Wide estimated spread ({option_liq['est_bid_ask_pct']:.1%})")

        if 'est_volume' in option_liq and option_liq['est_volume'] < 200:
            warnings.append(f"Low estimated option volume ({option_liq['est_volume']})")

        # Approval decision
        approved = stock_liq['liquid'] and option_liq.get('liquid', False) and overall_score >= 60

        # Recommendation
        if approved:
            if overall_score >= 80:
                recommendation = f"APPROVED - Excellent liquidity (score: {overall_score:.0f})"
            else:
                recommendation = f"APPROVED - Adequate liquidity (score: {overall_score:.0f})"
        else:
            recommendation = f"REJECTED - Insufficient liquidity (score: {overall_score:.0f})"

        return {
            'approved': approved,
            'stock_liquidity': stock_liq,
            'option_liquidity': option_liq,
            'overall_score': float(overall_score),
            'warnings': warnings,
            'recommendation': recommendation
        }


# Global instance
_liquidity_filter = None

def get_liquidity_filter() -> LiquidityFilter:
    """Get singleton liquidity filter"""
    global _liquidity_filter
    if _liquidity_filter is None:
        _liquidity_filter = LiquidityFilter()
    return _liquidity_filter


if __name__ == "__main__":
    # Test
    filter = LiquidityFilter()

    print("="*70)
    print("LIQUIDITY FILTERING TEST")
    print("="*70)

    # Test different stocks
    test_symbols = [
        ('AAPL', 'High volume tech stock'),
        ('SPY', 'Most liquid ETF'),
        ('TSLA', 'High volume volatile'),
        ('XYZ', 'Low volume (example)')
    ]

    for symbol, description in test_symbols:
        print(f"\n[{symbol} - {description}]")

        # Check stock liquidity
        stock_liq = filter.check_stock_liquidity(symbol)
        print(f"Stock: {stock_liq['reason']}")
        print(f"  Volume: {stock_liq.get('avg_volume', 0)/1e6:.1f}M")
        print(f"  Score: {stock_liq['score']:.0f}/100")

        # Full approval check
        approval = filter.approve_for_trading(symbol)
        print(f"\nApproval: {approval['approved']}")
        print(f"Overall Score: {approval['overall_score']:.0f}/100")
        print(f"Recommendation: {approval['recommendation']}")

        if approval['warnings']:
            print(f"Warnings: {', '.join(approval['warnings'])}")

        print("-" * 70)
