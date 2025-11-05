"""
SHARED KELLY CRITERION POSITION SIZING
Used by FOREX, FUTURES, and CRYPTO bots
Provides consistent risk-based position sizing across all markets
"""
import math


class KellyCriterion:
    """
    Kelly Criterion position sizing engine
    Calculates optimal position size based on signal confidence and win probability
    """

    def __init__(self, fraction=0.25):
        """
        Args:
            fraction: Kelly fraction (0.25 = quarter-Kelly, conservative)
                     0.25 = recommended (safer)
                     0.50 = aggressive
                     1.00 = full Kelly (very risky)
        """
        self.fraction = fraction

    def calculate_position_size(self, technical_score, fundamental_score, account_balance, risk_per_trade=0.01):
        """
        Calculate position size using Kelly Criterion

        Args:
            technical_score: 0-10 rating from technical analysis
            fundamental_score: -6 to +6 rating from fundamental analysis
            account_balance: Current account balance ($)
            risk_per_trade: Risk percentage per trade (default 1%)

        Returns:
            dict: {
                'base_size': Base position size in dollars,
                'kelly_multiplier': Kelly-based multiplier (0.5x - 1.5x),
                'final_size': Final position size after Kelly adjustment,
                'confidence': Combined confidence percentage
            }
        """
        # Normalize scores to percentages
        tech_confidence = technical_score / 10.0  # 0.0 to 1.0
        fund_confidence = abs(fundamental_score) / 6.0  # 0.0 to 1.0

        # Combined confidence (weighted average)
        combined_confidence = (tech_confidence * 0.7) + (fund_confidence * 0.3)

        # Calculate win probability from confidence
        # Assumes 50% base win rate + confidence boost
        win_probability = 0.50 + (combined_confidence * 0.15)  # 50% to 65% range

        # Kelly formula: f = (bp - q) / b
        # where:
        #   b = odds (assume 2:1 reward/risk)
        #   p = win probability
        #   q = loss probability (1 - p)
        odds = 2.0  # 2:1 reward/risk ratio
        loss_prob = 1 - win_probability

        kelly_fraction = (odds * win_probability - loss_prob) / odds

        # Apply quarter-Kelly for safety
        kelly_fraction = max(0, kelly_fraction) * self.fraction

        # Base position size (1% risk)
        base_size = account_balance * risk_per_trade

        # Kelly multiplier (scales base position from 0.5x to 1.5x)
        # Low confidence (0.2) = 0.5x multiplier
        # High confidence (0.8) = 1.5x multiplier
        kelly_multiplier = 0.5 + (combined_confidence * 1.0)
        kelly_multiplier = min(1.5, max(0.5, kelly_multiplier))

        # Final position size
        final_size = base_size * kelly_multiplier

        return {
            'base_size': base_size,
            'kelly_multiplier': kelly_multiplier,
            'final_size': final_size,
            'confidence': combined_confidence,
            'win_probability': win_probability,
            'kelly_fraction': kelly_fraction
        }

    def calculate_forex_units(self, position_size_dollars, current_price, leverage=5):
        """
        Convert dollar position size to forex units

        Args:
            position_size_dollars: Position size in dollars
            current_price: Current market price
            leverage: Leverage multiplier (default 5x)

        Returns:
            int: Number of units to trade
        """
        # Base units without leverage
        base_units = int(position_size_dollars / current_price * 100000)  # Standard lot = 100k

        # Apply leverage
        leveraged_units = base_units * leverage

        return leveraged_units

    def calculate_futures_contracts(self, position_size_dollars, point_value, current_price):
        """
        Convert dollar position size to futures contracts

        Args:
            position_size_dollars: Position size in dollars
            point_value: Dollar value per point (e.g., ES = $50/point)
            current_price: Current futures price

        Returns:
            int: Number of contracts to trade
        """
        # Calculate how many contracts fit in position size
        # Each contract represents (current_price * point_value) in notional value
        notional_per_contract = current_price * point_value
        contracts = int(position_size_dollars / notional_per_contract)

        # Minimum 1 contract
        return max(1, contracts)

    def calculate_crypto_units(self, position_size_dollars, current_price):
        """
        Convert dollar position size to crypto units

        Args:
            position_size_dollars: Position size in dollars
            current_price: Current crypto price

        Returns:
            float: Number of crypto units to trade (can be fractional)
        """
        # Crypto allows fractional units
        units = position_size_dollars / current_price
        return round(units, 8)  # 8 decimals for BTC precision


# Singleton instance
kelly = KellyCriterion(fraction=0.25)  # Conservative quarter-Kelly
