#!/usr/bin/env python3
"""
KELLY CRITERION POSITION SIZER - Week 5+ Feature
=================================================
Calculate optimal position size based on edge and risk

Kelly Criterion Formula:
f* = (p × b - q) / b

Where:
- f* = fraction of capital to risk
- p = probability of winning
- q = probability of losing (1 - p)
- b = odds received on the bet (profit/loss ratio)

For trading:
- Use fractional Kelly (0.25-0.50) to reduce volatility
- Adjust for correlation risk
- Cap maximum position size at 10-15%
"""

import numpy as np
from datetime import datetime


class KellyCriterionSizer:
    """Calculate optimal position sizes using Kelly Criterion"""

    def __init__(self, kelly_fraction=0.25):
        """
        Args:
            kelly_fraction: Fraction of Kelly to use (0.25 = Quarter Kelly)
                           Recommended: 0.25-0.50 for reduced volatility
        """
        self.kelly_fraction = kelly_fraction

    def calculate_kelly_size(self, win_prob, profit_loss_ratio, capital):
        """
        Calculate Kelly Criterion position size

        Args:
            win_prob: Probability of winning (0-1)
            profit_loss_ratio: Average win / average loss
            capital: Total capital available

        Returns:
            Recommended position size in dollars
        """

        # Kelly formula
        loss_prob = 1 - win_prob
        kelly_pct = (win_prob * profit_loss_ratio - loss_prob) / profit_loss_ratio

        # Apply fractional Kelly
        fractional_kelly_pct = kelly_pct * self.kelly_fraction

        # Cap at reasonable limits (5-15% of capital)
        MAX_POSITION_SIZE = 0.15
        MIN_POSITION_SIZE = 0.01

        final_kelly_pct = max(MIN_POSITION_SIZE, min(MAX_POSITION_SIZE, fractional_kelly_pct))

        position_size = capital * final_kelly_pct

        print(f"\n{'='*80}")
        print(f"KELLY CRITERION POSITION SIZING")
        print(f"{'='*80}")
        print(f"Win Probability: {win_prob*100:.1f}%")
        print(f"Profit/Loss Ratio: {profit_loss_ratio:.2f}x")
        print(f"Kelly %: {kelly_pct*100:.2f}%")
        print(f"Fractional Kelly ({self.kelly_fraction*100:.0f}%): {fractional_kelly_pct*100:.2f}%")
        print(f"Final Position Size: {final_kelly_pct*100:.2f}% of capital")
        print(f"Dollar Amount: ${position_size:,.2f} (of ${capital:,.2f})")

        if kelly_pct <= 0:
            print(f"\n[WARNING] No edge detected - Kelly ≤ 0")
            print(f"Do not take this trade (negative expected value)")
            return 0

        if final_kelly_pct >= MAX_POSITION_SIZE:
            print(f"\n[WARNING] Position capped at {MAX_POSITION_SIZE*100:.0f}% (risk management)")

        return position_size

    def calculate_from_greeks(self, delta, profit_target_pct, capital):
        """
        Calculate Kelly size from options Greeks

        Args:
            delta: Options delta (probability of ITM approximation)
            profit_target_pct: Target profit percentage
            capital: Total capital

        Returns:
            Recommended position size
        """

        # Approximate win probability from delta
        # For selling options: win_prob ≈ 1 - abs(delta)
        # For buying options: win_prob ≈ abs(delta)

        # Assume selling strategy (cash-secured puts, iron condors)
        win_prob = 1 - abs(delta)

        # Approximate profit/loss ratio
        # Selling: collect premium (profit) vs max loss (strike width - premium)
        # Estimate: profit = 20-30% of max risk
        profit_loss_ratio = 0.30 / (1 - 0.30)  # 30% profit on 70% risk

        return self.calculate_kelly_size(win_prob, profit_loss_ratio, capital)

    def calculate_from_historical_performance(self, trade_history):
        """
        Calculate Kelly size from actual trading history

        Args:
            trade_history: List of trades with 'pnl' field

        Returns:
            Recommended position size fraction
        """

        print(f"\n{'='*80}")
        print(f"KELLY FROM HISTORICAL PERFORMANCE")
        print(f"{'='*80}")

        if not trade_history or len(trade_history) < 10:
            print(f"  [WARNING] Insufficient history (need 10+ trades, have {len(trade_history)})")
            return None

        # Calculate win rate
        winning_trades = [t for t in trade_history if t['pnl'] > 0]
        losing_trades = [t for t in trade_history if t['pnl'] <= 0]

        win_rate = len(winning_trades) / len(trade_history)
        loss_rate = 1 - win_rate

        # Calculate average win and loss
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 1

        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1

        print(f"  Total Trades: {len(trade_history)}")
        print(f"  Win Rate: {win_rate*100:.1f}%")
        print(f"  Average Win: ${avg_win:,.2f}")
        print(f"  Average Loss: ${avg_loss:,.2f}")
        print(f"  Profit/Loss Ratio: {profit_loss_ratio:.2f}x")

        # Kelly calculation
        kelly_pct = (win_rate * profit_loss_ratio - loss_rate) / profit_loss_ratio
        fractional_kelly_pct = kelly_pct * self.kelly_fraction

        print(f"  Kelly %: {kelly_pct*100:.2f}%")
        print(f"  Recommended Position Size: {fractional_kelly_pct*100:.2f}% of capital")

        return fractional_kelly_pct

    def adjust_for_correlation(self, base_position_size, portfolio_correlation):
        """
        Adjust position size downward if portfolio is highly correlated

        Args:
            base_position_size: Calculated Kelly position size
            portfolio_correlation: Average correlation (0-1)

        Returns:
            Adjusted position size
        """

        # Higher correlation = smaller positions (reduce concentration risk)
        correlation_adjustment = 1 - (portfolio_correlation * 0.5)

        adjusted_size = base_position_size * correlation_adjustment

        print(f"\n{'='*80}")
        print(f"CORRELATION ADJUSTMENT")
        print(f"{'='*80}")
        print(f"Portfolio Correlation: {portfolio_correlation:.3f}")
        print(f"Correlation Adjustment: {correlation_adjustment:.2f}x")
        print(f"Base Position: ${base_position_size:,.2f}")
        print(f"Adjusted Position: ${adjusted_size:,.2f}")

        return adjusted_size


def test_kelly_sizer():
    """Test Kelly Criterion sizer"""

    sizer = KellyCriterionSizer(kelly_fraction=0.25)

    print("="*80)
    print("TEST: KELLY CRITERION POSITION SIZING")
    print("="*80)

    # Test 1: High-probability options trade
    print("\n\nTEST 1: High-Probability Iron Condor")
    print("Win Probability: 75%, Profit/Loss Ratio: 0.43x (30% profit on 70% max loss)")
    sizer.calculate_kelly_size(
        win_prob=0.75,
        profit_loss_ratio=0.43,
        capital=100000
    )

    # Test 2: Lower-probability directional trade
    print("\n\nTEST 2: Directional Options Trade")
    print("Win Probability: 40%, Profit/Loss Ratio: 2.0x (200% profit on 100% loss)")
    sizer.calculate_kelly_size(
        win_prob=0.40,
        profit_loss_ratio=2.0,
        capital=100000
    )

    # Test 3: From Greeks
    print("\n\nTEST 3: Calculate from Options Greeks")
    print("Delta: 0.30 (selling 0.30 delta puts)")
    sizer.calculate_from_greeks(
        delta=0.30,
        profit_target_pct=0.20,
        capital=100000
    )

    # Test 4: From historical performance
    print("\n\nTEST 4: Calculate from Trading History")
    sample_history = [
        {'pnl': 150}, {'pnl': 200}, {'pnl': -100}, {'pnl': 180},
        {'pnl': -80}, {'pnl': 220}, {'pnl': 190}, {'pnl': -120},
        {'pnl': 160}, {'pnl': 210}, {'pnl': -90}, {'pnl': 175}
    ]
    sizer.calculate_from_historical_performance(sample_history)

    # Test 5: Correlation adjustment
    print("\n\nTEST 5: Adjust for Portfolio Correlation")
    base_position = 10000
    sizer.adjust_for_correlation(base_position, portfolio_correlation=0.6)


if __name__ == "__main__":
    test_kelly_sizer()
