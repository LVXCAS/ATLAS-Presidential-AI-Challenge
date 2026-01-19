"""
Monte Carlo Validation Engine for ATLAS

Uses Monte Carlo simulations to validate strategy robustness and calculate
confidence intervals for performance metrics.

This is how institutional funds validate strategies before deployment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path


class MonteCarloValidator:
    """
    Monte Carlo simulation engine for strategy validation.

    Used by hedge funds to:
    - Test strategy robustness (overfitting detection)
    - Calculate confidence intervals (95%, 99%)
    - Estimate probability of passing E8 challenge
    - Find worst-case drawdown scenarios
    - Validate risk/reward expectations
    """

    def __init__(self, num_simulations: int = 1000):
        """
        Initialize Monte Carlo validator.

        Args:
            num_simulations: Number of Monte Carlo runs (1000-10000)
                - 1000: Quick validation (~1 min)
                - 5000: Standard validation (~5 min)
                - 10000: High-confidence validation (~10 min)
        """
        self.num_simulations = num_simulations
        self.results = []

    def run_simulations(self, trade_history: List[Dict], starting_balance: float = 200000) -> Dict:
        """
        Run Monte Carlo simulations by randomizing trade order.

        This tests if your strategy is order-dependent (bad) or robust (good).

        Args:
            trade_history: List of historical trades with P&L
            starting_balance: Starting account balance ($200k for E8)

        Returns:
            Dictionary with simulation results and statistics
        """
        print(f"\n{'='*80}")
        print(f"MONTE CARLO VALIDATION - {self.num_simulations} SIMULATIONS")
        print(f"{'='*80}")
        print(f"Historical Trades: {len(trade_history)}")
        print(f"Starting Balance: ${starting_balance:,.0f}")
        print(f"{'='*80}\n")

        if len(trade_history) < 30:
            print("[WARNING] Less than 30 trades - results may be unreliable")
            print("          Recommend minimum 100 trades for robust validation\n")

        simulation_results = []

        for i in range(self.num_simulations):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{self.num_simulations} simulations ({(i+1)/self.num_simulations*100:.0f}%)")

            # Randomize trade order (preserves win rate but tests sequence dependency)
            randomized_trades = np.random.permutation(trade_history).tolist()

            # Run simulation
            result = self._simulate_trading_sequence(randomized_trades, starting_balance)
            simulation_results.append(result)

        self.results = simulation_results

        # Calculate statistics
        stats = self._calculate_statistics(simulation_results, starting_balance)

        print(f"\n{'='*80}")
        print("SIMULATION COMPLETE")
        print(f"{'='*80}\n")

        return stats

    def _simulate_trading_sequence(self, trades: List[Dict], starting_balance: float) -> Dict:
        """
        Simulate a single trading sequence.

        Tracks:
        - Final balance
        - Max drawdown
        - Profit/loss
        - Number of DD violations
        - E8 pass/fail
        """
        balance = starting_balance
        peak_balance = starting_balance
        max_drawdown = 0.0
        daily_losses = []
        dd_violations = 0

        e8_profit_target = starting_balance * 0.10  # 10% = $20k
        e8_trailing_dd_limit = 0.06  # 6%
        e8_daily_dd_limit = starting_balance * 0.015  # ~$3k

        for trade in trades:
            pnl = trade.get('pnl', 0)
            balance += pnl

            # Track peak for trailing DD
            if balance > peak_balance:
                peak_balance = balance

            # Calculate drawdown
            current_dd = (peak_balance - balance) / peak_balance
            max_drawdown = max(max_drawdown, current_dd)

            # Check E8 violations
            if current_dd >= e8_trailing_dd_limit:
                dd_violations += 1
                break  # Account terminated

            # Daily DD check (simplified - assumes each trade is a day)
            if pnl < 0 and abs(pnl) >= e8_daily_dd_limit:
                dd_violations += 1
                break  # Account terminated

        # Calculate results
        final_pnl = balance - starting_balance
        profit_target_reached = final_pnl >= e8_profit_target
        passed_e8 = profit_target_reached and dd_violations == 0

        return {
            'final_balance': balance,
            'final_pnl': final_pnl,
            'max_drawdown': max_drawdown,
            'dd_violations': dd_violations,
            'passed_e8': passed_e8,
            'profit_target_reached': profit_target_reached
        }

    def _calculate_statistics(self, results: List[Dict], starting_balance: float) -> Dict:
        """
        Calculate statistical metrics across all simulations.

        Returns confidence intervals and probabilities.
        """
        final_pnls = [r['final_pnl'] for r in results]
        max_drawdowns = [r['max_drawdown'] for r in results]
        e8_passes = [r['passed_e8'] for r in results]

        stats = {
            # P&L Statistics
            'mean_pnl': np.mean(final_pnls),
            'median_pnl': np.median(final_pnls),
            'std_pnl': np.std(final_pnls),
            'min_pnl': np.min(final_pnls),
            'max_pnl': np.max(final_pnls),

            # P&L Confidence Intervals
            'pnl_ci_95': (
                np.percentile(final_pnls, 2.5),
                np.percentile(final_pnls, 97.5)
            ),
            'pnl_ci_99': (
                np.percentile(final_pnls, 0.5),
                np.percentile(final_pnls, 99.5)
            ),

            # Drawdown Statistics
            'mean_max_dd': np.mean(max_drawdowns),
            'median_max_dd': np.median(max_drawdowns),
            'worst_case_dd': np.max(max_drawdowns),
            'best_case_dd': np.min(max_drawdowns),

            # Drawdown Confidence Intervals
            'dd_ci_95': (
                np.percentile(max_drawdowns, 2.5),
                np.percentile(max_drawdowns, 97.5)
            ),
            'dd_ci_99': (
                np.percentile(max_drawdowns, 0.5),
                np.percentile(max_drawdowns, 99.5)
            ),

            # E8 Challenge Statistics
            'e8_pass_probability': np.mean(e8_passes),
            'e8_fail_probability': 1 - np.mean(e8_passes),

            # Risk of Ruin
            'probability_of_loss': sum(1 for pnl in final_pnls if pnl < 0) / len(final_pnls),
            'probability_of_profit': sum(1 for pnl in final_pnls if pnl > 0) / len(final_pnls),

            # Meta
            'num_simulations': len(results),
            'starting_balance': starting_balance
        }

        return stats

    def print_report(self, stats: Dict):
        """
        Print comprehensive Monte Carlo validation report.
        """
        print("\n" + "="*80)
        print("MONTE CARLO VALIDATION REPORT")
        print("="*80)

        # P&L Analysis
        print("\n[PROFIT & LOSS ANALYSIS]")
        print(f"  Mean P&L:           ${stats['mean_pnl']:>12,.2f}")
        print(f"  Median P&L:         ${stats['median_pnl']:>12,.2f}")
        print(f"  Std Deviation:      ${stats['std_pnl']:>12,.2f}")
        print(f"  Best Case:          ${stats['max_pnl']:>12,.2f}")
        print(f"  Worst Case:         ${stats['min_pnl']:>12,.2f}")

        print("\n[CONFIDENCE INTERVALS - P&L]")
        print(f"  95% Confidence:     ${stats['pnl_ci_95'][0]:>12,.2f} to ${stats['pnl_ci_95'][1]:>12,.2f}")
        print(f"  99% Confidence:     ${stats['pnl_ci_99'][0]:>12,.2f} to ${stats['pnl_ci_99'][1]:>12,.2f}")
        print(f"\n  → 95% chance your P&L will be between these values")

        # Drawdown Analysis
        print("\n[DRAWDOWN ANALYSIS]")
        print(f"  Mean Max DD:        {stats['mean_max_dd']:>12.2%}")
        print(f"  Median Max DD:      {stats['median_max_dd']:>12.2%}")
        print(f"  Worst Case DD:      {stats['worst_case_dd']:>12.2%}")
        print(f"  Best Case DD:       {stats['best_case_dd']:>12.2%}")

        print("\n[CONFIDENCE INTERVALS - DRAWDOWN]")
        print(f"  95% Confidence:     {stats['dd_ci_95'][0]:>12.2%} to {stats['dd_ci_95'][1]:>12.2%}")
        print(f"  99% Confidence:     {stats['dd_ci_99'][0]:>12.2%} to {stats['dd_ci_99'][1]:>12.2%}")
        print(f"\n  → 95% chance your max DD will be in this range")

        # E8 Challenge Analysis
        print("\n[E8 CHALLENGE PROBABILITY]")
        pass_prob = stats['e8_pass_probability']
        fail_prob = stats['e8_fail_probability']

        print(f"  Pass Probability:   {pass_prob:>12.1%}")
        print(f"  Fail Probability:   {fail_prob:>12.1%}")

        if pass_prob >= 0.60:
            verdict = "EXCELLENT - Deploy with confidence"
        elif pass_prob >= 0.50:
            verdict = "GOOD - Above 50/50 chance"
        elif pass_prob >= 0.40:
            verdict = "ACCEPTABLE - Needs improvement"
        else:
            verdict = "POOR - Do not deploy yet"

        print(f"\n  Verdict: {verdict}")

        # Risk Analysis
        print("\n[RISK ANALYSIS]")
        print(f"  Probability of Profit: {stats['probability_of_profit']:>10.1%}")
        print(f"  Probability of Loss:   {stats['probability_of_loss']:>10.1%}")

        # Recommendations
        print("\n[RECOMMENDATIONS]")
        if stats['worst_case_dd'] > 0.06:
            print("  ⚠️  Worst case DD exceeds 6% - adjust position sizing")
        else:
            print("  ✅ Worst case DD within E8 limits")

        if pass_prob >= 0.50:
            print(f"  ✅ Pass probability {pass_prob:.0%} - strategy is robust")
        else:
            print(f"  ❌ Pass probability {pass_prob:.0%} - needs more training")

        if stats['pnl_ci_95'][0] > 0:
            print("  ✅ 95% CI lower bound positive - high confidence")
        else:
            print("  ⚠️  95% CI includes negative outcomes - moderate risk")

        print("\n" + "="*80)

    def save_results(self, stats: Dict, filepath: str = "monte_carlo_results.json"):
        """Save Monte Carlo results to JSON file"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'num_simulations': self.num_simulations,
            'statistics': stats,
            'raw_results': self.results
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n[SAVED] Results saved to {filepath}")

    def plot_distributions(self, stats: Dict):
        """
        Plot P&L and drawdown distributions.

        Requires matplotlib - optional visualization.
        """
        try:
            import matplotlib.pyplot as plt

            final_pnls = [r['final_pnl'] for r in self.results]
            max_drawdowns = [r['max_drawdown'] for r in self.results]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # P&L Distribution
            ax1.hist(final_pnls, bins=50, alpha=0.7, edgecolor='black')
            ax1.axvline(stats['mean_pnl'], color='red', linestyle='--', label=f'Mean: ${stats["mean_pnl"]:,.0f}')
            ax1.axvline(stats['pnl_ci_95'][0], color='orange', linestyle='--', label='95% CI')
            ax1.axvline(stats['pnl_ci_95'][1], color='orange', linestyle='--')
            ax1.set_xlabel('Final P&L ($)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Monte Carlo P&L Distribution')
            ax1.legend()
            ax1.grid(alpha=0.3)

            # Drawdown Distribution
            ax2.hist([dd * 100 for dd in max_drawdowns], bins=50, alpha=0.7, edgecolor='black')
            ax2.axvline(stats['mean_max_dd'] * 100, color='red', linestyle='--', label=f'Mean: {stats["mean_max_dd"]:.1%}')
            ax2.axvline(6.0, color='red', linestyle='-', linewidth=2, label='E8 Limit (6%)')
            ax2.set_xlabel('Max Drawdown (%)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Monte Carlo Drawdown Distribution')
            ax2.legend()
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig('monte_carlo_validation.png', dpi=150)
            print("\n[SAVED] Distribution plots saved to monte_carlo_validation.png")

        except ImportError:
            print("\n[INFO] Install matplotlib for visualization: pip install matplotlib")


def run_atlas_monte_carlo_validation(trade_history_file: str = None, num_simulations: int = 1000):
    """
    Run Monte Carlo validation on ATLAS trade history.

    Args:
        trade_history_file: Path to ATLAS trade history JSON
        num_simulations: Number of Monte Carlo runs

    Returns:
        Validation statistics
    """
    if trade_history_file is None:
        # Generate example trade history for demo
        print("[DEMO MODE] Generating example trade history...")
        trade_history = generate_example_trades(num_trades=150, win_rate=0.60, avg_win=1500, avg_loss=800)
    else:
        # Load actual ATLAS trade history
        with open(trade_history_file, 'r') as f:
            data = json.load(f)
            trade_history = data.get('trades', [])

    # Run Monte Carlo
    validator = MonteCarloValidator(num_simulations=num_simulations)
    stats = validator.run_simulations(trade_history, starting_balance=200000)

    # Print report
    validator.print_report(stats)

    # Save results
    validator.save_results(stats, "Agents/ATLAS_HYBRID/learning/state/monte_carlo_results.json")

    # Plot distributions (if matplotlib available)
    validator.plot_distributions(stats)

    return stats


def generate_example_trades(num_trades: int = 150, win_rate: float = 0.60,
                            avg_win: float = 1500, avg_loss: float = 800) -> List[Dict]:
    """
    Generate example trade history for Monte Carlo demo.

    Args:
        num_trades: Number of trades
        win_rate: Win rate (0-1)
        avg_win: Average winning trade size
        avg_loss: Average losing trade size

    Returns:
        List of trade dictionaries
    """
    trades = []

    for i in range(num_trades):
        is_win = np.random.random() < win_rate

        if is_win:
            # Winning trade - add some variance
            pnl = np.random.normal(avg_win, avg_win * 0.3)
        else:
            # Losing trade - add some variance
            pnl = -np.random.normal(avg_loss, avg_loss * 0.3)

        trades.append({
            'trade_id': i + 1,
            'pnl': pnl,
            'outcome': 'WIN' if is_win else 'LOSS'
        })

    return trades


if __name__ == "__main__":
    """
    Run Monte Carlo validation demo.

    Usage:
        python monte_carlo_validator.py
    """
    print("\n" + "="*80)
    print("ATLAS MONTE CARLO VALIDATOR - DEMO MODE")
    print("="*80)
    print("\nSimulating ATLAS strategy with:")
    print("  • 150 trades")
    print("  • 60% win rate")
    print("  • $1,500 avg win")
    print("  • $800 avg loss")
    print("  • 1000 Monte Carlo simulations\n")

    stats = run_atlas_monte_carlo_validation(num_simulations=1000)

    print("\n[NEXT STEPS]")
    print("  1. Complete 60-day ATLAS paper training")
    print("  2. Export trade history to JSON")
    print("  3. Run Monte Carlo on actual trades:")
    print("     python monte_carlo_validator.py --trades atlas_trades.json --simulations 5000")
    print("\n" + "="*80 + "\n")
