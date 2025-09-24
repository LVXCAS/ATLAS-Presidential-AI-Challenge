#!/usr/bin/env python3
"""
Enhanced Monte Carlo Trading Simulation - More Aggressive Parameters
Demonstrates 5.75% profit target system with higher volatility scenarios
"""

import asyncio
import sys
import os
import time
import json
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Add current directory to path
sys.path.append('.')

class EnhancedTradingBotSimulation:
    """Enhanced Monte Carlo simulation with more aggressive trading scenarios"""

    def __init__(self, scenario: str = "aggressive"):
        self.starting_capital = 100000.0
        self.profit_target_pct = 5.75
        self.trading_days = 252

        # Set parameters based on scenario
        if scenario == "conservative":
            self.avg_trades_per_day = 5
            self.win_rate = 0.60
            self.avg_win_pct = 0.025
            self.avg_loss_pct = -0.015
            self.position_size_range = (0.003, 0.008)  # 0.3% to 0.8%
            self.volatility_factor = 0.8
        elif scenario == "aggressive":
            self.avg_trades_per_day = 10
            self.win_rate = 0.58
            self.avg_win_pct = 0.045
            self.avg_loss_pct = -0.025
            self.position_size_range = (0.008, 0.025)  # 0.8% to 2.5%
            self.volatility_factor = 1.5
        else:  # "very_aggressive"
            self.avg_trades_per_day = 15
            self.win_rate = 0.55
            self.avg_win_pct = 0.060
            self.avg_loss_pct = -0.035
            self.position_size_range = (0.015, 0.035)  # 1.5% to 3.5%
            self.volatility_factor = 2.0

        self.scenario = scenario
        self.profit_target_days = []
        self.daily_results = []
        self.equity_curve = []
        self.trade_log = []

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.profit_target_hits = 0
        self.consecutive_wins = 0
        self.max_consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0

    def simulate_high_volatility_trade(self, day: int, trade_num: int) -> Dict:
        """Simulate a trade with higher volatility for options trading"""
        is_winner = random.random() < self.win_rate

        if is_winner:
            # Winning options trades can be very profitable
            base_return = random.uniform(0.02, 0.15)  # 2% to 15%
            # Add momentum effect (big wins can lead to bigger wins)
            if self.consecutive_wins > 3:
                base_return *= 1.2
            # Add volatility
            volatility_adjustment = random.gauss(0, 0.02) * self.volatility_factor
            trade_return = base_return + volatility_adjustment
            trade_return = max(0.01, min(0.25, trade_return))  # 1% to 25%
        else:
            # Losing trades - options can lose quickly but risk is limited
            base_return = random.uniform(-0.08, -0.01)  # -8% to -1%
            # Add volatility
            volatility_adjustment = random.gauss(0, 0.015) * self.volatility_factor
            trade_return = base_return + volatility_adjustment
            trade_return = max(-0.12, min(-0.005, trade_return))  # -12% to -0.5%

        # Dynamic position sizing based on recent performance
        base_size = random.uniform(*self.position_size_range)

        # Reduce position size after losses
        if self.consecutive_losses > 2:
            base_size *= 0.7
        # Increase position size after wins (with limit)
        elif self.consecutive_wins > 2:
            base_size *= min(1.3, 1.0 + 0.1 * self.consecutive_wins)

        position_size_pct = base_size

        return {
            'day': day,
            'trade_number': trade_num,
            'is_winner': is_winner,
            'return_pct': trade_return,
            'position_size_pct': position_size_pct,
            'pnl_pct': trade_return * position_size_pct
        }

    def simulate_trading_day(self, day: int, current_equity: float) -> Tuple[float, List[Dict], bool]:
        """Simulate a trading day with profit target monitoring"""
        daily_pnl = 0.0
        daily_trades = []
        profit_target_hit = False

        # Market regime effects (some days are more volatile)
        market_regime = random.choice(['normal', 'normal', 'normal', 'volatile', 'trending'])

        if market_regime == 'volatile':
            num_trades = max(3, int(random.gauss(self.avg_trades_per_day * 1.3, 3)))
        elif market_regime == 'trending':
            num_trades = max(2, int(random.gauss(self.avg_trades_per_day * 0.8, 2)))
        else:
            num_trades = max(1, int(random.gauss(self.avg_trades_per_day, 3)))

        num_trades = min(num_trades, 20)  # Cap at 20 trades per day

        for trade_num in range(num_trades):
            # Check profit target BEFORE each trade
            current_daily_pnl_pct = (daily_pnl / current_equity) * 100

            if current_daily_pnl_pct >= self.profit_target_pct:
                profit_target_hit = True
                self.profit_target_hits += 1
                print(f"Day {day:3d}: 🎯 PROFIT TARGET HIT! {current_daily_pnl_pct:.2f}% >= {self.profit_target_pct}% - STOPPING TRADING")
                break

            # Daily loss limit check
            if current_daily_pnl_pct <= -3.0:  # 3% daily loss limit
                print(f"Day {day:3d}: 🛑 DAILY LOSS LIMIT HIT! {current_daily_pnl_pct:.2f}% - STOPPING TRADING")
                break

            # Simulate the trade
            trade = self.simulate_high_volatility_trade(day, trade_num + 1)

            # Apply market regime effects
            if market_regime == 'volatile':
                trade['pnl_pct'] *= random.uniform(0.8, 1.4)
            elif market_regime == 'trending':
                if trade['is_winner']:
                    trade['pnl_pct'] *= random.uniform(1.1, 1.3)

            # Calculate P&L
            trade_pnl = current_equity * trade['pnl_pct']
            daily_pnl += trade_pnl

            # Update trade record
            trade['pnl_dollars'] = trade_pnl
            trade['cumulative_daily_pnl'] = daily_pnl
            trade['market_regime'] = market_regime

            daily_trades.append(trade)
            self.trade_log.append(trade)
            self.total_trades += 1

            # Update streaks
            if trade['is_winner']:
                self.winning_trades += 1
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)
            else:
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)

        if profit_target_hit:
            self.profit_target_days.append(day)

        return daily_pnl, daily_trades, profit_target_hit

    def run_simulation(self) -> Dict:
        """Run the enhanced simulation"""
        print(f"Enhanced Monte Carlo Trading Simulation - {self.scenario.upper()} Strategy")
        print(f"Parameters:")
        print(f"  Starting Capital: ${self.starting_capital:,.0f}")
        print(f"  Profit Target: {self.profit_target_pct}%")
        print(f"  Win Rate: {self.win_rate:.1%}")
        print(f"  Avg Trades/Day: {self.avg_trades_per_day}")
        print(f"  Position Size Range: {self.position_size_range[0]:.1%} - {self.position_size_range[1]:.1%}")
        print()

        current_equity = self.starting_capital
        self.equity_curve.append(current_equity)

        for day in range(1, self.trading_days + 1):
            daily_pnl, daily_trades, profit_target_hit = self.simulate_trading_day(day, current_equity)

            current_equity += daily_pnl
            self.equity_curve.append(current_equity)

            daily_pnl_pct = (daily_pnl / (current_equity - daily_pnl)) * 100

            day_result = {
                'day': day,
                'starting_equity': current_equity - daily_pnl,
                'ending_equity': current_equity,
                'daily_pnl': daily_pnl,
                'daily_pnl_pct': daily_pnl_pct,
                'num_trades': len(daily_trades),
                'profit_target_hit': profit_target_hit
            }

            self.daily_results.append(day_result)

            # Progress reporting
            if day % 50 == 0 or profit_target_hit or abs(daily_pnl_pct) > 4:
                status = ""
                if profit_target_hit:
                    status = "🎯 TARGET HIT!"
                elif daily_pnl_pct > 4:
                    status = "🚀 BIG WIN!"
                elif daily_pnl_pct < -2:
                    status = "📉 BIG LOSS"

                print(f"Day {day:3d}: ${current_equity:8,.0f} | Daily P&L: ${daily_pnl:+7,.0f} ({daily_pnl_pct:+5.2f}%) | Trades: {len(daily_trades)} {status}")

        return self.calculate_results()

    def calculate_results(self) -> Dict:
        """Calculate comprehensive results"""
        total_return = (self.equity_curve[-1] - self.starting_capital) / self.starting_capital
        daily_returns = np.array([day['daily_pnl_pct'] for day in self.daily_results])

        # Risk metrics
        winning_days = len([day for day in self.daily_results if day['daily_pnl'] > 0])
        sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)) if np.std(daily_returns) > 0 else 0

        # Drawdown calculation
        equity_series = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity_series)
        drawdown = (peak - equity_series) / peak
        max_drawdown = np.max(drawdown)

        # Profit target analysis
        profit_target_rate = len(self.profit_target_days) / self.trading_days * 100
        target_day_profits = [self.daily_results[day-1]['daily_pnl_pct'] for day in self.profit_target_days]
        avg_target_day_profit = np.mean(target_day_profits) if target_day_profits else 0

        return {
            'scenario': self.scenario,
            'final_equity': self.equity_curve[-1],
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'profit_target_hits': len(self.profit_target_days),
            'profit_target_rate_pct': profit_target_rate,
            'avg_profit_on_target_days': avg_target_day_profit,
            'total_trades': self.total_trades,
            'win_rate': (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0,
            'winning_days': winning_days,
            'winning_days_rate': (winning_days / len(self.daily_results)) * 100,
            'best_day': np.max(daily_returns),
            'worst_day': np.min(daily_returns),
            'volatility': np.std(daily_returns),
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses
        }

async def run_all_scenarios():
    """Run simulations for all scenarios"""
    print("ENHANCED MONTE CARLO TRADING SIMULATIONS")
    print("=" * 60)
    print("Testing 5.75% profit target system across different trading styles")
    print()

    scenarios = ['conservative', 'aggressive', 'very_aggressive']
    all_results = {}

    for scenario in scenarios:
        print(f"\n{'=' * 20} {scenario.upper()} SCENARIO {'=' * 20}")

        simulator = EnhancedTradingBotSimulation(scenario)
        results = simulator.run_simulation()
        all_results[scenario] = results

        # Display results
        print(f"\n{scenario.upper()} SCENARIO RESULTS:")
        print(f"Final Equity: ${results['final_equity']:,.0f}")
        print(f"Total Return: {results['total_return_pct']:+.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"Profit Target Hits: {results['profit_target_hits']} days ({results['profit_target_rate_pct']:.1f}%)")
        print(f"Avg Profit on Target Days: {results['avg_profit_on_target_days']:+.2f}%")
        print(f"Total Trades: {results['total_trades']:,}")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Best Day: {results['best_day']:+.2f}%")
        print(f"Worst Day: {results['worst_day']:+.2f}%")

    # Comparative analysis
    print("\n" + "=" * 60)
    print("COMPARATIVE ANALYSIS - PROFIT TARGET SYSTEM EFFECTIVENESS")
    print("=" * 60)

    for scenario, results in all_results.items():
        print(f"\n{scenario.upper()}:")
        print(f"  Profit Target Hit Rate: {results['profit_target_rate_pct']:.1f}%")
        print(f"  Average Target Day Profit: {results['avg_profit_on_target_days']:+.2f}%")
        print(f"  Risk-Adjusted Return: {results['sharpe_ratio']:.2f}")

        if results['profit_target_hits'] > 0:
            protected_profit = results['profit_target_hits'] * results['avg_profit_on_target_days']
            print(f"  Estimated Protected Profit: {protected_profit:.1f}%")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'enhanced_simulation_results_{timestamp}.json', 'w') as f:
        # Convert numpy types for JSON serialization
        json_results = {}
        for scenario, results in all_results.items():
            json_results[scenario] = {k: float(v) if isinstance(v, np.floating) else v
                                    for k, v in results.items()}
        json.dump(json_results, f, indent=2)

    print(f"\nDetailed results saved to: enhanced_simulation_results_{timestamp}.json")

    return all_results

if __name__ == "__main__":
    results = asyncio.run(run_all_scenarios())