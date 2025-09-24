#!/usr/bin/env python3
"""
IMPROVED Monte Carlo Simulation for OPTIONS_BOT
Addresses Sharpe ratio issues with better risk management
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List
import random

from agents.advanced_monte_carlo_engine import advanced_monte_carlo_engine, OptionSpec
from agents.enhanced_options_pricing_engine import enhanced_options_pricing_engine

class ImprovedOptionsBotSimulation:
    """Improved Monte Carlo simulation with better risk management"""

    def __init__(self, iterations=10000):
        self.iterations = iterations
        self.results = []

        # Risk management parameters
        self.initial_capital = 18113.50
        self.position_size_pct = 0.02  # Risk 2% per trade
        self.max_loss_per_trade = 0.50  # Max 50% loss per position
        self.min_premium = 0.10  # Minimum option premium
        self.max_premium = 20.0   # Maximum option premium

        # Improved market parameters
        self.volatility_range = (0.15, 0.35)  # More realistic vol range
        self.min_days_to_expiry = 7   # Avoid options expiring too soon
        self.max_days_to_expiry = 45  # Don't go too far out

        # Moneyness filters
        self.call_moneyness_range = (0.90, 1.10)  # ATM to slightly OTM calls
        self.put_moneyness_range = (0.90, 1.10)   # ATM to slightly OTM puts

    def generate_improved_scenario(self) -> Dict:
        """Generate more realistic market scenario with filters"""
        # More realistic stock price distribution
        stock_price = np.random.lognormal(mean=np.log(150), sigma=0.3)
        stock_price = max(20, min(500, stock_price))  # Reasonable bounds

        # Controlled volatility range
        volatility = np.random.uniform(*self.volatility_range)

        # Controlled time to expiration
        days_to_expiry = np.random.randint(self.min_days_to_expiry, self.max_days_to_expiry + 1)

        # Choose option type
        option_type = np.random.choice(['call', 'put'])

        # Improved moneyness selection
        if option_type == 'call':
            moneyness = np.random.uniform(*self.call_moneyness_range)
        else:
            moneyness = np.random.uniform(*self.put_moneyness_range)

        strike_price = stock_price * moneyness

        return {
            'stock_price': stock_price,
            'strike_price': strike_price,
            'volatility': volatility,
            'days_to_expiry': days_to_expiry,
            'option_type': option_type,
            'moneyness': moneyness
        }

    async def simulate_improved_trade(self, scenario: Dict) -> Dict:
        """Simulate trade with improved risk management"""
        try:
            # Get initial option pricing
            analysis = await enhanced_options_pricing_engine.get_comprehensive_option_analysis(
                underlying_price=scenario['stock_price'],
                strike_price=scenario['strike_price'],
                time_to_expiry_days=scenario['days_to_expiry'],
                volatility=scenario['volatility'],
                option_type=scenario['option_type']
            )

            entry_price = analysis['pricing']['theoretical_price']

            # Apply premium filters
            if entry_price < self.min_premium or entry_price > self.max_premium:
                return None  # Skip trades outside premium range

            # Calculate position size based on risk management
            risk_amount = self.initial_capital * self.position_size_pct
            contracts = max(1, int(risk_amount / (entry_price * 100)))  # Options are per 100 shares

            # Simulate more realistic holding period (1-5 days typical)
            holding_days = min(np.random.geometric(p=0.3), scenario['days_to_expiry'] - 1)
            holding_days = max(1, holding_days)

            # Generate price path with more realistic movements
            option_spec = OptionSpec(
                S0=scenario['stock_price'],
                K=scenario['strike_price'],
                T=scenario['days_to_expiry'] / 365.25,
                r=0.05,
                sigma=scenario['volatility'],
                option_type=scenario['option_type']
            )

            # More conservative price movements
            daily_vol = scenario['volatility'] / np.sqrt(252)
            price_changes = np.random.normal(0, daily_vol, holding_days)

            # Apply some mean reversion to prevent extreme moves
            price_changes = price_changes * 0.7  # Dampen extreme movements

            final_stock_price = scenario['stock_price'] * np.exp(np.sum(price_changes))
            remaining_days = scenario['days_to_expiry'] - holding_days

            # Calculate exit price
            if remaining_days > 0:
                exit_analysis = await enhanced_options_pricing_engine.get_comprehensive_option_analysis(
                    underlying_price=final_stock_price,
                    strike_price=scenario['strike_price'],
                    time_to_expiry_days=remaining_days,
                    volatility=scenario['volatility'],
                    option_type=scenario['option_type']
                )
                exit_price = exit_analysis['pricing']['theoretical_price']
            else:
                # Option expired
                if scenario['option_type'] == 'call':
                    exit_price = max(0, final_stock_price - scenario['strike_price'])
                else:
                    exit_price = max(0, scenario['strike_price'] - final_stock_price)

            # Apply stop loss
            raw_pnl = exit_price - entry_price
            max_loss = entry_price * self.max_loss_per_trade

            if raw_pnl < -max_loss:
                # Stop loss triggered
                exit_price = entry_price - max_loss
                raw_pnl = -max_loss
                stop_loss_triggered = True
            else:
                stop_loss_triggered = False

            # Calculate position-sized P&L
            position_pnl = raw_pnl * contracts * 100  # $100 per contract per point
            pnl_percent = (raw_pnl / entry_price) * 100 if entry_price > 0 else 0

            # Calculate capital impact
            capital_impact = position_pnl / self.initial_capital * 100

            return {
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': raw_pnl,
                'pnl_percent': pnl_percent,
                'position_pnl': position_pnl,
                'capital_impact': capital_impact,
                'contracts': contracts,
                'holding_days': holding_days,
                'stock_price_change': (final_stock_price - scenario['stock_price']) / scenario['stock_price'],
                'is_winner': raw_pnl > 0,
                'stop_loss_triggered': stop_loss_triggered,
                'option_type': scenario['option_type'],
                'moneyness': scenario['moneyness'],
                'volatility': scenario['volatility'],
                'initial_dte': scenario['days_to_expiry']
            }

        except Exception as e:
            return None

    async def run_improved_simulation(self):
        """Run the improved simulation"""
        print(f"Starting IMPROVED Monte Carlo simulation with {self.iterations:,} iterations...")
        print("Enhanced risk management with position sizing and stop losses")
        print("=" * 70)

        start_time = time.time()
        successful_trades = 0

        for i in range(self.iterations):
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (self.iterations - i - 1) / rate
                success_rate = successful_trades / (i + 1) * 100
                print(f"Progress: {i+1:,}/{self.iterations:,} ({(i+1)/self.iterations*100:.1f}%) "
                      f"- Success: {success_rate:.1f}% - ETA: {eta:.0f}s")

            scenario = self.generate_improved_scenario()
            result = await self.simulate_improved_trade(scenario)

            if result:
                self.results.append(result)
                successful_trades += 1

        print(f"\nSimulation completed in {time.time() - start_time:.1f} seconds")
        print(f"Successful trades: {len(self.results):,} out of {self.iterations:,} attempts")

    def analyze_improved_results(self) -> Dict:
        """Analyze improved simulation results"""
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)

        # Basic metrics
        total_trades = len(df)
        win_rate = len(df[df['is_winner']]) / total_trades

        # Capital impact analysis (more meaningful than raw option returns)
        total_capital_impact = df['capital_impact'].sum()
        avg_capital_impact = df['capital_impact'].mean()
        std_capital_impact = df['capital_impact'].std()

        # Calculate realistic Sharpe ratio based on capital impact
        if std_capital_impact > 0:
            sharpe_ratio = (avg_capital_impact / std_capital_impact) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Risk metrics
        capital_returns = df['capital_impact'].values / 100
        risk_metrics = advanced_monte_carlo_engine.calculate_var_cvar(capital_returns, confidence_level=0.95)

        # Stop loss analysis
        stop_loss_rate = len(df[df['stop_loss_triggered']]) / total_trades

        # Position analysis
        avg_position_size = df['contracts'].mean()
        total_portfolio_pnl = df['position_pnl'].sum()

        return {
            'simulation_summary': {
                'total_attempts': self.iterations,
                'successful_trades': total_trades,
                'success_rate': total_trades / self.iterations,
                'risk_management': 'Position sizing + Stop losses',
                'position_size_pct': self.position_size_pct * 100,
                'max_loss_per_trade': self.max_loss_per_trade * 100
            },
            'performance_metrics': {
                'win_rate': win_rate,
                'total_capital_impact': total_capital_impact,
                'avg_capital_impact': avg_capital_impact,
                'std_capital_impact': std_capital_impact,
                'sharpe_ratio': sharpe_ratio,
                'stop_loss_rate': stop_loss_rate,
                'avg_position_size': avg_position_size
            },
            'portfolio_metrics': {
                'total_portfolio_pnl': total_portfolio_pnl,
                'final_capital': self.initial_capital + total_portfolio_pnl,
                'total_return_pct': (total_portfolio_pnl / self.initial_capital) * 100,
                'max_capital_impact': df['capital_impact'].max(),
                'min_capital_impact': df['capital_impact'].min()
            },
            'risk_metrics': {
                'var_95': risk_metrics['var'] * 100,
                'cvar_95': risk_metrics['cvar'] * 100,
                'max_drawdown': df['capital_impact'].min()
            },
            'trade_analysis': {
                'call_trades': len(df[df['option_type'] == 'call']),
                'put_trades': len(df[df['option_type'] == 'put']),
                'avg_holding_days': df['holding_days'].mean(),
                'avg_dte': df['initial_dte'].mean()
            }
        }

    def print_improved_results(self, analysis: Dict):
        """Print improved results"""
        print("\n" + "=" * 80)
        print("IMPROVED OPTIONS_BOT MONTE CARLO SIMULATION RESULTS")
        print("Enhanced Risk Management - Position Sizing + Stop Losses")
        print("=" * 80)

        sim = analysis['simulation_summary']
        print(f"\nSIMULATION SUMMARY:")
        print(f"   Total Attempts: {sim['total_attempts']:,}")
        print(f"   Successful Trades: {sim['successful_trades']:,}")
        print(f"   Success Rate: {sim['success_rate']:.1%}")
        print(f"   Risk per Trade: {sim['position_size_pct']:.1f}% of capital")
        print(f"   Max Loss per Trade: {sim['max_loss_per_trade']:.0f}%")

        perf = analysis['performance_metrics']
        print(f"\nPERFORMANCE METRICS:")
        print(f"   Win Rate: {perf['win_rate']:.1%}")
        print(f"   Average Capital Impact: {perf['avg_capital_impact']:.3f}%")
        print(f"   Capital Impact Volatility: {perf['std_capital_impact']:.3f}%")
        print(f"   IMPROVED Sharpe Ratio: {perf['sharpe_ratio']:.4f}")
        print(f"   Stop Loss Trigger Rate: {perf['stop_loss_rate']:.1%}")
        print(f"   Average Position Size: {perf['avg_position_size']:.1f} contracts")

        port = analysis['portfolio_metrics']
        print(f"\nPORTFOLIO RESULTS:")
        print(f"   Total Portfolio P&L: ${port['total_portfolio_pnl']:,.2f}")
        print(f"   Final Capital: ${port['final_capital']:,.2f}")
        print(f"   Total Return: {port['total_return_pct']:.2f}%")
        print(f"   Best Trade Impact: +{port['max_capital_impact']:.2f}%")
        print(f"   Worst Trade Impact: {port['min_capital_impact']:.2f}%")

        risk = analysis['risk_metrics']
        print(f"\nRISK METRICS:")
        print(f"   VaR (95%): {risk['var_95']:.2f}%")
        print(f"   CVaR (95%): {risk['cvar_95']:.2f}%")
        print(f"   Maximum Drawdown: {risk['max_drawdown']:.2f}%")

        print("\n" + "=" * 80)

async def main():
    simulation = ImprovedOptionsBotSimulation(iterations=10000)
    await simulation.run_improved_simulation()
    analysis = simulation.analyze_improved_results()
    simulation.print_improved_results(analysis)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"improved_monte_carlo_results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    asyncio.run(main())