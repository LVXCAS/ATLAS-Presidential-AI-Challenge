#!/usr/bin/env python3
"""
Monte Carlo Simulation for OPTIONS_BOT Performance Analysis
10,000 iterations using the enhanced Monte Carlo engine
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

class OptionsBotMonteCarloSimulation:
    """Monte Carlo simulation for OPTIONS_BOT trading performance"""

    def __init__(self, iterations=10000):
        self.iterations = iterations
        self.results = []
        self.portfolio_results = []

        # Trading parameters based on OPTIONS_BOT configuration
        self.initial_capital = 18113.50  # Current account value
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
        self.option_types = ['call', 'put']

        # Market parameters for simulation
        self.base_volatility = 0.25
        self.risk_free_rate = 0.05
        self.trading_days = 252

    def generate_market_scenario(self) -> Dict:
        """Generate random market scenario for simulation"""
        # Random stock price (realistic range)
        stock_price = np.random.uniform(50, 300)

        # Random volatility (market conditions)
        volatility = np.random.uniform(0.15, 0.50)

        # Random time to expiration (14-45 days as per bot settings)
        days_to_expiry = np.random.randint(14, 46)

        # Random strike relative to stock price
        moneyness = np.random.uniform(0.85, 1.15)  # 85% to 115% moneyness
        strike_price = stock_price * moneyness

        # Random option type
        option_type = np.random.choice(self.option_types)

        return {
            'stock_price': stock_price,
            'strike_price': strike_price,
            'volatility': volatility,
            'days_to_expiry': days_to_expiry,
            'option_type': option_type
        }

    async def simulate_single_trade(self, scenario: Dict) -> Dict:
        """Simulate a single options trade using enhanced pricing engine"""
        try:
            # Use the enhanced options pricing engine for realistic pricing
            analysis = await enhanced_options_pricing_engine.get_comprehensive_option_analysis(
                underlying_price=scenario['stock_price'],
                strike_price=scenario['strike_price'],
                time_to_expiry_days=scenario['days_to_expiry'],
                volatility=scenario['volatility'],
                option_type=scenario['option_type']
            )

            entry_price = analysis['pricing']['theoretical_price']
            entry_delta = analysis['greeks']['delta']
            entry_theta = analysis['greeks']['theta']
            entry_vega = analysis['greeks']['vega']

            # Simulate price movement over holding period
            holding_days = np.random.randint(1, min(10, scenario['days_to_expiry']))

            # Generate stock price path using geometric Brownian motion
            option_spec = OptionSpec(
                S0=scenario['stock_price'],
                K=scenario['strike_price'],
                T=scenario['days_to_expiry'] / 365.25,
                r=self.risk_free_rate,
                sigma=scenario['volatility'],
                option_type=scenario['option_type']
            )

            # Use Monte Carlo engine for price path simulation
            price_paths = advanced_monte_carlo_engine.generate_price_paths(
                S0=scenario['stock_price'],
                r=self.risk_free_rate,
                sigma=scenario['volatility'],
                T=holding_days / 365.25,
                steps=holding_days,
                paths=1
            )

            final_stock_price = price_paths[0, -1]
            remaining_days = scenario['days_to_expiry'] - holding_days

            # Calculate exit price if still time remaining
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
                # Option expired - calculate intrinsic value
                if scenario['option_type'] == 'call':
                    exit_price = max(0, final_stock_price - scenario['strike_price'])
                else:
                    exit_price = max(0, scenario['strike_price'] - final_stock_price)

            # Calculate P&L
            pnl = exit_price - entry_price
            pnl_percent = (pnl / entry_price) * 100 if entry_price > 0 else 0

            # Determine if trade was profitable (win/loss)
            is_winner = pnl > 0

            return {
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'holding_days': holding_days,
                'stock_price_change': (final_stock_price - scenario['stock_price']) / scenario['stock_price'],
                'is_winner': is_winner,
                'entry_delta': entry_delta,
                'entry_theta': entry_theta,
                'entry_vega': entry_vega,
                'option_type': scenario['option_type'],
                'moneyness': scenario['strike_price'] / scenario['stock_price'],
                'volatility': scenario['volatility'],
                'initial_dte': scenario['days_to_expiry']
            }

        except Exception as e:
            print(f"Error in trade simulation: {e}")
            return None

    async def run_simulation(self):
        """Run the full Monte Carlo simulation"""
        print(f"Starting Monte Carlo simulation with {self.iterations:,} iterations...")
        print("Using OPTIONS_BOT Enhanced Monte Carlo Engine")
        print("=" * 60)

        start_time = time.time()

        for i in range(self.iterations):
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (self.iterations - i - 1) / rate
                print(f"Progress: {i+1:,}/{self.iterations:,} ({(i+1)/self.iterations*100:.1f}%) "
                      f"- Rate: {rate:.0f}/sec - ETA: {eta:.0f}s")

            # Generate market scenario
            scenario = self.generate_market_scenario()

            # Simulate trade
            result = await self.simulate_single_trade(scenario)

            if result:
                self.results.append(result)

        print(f"\nSimulation completed in {time.time() - start_time:.1f} seconds")
        print(f"Successfully simulated {len(self.results):,} trades")

    def analyze_results(self) -> Dict:
        """Analyze simulation results and generate comprehensive statistics"""
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)

        # Basic performance metrics
        total_trades = len(df)
        winning_trades = len(df[df['is_winner'] == True])
        losing_trades = len(df[df['is_winner'] == False])
        win_rate = winning_trades / total_trades

        # P&L statistics
        total_pnl = df['pnl'].sum()
        avg_pnl = df['pnl'].mean()
        median_pnl = df['pnl'].median()
        std_pnl = df['pnl'].std()

        # Win/Loss analysis
        avg_winner = df[df['is_winner'] == True]['pnl'].mean() if winning_trades > 0 else 0
        avg_loser = df[df['is_winner'] == False]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_winner * winning_trades / (avg_loser * losing_trades)) if losing_trades > 0 and avg_loser != 0 else float('inf')

        # Percentage returns
        avg_return_pct = df['pnl_percent'].mean()
        median_return_pct = df['pnl_percent'].median()
        std_return_pct = df['pnl_percent'].std()

        # Risk metrics
        positive_returns = df[df['pnl_percent'] > 0]['pnl_percent']
        negative_returns = df[df['pnl_percent'] < 0]['pnl_percent']

        # Calculate VaR and CVaR using our Monte Carlo engine
        returns_array = df['pnl_percent'].values / 100  # Convert to decimal
        risk_metrics = advanced_monte_carlo_engine.calculate_var_cvar(returns_array, confidence_level=0.95)

        # Sharpe ratio (annualized)
        excess_return = avg_return_pct / 100  # Daily excess return
        sharpe_ratio = excess_return / (std_return_pct / 100) * np.sqrt(252) if std_return_pct > 0 else 0

        # Option type analysis
        call_trades = df[df['option_type'] == 'call']
        put_trades = df[df['option_type'] == 'put']

        # Moneyness analysis
        itm_trades = df[((df['option_type'] == 'call') & (df['moneyness'] < 1)) |
                       ((df['option_type'] == 'put') & (df['moneyness'] > 1))]
        otm_trades = df[((df['option_type'] == 'call') & (df['moneyness'] > 1)) |
                       ((df['option_type'] == 'put') & (df['moneyness'] < 1))]

        analysis = {
            'simulation_summary': {
                'total_iterations': self.iterations,
                'successful_trades': total_trades,
                'initial_capital': self.initial_capital,
                'engine_used': 'Enhanced Monte Carlo Engine'
            },
            'performance_metrics': {
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'average_pnl': avg_pnl,
                'median_pnl': median_pnl,
                'std_pnl': std_pnl,
                'average_winner': avg_winner,
                'average_loser': avg_loser,
                'profit_factor': profit_factor
            },
            'return_metrics': {
                'average_return_pct': avg_return_pct,
                'median_return_pct': median_return_pct,
                'std_return_pct': std_return_pct,
                'sharpe_ratio': sharpe_ratio,
                'best_trade_pct': df['pnl_percent'].max(),
                'worst_trade_pct': df['pnl_percent'].min()
            },
            'risk_metrics': {
                'var_95': risk_metrics['var'] * 100,  # Convert back to percentage
                'cvar_95': risk_metrics['cvar'] * 100,
                'worst_return': risk_metrics['worst_return'] * 100,
                'best_return': risk_metrics['best_return'] * 100,
                'max_drawdown': df['pnl_percent'].min()
            },
            'trade_analysis': {
                'call_trades': len(call_trades),
                'put_trades': len(put_trades),
                'call_win_rate': len(call_trades[call_trades['is_winner']]) / len(call_trades) if len(call_trades) > 0 else 0,
                'put_win_rate': len(put_trades[put_trades['is_winner']]) / len(put_trades) if len(put_trades) > 0 else 0,
                'itm_trades': len(itm_trades),
                'otm_trades': len(otm_trades),
                'itm_win_rate': len(itm_trades[itm_trades['is_winner']]) / len(itm_trades) if len(itm_trades) > 0 else 0,
                'otm_win_rate': len(otm_trades[otm_trades['is_winner']]) / len(otm_trades) if len(otm_trades) > 0 else 0
            },
            'holding_period': {
                'avg_holding_days': df['holding_days'].mean(),
                'median_holding_days': df['holding_days'].median(),
                'min_holding_days': df['holding_days'].min(),
                'max_holding_days': df['holding_days'].max()
            },
            'portfolio_projection': {
                'final_capital': self.initial_capital + total_pnl,
                'total_return_pct': (total_pnl / self.initial_capital) * 100,
                'annualized_return': ((self.initial_capital + total_pnl) / self.initial_capital) ** (252 / df['holding_days'].mean()) - 1
            }
        }

        return analysis

    def print_results(self, analysis: Dict):
        """Print comprehensive results in a formatted manner"""
        print("\n" + "=" * 80)
        print("OPTIONS_BOT MONTE CARLO SIMULATION RESULTS")
        print("Enhanced Monte Carlo Engine - 10,000 Iterations")
        print("=" * 80)

        # Simulation Summary
        sim = analysis['simulation_summary']
        print(f"\nSIMULATION SUMMARY:")
        print(f"   Total Iterations: {sim['total_iterations']:,}")
        print(f"   Successful Trades: {sim['successful_trades']:,}")
        print(f"   Initial Capital: ${sim['initial_capital']:,.2f}")
        print(f"   Engine: {sim['engine_used']}")

        # Performance Metrics
        perf = analysis['performance_metrics']
        print(f"\nPERFORMANCE METRICS:")
        print(f"   Win Rate: {perf['win_rate']:.1%}")
        print(f"   Total P&L: ${perf['total_pnl']:,.2f}")
        print(f"   Average P&L per Trade: ${perf['average_pnl']:,.4f}")
        print(f"   Median P&L per Trade: ${perf['median_pnl']:,.4f}")
        print(f"   Average Winner: ${perf['average_winner']:,.4f}")
        print(f"   Average Loser: ${perf['average_loser']:,.4f}")
        print(f"   Profit Factor: {perf['profit_factor']:.2f}")

        # Return Metrics
        ret = analysis['return_metrics']
        print(f"\nRETURN METRICS:")
        print(f"   Average Return: {ret['average_return_pct']:,.2f}%")
        print(f"   Median Return: {ret['median_return_pct']:,.2f}%")
        print(f"   Return Volatility: {ret['std_return_pct']:,.2f}%")
        print(f"   Sharpe Ratio: {ret['sharpe_ratio']:.3f}")
        print(f"   Best Trade: +{ret['best_trade_pct']:,.2f}%")
        print(f"   Worst Trade: {ret['worst_trade_pct']:,.2f}%")

        # Risk Metrics
        risk = analysis['risk_metrics']
        print(f"\nRISK METRICS:")
        print(f"   VaR (95%): {risk['var_95']:,.2f}%")
        print(f"   CVaR (95%): {risk['cvar_95']:,.2f}%")
        print(f"   Maximum Drawdown: {risk['max_drawdown']:,.2f}%")

        # Trade Analysis
        trade = analysis['trade_analysis']
        print(f"\nTRADE ANALYSIS:")
        print(f"   Call Trades: {trade['call_trades']:,} (Win Rate: {trade['call_win_rate']:.1%})")
        print(f"   Put Trades: {trade['put_trades']:,} (Win Rate: {trade['put_win_rate']:.1%})")
        print(f"   ITM Trades: {trade['itm_trades']:,} (Win Rate: {trade['itm_win_rate']:.1%})")
        print(f"   OTM Trades: {trade['otm_trades']:,} (Win Rate: {trade['otm_win_rate']:.1%})")

        # Portfolio Projection
        port = analysis['portfolio_projection']
        print(f"\nPORTFOLIO PROJECTION:")
        print(f"   Final Capital: ${port['final_capital']:,.2f}")
        print(f"   Total Return: {port['total_return_pct']:,.2f}%")
        print(f"   Annualized Return: {port['annualized_return']:.1%}")

        print("\n" + "=" * 80)

async def main():
    """Run the Monte Carlo simulation"""

    # Initialize simulation
    simulation = OptionsBotMonteCarloSimulation(iterations=10000)

    # Run simulation
    await simulation.run_simulation()

    # Analyze results
    analysis = simulation.analyze_results()

    # Print results
    simulation.print_results(analysis)

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"options_bot_monte_carlo_results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    # Also save raw trade data
    if simulation.results:
        df = pd.DataFrame(simulation.results)
        csv_file = f"options_bot_trade_data_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Raw trade data saved to: {csv_file}")

if __name__ == "__main__":
    asyncio.run(main())