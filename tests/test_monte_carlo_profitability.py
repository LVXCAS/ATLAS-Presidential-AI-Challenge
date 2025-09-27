#!/usr/bin/env python3
"""
Monte Carlo Profitability Test for Options Trading Bot
Tests bot performance across thousands of simulated market scenarios
"""

import asyncio
import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import random
# import matplotlib.pyplot as plt
# import seaborn as sns

# Add current directory to path
sys.path.append('.')

from agents.options_trading_agent import OptionsTrader, OptionsStrategy
from agents.position_manager import PositionManager
from agents.risk_management import RiskManager
from agents.quantlib_pricing import quantlib_pricer

class MarketSimulator:
    """Simulates realistic market conditions for backtesting"""
    
    def __init__(self):
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'SPY']
        
    def generate_market_scenario(self) -> Dict:
        """Generate a random but realistic market scenario"""
        
        # Random market regime
        regimes = ['trending_up', 'trending_down', 'sideways', 'volatile', 'crash', 'recovery']
        regime = random.choice(regimes)
        
        symbol = random.choice(self.symbols)
        base_price = random.uniform(150.0, 300.0)
        
        if regime == 'trending_up':
            price_change = random.uniform(2.0, 8.0)
            volatility = random.uniform(0.15, 0.25)
            volume_multiplier = random.uniform(1.2, 2.0)
            rsi = random.uniform(55.0, 80.0)
        elif regime == 'trending_down':
            price_change = random.uniform(-8.0, -2.0)
            volatility = random.uniform(0.20, 0.35)
            volume_multiplier = random.uniform(1.5, 3.0)
            rsi = random.uniform(20.0, 45.0)
        elif regime == 'sideways':
            price_change = random.uniform(-1.5, 1.5)
            volatility = random.uniform(0.12, 0.20)
            volume_multiplier = random.uniform(0.8, 1.2)
            rsi = random.uniform(45.0, 55.0)
        elif regime == 'volatile':
            price_change = random.uniform(-5.0, 5.0)
            volatility = random.uniform(0.30, 0.50)
            volume_multiplier = random.uniform(2.0, 4.0)
            rsi = random.uniform(30.0, 70.0)
        elif regime == 'crash':
            price_change = random.uniform(-15.0, -5.0)
            volatility = random.uniform(0.40, 0.80)
            volume_multiplier = random.uniform(3.0, 6.0)
            rsi = random.uniform(10.0, 30.0)
        else:  # recovery
            price_change = random.uniform(5.0, 15.0)
            volatility = random.uniform(0.25, 0.40)
            volume_multiplier = random.uniform(2.0, 4.0)
            rsi = random.uniform(60.0, 85.0)
        
        return {
            'symbol': symbol,
            'regime': regime,
            'base_price': base_price,
            'price_change_pct': price_change,
            'volatility': volatility,
            'volume_multiplier': volume_multiplier,
            'rsi': rsi,
            'current_price': base_price * (1 + price_change / 100)
        }
    
    def simulate_price_path(self, scenario: Dict, days: int = 30) -> List[float]:
        """Simulate realistic price movement over time"""
        
        start_price = scenario['current_price']
        volatility = scenario['volatility']
        drift = scenario['price_change_pct'] / 100 / 252  # Daily drift
        
        prices = [start_price]
        
        for i in range(days):
            # Geometric Brownian Motion with some mean reversion
            random_shock = np.random.normal(0, volatility / np.sqrt(252))
            mean_reversion = 0.001 * (start_price - prices[-1]) / start_price
            
            daily_return = drift + random_shock + mean_reversion
            new_price = prices[-1] * (1 + daily_return)
            
            # Add some realistic bounds
            new_price = max(new_price, start_price * 0.5)  # No more than 50% crash
            new_price = min(new_price, start_price * 2.0)   # No more than 100% gain
            
            prices.append(new_price)
        
        return prices[1:]  # Return without initial price

class MonteCarloTester:
    """Runs comprehensive Monte Carlo testing"""
    
    def __init__(self, num_simulations: int = 1000):
        self.num_simulations = num_simulations
        self.market_simulator = MarketSimulator()
        self.results = []
        
    async def run_single_simulation(self, simulation_id: int) -> Dict:
        """Run a single trading simulation"""
        
        try:
            # Generate market scenario
            scenario = self.market_simulator.generate_market_scenario()
            
            # Initialize bot components
            trader = OptionsTrader(None)
            
            # Simulate bot decision making
            strategy_result = trader.find_best_options_strategy(
                scenario['symbol'],
                scenario['current_price'],
                scenario['volatility'] * 100,  # Convert to percentage
                scenario['rsi'],
                scenario['price_change_pct'] / 100  # Convert to decimal
            )
            
            if not strategy_result:
                return {
                    'simulation_id': simulation_id,
                    'symbol': scenario['symbol'],
                    'regime': scenario['regime'],
                    'strategy': 'NO_TRADE',
                    'entry_price': scenario['current_price'],
                    'pnl': 0.0,
                    'pnl_pct': 0.0,
                    'max_loss': 0.0,
                    'max_gain': 0.0,
                    'trade_duration': 0,
                    'reason': 'No suitable strategy found'
                }
            
            strategy, contracts = strategy_result
            
            if not contracts:
                return {
                    'simulation_id': simulation_id,
                    'symbol': scenario['symbol'],
                    'regime': scenario['regime'],
                    'strategy': 'NO_TRADE',
                    'entry_price': scenario['current_price'],
                    'pnl': 0.0,
                    'pnl_pct': 0.0,
                    'max_loss': 0.0,
                    'max_gain': 0.0,
                    'trade_duration': 0,
                    'reason': 'No contracts available'
                }
            
            # Simulate trade execution and price movement
            entry_price = scenario['current_price']
            trade_days = random.randint(1, 21)  # 1-21 days trade duration
            
            # Generate price path
            price_path = self.market_simulator.simulate_price_path(scenario, trade_days)
            
            # Calculate theoretical P&L using QuantLib
            pnl_path = []
            entry_cost = 0.0
            
            for day, current_price in enumerate(price_path):
                daily_pnl = 0.0
                
                for contract in contracts:
                    if day == 0:  # Entry day
                        if quantlib_pricer:
                            entry_pricing = quantlib_pricer.price_european_option(
                                contract.option_type,
                                entry_price,
                                contract.strike,
                                contract.expiration,
                                scenario['symbol'],
                                scenario['volatility']
                            )
                            contract_cost = entry_pricing['price'] * 100  # Per contract
                            if strategy in [OptionsStrategy.LONG_CALL, OptionsStrategy.LONG_PUT]:
                                entry_cost += contract_cost
                            elif strategy == OptionsStrategy.BULL_CALL_SPREAD:
                                # Long lower strike, short higher strike
                                if contract.strike == min(c.strike for c in contracts):
                                    entry_cost += contract_cost  # Buy
                                else:
                                    entry_cost -= contract_cost  # Sell
                            elif strategy == OptionsStrategy.BEAR_PUT_SPREAD:
                                # Long higher strike, short lower strike  
                                if contract.strike == max(c.strike for c in contracts):
                                    entry_cost += contract_cost  # Buy
                                else:
                                    entry_cost -= contract_cost  # Sell
                            elif strategy == OptionsStrategy.STRADDLE:
                                entry_cost += contract_cost  # Buy both call and put
                    
                    # Calculate current value
                    if quantlib_pricer:
                        current_pricing = quantlib_pricer.price_european_option(
                            contract.option_type,
                            current_price,
                            contract.strike,
                            contract.expiration - timedelta(days=day),
                            scenario['symbol'],
                            scenario['volatility']
                        )
                        current_value = current_pricing['price'] * 100
                        
                        if strategy in [OptionsStrategy.LONG_CALL, OptionsStrategy.LONG_PUT, OptionsStrategy.STRADDLE]:
                            daily_pnl += current_value
                        elif strategy == OptionsStrategy.BULL_CALL_SPREAD:
                            if contract.strike == min(c.strike for c in contracts):
                                daily_pnl += current_value  # Long position
                            else:
                                daily_pnl -= current_value  # Short position
                        elif strategy == OptionsStrategy.BEAR_PUT_SPREAD:
                            if contract.strike == max(c.strike for c in contracts):
                                daily_pnl += current_value  # Long position
                            else:
                                daily_pnl -= current_value  # Short position
                
                net_pnl = daily_pnl - entry_cost
                pnl_path.append(net_pnl)
            
            # Calculate final metrics
            if pnl_path:
                final_pnl = pnl_path[-1]
                max_gain = max(pnl_path)
                max_loss = min(pnl_path)
                pnl_pct = (final_pnl / max(abs(entry_cost), 1)) * 100
            else:
                final_pnl = -abs(entry_cost)  # Total loss if no pricing available
                max_gain = 0.0
                max_loss = final_pnl
                pnl_pct = -100.0
            
            return {
                'simulation_id': simulation_id,
                'symbol': scenario['symbol'],
                'regime': scenario['regime'],
                'strategy': strategy.name if hasattr(strategy, 'name') else str(strategy),
                'entry_price': entry_price,
                'final_price': price_path[-1] if price_path else entry_price,
                'pnl': final_pnl,
                'pnl_pct': pnl_pct,
                'max_loss': max_loss,
                'max_gain': max_gain,
                'trade_duration': trade_days,
                'entry_cost': entry_cost,
                'volatility': scenario['volatility'],
                'rsi': scenario['rsi'],
                'reason': 'Completed'
            }
            
        except Exception as e:
            return {
                'simulation_id': simulation_id,
                'symbol': scenario.get('symbol', 'UNKNOWN'),
                'regime': scenario.get('regime', 'UNKNOWN'),
                'strategy': 'ERROR',
                'entry_price': 0.0,
                'pnl': -1000.0,  # Assume significant loss for errors
                'pnl_pct': -100.0,
                'max_loss': -1000.0,
                'max_gain': 0.0,
                'trade_duration': 0,
                'reason': f'Error: {str(e)}'
            }
    
    async def run_monte_carlo_test(self) -> pd.DataFrame:
        """Run full Monte Carlo simulation"""
        
        print(f"MONTE CARLO PROFITABILITY TEST")
        print(f"=" * 60)
        print(f"Running {self.num_simulations} simulations...")
        print(f"Testing bot performance across diverse market conditions")
        print(f"=" * 60)
        
        # Run simulations in batches to avoid memory issues
        batch_size = 50
        all_results = []
        
        for batch_start in range(0, self.num_simulations, batch_size):
            batch_end = min(batch_start + batch_size, self.num_simulations)
            batch_tasks = []
            
            print(f"Processing simulations {batch_start + 1}-{batch_end}...")
            
            for sim_id in range(batch_start, batch_end):
                task = self.run_single_simulation(sim_id + 1)
                batch_tasks.append(task)
            
            # Run batch
            batch_results = await asyncio.gather(*batch_tasks)
            all_results.extend(batch_results)
            
            # Progress update
            progress = (batch_end / self.num_simulations) * 100
            print(f"Progress: {progress:.1f}% complete")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_results)
        self.results = df
        
        return df
    
    def analyze_results(self) -> Dict:
        """Analyze Monte Carlo results"""
        
        if self.results is None or len(self.results) == 0:
            return {"error": "No results to analyze"}
        
        df = self.results
        
        # Filter out error cases and no-trades for P&L analysis
        trading_results = df[df['strategy'] != 'NO_TRADE']
        successful_trades = trading_results[trading_results['strategy'] != 'ERROR']
        
        # Overall statistics
        total_simulations = len(df)
        trading_opportunities = len(trading_results)
        successful_executions = len(successful_trades)
        
        # P&L Analysis (only successful trades)
        if len(successful_trades) > 0:
            winning_trades = successful_trades[successful_trades['pnl'] > 0]
            losing_trades = successful_trades[successful_trades['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(successful_trades) * 100
            
            avg_pnl = successful_trades['pnl'].mean()
            median_pnl = successful_trades['pnl'].median()
            std_pnl = successful_trades['pnl'].std()
            
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            
            max_win = successful_trades['pnl'].max()
            max_loss = successful_trades['pnl'].min()
            
            # Percentage returns
            avg_return_pct = successful_trades['pnl_pct'].mean()
            median_return_pct = successful_trades['pnl_pct'].median()
            
            # Risk metrics
            sharpe_ratio = avg_return_pct / (successful_trades['pnl_pct'].std() + 1e-6) if successful_trades['pnl_pct'].std() > 0 else 0
            
            # Drawdown analysis
            max_drawdown = successful_trades['max_loss'].min()
            
        else:
            win_rate = 0
            avg_pnl = median_pnl = std_pnl = 0
            avg_win = avg_loss = max_win = max_loss = 0
            avg_return_pct = median_return_pct = sharpe_ratio = max_drawdown = 0
        
        # Strategy performance
        strategy_performance = {}
        if len(successful_trades) > 0:
            strategy_stats = successful_trades.groupby('strategy').agg({
                'pnl': ['count', 'mean', 'std'],
                'pnl_pct': 'mean'
            }).round(2)
            
            for strategy in successful_trades['strategy'].unique():
                strategy_data = successful_trades[successful_trades['strategy'] == strategy]
                strategy_wins = len(strategy_data[strategy_data['pnl'] > 0])
                strategy_total = len(strategy_data)
                strategy_win_rate = strategy_wins / strategy_total * 100 if strategy_total > 0 else 0
                
                strategy_performance[strategy] = {
                    'trades': strategy_total,
                    'win_rate': strategy_win_rate,
                    'avg_pnl': strategy_data['pnl'].mean(),
                    'avg_return_pct': strategy_data['pnl_pct'].mean()
                }
        
        # Market regime performance
        regime_performance = {}
        if len(successful_trades) > 0:
            for regime in successful_trades['regime'].unique():
                regime_data = successful_trades[successful_trades['regime'] == regime]
                regime_wins = len(regime_data[regime_data['pnl'] > 0])
                regime_total = len(regime_data)
                regime_win_rate = regime_wins / regime_total * 100 if regime_total > 0 else 0
                
                regime_performance[regime] = {
                    'trades': regime_total,
                    'win_rate': regime_win_rate,
                    'avg_pnl': regime_data['pnl'].mean(),
                    'avg_return_pct': regime_data['pnl_pct'].mean()
                }
        
        return {
            'overview': {
                'total_simulations': total_simulations,
                'trading_opportunities': trading_opportunities,
                'successful_executions': successful_executions,
                'execution_rate': (successful_executions / total_simulations * 100) if total_simulations > 0 else 0,
                'trading_opportunity_rate': (trading_opportunities / total_simulations * 100) if total_simulations > 0 else 0
            },
            'profitability': {
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'median_pnl': median_pnl,
                'std_pnl': std_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'max_win': max_win,
                'max_loss': max_loss,
                'avg_return_pct': avg_return_pct,
                'median_return_pct': median_return_pct,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            },
            'strategy_performance': strategy_performance,
            'regime_performance': regime_performance
        }
    
    def print_results(self, analysis: Dict):
        """Print formatted results"""
        
        print(f"\nMONTE CARLO TEST RESULTS")
        print(f"=" * 60)
        
        # Overview
        overview = analysis['overview']
        print(f"EXECUTION SUMMARY:")
        print(f"  Total Simulations: {overview['total_simulations']:,}")
        print(f"  Trading Opportunities: {overview['trading_opportunities']:,} ({overview['trading_opportunity_rate']:.1f}%)")
        print(f"  Successful Executions: {overview['successful_executions']:,} ({overview['execution_rate']:.1f}%)")
        
        # Profitability
        profit = analysis['profitability']
        print(f"\nPROFITABILITY ANALYSIS:")
        print(f"  Win Rate: {profit['win_rate']:.1f}%")
        print(f"  Average P&L: ${profit['avg_pnl']:.2f}")
        print(f"  Median P&L: ${profit['median_pnl']:.2f}")
        print(f"  Average Return: {profit['avg_return_pct']:.1f}%")
        print(f"  Median Return: {profit['median_return_pct']:.1f}%")
        print(f"  Sharpe Ratio: {profit['sharpe_ratio']:.2f}")
        print(f"  Best Trade: ${profit['max_win']:.2f}")
        print(f"  Worst Trade: ${profit['max_loss']:.2f}")
        print(f"  Max Drawdown: ${profit['max_drawdown']:.2f}")
        
        # Strategy Performance
        if analysis['strategy_performance']:
            print(f"\nSTRATEGY PERFORMANCE:")
            for strategy, stats in analysis['strategy_performance'].items():
                print(f"  {strategy}:")
                print(f"    Trades: {stats['trades']}")
                print(f"    Win Rate: {stats['win_rate']:.1f}%")
                print(f"    Avg P&L: ${stats['avg_pnl']:.2f}")
                print(f"    Avg Return: {stats['avg_return_pct']:.1f}%")
        
        # Market Regime Performance  
        if analysis['regime_performance']:
            print(f"\nMARKET REGIME PERFORMANCE:")
            for regime, stats in analysis['regime_performance'].items():
                print(f"  {regime.upper()}:")
                print(f"    Trades: {stats['trades']}")
                print(f"    Win Rate: {stats['win_rate']:.1f}%")
                print(f"    Avg P&L: ${stats['avg_pnl']:.2f}")
                print(f"    Avg Return: {stats['avg_return_pct']:.1f}%")
        
        # Overall Assessment
        print(f"\nOVERALL ASSESSMENT:")
        if profit['win_rate'] >= 55 and profit['avg_return_pct'] > 5:
            print(f"  [EXCELLENT] High win rate and strong returns!")
        elif profit['win_rate'] >= 50 and profit['avg_return_pct'] > 0:
            print(f"  [PROFITABLE] Bot shows positive expected returns")
        elif profit['win_rate'] >= 45:
            print(f"  [MARGINAL] Borderline profitability, needs optimization")
        else:
            print(f"  [UNPROFITABLE] Bot needs significant improvements")
        
        print(f"=" * 60)

async def main():
    """Run Monte Carlo profitability test"""
    
    # Run with 500 simulations for thorough testing
    tester = MonteCarloTester(num_simulations=500)
    
    # Run the test
    results_df = await tester.run_monte_carlo_test()
    
    # Analyze results
    analysis = tester.analyze_results()
    
    # Print results
    tester.print_results(analysis)
    
    # Save detailed results
    results_df.to_csv('monte_carlo_results.csv', index=False)
    print(f"\nDetailed results saved to: monte_carlo_results.csv")
    
    return analysis

if __name__ == "__main__":
    asyncio.run(main())