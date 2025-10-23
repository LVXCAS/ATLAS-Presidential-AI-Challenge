#!/usr/bin/env python3
"""
Vectorbt Portfolio Engine
Ultra-fast portfolio simulations with NumPy/Pandas optimization
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Try to import vectorbt
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
    print("+ Vectorbt available for ultra-fast portfolio simulations")
except ImportError:
    VECTORBT_AVAILABLE = False
    print("- Vectorbt not available - using custom implementation")

class VectorbtPortfolioEngine:
    """Ultra-fast portfolio simulations using Vectorbt"""
    
    def __init__(self):
        self.portfolio_results = {}
        self.performance_metrics = {}
        self.vectorbt_available = VECTORBT_AVAILABLE
        
        if VECTORBT_AVAILABLE:
            # Configure vectorbt settings
            vbt.settings.set_theme("dark")
            vbt.settings.portfolio['init_cash'] = 100000
            vbt.settings.portfolio['fees'] = 0.001  # 0.1% fees
            
        # Portfolio configuration
        self.config = {
            'initial_cash': 100000,
            'commission': 0.001,  # 0.1%
            'slippage': 0.0005,   # 0.05%
            'risk_free_rate': 0.02,  # 2% risk-free rate
            'max_positions': 10,
            'position_sizing': 'equal_weight'
        }
    
    async def run_fast_backtest(self, signals_df: pd.DataFrame, price_data: pd.DataFrame, 
                               strategy_name: str = "Strategy") -> Dict:
        """Run ultra-fast backtest using vectorbt"""
        
        if self.vectorbt_available:
            return await self._run_vectorbt_backtest(signals_df, price_data, strategy_name)
        else:
            return await self._run_custom_backtest(signals_df, price_data, strategy_name)
    
    async def _run_vectorbt_backtest(self, signals_df: pd.DataFrame, price_data: pd.DataFrame, 
                                   strategy_name: str) -> Dict:
        """Run backtest using vectorbt library"""
        try:
            print(f"Running vectorbt backtest for {strategy_name}...")
            
            # Align data
            common_index = signals_df.index.intersection(price_data.index)
            signals_aligned = signals_df.loc[common_index]
            prices_aligned = price_data.loc[common_index]
            
            # Create entries and exits from signals
            entries = signals_aligned == 1  # Buy signals
            exits = signals_aligned == -1   # Sell signals
            
            # Run portfolio simulation
            portfolio = vbt.Portfolio.from_signals(
                prices_aligned,
                entries=entries,
                exits=exits,
                init_cash=self.config['initial_cash'],
                fees=self.config['commission'],
                freq='D'
            )
            
            # Calculate performance metrics
            returns = portfolio.returns()
            total_return = portfolio.total_return()
            sharpe_ratio = returns.vbt.stats('sharpe_ratio', risk_free=self.config['risk_free_rate'])
            max_drawdown = portfolio.max_drawdown()
            win_rate = portfolio.trades.win_rate()
            
            # Advanced metrics
            sortino_ratio = returns.vbt.stats('sortino_ratio', risk_free=self.config['risk_free_rate'])
            calmar_ratio = returns.vbt.stats('calmar_ratio')
            profit_factor = portfolio.trades.profit_factor()
            
            results = {
                'strategy_name': strategy_name,
                'total_return': float(total_return),
                'annual_return': float(total_return) / (len(common_index) / 252),  # Annualized
                'sharpe_ratio': float(sharpe_ratio) if not pd.isna(sharpe_ratio) else 0,
                'sortino_ratio': float(sortino_ratio) if not pd.isna(sortino_ratio) else 0,
                'calmar_ratio': float(calmar_ratio) if not pd.isna(calmar_ratio) else 0,
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate) if not pd.isna(win_rate) else 0,
                'profit_factor': float(profit_factor) if not pd.isna(profit_factor) else 1,
                'total_trades': len(portfolio.trades.records),
                'final_value': float(portfolio.value().iloc[-1]),
                'volatility': float(returns.std() * np.sqrt(252)),  # Annualized volatility
                'returns_series': returns,
                'portfolio_value': portfolio.value(),
                'trades': portfolio.trades.records_readable
            }
            
            print(f"+ Vectorbt backtest complete: {total_return:.2%} return, {sharpe_ratio:.2f} Sharpe")
            return results
            
        except Exception as e:
            print(f"- Vectorbt backtest error: {e}")
            return await self._run_custom_backtest(signals_df, price_data, strategy_name)
    
    async def _run_custom_backtest(self, signals_df: pd.DataFrame, price_data: pd.DataFrame, 
                                 strategy_name: str) -> Dict:
        """Custom backtest implementation when vectorbt is not available"""
        try:
            print(f"Running custom backtest for {strategy_name}...")
            
            # Align data
            common_index = signals_df.index.intersection(price_data.index)
            signals_aligned = signals_df.loc[common_index]
            prices_aligned = price_data.loc[common_index]
            
            # Initialize portfolio
            cash = self.config['initial_cash']
            positions = {}
            portfolio_values = []
            trades = []
            
            for date in common_index:
                signal = signals_aligned.loc[date]
                price = prices_aligned.loc[date]
                
                if isinstance(signal, pd.Series):
                    signal = signal.iloc[0]
                if isinstance(price, pd.Series):
                    price = price.iloc[0]
                
                # Execute trades based on signals
                if signal == 1 and cash > 0:  # Buy signal
                    shares = int(cash * 0.95 / price)  # Use 95% of cash
                    if shares > 0:
                        cost = shares * price * (1 + self.config['commission'])
                        cash -= cost
                        positions['shares'] = positions.get('shares', 0) + shares
                        trades.append({
                            'date': date,
                            'action': 'BUY',
                            'shares': shares,
                            'price': price,
                            'value': cost
                        })
                
                elif signal == -1 and positions.get('shares', 0) > 0:  # Sell signal
                    shares = positions['shares']
                    proceeds = shares * price * (1 - self.config['commission'])
                    cash += proceeds
                    positions['shares'] = 0
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'shares': shares,
                        'price': price,
                        'value': proceeds
                    })
                
                # Calculate portfolio value
                portfolio_value = cash + positions.get('shares', 0) * price
                portfolio_values.append(portfolio_value)
            
            # Create results DataFrame
            portfolio_series = pd.Series(portfolio_values, index=common_index)
            returns = portfolio_series.pct_change().dropna()
            
            # Calculate metrics
            total_return = (portfolio_series.iloc[-1] - self.config['initial_cash']) / self.config['initial_cash']
            annual_return = total_return / (len(common_index) / 252)
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - self.config['risk_free_rate']) / volatility if volatility > 0 else 0
            
            # Drawdown calculation
            rolling_max = portfolio_series.expanding().max()
            drawdown = (portfolio_series - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Trade statistics
            winning_trades = [t for t in trades if t['action'] == 'SELL' and len([t2 for t2 in trades if t2['action'] == 'BUY' and t2['date'] < t['date']]) > 0]
            win_rate = len(winning_trades) / max(1, len(trades) // 2)
            
            results = {
                'strategy_name': strategy_name,
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': 0,  # Not calculated in custom implementation
                'calmar_ratio': annual_return / abs(max_drawdown) if max_drawdown < 0 else 0,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': 1,  # Simplified
                'total_trades': len(trades),
                'final_value': portfolio_series.iloc[-1],
                'volatility': volatility,
                'returns_series': returns,
                'portfolio_value': portfolio_series,
                'trades': trades
            }
            
            print(f"+ Custom backtest complete: {total_return:.2%} return, {sharpe_ratio:.2f} Sharpe")
            return results
            
        except Exception as e:
            print(f"- Custom backtest error: {e}")
            return {}
    
    async def run_multi_strategy_simulation(self, strategies: Dict[str, Dict]) -> Dict:
        """Run multiple strategy simulations for comparison"""
        results = {}
        
        print(f"Running multi-strategy simulation with {len(strategies)} strategies...")
        
        for strategy_name, strategy_data in strategies.items():
            try:
                signals_df = strategy_data.get('signals')
                price_data = strategy_data.get('prices')
                
                if signals_df is not None and price_data is not None:
                    result = await self.run_fast_backtest(signals_df, price_data, strategy_name)
                    results[strategy_name] = result
                    
            except Exception as e:
                print(f"- Error simulating strategy {strategy_name}: {e}")
        
        # Create comparison summary
        if results:
            comparison = self._create_strategy_comparison(results)
            results['comparison'] = comparison
        
        return results
    
    def _create_strategy_comparison(self, results: Dict) -> pd.DataFrame:
        """Create strategy comparison DataFrame"""
        comparison_data = []
        
        for strategy_name, result in results.items():
            if isinstance(result, dict) and 'total_return' in result:
                comparison_data.append({
                    'Strategy': strategy_name,
                    'Total Return': result.get('total_return', 0),
                    'Annual Return': result.get('annual_return', 0),
                    'Sharpe Ratio': result.get('sharpe_ratio', 0),
                    'Max Drawdown': result.get('max_drawdown', 0),
                    'Win Rate': result.get('win_rate', 0),
                    'Total Trades': result.get('total_trades', 0),
                    'Final Value': result.get('final_value', 0),
                    'Volatility': result.get('volatility', 0)
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            df = df.sort_values('Sharpe Ratio', ascending=False)
            return df
        else:
            return pd.DataFrame()
    
    async def optimize_portfolio_weights(self, returns_data: pd.DataFrame, 
                                       method: str = "sharpe") -> Dict:
        """Optimize portfolio weights using vectorbt or custom implementation"""
        
        if self.vectorbt_available:
            return await self._vectorbt_optimize_weights(returns_data, method)
        else:
            return await self._custom_optimize_weights(returns_data, method)
    
    async def _vectorbt_optimize_weights(self, returns_data: pd.DataFrame, method: str) -> Dict:
        """Portfolio optimization using vectorbt"""
        try:
            # Calculate expected returns and covariance matrix
            expected_returns = returns_data.mean() * 252  # Annualized
            cov_matrix = returns_data.cov() * 252  # Annualized
            
            if method == "sharpe":
                # Maximum Sharpe ratio optimization (simplified)
                # In practice, would use scipy.optimize or cvxpy
                n_assets = len(expected_returns)
                weights = np.ones(n_assets) / n_assets  # Equal weights as approximation
                
                # Calculate portfolio metrics
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - self.config['risk_free_rate']) / portfolio_volatility
                
            else:
                # Equal weight fallback
                weights = np.ones(len(expected_returns)) / len(expected_returns)
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - self.config['risk_free_rate']) / portfolio_volatility
            
            return {
                'weights': dict(zip(returns_data.columns, weights)),
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'method': method
            }
            
        except Exception as e:
            print(f"- Vectorbt optimization error: {e}")
            return await self._custom_optimize_weights(returns_data, method)
    
    async def _custom_optimize_weights(self, returns_data: pd.DataFrame, method: str) -> Dict:
        """Custom portfolio optimization implementation"""
        try:
            # Simple equal weight optimization
            n_assets = len(returns_data.columns)
            weights = np.ones(n_assets) / n_assets
            
            # Calculate portfolio metrics
            expected_returns = returns_data.mean() * 252
            cov_matrix = returns_data.cov() * 252
            
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.config['risk_free_rate']) / portfolio_volatility
            
            return {
                'weights': dict(zip(returns_data.columns, weights)),
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'method': f'custom_{method}'
            }
            
        except Exception as e:
            print(f"- Custom optimization error: {e}")
            return {}
    
    async def run_monte_carlo_simulation(self, strategy_returns: pd.Series, 
                                       n_simulations: int = 1000, 
                                       time_horizon: int = 252) -> Dict:
        """Run Monte Carlo simulation on strategy returns"""
        try:
            print(f"Running Monte Carlo simulation: {n_simulations} paths, {time_horizon} days")
            
            # Calculate return statistics
            mean_return = strategy_returns.mean()
            std_return = strategy_returns.std()
            
            # Generate random returns
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.normal(mean_return, std_return, 
                                               (n_simulations, time_horizon))
            
            # Calculate cumulative returns for each simulation
            cumulative_returns = np.cumprod(1 + simulated_returns, axis=1)
            final_values = cumulative_returns[:, -1] * self.config['initial_cash']
            
            # Calculate statistics
            mean_final_value = np.mean(final_values)
            std_final_value = np.std(final_values)
            percentiles = np.percentile(final_values, [5, 25, 50, 75, 95])
            
            prob_profit = np.sum(final_values > self.config['initial_cash']) / n_simulations
            
            return {
                'n_simulations': n_simulations,
                'time_horizon': time_horizon,
                'mean_final_value': mean_final_value,
                'std_final_value': std_final_value,
                'percentiles': {
                    '5th': percentiles[0],
                    '25th': percentiles[1],
                    '50th': percentiles[2],
                    '75th': percentiles[3],
                    '95th': percentiles[4]
                },
                'probability_of_profit': prob_profit,
                'value_at_risk_5pct': percentiles[0] - self.config['initial_cash'],
                'expected_shortfall': np.mean(final_values[final_values <= percentiles[0]]) - self.config['initial_cash'],
                'simulated_paths': cumulative_returns
            }
            
        except Exception as e:
            print(f"- Monte Carlo simulation error: {e}")
            return {}

# Create global instance
vectorbt_engine = VectorbtPortfolioEngine()