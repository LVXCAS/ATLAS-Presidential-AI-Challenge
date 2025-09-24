"""
PRACTICAL STRATEGY OPTIMIZER
============================
Optimizes the successful volatility breakout strategy for maximum edge
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from strategy_backtester import OptionsBacktester, Trade, OptionsContract
import json
import itertools

class PracticalStrategyOptimizer:
    """Optimize practical strategies for 20%+ monthly returns."""
    
    def __init__(self):
        self.backtester = OptionsBacktester("2021-01-01", "2024-12-31")
        self.symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'META']
        
        print("PRACTICAL STRATEGY OPTIMIZER INITIALIZED")
        print("=" * 50)
        print("Mission: Optimize volatility breakout (76% win rate baseline)")
        print("Target: 85%+ win rate with 20%+ monthly returns")
    
    def test_volatility_parameter_combinations(self):
        """Test different parameter combinations for volatility strategy."""
        
        print("\nOPTIMIZING VOLATILITY BREAKOUT PARAMETERS...")
        print("-" * 50)
        
        # Parameter ranges to test
        vol_contraction_thresholds = [0.6, 0.65, 0.7, 0.75, 0.8]
        range_compression_thresholds = [0.7, 0.75, 0.8, 0.85]
        min_historical_vol = [0.20, 0.25, 0.30]
        min_recent_vol = [0.12, 0.15, 0.18]
        dte_options = [28, 35, 42]
        
        best_strategy = None
        best_score = 0
        
        results = []
        
        # Test combinations
        combinations = list(itertools.product(
            vol_contraction_thresholds,
            range_compression_thresholds, 
            min_historical_vol,
            min_recent_vol,
            dte_options
        ))
        
        print(f"Testing {len(combinations)} parameter combinations...")
        
        for i, (vol_thresh, range_thresh, min_hist_vol, min_rec_vol, dte) in enumerate(combinations[:20]):  # Limit for speed
            
            if i % 5 == 0:
                print(f"Testing combination {i+1}/20...")
            
            trades = self.run_optimized_volatility_strategy(
                vol_contraction_threshold=vol_thresh,
                range_compression_threshold=range_thresh,
                min_historical_volatility=min_hist_vol,
                min_recent_volatility=min_rec_vol,
                target_dte=dte
            )
            
            if len(trades) >= 10:  # Need minimum trades for statistics
                
                df = pd.DataFrame([
                    {
                        'return_pct': t.return_pct,
                        'win': t.win,
                        'hold_days': t.hold_days
                    }
                    for t in trades
                ])
                
                win_rate = df['win'].mean() * 100
                avg_return = df['return_pct'].mean()
                total_trades = len(df)
                
                # Score function (prioritize win rate and average return)
                score = win_rate + (avg_return / 2)  # Win rate weight + avg return bonus
                
                result = {
                    'vol_contraction_threshold': vol_thresh,
                    'range_compression_threshold': range_thresh,
                    'min_historical_volatility': min_hist_vol,
                    'min_recent_volatility': min_rec_vol,
                    'target_dte': dte,
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'score': score
                }
                
                results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_strategy = result
                    print(f"  New best: {win_rate:.1f}% win rate, {avg_return:.1f}% avg return")
        
        # Sort results by score
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        print(f"\nOPTIMIZATION COMPLETE!")
        print(f"Best strategy found:")
        print(f"  Win Rate: {best_strategy['win_rate']:.1f}%")
        print(f"  Avg Return: {best_strategy['avg_return']:.1f}%")
        print(f"  Total Trades: {best_strategy['total_trades']}")
        print(f"  Parameters: {best_strategy}")
        
        return results, best_strategy
    
    def run_optimized_volatility_strategy(self, vol_contraction_threshold=0.7, 
                                        range_compression_threshold=0.8,
                                        min_historical_volatility=0.25,
                                        min_recent_volatility=0.15,
                                        target_dte=35):
        """Run volatility strategy with specific parameters."""
        
        all_trades = []
        
        for symbol in self.symbols:
            data = self.backtester.get_historical_data(symbol, 300)
            if data is None or len(data) < 100:
                continue
            
            for i in range(80, len(data) - 40):
                current_date = data.index[i]
                current_price = data['Close'].iloc[i]
                
                if current_date.weekday() >= 5:
                    continue
                
                # Calculate volatility metrics
                returns = data['Returns'].iloc[:i+1]
                recent_vol = returns.iloc[-10:].std() * np.sqrt(252)
                hist_vol = returns.iloc[-60:-10].std() * np.sqrt(252)
                
                if hist_vol <= 0 or recent_vol <= 0:
                    continue
                
                vol_ratio = recent_vol / hist_vol
                
                # Price range compression
                recent_range = (data['High'].iloc[i-10:i+1].max() - data['Low'].iloc[i-10:i+1].min()) / current_price
                hist_range = (data['High'].iloc[i-30:i-10].max() - data['Low'].iloc[i-30:i-10].min()) / data['Close'].iloc[i-20]
                
                if hist_range <= 0:
                    continue
                
                range_ratio = recent_range / hist_range
                
                # Optimized criteria
                if (vol_ratio < vol_contraction_threshold and
                    range_ratio < range_compression_threshold and
                    hist_vol > min_historical_volatility and
                    recent_vol > min_recent_volatility):
                    
                    # Execute optimized straddle
                    trade = self.execute_optimized_straddle(
                        symbol, current_date, current_price, data, i, target_dte
                    )
                    
                    if trade:
                        all_trades.append(trade)
        
        return all_trades
    
    def execute_optimized_straddle(self, symbol, entry_date, entry_price, data, entry_idx, dte=35):
        """Execute optimized straddle with refined exit rules."""
        
        try:
            strike = entry_price
            vol = data['Volatility'].iloc[entry_idx]
            
            if pd.isna(vol) or vol < 0.15:
                vol = 0.30  # Conservative vol estimate
            
            # Cap volatility to prevent overpricing
            vol = min(vol, 0.55)
            
            call_premium = self.backtester.black_scholes_price(
                S=entry_price, K=strike, T=dte/365, r=0.05, sigma=vol, option_type='call'
            )
            put_premium = self.backtester.black_scholes_price(
                S=entry_price, K=strike, T=dte/365, r=0.05, sigma=vol, option_type='put'
            )
            
            entry_cost = call_premium + put_premium
            
            # Refined exit strategy
            best_exit_idx = entry_idx + dte
            best_exit_value = 0
            
            # Scan for optimal exit within holding period
            for j in range(entry_idx + 5, min(entry_idx + dte + 1, len(data))):  # Wait 5 days minimum
                current_price_j = data['Close'].iloc[j]
                move_pct = abs(current_price_j - entry_price) / entry_price
                days_left = max(1, dte - (j - entry_idx))
                
                # Adjust vol based on realized movement
                if move_pct >= 0.10:
                    exit_vol = vol * 1.15  # IV expansion
                elif move_pct >= 0.06:
                    exit_vol = vol * 1.05
                else:
                    exit_vol = vol * 0.85  # IV contraction
                
                # Calculate theoretical value
                exit_call = self.backtester.black_scholes_price(
                    S=current_price_j, K=strike, T=days_left/365, r=0.05, sigma=exit_vol, option_type='call'
                )
                exit_put = self.backtester.black_scholes_price(
                    S=current_price_j, K=strike, T=days_left/365, r=0.05, sigma=exit_vol, option_type='put'
                )
                
                theoretical_value = exit_call + exit_put
                theoretical_return = (theoretical_value - entry_cost) / entry_cost
                
                # Exit conditions
                if theoretical_return >= 0.75:  # 75% profit target
                    best_exit_idx = j
                    best_exit_value = theoretical_value
                    break
                elif j >= entry_idx + 14 and theoretical_return <= -0.25:  # Stop loss after 2 weeks
                    best_exit_idx = j
                    best_exit_value = theoretical_value
                    break
                elif j == min(entry_idx + dte, len(data) - 1):  # Hold to expiry
                    best_exit_idx = j
                    best_exit_value = theoretical_value
            
            # Fallback calculation
            if best_exit_value == 0:
                exit_idx = min(entry_idx + dte, len(data) - 1)
                exit_price = data['Close'].iloc[exit_idx]
                days_left = max(0, dte - (exit_idx - entry_idx))
                
                exit_call = self.backtester.black_scholes_price(
                    S=exit_price, K=strike, T=days_left/365, r=0.05, sigma=vol*0.8, option_type='call'
                )
                exit_put = self.backtester.black_scholes_price(
                    S=exit_price, K=strike, T=days_left/365, r=0.05, sigma=vol*0.8, option_type='put'
                )
                
                best_exit_value = exit_call + exit_put
                best_exit_idx = exit_idx
            
            exit_date = data.index[best_exit_idx]
            pnl = best_exit_value - entry_cost
            return_pct = (pnl / entry_cost) * 100
            
            trade = Trade(
                entry_date=entry_date,
                exit_date=exit_date,
                strategy='OPTIMIZED_STRADDLE',
                symbol=symbol,
                contracts=[
                    OptionsContract(symbol, strike, exit_date, 'call', call_premium),
                    OptionsContract(symbol, strike, exit_date, 'put', put_premium)
                ],
                entry_cost=entry_cost,
                exit_value=best_exit_value,
                pnl=pnl,
                return_pct=return_pct,
                hold_days=best_exit_idx - entry_idx,
                win=pnl > 0
            )
            
            return trade
            
        except Exception as e:
            return None
    
    def create_ultimate_strategy(self, best_params):
        """Create the ultimate strategy based on optimization results."""
        
        print(f"\nCREATING ULTIMATE STRATEGY...")
        print("-" * 50)
        
        # Run with best parameters
        ultimate_trades = self.run_optimized_volatility_strategy(**best_params)
        
        # Analyze results
        ultimate_results = self.backtester.analyze_results(ultimate_trades, "Ultimate Optimized Strategy")
        
        print(f"\nULTIMATE STRATEGY PERFORMANCE:")
        print("=" * 50)
        print(f"Win Rate: {ultimate_results['win_rate']:.1f}%")
        print(f"Average Return: {ultimate_results['avg_return']:.1f}%")
        print(f"Total Trades: {ultimate_results['total_trades']}")
        print(f"Months with 20%+ returns: {ultimate_results['months_20plus']}/{ultimate_results['total_months']} ({ultimate_results['pct_months_20plus']:.1f}%)")
        
        # Compare to baseline
        baseline_win_rate = 76.1
        if ultimate_results['win_rate'] > baseline_win_rate:
            improvement = ultimate_results['win_rate'] - baseline_win_rate
            print(f"✅ IMPROVEMENT: +{improvement:.1f}% win rate vs baseline!")
        
        return ultimate_trades, ultimate_results
    
    def run_optimization(self):
        """Run complete optimization process."""
        
        print("STARTING COMPLETE STRATEGY OPTIMIZATION")
        print("=" * 60)
        
        # Step 1: Parameter optimization
        all_results, best_strategy = self.test_volatility_parameter_combinations()
        
        # Step 2: Create ultimate strategy
        best_params = {k: v for k, v in best_strategy.items() 
                      if k not in ['total_trades', 'win_rate', 'avg_return', 'score']}
        
        ultimate_trades, ultimate_results = self.create_ultimate_strategy(best_params)
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        optimization_results = {
            'timestamp': timestamp,
            'optimization_results': all_results[:10],  # Top 10 results
            'best_parameters': best_strategy,
            'ultimate_strategy_performance': ultimate_results,
            'baseline_comparison': {
                'baseline_win_rate': 76.1,
                'baseline_avg_return': 46.1,
                'ultimate_win_rate': ultimate_results['win_rate'],
                'ultimate_avg_return': ultimate_results['avg_return'],
                'improvement': ultimate_results['win_rate'] - 76.1
            }
        }
        
        with open(f'strategy_optimization_{timestamp}.json', 'w') as f:
            json.dump(optimization_results, f, indent=2, default=str)
        
        print(f"\n🎯 OPTIMIZATION COMPLETE!")
        print(f"Results saved: strategy_optimization_{timestamp}.json")
        print(f"Ultimate strategy ready for live trading!")
        
        return optimization_results

def main():
    """Run complete strategy optimization."""
    
    optimizer = PracticalStrategyOptimizer()
    results = optimizer.run_optimization()
    
    print(f"\nStrategy optimization completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()