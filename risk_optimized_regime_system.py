"""
RISK-OPTIMIZED REGIME SYSTEM
============================
Day 3 - Optimize risk parameters for our breakthrough regime system
to maximize returns while controlling downside risk.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class RiskOptimizedRegimeSystem:
    """Optimized risk management for regime-based trading"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        
        # Our breakthrough regime allocations
        self.regime_allocations = {
            "Bull_Low_Vol": {'asset': 'JPM', 'accuracy': 0.695, 'base_allocation': 0.8},
            "Bull_High_Vol": {'asset': 'XOM', 'accuracy': 0.589, 'base_allocation': 0.6}, 
            "Bear_High_Vol": {'asset': 'WMT', 'accuracy': 0.569, 'base_allocation': 0.5},
            "Bear_Low_Vol": {'asset': 'JNJ', 'accuracy': 0.486, 'base_allocation': 0.2}  # Minimal allocation
        }
        
        print("DAY 3 - RISK-OPTIMIZED REGIME SYSTEM")
        print("=" * 50)
        print("Optimizing our breakthrough regime system:")
        print("  • Bull_Low_Vol -> JPM (69.5% accuracy)")
        print("  • Bull_High_Vol -> XOM (58.9% accuracy)")
        print("  • Bear_High_Vol -> WMT (56.9% accuracy)")
        print("  • Bear_Low_Vol -> JNJ (48.6% accuracy)")
        print(f"\nOptimizing risk parameters for maximum returns")
        print(f"Starting capital: ${initial_capital:,.0f}")
        print("=" * 50)
    
    def get_regime_data(self):
        """Get market regime data"""
        print("\nGETTING REGIME AND ASSET DATA...")
        
        try:
            # Get SPY for regime detection
            spy_data = yf.download('SPY', period='2y', progress=False)
            spy_data.index = spy_data.index.tz_localize(None)
            
            if isinstance(spy_data.columns, pd.MultiIndex):
                spy = spy_data.droplevel(1, axis=1)
            else:
                spy = spy_data
            
            # Detect regimes (simplified from previous analysis)
            spy_close = spy['Close']
            sma_50 = spy_close.rolling(50).mean()
            sma_200 = spy_close.rolling(200).mean()
            is_bull = (spy_close > sma_200) & (sma_50 > sma_200)
            
            volatility = spy_close.pct_change().rolling(20).std() * np.sqrt(252)
            vol_median = volatility.median()
            is_high_vol = volatility > vol_median
            
            # Create regime labels
            regimes = []
            bull_array = is_bull.fillna(False).values
            vol_array = is_high_vol.fillna(False).values
            
            for i in range(len(spy)):
                bull = bull_array[i]
                high_vol = vol_array[i]
                
                if bull and not high_vol:
                    regimes.append("Bull_Low_Vol")
                elif bull and high_vol:
                    regimes.append("Bull_High_Vol")
                elif not bull and not high_vol:
                    regimes.append("Bear_Low_Vol")
                else:
                    regimes.append("Bear_High_Vol")
            
            # Create regime dataframe
            regime_df = pd.DataFrame({
                'SPY_Close': spy_close,
                'Regime': regimes,
                'Volatility': volatility
            }, index=spy.index)
            
            print(f"   Regime data: {len(regime_df)} days")
            
            # Get asset price data
            asset_data = {}
            for regime_info in self.regime_allocations.values():
                asset = regime_info['asset']
                if asset not in asset_data:
                    print(f"   Getting {asset} data...")
                    ticker_data = yf.download(asset, period='2y', progress=False)
                    ticker_data.index = ticker_data.index.tz_localize(None)
                    
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        asset_data[asset] = ticker_data.droplevel(1, axis=1)
                    else:
                        asset_data[asset] = ticker_data
            
            return regime_df, asset_data
            
        except Exception as e:
            print(f"   ERROR: {e}")
            return None, None
    
    def optimize_position_sizing(self, regime_info):
        """Optimize position sizing for each regime"""
        
        accuracy = regime_info['accuracy']
        base_allocation = regime_info['base_allocation']
        
        # Kelly Criterion with modifications
        win_rate = accuracy
        loss_rate = 1 - accuracy
        
        # Assume average win is slightly larger than average loss (1.2:1 ratio)
        win_loss_ratio = 1.2
        
        # Kelly fraction
        kelly_fraction = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        kelly_fraction = max(0, min(kelly_fraction, 0.5))  # Cap at 50%
        
        # Combine with base allocation and confidence
        confidence_multiplier = (accuracy - 0.5) * 2  # Scale excess accuracy
        confidence_multiplier = max(0.1, min(confidence_multiplier, 1.0))
        
        # Final position size
        position_size = base_allocation * kelly_fraction * confidence_multiplier
        position_size = max(0.05, min(position_size, 0.8))  # 5-80% range
        
        return {
            'position_size': position_size,
            'kelly_fraction': kelly_fraction,
            'confidence_multiplier': confidence_multiplier,
            'win_rate': win_rate
        }
    
    def backtest_optimized_regime_system(self, regime_df, asset_data):
        """Backtest the optimized regime system"""
        print("\nBACKTESTING OPTIMIZED REGIME SYSTEM...")
        
        portfolio_value = self.initial_capital
        positions = []
        trades = []
        daily_returns = []
        
        # Use last 6 months for out-of-sample testing
        test_start = len(regime_df) - 126  # ~6 months
        test_regime_df = regime_df.iloc[test_start:]
        
        print(f"   Test period: {len(test_regime_df)} days")
        print(f"   Starting portfolio: ${portfolio_value:,.0f}")
        
        current_position = None
        current_asset = None
        entry_price = None
        entry_date = None
        
        for i, (date, row) in enumerate(test_regime_df.iterrows()):
            current_regime = row['Regime']
            current_vol = row['Volatility']
            
            # Get recommended allocation for this regime
            if current_regime not in self.regime_allocations:
                continue
                
            regime_info = self.regime_allocations[current_regime]
            recommended_asset = regime_info['asset']
            
            # Get asset price
            if recommended_asset not in asset_data:
                continue
                
            asset_prices = asset_data[recommended_asset]
            if date not in asset_prices.index:
                continue
                
            current_price = asset_prices.loc[date, 'Close']
            
            # Check if we need to change position
            if current_asset != recommended_asset:
                
                # Close existing position if any
                if current_position is not None:
                    # Calculate return
                    position_return = (current_price - entry_price) / entry_price
                    if current_asset != recommended_asset:
                        # Different asset, use SPY as proxy for exit
                        spy_price = regime_df.loc[date, 'SPY_Close']
                        spy_entry_price = regime_df.loc[entry_date, 'SPY_Close'] if entry_date in regime_df.index else spy_price
                        position_return = (spy_price - spy_entry_price) / spy_entry_price
                    
                    dollar_return = position_return * current_position
                    portfolio_value += dollar_return
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'asset': current_asset,
                        'regime': current_regime,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position_size': current_position / portfolio_value if portfolio_value > 0 else 0,
                        'return': position_return,
                        'dollar_return': dollar_return,
                        'portfolio_value': portfolio_value
                    })
                
                # Open new position
                position_sizing = self.optimize_position_sizing(regime_info)
                position_size = position_sizing['position_size']
                
                # Adjust for volatility (reduce size in high vol)
                vol_adjustment = 1.0 if current_vol < 0.20 else 0.7  # Reduce by 30% in high vol
                position_size *= vol_adjustment
                
                current_position = portfolio_value * position_size
                current_asset = recommended_asset
                entry_price = current_price
                entry_date = date
            
            # Calculate daily return
            if len(daily_returns) > 0:
                daily_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
                daily_returns.append(daily_return)
            
            prev_portfolio_value = portfolio_value
        
        # Close final position
        if current_position is not None:
            final_price = asset_data[current_asset].iloc[-1]['Close']
            position_return = (final_price - entry_price) / entry_price
            dollar_return = position_return * current_position
            portfolio_value += dollar_return
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': test_regime_df.index[-1],
                'asset': current_asset,
                'entry_price': entry_price,
                'exit_price': final_price,
                'return': position_return,
                'dollar_return': dollar_return,
                'portfolio_value': portfolio_value
            })
        
        return trades, portfolio_value, daily_returns
    
    def analyze_optimized_performance(self, trades, final_portfolio_value, daily_returns):
        """Analyze the optimized system performance"""
        print(f"\nOPTIMIZED SYSTEM PERFORMANCE ANALYSIS:")
        print("=" * 50)
        
        if not trades:
            print("No trades executed!")
            return None
        
        # Basic performance
        total_return = (final_portfolio_value - self.initial_capital) / self.initial_capital
        num_trades = len(trades)
        
        print(f"PERFORMANCE SUMMARY:")
        print(f"  Starting Capital: ${self.initial_capital:,.0f}")
        print(f"  Final Portfolio: ${final_portfolio_value:,.0f}")
        print(f"  Total Return: {total_return:.1%}")
        print(f"  Profit/Loss: ${final_portfolio_value - self.initial_capital:,.0f}")
        print(f"  Number of Trades: {num_trades}")
        
        # Trade analysis
        df_trades = pd.DataFrame(trades)
        winning_trades = len(df_trades[df_trades['return'] > 0])
        win_rate = winning_trades / num_trades
        
        avg_return = df_trades['return'].mean()
        avg_winner = df_trades[df_trades['return'] > 0]['return'].mean()
        avg_loser = df_trades[df_trades['return'] < 0]['return'].mean()
        
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Average Return per Trade: {avg_return:.2%}")
        if not pd.isna(avg_winner):
            print(f"  Average Winner: {avg_winner:.2%}")
        if not pd.isna(avg_loser):
            print(f"  Average Loser: {avg_loser:.2%}")
        
        # Risk metrics
        if daily_returns:
            daily_volatility = np.std(daily_returns) * np.sqrt(252)
            sharpe_ratio = (avg_return * num_trades * 4) / daily_volatility if daily_volatility > 0 else 0  # Approximate annual
            print(f"  Annual Volatility: {daily_volatility:.1%}")
            print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Regime breakdown
        print(f"\nREGIME PERFORMANCE BREAKDOWN:")
        regime_performance = df_trades.groupby('regime').agg({
            'return': ['count', 'mean', 'sum'],
            'asset': lambda x: x.iloc[0]
        }).round(3)
        
        for regime in regime_performance.index:
            count = regime_performance.loc[regime, ('return', 'count')]
            avg_ret = regime_performance.loc[regime, ('return', 'mean')]
            total_ret = regime_performance.loc[regime, ('return', 'sum')]
            asset = regime_performance.loc[regime, ('asset', '<lambda>')]
            
            print(f"  {regime} ({asset}): {count} trades, {avg_ret:.1%} avg, {total_ret:.1%} total")
        
        # Compare to buy and hold SPY
        spy_return = 0.15  # Approximate 15% annual return for comparison
        test_period_spy_return = spy_return * (len(daily_returns) / 252) if daily_returns else 0.15
        
        print(f"\nCOMPARISON:")
        print(f"  SPY Buy & Hold (est): {test_period_spy_return:.1%}")
        print(f"  Our System: {total_return:.1%}")
        print(f"  Excess Return: {total_return - test_period_spy_return:+.1%}")
        
        if total_return > test_period_spy_return:
            print("SUCCESS! System outperforms market!")
        else:
            print("Underperformed market - needs optimization")
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'sharpe_ratio': sharpe_ratio if daily_returns else 0,
            'excess_return': total_return - test_period_spy_return
        }
    
    def run_optimized_system(self):
        """Run complete risk-optimized regime system"""
        
        # Get data
        regime_df, asset_data = self.get_regime_data()
        if regime_df is None:
            return None
        
        # Backtest optimized system
        trades, final_value, daily_returns = self.backtest_optimized_regime_system(regime_df, asset_data)
        
        # Analyze performance
        performance = self.analyze_optimized_performance(trades, final_value, daily_returns)
        
        print(f"\nRISK-OPTIMIZED REGIME SYSTEM COMPLETE!")
        print("=" * 50)
        
        if performance and performance['total_return'] > 0.1:
            print("EXCELLENT! Risk-optimized system shows strong returns!")
        elif performance and performance['total_return'] > 0:
            print("GOOD! System is profitable with optimization!")
        else:
            print("NEEDS WORK: Further risk optimization required")
        
        return performance

if __name__ == "__main__":
    system = RiskOptimizedRegimeSystem()
    results = system.run_optimized_system()