"""
ADVANCED STRATEGY GENERATOR
===========================
Develops new high-edge options strategies based on backtesting insights
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from strategy_backtester import OptionsBacktester
import json

class AdvancedStrategyGenerator:
    """Generate and test new high-edge strategies."""
    
    def __init__(self):
        self.backtester = OptionsBacktester("2021-01-01", "2024-12-31")
        self.symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'META']
        
        print("ADVANCED STRATEGY GENERATOR INITIALIZED")
        print("=" * 50)
        print("Mission: Create new 20%+ monthly return strategies")
        print("Based on: Volatility breakout success (76% win rate)")
    
    def multi_timeframe_volatility_strategy(self):
        """Enhanced volatility strategy with multiple timeframes."""
        
        print("\nTESTING MULTI-TIMEFRAME VOLATILITY STRATEGY...")
        print("-" * 50)
        
        all_trades = []
        
        for symbol in self.symbols:
            print(f"Analyzing {symbol}...")
            
            data = self.backtester.get_historical_data(symbol, 300)
            if data is None or len(data) < 150:
                continue
            
            for i in range(80, len(data) - 30):
                current_date = data.index[i]
                current_price = data['Close'].iloc[i]
                
                if current_date.weekday() >= 5:
                    continue
                
                # Multiple timeframe volatility analysis
                vol_5d = data['Returns'].iloc[i-5:i].std() * np.sqrt(252)
                vol_10d = data['Returns'].iloc[i-10:i].std() * np.sqrt(252)
                vol_20d = data['Returns'].iloc[i-20:i].std() * np.sqrt(252)
                vol_50d = data['Returns'].iloc[i-50:i].std() * np.sqrt(252)
                
                # Volume confirmation across timeframes
                vol_ratio_5d = data['Volume'].iloc[i-5:i].mean() / data['Volume'].iloc[i-25:i-5].mean()
                
                # Price action confirmation
                price_compression_10d = (data['High'].iloc[i-10:i].max() - data['Low'].iloc[i-10:i].min()) / current_price
                price_compression_20d = (data['High'].iloc[i-30:i-10].max() - data['Low'].iloc[i-30:i-10].min()) / data['Close'].iloc[i-20]
                
                # RSI for oversold/overbought
                rsi = data['RSI'].iloc[i]
                
                # ENHANCED VOLATILITY CRITERIA
                if (vol_10d / vol_50d < 0.65 and          # 10d vol contracted vs 50d
                    vol_5d / vol_20d < 0.75 and           # Recent vol still contracting
                    price_compression_10d / price_compression_20d < 0.7 and  # Price range tight
                    vol_ratio_5d > 1.3 and               # Volume picking up
                    vol_50d > 0.22 and                   # Normally volatile stock
                    30 < rsi < 70):                      # Not extreme RSI
                    
                    # Execute enhanced straddle
                    trade = self.execute_enhanced_straddle(
                        symbol, current_date, current_price, data, i,
                        vol_10d, vol_50d, dte=28
                    )
                    
                    if trade:
                        all_trades.append(trade)
        
        return all_trades
    
    def execute_enhanced_straddle(self, symbol, entry_date, entry_price, data, entry_idx, recent_vol, hist_vol, dte=28):
        """Execute enhanced straddle with better exit rules."""
        
        try:
            strike = entry_price
            vol = min(hist_vol * 1.1, 0.6)  # Use elevated vol but cap it
            
            call_premium = self.backtester.black_scholes_price(
                S=entry_price, K=strike, T=dte/365, r=0.05, sigma=vol, option_type='call'
            )
            put_premium = self.backtester.black_scholes_price(
                S=entry_price, K=strike, T=dte/365, r=0.05, sigma=vol, option_type='put'
            )
            
            entry_cost = call_premium + put_premium
            
            # Enhanced exit logic
            exit_idx = entry_idx + dte
            exit_date = data.index[min(exit_idx, len(data)-1)]
            exit_price = entry_price  # Default
            
            # Look for breakout within holding period
            max_profit_reached = False
            for j in range(entry_idx + 3, min(entry_idx + dte + 1, len(data))):  # Wait 3 days minimum
                current_price_j = data['Close'].iloc[j]
                move_pct = abs(current_price_j - entry_price) / entry_price
                
                # Take profits on 12%+ moves
                if move_pct >= 0.12:
                    exit_idx = j
                    exit_date = data.index[j]
                    exit_price = current_price_j
                    max_profit_reached = True
                    break
                
                # Stop loss on time decay without movement (after 2 weeks)
                elif j >= entry_idx + 14 and move_pct < 0.04:
                    exit_idx = j
                    exit_date = data.index[j]
                    exit_price = current_price_j
                    break
            
            if not max_profit_reached:
                exit_price = data['Close'].iloc[min(exit_idx, len(data)-1)]
            
            # Calculate exit value
            days_left = max(0, dte - (exit_idx - entry_idx))
            
            # Adjust vol for realized movement
            realized_move = abs(exit_price - entry_price) / entry_price
            if realized_move > 0.08:
                exit_vol = vol * 1.2  # IV expansion on big moves
            else:
                exit_vol = vol * 0.8  # IV contraction
            
            exit_call = self.backtester.black_scholes_price(
                S=exit_price, K=strike, T=days_left/365, r=0.05, sigma=exit_vol, option_type='call'
            )
            exit_put = self.backtester.black_scholes_price(
                S=exit_price, K=strike, T=days_left/365, r=0.05, sigma=exit_vol, option_type='put'
            )
            
            exit_value = exit_call + exit_put
            pnl = exit_value - entry_cost
            return_pct = (pnl / entry_cost) * 100
            
            from strategy_backtester import Trade, OptionsContract
            
            trade = Trade(
                entry_date=entry_date,
                exit_date=exit_date,
                strategy='ENHANCED_STRADDLE',
                symbol=symbol,
                contracts=[
                    OptionsContract(symbol, strike, exit_date, 'call', call_premium),
                    OptionsContract(symbol, strike, exit_date, 'put', put_premium)
                ],
                entry_cost=entry_cost,
                exit_value=exit_value,
                pnl=pnl,
                return_pct=return_pct,
                hold_days=exit_idx - entry_idx,
                win=pnl > 0
            )
            
            return trade
            
        except Exception as e:
            return None
    
    def momentum_reversal_strategy(self):
        """Strategy that catches momentum reversals with options."""
        
        print("\nTESTING MOMENTUM REVERSAL STRATEGY...")
        print("-" * 50)
        
        all_trades = []
        
        for symbol in self.symbols:
            print(f"Analyzing {symbol}...")
            
            data = self.backtester.get_historical_data(symbol, 300)
            if data is None or len(data) < 100:
                continue
            
            for i in range(60, len(data) - 30):
                current_date = data.index[i]
                current_price = data['Close'].iloc[i]
                
                if current_date.weekday() >= 5:
                    continue
                
                # Calculate momentum metrics
                returns_20d = (current_price / data['Close'].iloc[i-20] - 1)
                returns_5d = (current_price / data['Close'].iloc[i-5] - 1)
                returns_2d = (current_price / data['Close'].iloc[i-2] - 1)
                
                # Volume and volatility
                vol_surge = data['Volume_Ratio'].iloc[i]
                volatility = data['Volatility'].iloc[i]
                rsi = data['RSI'].iloc[i]
                
                # Price vs moving averages
                above_sma20 = current_price > data['SMA_20'].iloc[i]
                above_sma50 = current_price > data['SMA_50'].iloc[i]
                
                # MOMENTUM EXHAUSTION CRITERIA (for reversal)
                if (returns_20d > 0.25 and           # Strong 25% run
                    returns_5d > 0.08 and           # Recent acceleration
                    returns_2d < 0.02 and           # But slowing down
                    rsi > 75 and                   # Overbought
                    vol_surge > 2.0 and            # High volume
                    volatility > 0.30):            # High volatility
                    
                    # Execute BEARISH reversal (PUT)
                    trade = self.execute_reversal_put(
                        symbol, current_date, current_price, data, i,
                        dte=21, otm_pct=0.03
                    )
                    
                    if trade:
                        all_trades.append(trade)
                
                # OVERSOLD BOUNCE CRITERIA
                elif (returns_20d < -0.20 and        # Strong 20% decline
                      returns_5d < -0.06 and        # Recent selling
                      returns_2d > -0.01 and        # But stabilizing
                      rsi < 25 and                 # Oversold
                      vol_surge > 1.8 and          # Volume
                      above_sma50):                # Still above 50-day trend
                    
                    # Execute BULLISH reversal (CALL)
                    trade = self.execute_reversal_call(
                        symbol, current_date, current_price, data, i,
                        dte=21, otm_pct=0.03
                    )
                    
                    if trade:
                        all_trades.append(trade)
        
        return all_trades
    
    def execute_reversal_put(self, symbol, entry_date, entry_price, data, entry_idx, dte=21, otm_pct=0.03):
        """Execute momentum reversal PUT trade."""
        
        try:
            strike = entry_price * (1 - otm_pct)  # Slightly OTM put
            vol = data['Volatility'].iloc[entry_idx]
            if pd.isna(vol) or vol < 0.2:
                vol = 0.35  # Use higher vol for reversal trades
            
            entry_premium = self.backtester.black_scholes_price(
                S=entry_price, K=strike, T=dte/365, r=0.05, sigma=vol, option_type='put'
            )
            
            # Quick exit strategy (5-15 days)
            max_hold = min(15, dte)
            exit_idx = entry_idx + max_hold
            
            # Look for reversal within holding period
            for j in range(entry_idx + 2, min(entry_idx + max_hold + 1, len(data))):
                price_decline = (data['Close'].iloc[j] - entry_price) / entry_price
                
                # Exit on 8% decline (profit taking)
                if price_decline < -0.08:
                    exit_idx = j
                    break
                
                # Exit on bounce back above entry (stop loss)
                elif j >= entry_idx + 5 and price_decline > 0.03:
                    exit_idx = j
                    break
            
            exit_idx = min(exit_idx, len(data)-1)
            exit_date = data.index[exit_idx]
            exit_price = data['Close'].iloc[exit_idx]
            
            days_left = max(0, dte - (exit_idx - entry_idx))
            exit_premium = self.backtester.black_scholes_price(
                S=exit_price, K=strike, T=days_left/365, r=0.05, sigma=vol*0.9, option_type='put'
            )
            
            pnl = exit_premium - entry_premium
            return_pct = (pnl / entry_premium) * 100
            
            from strategy_backtester import Trade, OptionsContract
            
            trade = Trade(
                entry_date=entry_date,
                exit_date=exit_date,
                strategy='REVERSAL_PUT',
                symbol=symbol,
                contracts=[OptionsContract(symbol, strike, exit_date, 'put', entry_premium)],
                entry_cost=entry_premium,
                exit_value=exit_premium,
                pnl=pnl,
                return_pct=return_pct,
                hold_days=exit_idx - entry_idx,
                win=pnl > 0
            )
            
            return trade
            
        except Exception as e:
            return None
    
    def execute_reversal_call(self, symbol, entry_date, entry_price, data, entry_idx, dte=21, otm_pct=0.03):
        """Execute momentum reversal CALL trade."""
        
        try:
            strike = entry_price * (1 + otm_pct)
            vol = data['Volatility'].iloc[entry_idx]
            if pd.isna(vol) or vol < 0.2:
                vol = 0.35
            
            entry_premium = self.backtester.black_scholes_price(
                S=entry_price, K=strike, T=dte/365, r=0.05, sigma=vol, option_type='call'
            )
            
            max_hold = min(15, dte)
            exit_idx = entry_idx + max_hold
            
            for j in range(entry_idx + 2, min(entry_idx + max_hold + 1, len(data))):
                price_gain = (data['Close'].iloc[j] - entry_price) / entry_price
                
                if price_gain > 0.08:  # 8% gain
                    exit_idx = j
                    break
                elif j >= entry_idx + 5 and price_gain < -0.03:  # Stop loss
                    exit_idx = j
                    break
            
            exit_idx = min(exit_idx, len(data)-1)
            exit_date = data.index[exit_idx]
            exit_price = data['Close'].iloc[exit_idx]
            
            days_left = max(0, dte - (exit_idx - entry_idx))
            exit_premium = self.backtester.black_scholes_price(
                S=exit_price, K=strike, T=days_left/365, r=0.05, sigma=vol*0.9, option_type='call'
            )
            
            pnl = exit_premium - entry_premium
            return_pct = (pnl / entry_premium) * 100
            
            from strategy_backtester import Trade, OptionsContract
            
            trade = Trade(
                entry_date=entry_date,
                exit_date=exit_date,
                strategy='REVERSAL_CALL',
                symbol=symbol,
                contracts=[OptionsContract(symbol, strike, exit_date, 'call', entry_premium)],
                entry_cost=entry_premium,
                exit_value=exit_premium,
                pnl=pnl,
                return_pct=return_pct,
                hold_days=exit_idx - entry_idx,
                win=pnl > 0
            )
            
            return trade
            
        except Exception as e:
            return None
    
    def test_new_strategies(self):
        """Test all new advanced strategies."""
        
        print("TESTING ADVANCED STRATEGIES")
        print("=" * 60)
        print("Goal: Beat 76% win rate of volatility breakout")
        print("=" * 60)
        
        # Test new strategies
        multi_tf_trades = self.multi_timeframe_volatility_strategy()
        reversal_trades = self.momentum_reversal_strategy()
        
        # Analyze results
        print("\n" + "="*60)
        print("ADVANCED STRATEGY RESULTS")
        print("="*60)
        
        multi_tf_results = self.backtester.analyze_results(multi_tf_trades, "Multi-Timeframe Volatility")
        reversal_results = self.backtester.analyze_results(reversal_trades, "Momentum Reversal")
        
        # Combined advanced strategy
        all_advanced = multi_tf_trades + reversal_trades
        combined_results = self.backtester.analyze_results(all_advanced, "Combined Advanced Strategies")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        advanced_results = {
            'timestamp': timestamp,
            'multi_timeframe_volatility': multi_tf_results,
            'momentum_reversal': reversal_results,
            'combined_advanced': combined_results,
            'benchmark_volatility_breakout': {
                'win_rate': 76.1,
                'avg_return': 46.1,
                'months_20plus_pct': 85.7
            }
        }
        
        with open(f'advanced_strategy_results_{timestamp}.json', 'w') as f:
            json.dump(advanced_results, f, indent=2, default=str)
        
        print(f"\n\nADVANCED STRATEGY TESTING COMPLETE!")
        print(f"Results saved: advanced_strategy_results_{timestamp}.json")
        
        # Recommendations
        print(f"\nSTRATEGY RECOMMENDATIONS:")
        print("-" * 30)
        
        if multi_tf_results.get('win_rate', 0) > 76:
            print(f"✅ Multi-Timeframe Volatility BEATS benchmark!")
            print(f"   Win Rate: {multi_tf_results['win_rate']:.1f}% vs 76.1%")
        
        if reversal_results.get('win_rate', 0) > 76:
            print(f"✅ Momentum Reversal BEATS benchmark!")
            print(f"   Win Rate: {reversal_results['win_rate']:.1f}% vs 76.1%")
        
        if combined_results.get('win_rate', 0) > 76:
            print(f"✅ Combined Advanced BEATS benchmark!")
            print(f"   Win Rate: {combined_results['win_rate']:.1f}% vs 76.1%")
        
        return advanced_results

def main():
    """Run advanced strategy generation and testing."""
    
    generator = AdvancedStrategyGenerator()
    results = generator.test_new_strategies()
    
    print(f"\nAdvanced strategy testing completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()