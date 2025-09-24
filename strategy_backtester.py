"""
ADVANCED OPTIONS STRATEGY BACKTESTER
====================================
Comprehensive backtesting system for options strategies targeting 20%+ returns
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import norm
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

@dataclass
class OptionsContract:
    """Represents an options contract."""
    symbol: str
    strike: float
    expiry: datetime
    contract_type: str  # 'call' or 'put'
    premium: float
    iv: float = 0.25  # Implied volatility default

@dataclass
class Trade:
    """Represents a complete trade."""
    entry_date: datetime
    exit_date: datetime
    strategy: str
    symbol: str
    contracts: List[OptionsContract]
    entry_cost: float
    exit_value: float
    pnl: float
    return_pct: float
    hold_days: int
    win: bool

class OptionsBacktester:
    """Advanced options backtesting engine."""
    
    def __init__(self, start_date="2020-01-01", end_date="2024-12-31"):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.trades = []
        self.symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'META']
        
        print("OPTIONS BACKTESTING ENGINE INITIALIZED")
        print("=" * 50)
        print(f"Backtest period: {start_date} to {end_date}")
        print(f"Target symbols: {len(self.symbols)}")
    
    def black_scholes_price(self, S, K, T, r, sigma, option_type='call'):
        """Calculate Black-Scholes option price."""
        if T <= 0:
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        return max(price, 0.01)  # Minimum 1 cent
    
    def get_historical_data(self, symbol, days=252):
        """Get historical price data."""
        try:
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days+50)
            
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                return None
            
            # Add technical indicators
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(20).std() * np.sqrt(252)
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['SMA_50'] = data['Close'].rolling(50).mean()
            data['RSI'] = self.calculate_rsi(data['Close'])
            data['Volume_MA'] = data['Volume'].rolling(20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
            
            return data.dropna()
            
        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def momentum_strategy_backtest(self, lookback_days=252):
        """Backtest momentum-based options strategies."""
        
        print("\nBACKTESTING MOMENTUM STRATEGIES...")
        print("-" * 40)
        
        all_trades = []
        
        for symbol in self.symbols:
            print(f"Testing {symbol}...")
            
            data = self.get_historical_data(symbol, lookback_days + 100)
            if data is None or len(data) < 100:
                continue
            
            for i in range(60, len(data) - 30):  # Leave room for option expiry
                current_date = data.index[i]
                current_price = data['Close'].iloc[i]
                
                # Skip weekends
                if current_date.weekday() >= 5:
                    continue
                
                # Calculate momentum signals
                returns_5d = (current_price / data['Close'].iloc[i-5] - 1)
                returns_20d = (current_price / data['Close'].iloc[i-20] - 1)
                vol_ratio = data['Volume_Ratio'].iloc[i]
                volatility = data['Volatility'].iloc[i]
                rsi = data['RSI'].iloc[i]
                
                # BULLISH MOMENTUM CRITERIA
                if (returns_20d > 0.08 and      # 8%+ move in 20 days
                    returns_5d > 0.02 and      # 2%+ recent acceleration  
                    vol_ratio > 1.5 and       # Volume surge
                    volatility > 0.25 and     # High volatility
                    rsi < 75):                 # Not overbought
                    
                    # Execute CALL option trade
                    trade = self.execute_call_trade(
                        symbol, current_date, current_price, data, i,
                        dte=30, otm_pct=0.05
                    )
                    
                    if trade:
                        all_trades.append(trade)
                
                # BEARISH MOMENTUM CRITERIA  
                elif (returns_20d < -0.08 and   # 8%+ decline in 20 days
                      returns_5d < -0.02 and   # 2%+ recent acceleration
                      vol_ratio > 1.5 and      # Volume surge
                      volatility > 0.25 and    # High volatility
                      rsi > 25):                # Not oversold
                    
                    # Execute PUT option trade
                    trade = self.execute_put_trade(
                        symbol, current_date, current_price, data, i,
                        dte=30, otm_pct=0.05
                    )
                    
                    if trade:
                        all_trades.append(trade)
        
        return all_trades
    
    def volatility_breakout_backtest(self, lookback_days=252):
        """Backtest volatility breakout strategies."""
        
        print("\nBACKTESTING VOLATILITY BREAKOUT STRATEGIES...")
        print("-" * 40)
        
        all_trades = []
        
        for symbol in self.symbols:
            print(f"Testing {symbol}...")
            
            data = self.get_historical_data(symbol, lookback_days + 100)
            if data is None or len(data) < 100:
                continue
            
            for i in range(60, len(data) - 30):
                current_date = data.index[i]
                current_price = data['Close'].iloc[i]
                
                if current_date.weekday() >= 5:
                    continue
                
                # Calculate volatility metrics
                recent_vol = data['Returns'].iloc[i-10:i].std() * np.sqrt(252)
                hist_vol = data['Returns'].iloc[i-50:i-10].std() * np.sqrt(252)
                vol_ratio = recent_vol / hist_vol if hist_vol > 0 else 1
                
                # Price range compression
                recent_range = (data['High'].iloc[i-10:i].max() - 
                               data['Low'].iloc[i-10:i].min()) / current_price
                hist_range = (data['High'].iloc[i-30:i-10].max() - 
                             data['Low'].iloc[i-30:i-10].min()) / data['Close'].iloc[i-20]
                range_ratio = recent_range / hist_range if hist_range > 0 else 1
                
                # VOLATILITY COMPRESSION CRITERIA
                if (vol_ratio < 0.7 and         # Vol contracted 30%+
                    range_ratio < 0.8 and       # Range compressed 20%+
                    hist_vol > 0.25 and         # Normally volatile
                    recent_vol > 0.15):         # Still has min volatility
                    
                    # Execute STRADDLE trade
                    trade = self.execute_straddle_trade(
                        symbol, current_date, current_price, data, i,
                        dte=35
                    )
                    
                    if trade:
                        all_trades.append(trade)
        
        return all_trades
    
    def earnings_strategy_backtest(self, lookback_days=252):
        """Backtest earnings-based strategies."""
        
        print("\nBACKTESTING EARNINGS STRATEGIES...")
        print("-" * 40)
        
        all_trades = []
        
        # Simulate earnings dates (quarterly, roughly)
        for symbol in self.symbols[:4]:  # Limit for demo
            print(f"Testing {symbol}...")
            
            data = self.get_historical_data(symbol, lookback_days + 100)
            if data is None:
                continue
            
            # Create simulated earnings dates (every ~90 days)
            earnings_dates = []
            start_idx = 60
            while start_idx < len(data) - 30:
                earnings_dates.append(start_idx)
                start_idx += 65 + np.random.randint(-10, 15)  # 65±12 days
            
            for earnings_idx in earnings_dates:
                earnings_date = data.index[earnings_idx]
                pre_earnings_idx = earnings_idx - 5  # 5 days before
                
                if pre_earnings_idx < 20:
                    continue
                
                pre_earnings_date = data.index[pre_earnings_idx]
                current_price = data['Close'].iloc[pre_earnings_idx]
                
                # Calculate expected earnings move
                hist_vol = data['Returns'].iloc[pre_earnings_idx-20:pre_earnings_idx].std()
                expected_move = hist_vol * 2.5  # Rough earnings move estimate
                
                if expected_move > 0.03:  # At least 3% expected move
                    
                    # Execute earnings straddle
                    trade = self.execute_earnings_straddle(
                        symbol, pre_earnings_date, current_price, data, 
                        pre_earnings_idx, earnings_idx, expected_move
                    )
                    
                    if trade:
                        all_trades.append(trade)
        
        return all_trades
    
    def execute_call_trade(self, symbol, entry_date, entry_price, data, entry_idx, dte=30, otm_pct=0.05):
        """Execute a call option trade."""
        
        try:
            # Calculate strike price (OTM)
            strike = entry_price * (1 + otm_pct)
            
            # Estimate volatility for pricing
            vol = data['Volatility'].iloc[entry_idx]
            if pd.isna(vol) or vol < 0.1:
                vol = 0.25
            
            # Entry option price
            entry_premium = self.black_scholes_price(
                S=entry_price, K=strike, T=dte/365, r=0.05, sigma=vol, option_type='call'
            )
            
            # Find exit date (hold until expiry or big move)
            exit_idx = min(entry_idx + dte, len(data) - 1)
            exit_date = data.index[exit_idx]
            exit_price = data['Close'].iloc[exit_idx]
            
            # Check for early exit on big moves
            max_gain_reached = False
            for j in range(entry_idx + 1, exit_idx):
                if data['Close'].iloc[j] > strike * 1.1:  # 10% beyond strike
                    exit_idx = j
                    exit_date = data.index[j]
                    exit_price = data['Close'].iloc[j]
                    max_gain_reached = True
                    break
            
            # Exit option price
            days_left = max(0, dte - (exit_idx - entry_idx))
            exit_premium = self.black_scholes_price(
                S=exit_price, K=strike, T=days_left/365, r=0.05, sigma=vol, option_type='call'
            )
            
            # Calculate P&L
            pnl = exit_premium - entry_premium
            return_pct = (pnl / entry_premium) * 100
            hold_days = exit_idx - entry_idx
            
            trade = Trade(
                entry_date=entry_date,
                exit_date=exit_date,
                strategy='LONG_CALL',
                symbol=symbol,
                contracts=[OptionsContract(symbol, strike, exit_date, 'call', entry_premium)],
                entry_cost=entry_premium,
                exit_value=exit_premium,
                pnl=pnl,
                return_pct=return_pct,
                hold_days=hold_days,
                win=pnl > 0
            )
            
            return trade
            
        except Exception as e:
            return None
    
    def execute_put_trade(self, symbol, entry_date, entry_price, data, entry_idx, dte=30, otm_pct=0.05):
        """Execute a put option trade."""
        
        try:
            strike = entry_price * (1 - otm_pct)
            vol = data['Volatility'].iloc[entry_idx]
            if pd.isna(vol) or vol < 0.1:
                vol = 0.25
            
            entry_premium = self.black_scholes_price(
                S=entry_price, K=strike, T=dte/365, r=0.05, sigma=vol, option_type='put'
            )
            
            exit_idx = min(entry_idx + dte, len(data) - 1)
            exit_date = data.index[exit_idx]
            exit_price = data['Close'].iloc[exit_idx]
            
            # Check for early exit
            for j in range(entry_idx + 1, exit_idx):
                if data['Close'].iloc[j] < strike * 0.9:
                    exit_idx = j
                    exit_date = data.index[j]
                    exit_price = data['Close'].iloc[j]
                    break
            
            days_left = max(0, dte - (exit_idx - entry_idx))
            exit_premium = self.black_scholes_price(
                S=exit_price, K=strike, T=days_left/365, r=0.05, sigma=vol, option_type='put'
            )
            
            pnl = exit_premium - entry_premium
            return_pct = (pnl / entry_premium) * 100
            
            trade = Trade(
                entry_date=entry_date,
                exit_date=exit_date,
                strategy='LONG_PUT',
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
    
    def execute_straddle_trade(self, symbol, entry_date, entry_price, data, entry_idx, dte=35):
        """Execute a long straddle trade."""
        
        try:
            strike = entry_price
            vol = data['Volatility'].iloc[entry_idx]
            if pd.isna(vol) or vol < 0.1:
                vol = 0.25
            
            # Price both call and put
            call_premium = self.black_scholes_price(
                S=entry_price, K=strike, T=dte/365, r=0.05, sigma=vol, option_type='call'
            )
            put_premium = self.black_scholes_price(
                S=entry_price, K=strike, T=dte/365, r=0.05, sigma=vol, option_type='put'
            )
            
            entry_cost = call_premium + put_premium
            
            exit_idx = min(entry_idx + dte, len(data) - 1)
            exit_date = data.index[exit_idx]
            exit_price = data['Close'].iloc[exit_idx]
            
            # Check for early exit on big moves
            for j in range(entry_idx + 5, exit_idx):  # Wait at least 5 days
                price_move = abs(data['Close'].iloc[j] - entry_price) / entry_price
                if price_move > 0.15:  # 15% move triggers exit
                    exit_idx = j
                    exit_date = data.index[j]
                    exit_price = data['Close'].iloc[j]
                    break
            
            days_left = max(0, dte - (exit_idx - entry_idx))
            
            exit_call_premium = self.black_scholes_price(
                S=exit_price, K=strike, T=days_left/365, r=0.05, sigma=vol, option_type='call'
            )
            exit_put_premium = self.black_scholes_price(
                S=exit_price, K=strike, T=days_left/365, r=0.05, sigma=vol, option_type='put'
            )
            
            exit_value = exit_call_premium + exit_put_premium
            pnl = exit_value - entry_cost
            return_pct = (pnl / entry_cost) * 100
            
            trade = Trade(
                entry_date=entry_date,
                exit_date=exit_date,
                strategy='LONG_STRADDLE',
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
    
    def execute_earnings_straddle(self, symbol, entry_date, entry_price, data, entry_idx, earnings_idx, expected_move):
        """Execute earnings straddle trade."""
        
        try:
            strike = entry_price
            vol = max(0.4, expected_move * 4)  # Elevated IV for earnings
            
            call_premium = self.black_scholes_price(
                S=entry_price, K=strike, T=5/365, r=0.05, sigma=vol, option_type='call'
            )
            put_premium = self.black_scholes_price(
                S=entry_price, K=strike, T=5/365, r=0.05, sigma=vol, option_type='put'
            )
            
            entry_cost = call_premium + put_premium
            
            # Exit day after earnings
            exit_idx = earnings_idx + 1
            exit_date = data.index[exit_idx]
            exit_price = data['Close'].iloc[exit_idx]
            
            # Post-earnings IV crush (reduce vol by 60%)
            post_vol = vol * 0.4
            
            exit_call_premium = self.black_scholes_price(
                S=exit_price, K=strike, T=4/365, r=0.05, sigma=post_vol, option_type='call'
            )
            exit_put_premium = self.black_scholes_price(
                S=exit_price, K=strike, T=4/365, r=0.05, sigma=post_vol, option_type='put'
            )
            
            exit_value = exit_call_premium + exit_put_premium
            pnl = exit_value - entry_cost
            return_pct = (pnl / entry_cost) * 100
            
            trade = Trade(
                entry_date=entry_date,
                exit_date=exit_date,
                strategy='EARNINGS_STRADDLE',
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
    
    def analyze_results(self, trades, strategy_name):
        """Analyze backtest results."""
        
        if not trades:
            print(f"No trades found for {strategy_name}")
            return {}
        
        df = pd.DataFrame([
            {
                'date': t.entry_date,
                'symbol': t.symbol,
                'strategy': t.strategy,
                'return_pct': t.return_pct,
                'pnl': t.pnl,
                'hold_days': t.hold_days,
                'win': t.win
            }
            for t in trades
        ])
        
        # Performance metrics
        total_trades = len(df)
        win_rate = df['win'].mean() * 100
        avg_return = df['return_pct'].mean()
        avg_winner = df[df['win']]['return_pct'].mean()
        avg_loser = df[~df['win']]['return_pct'].mean()
        best_trade = df['return_pct'].max()
        worst_trade = df['return_pct'].min()
        
        # Monthly returns
        df['month'] = df['date'].dt.to_period('M')
        monthly_returns = df.groupby('month')['return_pct'].mean()
        months_20plus = (monthly_returns >= 20).sum()
        total_months = len(monthly_returns)
        
        results = {
            'strategy': strategy_name,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'months_20plus': months_20plus,
            'total_months': total_months,
            'pct_months_20plus': (months_20plus / total_months * 100) if total_months > 0 else 0
        }
        
        print(f"\n{strategy_name.upper()} RESULTS:")
        print("=" * 50)
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Avg Return: {avg_return:.1f}%")
        print(f"Avg Winner: {avg_winner:.1f}%")
        print(f"Avg Loser: {avg_loser:.1f}%")
        print(f"Best Trade: {best_trade:.1f}%")
        print(f"Worst Trade: {worst_trade:.1f}%")
        print(f"Months with 20%+ returns: {months_20plus}/{total_months} ({(months_20plus/total_months*100):.1f}%)")
        
        return results
    
    def run_comprehensive_backtest(self):
        """Run comprehensive backtest of all strategies."""
        
        print("STARTING COMPREHENSIVE OPTIONS BACKTEST")
        print("=" * 60)
        print("Target: Validate 20%+ monthly return strategies")
        print("=" * 60)
        
        # Test all strategies
        momentum_trades = self.momentum_strategy_backtest()
        volatility_trades = self.volatility_breakout_backtest() 
        earnings_trades = self.earnings_strategy_backtest()
        
        # Analyze results
        momentum_results = self.analyze_results(momentum_trades, "Momentum Strategy")
        volatility_results = self.analyze_results(volatility_trades, "Volatility Breakout")
        earnings_results = self.analyze_results(earnings_trades, "Earnings Strategy")
        
        # Combined analysis
        all_trades = momentum_trades + volatility_trades + earnings_trades
        combined_results = self.analyze_results(all_trades, "Combined Strategy")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        results_summary = {
            'timestamp': timestamp,
            'backtest_period': f"{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}",
            'momentum': momentum_results,
            'volatility': volatility_results,
            'earnings': earnings_results,
            'combined': combined_results
        }
        
        with open(f'backtest_results_{timestamp}.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print(f"\n\nBACKTEST COMPLETE!")
        print(f"Results saved: backtest_results_{timestamp}.json")
        
        return results_summary

def main():
    """Run the comprehensive backtest."""
    
    # Initialize backtester (last 4 years of data)
    backtester = OptionsBacktester(
        start_date="2021-01-01",
        end_date="2024-12-31"
    )
    
    # Run comprehensive backtest
    results = backtester.run_comprehensive_backtest()
    
    print(f"\nBacktest completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()