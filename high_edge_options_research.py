"""
HIGH-EDGE OPTIONS RESEARCH SYSTEM
=================================================================
Target: 20%+ Monthly Returns through Advanced Options Strategies
Focus: Statistical Arbitrage, Volatility Mispricing, Event-Driven
"""

import pandas as pd
import numpy as np
import QuantLib as ql
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.implied_volatility import implied_volatility
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HighEdgeOptionsResearcher:
    """Advanced options research for high-edge opportunities."""
    
    def __init__(self):
        """Initialize the research system."""
        self.risk_free_rate = 0.05  # 5% risk-free rate
        
        # High-edge focus symbols (liquid options, high volatility)
        self.focus_symbols = [
            # High Beta Tech (volatile, high option volume)
            'TSLA', 'NVDA', 'AMD', 'NFLX', 'META', 'GOOGL', 'AMZN',
            
            # Biotech (high volatility, event-driven)
            'MRNA', 'BNTX', 'GILD', 'BIIB',
            
            # Meme/Volatile Stocks
            'GME', 'AMC', 'PLTR', 'RIVN',
            
            # ETFs (liquid options, good for spreads)
            'SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLK',
            
            # Energy (volatile sector)
            'XOM', 'CVX', 'SLB', 'COP'
        ]
    
    def calculate_realized_volatility(self, symbol, days=30):
        """Calculate realized volatility for comparison with IV."""
        try:
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days*2)  # Extra buffer
            
            data = ticker.history(start=start_date, end=end_date)
            if len(data) < days:
                return None
                
            returns = data['Close'].pct_change().dropna()
            realized_vol = returns.std() * np.sqrt(252)  # Annualized
            
            return realized_vol
            
        except Exception as e:
            print(f"Error calculating realized vol for {symbol}: {e}")
            return None
    
    def get_options_chain(self, symbol):
        """Get options chain with calculated Greeks and edge metrics."""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current stock price
            current_price = ticker.history(period="1d")['Close'][-1]
            
            # Get options expirations
            expirations = ticker.options
            if not expirations:
                return None
            
            # Focus on near-term expirations (high gamma, fast profits)
            near_expirations = [exp for exp in expirations[:4]]  # Next 4 expirations
            
            options_data = []
            
            for exp_date in near_expirations:
                try:
                    opt_chain = ticker.option_chain(exp_date)
                    
                    # Days to expiration
                    exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                    dte = (exp_datetime - datetime.now()).days
                    
                    if dte < 5 or dte > 60:  # Focus on 5-60 DTE
                        continue
                    
                    # Process calls and puts
                    for option_type in ['calls', 'puts']:
                        options = getattr(opt_chain, option_type)
                        
                        for _, option in options.iterrows():
                            try:
                                strike = option['strike']
                                last_price = option['lastPrice']
                                bid = option['bid']
                                ask = option['ask']
                                volume = option['volume']
                                open_interest = option['openInterest']
                                
                                # Skip illiquid options
                                if bid <= 0 or ask <= 0 or volume < 10:
                                    continue
                                
                                # Calculate implied volatility
                                mid_price = (bid + ask) / 2
                                
                                try:
                                    iv = implied_volatility(
                                        mid_price, current_price, strike, 
                                        dte/365, self.risk_free_rate, 
                                        option_type[:-1]  # 'call' or 'put'
                                    )
                                except:
                                    iv = None
                                
                                if iv is None or iv <= 0:
                                    continue
                                
                                # Calculate theoretical price using Black-Scholes
                                theoretical_price = bs(
                                    option_type[:-1], current_price, strike,
                                    dte/365, self.risk_free_rate, iv
                                )
                                
                                # Edge metrics
                                bid_ask_spread = (ask - bid) / mid_price
                                edge_ratio = (theoretical_price - mid_price) / mid_price
                                
                                # Moneyness for filtering
                                if option_type == 'calls':
                                    moneyness = current_price / strike
                                else:
                                    moneyness = strike / current_price
                                
                                options_data.append({
                                    'symbol': symbol,
                                    'expiration': exp_date,
                                    'dte': dte,
                                    'type': option_type[:-1],
                                    'strike': strike,
                                    'current_price': current_price,
                                    'bid': bid,
                                    'ask': ask,
                                    'mid_price': mid_price,
                                    'last_price': last_price,
                                    'volume': volume,
                                    'open_interest': open_interest,
                                    'implied_vol': iv,
                                    'theoretical_price': theoretical_price,
                                    'edge_ratio': edge_ratio,
                                    'bid_ask_spread': bid_ask_spread,
                                    'moneyness': moneyness
                                })
                                
                            except Exception as e:
                                continue
                                
                except Exception as e:
                    continue
            
            return pd.DataFrame(options_data) if options_data else None
            
        except Exception as e:
            print(f"Error getting options for {symbol}: {e}")
            return None
    
    def find_volatility_arbitrage_opportunities(self):
        """Find options where IV significantly differs from realized volatility."""
        print("SCANNING FOR VOLATILITY ARBITRAGE OPPORTUNITIES...")
        print("=" * 60)
        
        opportunities = []
        
        for symbol in self.focus_symbols:
            print(f"Analyzing {symbol}...", end=" ")
            
            # Get realized volatility
            realized_vol = self.calculate_realized_volatility(symbol, days=30)
            if realized_vol is None:
                print("ERROR")
                continue
            
            # Get options data
            options_df = self.get_options_chain(symbol)
            if options_df is None or options_df.empty:
                print("ERROR")
                continue
            
            # Filter for liquid, near-the-money options
            liquid_options = options_df[
                (options_df['volume'] >= 50) &
                (options_df['open_interest'] >= 100) &
                (options_df['moneyness'] >= 0.8) &
                (options_df['moneyness'] <= 1.2) &
                (options_df['bid_ask_spread'] <= 0.15)  # Tight spreads
            ].copy()
            
            if liquid_options.empty:
                print("ERROR")
                continue
            
            # Calculate volatility edge
            liquid_options['vol_edge'] = liquid_options['implied_vol'] - realized_vol
            liquid_options['vol_edge_pct'] = liquid_options['vol_edge'] / realized_vol
            
            # Find significant volatility mispricing
            high_iv_options = liquid_options[
                liquid_options['vol_edge_pct'] > 0.30  # IV > 30% above realized
            ].copy()
            
            low_iv_options = liquid_options[
                liquid_options['vol_edge_pct'] < -0.20  # IV > 20% below realized
            ].copy()
            
            # Calculate expected profit potential
            for df, strategy in [(high_iv_options, 'SELL'), (low_iv_options, 'BUY')]:
                if not df.empty:
                    df = df.copy()
                    df['strategy'] = strategy
                    df['realized_vol'] = realized_vol
                    df['profit_potential'] = abs(df['vol_edge_pct']) * 100  # Rough estimate
                    
                    opportunities.extend(df.to_dict('records'))
            
            print("OK")
        
        if not opportunities:
            print("\nERROR No high-edge volatility opportunities found.")
            return pd.DataFrame()
        
        # Sort by profit potential
        opp_df = pd.DataFrame(opportunities)
        opp_df = opp_df.sort_values('profit_potential', ascending=False)
        
        return opp_df
    
    def find_earnings_straddle_opportunities(self):
        """Find high-IV options before earnings (straddle/strangle plays)."""
        print("\n SCANNING FOR EARNINGS VOLATILITY PLAYS...")
        print("=" * 60)
        
        earnings_plays = []
        
        for symbol in self.focus_symbols[:10]:  # Focus on most liquid
            try:
                ticker = yf.Ticker(symbol)
                
                # Get options data
                options_df = self.get_options_chain(symbol)
                if options_df is None or options_df.empty:
                    continue
                
                # Look for extremely high IV (potential earnings)
                high_iv_options = options_df[
                    (options_df['implied_vol'] > 0.5) &  # >50% IV
                    (options_df['dte'] <= 30) &  # Short-term
                    (options_df['volume'] >= 100)  # High volume
                ]
                
                if not high_iv_options.empty:
                    avg_iv = high_iv_options['implied_vol'].mean()
                    current_price = high_iv_options['current_price'].iloc[0]
                    
                    # Calculate straddle cost and breakevens
                    atm_calls = high_iv_options[
                        (high_iv_options['type'] == 'call') &
                        (abs(high_iv_options['strike'] - current_price) <= 5)
                    ]
                    atm_puts = high_iv_options[
                        (high_iv_options['type'] == 'put') &
                        (abs(high_iv_options['strike'] - current_price) <= 5)
                    ]
                    
                    if not atm_calls.empty and not atm_puts.empty:
                        call_price = atm_calls['mid_price'].iloc[0]
                        put_price = atm_puts['mid_price'].iloc[0]
                        strike = atm_calls['strike'].iloc[0]
                        
                        straddle_cost = call_price + put_price
                        breakeven_up = strike + straddle_cost
                        breakeven_down = strike - straddle_cost
                        
                        move_needed = straddle_cost / current_price
                        
                        earnings_plays.append({
                            'symbol': symbol,
                            'strategy': 'LONG_STRADDLE',
                            'current_price': current_price,
                            'strike': strike,
                            'call_price': call_price,
                            'put_price': put_price,
                            'straddle_cost': straddle_cost,
                            'breakeven_up': breakeven_up,
                            'breakeven_down': breakeven_down,
                            'move_needed_pct': move_needed * 100,
                            'avg_iv': avg_iv * 100,
                            'dte': atm_calls['dte'].iloc[0]
                        })
                
            except Exception as e:
                continue
        
        return pd.DataFrame(earnings_plays)
    
    def find_spread_opportunities(self):
        """Find high-probability spread opportunities."""
        print("\n SCANNING FOR HIGH-EDGE SPREAD OPPORTUNITIES...")
        print("=" * 60)
        
        spread_opportunities = []
        
        for symbol in self.focus_symbols[:8]:  # Focus on most liquid
            try:
                options_df = self.get_options_chain(symbol)
                if options_df is None or options_df.empty:
                    continue
                
                current_price = options_df['current_price'].iloc[0]
                
                # Find bull call spreads (bullish bias)
                calls = options_df[options_df['type'] == 'call'].copy()
                if len(calls) < 2:
                    continue
                
                # Look for spreads with good risk/reward
                for _, long_call in calls.iterrows():
                    if long_call['moneyness'] < 0.95 or long_call['moneyness'] > 1.05:
                        continue
                    
                    # Find short call 5-10 strikes higher
                    short_calls = calls[
                        (calls['strike'] > long_call['strike']) &
                        (calls['strike'] <= long_call['strike'] + 15) &
                        (calls['expiration'] == long_call['expiration'])
                    ]
                    
                    for _, short_call in short_calls.iterrows():
                        # Calculate spread metrics
                        net_debit = long_call['mid_price'] - short_call['mid_price']
                        max_profit = (short_call['strike'] - long_call['strike']) - net_debit
                        max_loss = net_debit
                        
                        if net_debit <= 0 or max_profit <= 0:
                            continue
                        
                        risk_reward = max_profit / max_loss
                        breakeven = long_call['strike'] + net_debit
                        
                        # High-edge criteria
                        if (risk_reward >= 1.5 and  # Good risk/reward
                            net_debit <= current_price * 0.05 and  # Low cost
                            long_call['volume'] >= 20 and short_call['volume'] >= 10):
                            
                            spread_opportunities.append({
                                'symbol': symbol,
                                'strategy': 'BULL_CALL_SPREAD',
                                'long_strike': long_call['strike'],
                                'short_strike': short_call['strike'],
                                'expiration': long_call['expiration'],
                                'dte': long_call['dte'],
                                'net_debit': net_debit,
                                'max_profit': max_profit,
                                'max_loss': max_loss,
                                'risk_reward': risk_reward,
                                'breakeven': breakeven,
                                'current_price': current_price,
                                'profit_potential_pct': (max_profit / max_loss) * 100
                            })
                
            except Exception as e:
                continue
        
        return pd.DataFrame(spread_opportunities)
    
    def generate_high_edge_report(self):
        """Generate comprehensive high-edge opportunities report."""
        print("\n" + "=" * 60)
        print("   HIGH-EDGE OPTIONS RESEARCH REPORT")
        print("   Target: 20%+ Monthly Returns")
        print("=" * 60)
        
        # 1. Volatility Arbitrage
        vol_arb = self.find_volatility_arbitrage_opportunities()
        
        if not vol_arb.empty:
            print("\nTOP VOLATILITY ARBITRAGE OPPORTUNITIES:")
            print("-" * 50)
            
            top_vol_arb = vol_arb.head(5)
            for _, opp in top_vol_arb.iterrows():
                print(f"{opp['symbol']} {opp['type'].upper()} ${opp['strike']:.0f} "
                      f"({opp['dte']}DTE) - {opp['strategy']}")
                print(f"  IV: {opp['implied_vol']*100:.1f}% vs RV: {opp['realized_vol']*100:.1f}% "
                      f"(Edge: {opp['vol_edge_pct']*100:+.1f}%)")
                print(f"  Mid: ${opp['mid_price']:.2f}, Volume: {opp['volume']:.0f}")
                print(f"  >> Profit Potential: {opp['profit_potential']:.1f}%\n")
        
        # 2. Earnings Plays
        earnings = self.find_earnings_straddle_opportunities()
        
        if not earnings.empty:
            print("\n>> TOP EARNINGS VOLATILITY PLAYS:")
            print("-" * 50)
            
            for _, play in earnings.head(3).iterrows():
                print(f"{play['symbol']} STRADDLE ${play['strike']:.0f} "
                      f"({play['dte']}DTE)")
                print(f"  Cost: ${play['straddle_cost']:.2f} "
                      f"(Move needed: {play['move_needed_pct']:.1f}%)")
                print(f"  Breakevens: ${play['breakeven_down']:.2f} - ${play['breakeven_up']:.2f}")
                print(f"  IV: {play['avg_iv']:.1f}%\n")
        
        # 3. Spread Opportunities
        spreads = self.find_spread_opportunities()
        
        if not spreads.empty:
            print("\nTOP SPREAD OPPORTUNITIES:")
            print("-" * 50)
            
            top_spreads = spreads.head(5)
            for _, spread in top_spreads.iterrows():
                print(f"{spread['symbol']} {spread['long_strike']:.0f}/"
                      f"{spread['short_strike']:.0f} BULL CALL ({spread['dte']}DTE)")
                print(f"  Cost: ${spread['net_debit']:.2f}, Max Profit: ${spread['max_profit']:.2f}")
                print(f"  Risk/Reward: {spread['risk_reward']:.1f}:1")
                print(f"  >> Profit Potential: {spread['profit_potential_pct']:.1f}%\n")
        
        print("\n" + "=" * 60)
        print("SUMMARY: Focus on highest profit potential opportunities")
        print("Risk Management: Never risk more than 2-3% per trade")
        print("Target: 5-7 high-edge trades per month for 20%+ returns")
        print("=" * 60)
        
        return {
            'volatility_arbitrage': vol_arb,
            'earnings_plays': earnings,
            'spread_opportunities': spreads
        }

def main():
    """Run the high-edge options research."""
    researcher = HighEdgeOptionsResearcher()
    
    # Generate comprehensive report
    opportunities = researcher.generate_high_edge_report()
    
    # Save to CSV for further analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    if not opportunities['volatility_arbitrage'].empty:
        opportunities['volatility_arbitrage'].to_csv(
            f'vol_arbitrage_opportunities_{timestamp}.csv', index=False
        )
    
    if not opportunities['earnings_plays'].empty:
        opportunities['earnings_plays'].to_csv(
            f'earnings_opportunities_{timestamp}.csv', index=False
        )
    
    if not opportunities['spread_opportunities'].empty:
        opportunities['spread_opportunities'].to_csv(
            f'spread_opportunities_{timestamp}.csv', index=False
        )
    
    print(f"\nOpportunity data saved with timestamp: {timestamp}")

if __name__ == "__main__":
    main()