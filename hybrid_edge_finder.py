"""
HYBRID HIGH-EDGE OPTIONS FINDER
===============================
Uses multiple data sources for maximum reliability in finding 20%+ opportunities
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time

class HybridEdgeFinder:
    """Find high-edge options opportunities using hybrid data approach."""
    
    def __init__(self):
        """Initialize with robust data sourcing."""
        
        # High-edge focus symbols (most liquid options)
        self.edge_symbols = [
            'SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT',
            'AMZN', 'GOOGL', 'META', 'NFLX', 'IWM', 'XLF', 'XLE'
        ]
        
        print("HYBRID EDGE FINDER INITIALIZED")
        print("=" * 50)
        print(f"Target symbols: {len(self.edge_symbols)}")
        print("Data sources: Yahoo Finance (primary)")
    
    def get_stock_data(self, symbol, days=60):
        """Get comprehensive stock data."""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            hist = ticker.history(
                period=f"{days}d",
                interval="1d",
                auto_adjust=True,
                prepost=False
            )
            
            if hist.empty:
                return None
                
            return hist
            
        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
            return None
    
    def find_momentum_opportunities(self):
        """Find stocks with strong momentum for directional plays."""
        print("\nSCANNING FOR MOMENTUM OPPORTUNITIES...")
        print("-" * 40)
        
        momentum_plays = []
        
        for symbol in self.edge_symbols:
            try:
                data = self.get_stock_data(symbol, days=30)
                
                if data is None or len(data) < 20:
                    print(f"Insufficient data for {symbol}")
                    continue
                
                prices = data['Close'].values
                volumes = data['Volume'].values
                
                # Calculate momentum metrics
                current_price = prices[-1]
                price_20d_ago = prices[-20] if len(prices) >= 20 else prices[0]
                price_5d_ago = prices[-5] if len(prices) >= 5 else prices[-1]
                
                # Returns
                momentum_20d = (current_price - price_20d_ago) / price_20d_ago
                momentum_5d = (current_price - price_5d_ago) / price_5d_ago
                
                # Volume analysis
                recent_volume = np.mean(volumes[-3:])
                avg_volume = np.mean(volumes[:-3]) if len(volumes) > 3 else recent_volume
                volume_surge = recent_volume / avg_volume if avg_volume > 0 else 1
                
                # Volatility
                returns = np.diff(np.log(prices))
                volatility = np.std(returns) * np.sqrt(252)
                
                # HIGH-EDGE CRITERIA (aggressive for 20%+ monthly targets)
                if (abs(momentum_20d) > 0.10 and    # Strong 20-day move (10%+)
                    abs(momentum_5d) > 0.03 and     # Recent acceleration (3%+)
                    volume_surge > 1.2 and         # Volume confirmation
                    volatility > 0.20):            # High volatility (20%+)
                    
                    direction = "BULLISH" if momentum_5d > 0 else "BEARISH"
                    
                    # Calculate edge score (higher = better opportunity)
                    edge_score = (abs(momentum_20d) * 100 * 
                                volume_surge * 
                                volatility * 100)
                    
                    momentum_plays.append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'momentum_20d': momentum_20d * 100,
                        'momentum_5d': momentum_5d * 100,
                        'volume_surge': volume_surge,
                        'volatility': volatility * 100,
                        'direction': direction,
                        'edge_score': edge_score,
                        'target_return': abs(momentum_20d) * 100 * 2  # Estimated options multiplier
                    })
                    
                    print(f"FOUND: {symbol} - {direction} momentum")
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
        
        return sorted(momentum_plays, key=lambda x: x['edge_score'], reverse=True)
    
    def find_volatility_compression_setups(self):
        """Find stocks coiling for breakouts."""
        print("\nSCANNING FOR VOLATILITY BREAKOUT SETUPS...")
        print("-" * 40)
        
        breakout_setups = []
        
        for symbol in self.edge_symbols:
            try:
                data = self.get_stock_data(symbol, days=90)
                
                if data is None or len(data) < 60:
                    print(f"Insufficient data for {symbol}")
                    continue
                
                prices = data['Close'].values
                
                # Calculate volatility metrics
                returns = np.diff(np.log(prices))
                
                # Recent volatility (last 10 days)
                recent_vol = np.std(returns[-10:]) * np.sqrt(252)
                
                # Historical volatility (previous 50 days)
                historical_vol = np.std(returns[-60:-10]) * np.sqrt(252)
                
                # Volatility ratio
                vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1
                
                # Price compression
                recent_range = (np.max(prices[-10:]) - np.min(prices[-10:])) / prices[-1]
                hist_range = (np.max(prices[-30:-10]) - np.min(prices[-30:-10])) / prices[-20]
                range_compression = recent_range / hist_range if hist_range > 0 else 1
                
                current_price = prices[-1]
                
                # VOLATILITY CONTRACTION CRITERIA
                if (vol_ratio < 0.75 and           # Volatility contracted 25%+
                    range_compression < 0.85 and   # Price range tightening
                    historical_vol > 0.20 and     # Stock normally volatile
                    recent_vol > 0.12):           # Still has minimum volatility
                    
                    breakout_potential = historical_vol / recent_vol
                    edge_score = breakout_potential * (1 - vol_ratio) * 1000
                    
                    breakout_setups.append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'recent_vol': recent_vol * 100,
                        'historical_vol': historical_vol * 100,
                        'vol_contraction': (1 - vol_ratio) * 100,
                        'range_compression': (1 - range_compression) * 100,
                        'breakout_potential': breakout_potential,
                        'edge_score': edge_score,
                        'target_return': breakout_potential * 15  # Estimated straddle return
                    })
                    
                    print(f"FOUND: {symbol} - volatility compression setup")
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
        
        return sorted(breakout_setups, key=lambda x: x['edge_score'], reverse=True)
    
    def analyze_earnings_opportunities(self):
        """Find upcoming earnings plays."""
        print("\nSCANNING FOR EARNINGS OPPORTUNITIES...")
        print("-" * 40)
        
        earnings_plays = []
        
        for symbol in self.edge_symbols[:5]:  # Limit for demo
            try:
                ticker = yf.Ticker(symbol)
                calendar = ticker.calendar
                
                if calendar is not None and not calendar.empty:
                    # Check if earnings are within next 2 weeks
                    next_earnings = calendar.index[0]
                    days_to_earnings = (next_earnings - datetime.now()).days
                    
                    if 1 <= days_to_earnings <= 14:  # 1-14 days out
                        data = self.get_stock_data(symbol, days=90)
                        
                        if data is not None and len(data) >= 20:
                            # Calculate historical earnings moves
                            returns = data['Close'].pct_change()
                            volatility = returns.std() * np.sqrt(252)
                            
                            # Estimate earnings move (simplified)
                            estimated_move = volatility * 0.15  # 15% of annual vol
                            
                            earnings_plays.append({
                                'symbol': symbol,
                                'current_price': data['Close'].iloc[-1],
                                'earnings_date': next_earnings.strftime('%Y-%m-%d'),
                                'days_to_earnings': days_to_earnings,
                                'estimated_move': estimated_move * 100,
                                'volatility': volatility * 100,
                                'strategy': 'LONG STRADDLE',
                                'target_return': estimated_move * 200  # Options leverage
                            })
                            
                            print(f"FOUND: {symbol} earnings on {next_earnings.strftime('%Y-%m-%d')}")
                
                time.sleep(0.2)  # Rate limiting for calendar requests
                
            except Exception as e:
                print(f"Error getting earnings for {symbol}: {e}")
                continue
        
        return earnings_plays
    
    def generate_comprehensive_report(self):
        """Generate comprehensive high-edge opportunities report."""
        print("=" * 70)
        print("        HYBRID HIGH-EDGE OPTIONS OPPORTUNITIES")
        print("           TARGET: 20%+ MONTHLY RETURNS")
        print("=" * 70)
        
        # Get all opportunity types
        momentum_ops = self.find_momentum_opportunities()
        breakout_ops = self.find_volatility_compression_setups()
        earnings_ops = self.analyze_earnings_opportunities()
        
        # Display results
        if momentum_ops:
            print("\nTOP MOMENTUM PLAYS (Directional Options):")
            print("=" * 50)
            
            for i, play in enumerate(momentum_ops[:3], 1):
                print(f"\n#{i}. {play['symbol']} - {play['direction']}")
                print(f"   Price: ${play['current_price']:.2f}")
                print(f"   20D Momentum: {play['momentum_20d']:+.1f}%")
                print(f"   5D Momentum: {play['momentum_5d']:+.1f}%")
                print(f"   Volume Surge: {play['volume_surge']:.1f}x")
                print(f"   Volatility: {play['volatility']:.1f}%")
                print(f"   Strategy: Buy {'CALLS' if play['direction'] == 'BULLISH' else 'PUTS'}")
                print(f"   Target Return: {play['target_return']:.0f}%")
                print(f"   Edge Score: {play['edge_score']:.0f}")
        
        if breakout_ops:
            print("\n\nTOP BREAKOUT SETUPS (Volatility Plays):")
            print("=" * 50)
            
            for i, setup in enumerate(breakout_ops[:3], 1):
                print(f"\n#{i}. {setup['symbol']} - COILED FOR BREAKOUT")
                print(f"   Price: ${setup['current_price']:.2f}")
                print(f"   Vol Contraction: {setup['vol_contraction']:.1f}%")
                print(f"   Range Compression: {setup['range_compression']:.1f}%")
                print(f"   Breakout Potential: {setup['breakout_potential']:.1f}x")
                print(f"   Strategy: LONG STRADDLE/STRANGLE")
                print(f"   Target Return: {setup['target_return']:.0f}%")
                print(f"   Edge Score: {setup['edge_score']:.0f}")
        
        if earnings_ops:
            print("\n\nUPCOMING EARNINGS PLAYS:")
            print("=" * 50)
            
            for play in earnings_ops:
                print(f"\n{play['symbol']} - {play['earnings_date']} ({play['days_to_earnings']} days)")
                print(f"   Price: ${play['current_price']:.2f}")
                print(f"   Estimated Move: {play['estimated_move']:.1f}%")
                print(f"   Strategy: {play['strategy']}")
                print(f"   Target Return: {play['target_return']:.0f}%")
        
        print("\n\n" + "=" * 70)
        print("OPPORTUNITY SUMMARY:")
        print(f"  Momentum opportunities: {len(momentum_ops)}")
        print(f"  Breakout setups: {len(breakout_ops)}")
        print(f"  Earnings plays: {len(earnings_ops)}")
        print("\nHIGH-EDGE TRADING RULES:")
        print("  1. Risk 2-3% of account per trade maximum")
        print("  2. Use 30-60 DTE options for time buffer")
        print("  3. Take profits at 50-100% gains")
        print("  4. Cut losses at 25-30%")
        print("  5. Never risk more than 10% of account daily")
        print("=" * 70)
        
        # Save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        if momentum_ops:
            df = pd.DataFrame(momentum_ops)
            df.to_csv(f'high_edge_momentum_{timestamp}.csv', index=False)
            print(f"\nMomentum data saved: high_edge_momentum_{timestamp}.csv")
        
        if breakout_ops:
            df = pd.DataFrame(breakout_ops)
            df.to_csv(f'high_edge_breakouts_{timestamp}.csv', index=False)
            print(f"Breakout data saved: high_edge_breakouts_{timestamp}.csv")
        
        return {
            'momentum': momentum_ops,
            'breakouts': breakout_ops,
            'earnings': earnings_ops
        }

def main():
    """Run the hybrid edge finder."""
    finder = HybridEdgeFinder()
    
    # Generate comprehensive opportunities
    opportunities = finder.generate_comprehensive_report()
    
    print(f"\nScan completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()