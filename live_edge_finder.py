"""
LIVE HIGH-EDGE OPTIONS FINDER
============================
Uses Alpaca API for real-time options data to find 20%+ monthly opportunities
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LiveEdgeFinder:
    """Find high-edge options opportunities using live data."""
    
    def __init__(self):
        """Initialize with Alpaca credentials."""
        self.alpaca_key = os.getenv('ALPACA_API_KEY')
        self.alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = "https://paper-api.alpaca.markets"
        
        if not self.alpaca_key:
            print("ERROR: ALPACA_API_KEY not found in environment")
            return
            
        self.headers = {
            'APCA-API-KEY-ID': self.alpaca_key,
            'APCA-API-SECRET-KEY': self.alpaca_secret,
        }
        
        # High-edge focus symbols (most liquid options)
        self.edge_symbols = [
            'SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT',
            'AMZN', 'GOOGL', 'META', 'NFLX', 'IWM', 'XLF', 'XLE'
        ]
        
        print("LIVE EDGE FINDER INITIALIZED")
        print("=" * 50)
        print(f"Connected to Alpaca: {self.alpaca_key[:8]}...")
        print(f"Target symbols: {len(self.edge_symbols)}")
    
    def get_stock_price(self, symbol):
        """Get current stock price from Alpaca."""
        try:
            url = f"{self.base_url}/v2/stocks/{symbol}/bars/latest"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                return data['bar']['c']  # Close price
            else:
                print(f"API Error for {symbol}: {response.status_code} - {response.text}")
            
            return None
            
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return None
    
    def calculate_historical_volatility(self, symbol, days=30):
        """Calculate historical volatility for comparison."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days*2)
            
            url = f"{self.base_url}/v2/stocks/{symbol}/bars"
            params = {
                'start': start_date.strftime('%Y-%m-%dT00:00:00Z'),
                'end': end_date.strftime('%Y-%m-%dT00:00:00Z'),
                'timeframe': '1Day'
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                bars = data.get('bars', [])
                
                if len(bars) < days:
                    return None
                
                prices = [bar['c'] for bar in bars]
                returns = np.diff(np.log(prices))
                
                historical_vol = np.std(returns) * np.sqrt(252)
                return historical_vol
            
            return None
            
        except Exception as e:
            print(f"Error calculating volatility for {symbol}: {e}")
            return None
    
    def find_momentum_opportunities(self):
        """Find stocks with strong momentum for directional plays."""
        print("\nSCANNING FOR MOMENTUM OPPORTUNITIES...")
        print("-" * 40)
        
        momentum_plays = []
        
        for symbol in self.edge_symbols:
            try:
                # Get recent price data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                url = f"{self.base_url}/v2/stocks/{symbol}/bars"
                params = {
                    'start': start_date.strftime('%Y-%m-%dT00:00:00Z'),
                    'end': end_date.strftime('%Y-%m-%dT00:00:00Z'),
                    'timeframe': '1Day'
                }
                
                response = requests.get(url, headers=self.headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    bars = data.get('bars', [])
                    print(f"Got {len(bars)} bars for {symbol}")
                    
                    if len(bars) < 20:
                        print(f"Insufficient data for {symbol}: only {len(bars)} bars")
                        continue
                else:
                    print(f"API Error for {symbol}: {response.status_code} - {response.text}")
                    continue
                    
                    prices = [bar['c'] for bar in bars]
                    volumes = [bar['v'] for bar in bars]
                    
                    # Calculate momentum metrics
                    current_price = prices[-1]
                    price_20d_ago = prices[-20]
                    price_5d_ago = prices[-5]
                    
                    # 20-day return
                    momentum_20d = (current_price - price_20d_ago) / price_20d_ago
                    
                    # 5-day return
                    momentum_5d = (current_price - price_5d_ago) / price_5d_ago
                    
                    # Volume surge (recent vs average)
                    recent_volume = np.mean(volumes[-3:])
                    avg_volume = np.mean(volumes[:-3])
                    volume_surge = recent_volume / avg_volume if avg_volume > 0 else 1
                    
                    # Historical volatility
                    returns = np.diff(np.log(prices))
                    volatility = np.std(returns) * np.sqrt(252)
                    
                    # High-edge criteria
                    if (abs(momentum_20d) > 0.15 and  # Strong 20-day move
                        abs(momentum_5d) > 0.05 and   # Recent acceleration
                        volume_surge > 1.5 and       # Volume confirmation
                        volatility > 0.25):          # High volatility
                        
                        direction = "BULLISH" if momentum_5d > 0 else "BEARISH"
                        
                        momentum_plays.append({
                            'symbol': symbol,
                            'current_price': current_price,
                            'momentum_20d': momentum_20d * 100,
                            'momentum_5d': momentum_5d * 100,
                            'volume_surge': volume_surge,
                            'volatility': volatility * 100,
                            'direction': direction,
                            'edge_score': abs(momentum_20d) * volume_surge * volatility * 1000
                        })
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                continue
        
        return sorted(momentum_plays, key=lambda x: x['edge_score'], reverse=True)
    
    def find_volatility_contraction_opportunities(self):
        """Find stocks with volatility contraction (coiling for breakout)."""
        print("\nSCANNING FOR VOLATILITY BREAKOUT SETUPS...")
        print("-" * 40)
        
        breakout_setups = []
        
        for symbol in self.edge_symbols:
            try:
                # Get 60 days of data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=90)
                
                url = f"{self.base_url}/v2/stocks/{symbol}/bars"
                params = {
                    'start': start_date.strftime('%Y-%m-%dT00:00:00Z'),
                    'end': end_date.strftime('%Y-%m-%dT00:00:00Z'),
                    'timeframe': '1Day'
                }
                
                response = requests.get(url, headers=self.headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    bars = data.get('bars', [])
                    
                    if len(bars) < 60:
                        continue
                    
                    prices = [bar['c'] for bar in bars]
                    
                    # Calculate rolling volatility
                    returns = np.diff(np.log(prices))
                    
                    # Recent volatility (last 10 days)
                    recent_vol = np.std(returns[-10:]) * np.sqrt(252)
                    
                    # Historical volatility (previous 50 days)
                    historical_vol = np.std(returns[-60:-10]) * np.sqrt(252)
                    
                    # Volatility ratio
                    vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1
                    
                    # Price compression (trading range tightening)
                    recent_range = (max(prices[-10:]) - min(prices[-10:])) / prices[-1]
                    historical_range = (max(prices[-30:-10]) - min(prices[-30:-10])) / prices[-20]
                    
                    range_compression = recent_range / historical_range if historical_range > 0 else 1
                    
                    current_price = prices[-1]
                    
                    # Volatility contraction criteria
                    if (vol_ratio < 0.7 and           # Volatility contracted 30%+
                        range_compression < 0.8 and   # Price range tightening
                        historical_vol > 0.25 and     # Stock normally volatile
                        recent_vol > 0.15):           # Still has minimum volatility
                        
                        # Calculate breakout potential
                        breakout_potential = historical_vol / recent_vol
                        
                        breakout_setups.append({
                            'symbol': symbol,
                            'current_price': current_price,
                            'recent_vol': recent_vol * 100,
                            'historical_vol': historical_vol * 100,
                            'vol_contraction': (1 - vol_ratio) * 100,
                            'range_compression': (1 - range_compression) * 100,
                            'breakout_potential': breakout_potential,
                            'edge_score': breakout_potential * (1 - vol_ratio) * 100
                        })
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                continue
        
        return sorted(breakout_setups, key=lambda x: x['edge_score'], reverse=True)
    
    def generate_edge_report(self):
        """Generate comprehensive edge opportunities report."""
        print("=" * 60)
        print("   LIVE HIGH-EDGE OPTIONS OPPORTUNITIES")
        print("   Target: 20%+ Monthly Returns")
        print("=" * 60)
        
        # 1. Momentum opportunities
        momentum_ops = self.find_momentum_opportunities()
        
        if momentum_ops:
            print("\nTOP MOMENTUM PLAYS (for directional options):")
            print("-" * 50)
            
            for play in momentum_ops[:5]:
                print(f"{play['symbol']} - {play['direction']}")
                print(f"  Price: ${play['current_price']:.2f}")
                print(f"  20D Momentum: {play['momentum_20d']:+.1f}%")
                print(f"  5D Momentum: {play['momentum_5d']:+.1f}%")
                print(f"  Volume Surge: {play['volume_surge']:.1f}x")
                print(f"  Volatility: {play['volatility']:.1f}%")
                print(f"  >> Strategy: Buy {'CALLS' if play['direction'] == 'BULLISH' else 'PUTS'}")
                print(f"  >> Edge Score: {play['edge_score']:.0f}\n")
        
        # 2. Volatility breakout setups
        breakout_ops = self.find_volatility_contraction_opportunities()
        
        if breakout_ops:
            print("\nTOP BREAKOUT SETUPS (for straddle/strangle plays):")
            print("-" * 50)
            
            for setup in breakout_ops[:5]:
                print(f"{setup['symbol']} - VOLATILITY CONTRACTION")
                print(f"  Price: ${setup['current_price']:.2f}")
                print(f"  Volatility Contraction: {setup['vol_contraction']:.1f}%")
                print(f"  Range Compression: {setup['range_compression']:.1f}%")
                print(f"  Breakout Potential: {setup['breakout_potential']:.1f}x")
                print(f"  >> Strategy: LONG STRADDLE/STRANGLE")
                print(f"  >> Edge Score: {setup['edge_score']:.0f}\n")
        
        print("\n" + "=" * 60)
        print("LIVE EDGE SUMMARY:")
        print(f"Momentum opportunities: {len(momentum_ops)}")
        print(f"Breakout setups: {len(breakout_ops)}")
        print("\nSTRATEGY RECOMMENDATIONS:")
        print("1. Use momentum plays for directional options (calls/puts)")
        print("2. Use breakout setups for volatility plays (straddles)")
        print("3. Risk 2-3% per trade maximum")
        print("4. Target 30-60 DTE for time decay buffer")
        print("=" * 60)
        
        return {
            'momentum_opportunities': momentum_ops,
            'breakout_setups': breakout_ops
        }

def main():
    """Run the live edge finder."""
    finder = LiveEdgeFinder()
    
    if not finder.alpaca_key:
        return
    
    # Generate live opportunities
    opportunities = finder.generate_edge_report()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    if opportunities['momentum_opportunities']:
        df_momentum = pd.DataFrame(opportunities['momentum_opportunities'])
        df_momentum.to_csv(f'live_momentum_opportunities_{timestamp}.csv', index=False)
    
    if opportunities['breakout_setups']:
        df_breakout = pd.DataFrame(opportunities['breakout_setups'])
        df_breakout.to_csv(f'live_breakout_setups_{timestamp}.csv', index=False)
    
    print(f"\nLive opportunity data saved with timestamp: {timestamp}")

if __name__ == "__main__":
    main()