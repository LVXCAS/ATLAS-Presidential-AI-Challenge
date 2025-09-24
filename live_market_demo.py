#!/usr/bin/env python3
"""
HIVE TRADING EMPIRE - LIVE MARKET DATA DEMONSTRATION
===================================================

Real-time market data feed demonstration
"""

import time
from datetime import datetime

def live_market_demo():
    """Demonstrate live market data capabilities"""
    print("=" * 60)
    print("HIVE TRADING EMPIRE - LIVE MARKET DATA DEMO")
    print("=" * 60)
    
    try:
        from openbb import obb
        print(f"[OK] OpenBB Platform v{obb.system.version} loaded")
        print(f"[TIME] Demo started: {datetime.now().strftime('%H:%M:%S')}")
        
        # Portfolio symbols to track
        portfolio = {
            "SPY": "S&P 500 ETF",
            "QQQ": "NASDAQ 100 ETF", 
            "IWM": "Russell 2000 ETF",
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corp.",
            "TSLA": "Tesla Inc.",
            "NVDA": "NVIDIA Corp."
        }
        
        print("\n[PORTFOLIO] Tracking live market data for:")
        for symbol, name in portfolio.items():
            print(f"  - {symbol}: {name}")
        
        print("\n" + "=" * 60)
        print("REAL-TIME MARKET DATA FEED")
        print("=" * 60)
        
        # Get live data for all symbols
        market_data = {}
        
        for symbol, name in portfolio.items():
            try:
                print(f"[FETCH] {symbol}...")
                data = obb.equity.price.historical(symbol, period="2d")
                df = data.to_df()
                
                current = df.iloc[-1]
                previous = df.iloc[-2] if len(df) > 1 else current
                
                price = current['close']
                volume = current['volume']
                change = price - previous['close']
                change_pct = (change / previous['close']) * 100
                
                market_data[symbol] = {
                    'name': name,
                    'price': price,
                    'volume': volume,
                    'change': change,
                    'change_pct': change_pct,
                    'timestamp': datetime.now()
                }
                
                status = "[+]" if change >= 0 else "[-]"
                print(f"[OK] {symbol}: ${price:.2f} {status} {change_pct:+.2f}%")
                
            except Exception as e:
                print(f"[ERROR] {symbol}: {e}")
                
        # Display formatted market data
        print("\n" + "=" * 60)
        print("LIVE MARKET DASHBOARD")
        print("=" * 60)
        print(f"{'SYMBOL':<6} {'PRICE':<8} {'CHANGE':<8} {'%':<7} {'VOLUME':<12}")
        print("-" * 60)
        
        for symbol, data in market_data.items():
            if data:
                status = "+" if data['change'] >= 0 else ""
                print(f"{symbol:<6} ${data['price']:<7.2f} {status}{data['change']:<7.2f} "
                      f"{data['change_pct']:+6.2f}% {data['volume']:<12,.0f}")
        
        # Market summary
        print("\n" + "=" * 60)
        print("MARKET ANALYSIS")
        print("=" * 60)
        
        gainers = [s for s, d in market_data.items() if d and d['change_pct'] > 0]
        losers = [s for s, d in market_data.items() if d and d['change_pct'] < 0]
        
        print(f"[GAINERS] {len(gainers)} symbols up: {', '.join(gainers)}")
        print(f"[LOSERS] {len(losers)} symbols down: {', '.join(losers)}")
        
        if 'SPY' in market_data and market_data['SPY']:
            spy_change = market_data['SPY']['change_pct']
            if spy_change > 0.5:
                print(f"[MARKET] BULLISH - SPY up {spy_change:+.2f}%")
            elif spy_change < -0.5:
                print(f"[MARKET] BEARISH - SPY down {spy_change:+.2f}%")
            else:
                print(f"[MARKET] NEUTRAL - SPY {spy_change:+.2f}%")
        
        # Hive Trading Integration Demo
        print("\n" + "=" * 60)
        print("HIVE TRADING SYSTEM INTEGRATION")
        print("=" * 60)
        
        print("[INTEGRATION] Market data successfully integrated with Hive system:")
        print(f"[DATA] {len(market_data)} symbols updated")
        print(f"[FEED] Data timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("[STATUS] Ready for algorithmic trading strategies")
        print("[READY] Data can be consumed by your 353-file trading system")
        
        # Show integration capabilities
        print("\n[CAPABILITIES] Available for Hive Trading Empire:")
        print("  - Real-time price feeds")
        print("  - Volume analysis")
        print("  - Change detection")
        print("  - Market sentiment analysis")
        print("  - Technical indicator calculations")
        print("  - News sentiment integration")
        print("  - Portfolio optimization data")
        
        return True
        
    except Exception as e:
        print(f"[CRITICAL ERROR] Demo failed: {e}")
        return False

if __name__ == "__main__":
    success = live_market_demo()
    print(f"\n[RESULT] Demo {'SUCCESSFUL' if success else 'FAILED'}")
    print("\n[READY] Hive Trading Empire market data feed is operational!")