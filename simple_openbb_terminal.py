#!/usr/bin/env python3
"""
HIVE TRADING EMPIRE - SIMPLE OPENBB-LIKE TERMINAL
==============================================

A simple terminal interface for market data analysis using your existing infrastructure.
"""

import os
import sys
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HiveMarketTerminal:
    """Simple market data terminal for Hive Trading Empire"""
    
    def __init__(self):
        self.current_symbol = "SPY"
        print("🚀 HIVE TRADING EMPIRE - MARKET TERMINAL")
        print("=" * 50)
        print("Your simplified OpenBB-like terminal is ready!")
        print("")
        
    def show_help(self):
        """Display available commands"""
        print("\n📋 AVAILABLE COMMANDS:")
        print("-" * 30)
        print("General:")
        print("  help     - Show this help")
        print("  quit     - Exit terminal")
        print("")
        print("Market Data:")
        print("  quote    - Get current quote for symbol")
        print("  chart    - Show price chart")
        print("  info     - Company information")
        print("  news     - Recent news")
        print("  set      - Set current symbol (e.g., 'set AAPL')")
        print("")
        print("Analytics:")
        print("  ma       - Moving averages")
        print("  volume   - Volume analysis")
        print("  summary  - Market summary")
        print("")
        print(f"Current Symbol: {self.current_symbol}")
        print("")
        
    def get_quote(self, symbol=None):
        """Get current quote"""
        symbol = symbol or self.current_symbol
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="2d")
            
            if hist.empty:
                print(f"❌ No data found for {symbol}")
                return
                
            current = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current
            change = current - prev_close
            change_pct = (change / prev_close) * 100
            
            print(f"\n📊 QUOTE: {symbol.upper()}")
            print(f"Price: ${current:.2f}")
            print(f"Change: ${change:+.2f} ({change_pct:+.2f}%)")
            print(f"Volume: {hist['Volume'].iloc[-1]:,.0f}")
            
            if 'marketCap' in info:
                print(f"Market Cap: ${info['marketCap']:,.0f}")
                
        except Exception as e:
            print(f"❌ Error getting quote for {symbol}: {e}")
            
    def show_chart(self, symbol=None, period="1mo"):
        """Show price chart"""
        symbol = symbol or self.current_symbol
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                print(f"❌ No data found for {symbol}")
                return
            
            plt.figure(figsize=(12, 8))
            
            # Price chart
            plt.subplot(2, 1, 1)
            plt.plot(hist.index, hist['Close'], linewidth=2, label='Close Price')
            plt.plot(hist.index, hist['Close'].rolling(20).mean(), '--', label='20-day MA')
            plt.title(f"{symbol.upper()} - Price Chart ({period})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Volume chart
            plt.subplot(2, 1, 2)
            plt.bar(hist.index, hist['Volume'], alpha=0.7)
            plt.title("Volume")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            print(f"✅ Chart displayed for {symbol.upper()}")
            
        except Exception as e:
            print(f"❌ Error creating chart for {symbol}: {e}")
            
    def get_info(self, symbol=None):
        """Get company information"""
        symbol = symbol or self.current_symbol
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            print(f"\n🏢 COMPANY INFO: {symbol.upper()}")
            print("-" * 30)
            
            fields = [
                ('Name', 'longName'),
                ('Sector', 'sector'),
                ('Industry', 'industry'),
                ('Market Cap', 'marketCap'),
                ('P/E Ratio', 'trailingPE'),
                ('52-Week High', 'fiftyTwoWeekHigh'),
                ('52-Week Low', 'fiftyTwoWeekLow'),
                ('Dividend Yield', 'dividendYield')
            ]
            
            for label, key in fields:
                if key in info and info[key] is not None:
                    value = info[key]
                    if key == 'marketCap' and isinstance(value, (int, float)):
                        print(f"{label}: ${value:,.0f}")
                    elif key == 'dividendYield' and isinstance(value, (int, float)):
                        print(f"{label}: {value*100:.2f}%")
                    else:
                        print(f"{label}: {value}")
                        
        except Exception as e:
            print(f"❌ Error getting info for {symbol}: {e}")
            
    def get_news(self, symbol=None):
        """Get recent news"""
        symbol = symbol or self.current_symbol
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            print(f"\n📰 RECENT NEWS: {symbol.upper()}")
            print("-" * 40)
            
            for i, article in enumerate(news[:5]):  # Show top 5
                print(f"{i+1}. {article['title']}")
                print(f"   Source: {article.get('publisher', 'N/A')}")
                print(f"   Time: {datetime.fromtimestamp(article['providerPublishTime'])}")
                print()
                
        except Exception as e:
            print(f"❌ Error getting news for {symbol}: {e}")
            
    def moving_averages(self, symbol=None):
        """Calculate moving averages"""
        symbol = symbol or self.current_symbol
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            
            if hist.empty:
                print(f"❌ No data found for {symbol}")
                return
            
            current_price = hist['Close'].iloc[-1]
            ma_5 = hist['Close'].rolling(5).mean().iloc[-1]
            ma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            ma_50 = hist['Close'].rolling(50).mean().iloc[-1]
            
            print(f"\n📈 MOVING AVERAGES: {symbol.upper()}")
            print("-" * 35)
            print(f"Current Price: ${current_price:.2f}")
            print(f"5-day MA:      ${ma_5:.2f}")
            print(f"20-day MA:     ${ma_20:.2f}")
            print(f"50-day MA:     ${ma_50:.2f}")
            print()
            
            # Signal analysis
            if current_price > ma_20 > ma_50:
                print("🟢 BULLISH: Price above key moving averages")
            elif current_price < ma_20 < ma_50:
                print("🔴 BEARISH: Price below key moving averages")
            else:
                print("🟡 MIXED: Mixed signals from moving averages")
                
        except Exception as e:
            print(f"❌ Error calculating moving averages for {symbol}: {e}")
            
    def market_summary(self):
        """Show market summary"""
        indices = ['SPY', 'QQQ', 'IWM', 'VXX']  # S&P 500, NASDAQ, Russell 2000, VIX
        
        print("\n📊 MARKET SUMMARY")
        print("-" * 50)
        
        for symbol in indices:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current
                    change_pct = ((current - prev_close) / prev_close) * 100
                    
                    status = "🟢" if change_pct > 0 else "🔴"
                    print(f"{status} {symbol}: ${current:.2f} ({change_pct:+.2f}%)")
            except:
                print(f"❌ {symbol}: Data unavailable")
                
    def run(self):
        """Main terminal loop"""
        self.show_help()
        
        while True:
            try:
                command = input(f"\n[{self.current_symbol}] hive> ").strip().lower()
                
                if command in ['quit', 'exit', 'q']:
                    print("👋 Thanks for using Hive Market Terminal!")
                    break
                    
                elif command == 'help':
                    self.show_help()
                    
                elif command == 'quote':
                    self.get_quote()
                    
                elif command == 'chart':
                    self.show_chart()
                    
                elif command == 'info':
                    self.get_info()
                    
                elif command == 'news':
                    self.get_news()
                    
                elif command == 'ma':
                    self.moving_averages()
                    
                elif command == 'summary':
                    self.market_summary()
                    
                elif command.startswith('set '):
                    new_symbol = command.split()[1].upper()
                    self.current_symbol = new_symbol
                    print(f"✅ Current symbol set to {new_symbol}")
                    
                elif command == '':
                    continue
                    
                else:
                    print(f"❌ Unknown command: {command}")
                    print("Type 'help' for available commands")
                    
            except KeyboardInterrupt:
                print("\n👋 Thanks for using Hive Market Terminal!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    terminal = HiveMarketTerminal()
    terminal.run()