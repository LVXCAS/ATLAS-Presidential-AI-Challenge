#!/usr/bin/env python3
"""
HIVE TRADING EMPIRE - OpenBB Terminal Demo
=========================================

Demo of market data terminal capabilities
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def demo_terminal():
    """Demonstrate terminal capabilities"""
    print("[HIVE] HIVE TRADING EMPIRE - MARKET TERMINAL DEMO")
    print("=" * 55)
    print("")
    
    # Demo: Market Quote
    print("[QUOTE] GETTING MARKET QUOTE FOR SPY...")
    try:
        ticker = yf.Ticker("SPY")
        hist = ticker.history(period="2d")
        
        if not hist.empty:
            current = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current
            change = current - prev_close
            change_pct = (change / prev_close) * 100
            
            print(f"Price: ${current:.2f}")
            print(f"Change: ${change:+.2f} ({change_pct:+.2f}%)")
            print(f"Volume: {hist['Volume'].iloc[-1]:,.0f}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "-" * 50)
    
    # Demo: Market Summary
    print("\n[MARKETS] MARKET SUMMARY:")
    indices = [
        ('SPY', 'S&P 500'),
        ('QQQ', 'NASDAQ'),
        ('IWM', 'Russell 2000'),
        ('AAPL', 'Apple'),
        ('MSFT', 'Microsoft')
    ]
    
    for symbol, name in indices:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current
                change_pct = ((current - prev_close) / prev_close) * 100
                
                status = "[+]" if change_pct > 0 else "[-]"
                print(f"{status} {name:15} ({symbol}): ${current:7.2f} ({change_pct:+5.2f}%)")
        except:
            print(f"[X] {name:15} ({symbol}): Data unavailable")
    
    print("\n" + "-" * 50)
    
    # Demo: Moving Averages for AAPL
    print("\n[MA] MOVING AVERAGES FOR AAPL:")
    try:
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="6mo")
        
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            ma_5 = hist['Close'].rolling(5).mean().iloc[-1]
            ma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            ma_50 = hist['Close'].rolling(50).mean().iloc[-1]
            
            print(f"Current Price: ${current_price:.2f}")
            print(f"5-day MA:      ${ma_5:.2f}")
            print(f"20-day MA:     ${ma_20:.2f}")
            print(f"50-day MA:     ${ma_50:.2f}")
            
            # Signal analysis
            if current_price > ma_20 > ma_50:
                print("[+] SIGNAL: Bullish - Price above key moving averages")
            elif current_price < ma_20 < ma_50:
                print("[-] SIGNAL: Bearish - Price below key moving averages")
            else:
                print("[~] SIGNAL: Mixed - Neutral trend")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "-" * 50)
    
    # Demo: Company Info
    print("\n[INFO] COMPANY INFO FOR TESLA (TSLA):")
    try:
        ticker = yf.Ticker("TSLA")
        info = ticker.info
        
        fields = [
            ('Name', 'longName'),
            ('Sector', 'sector'),
            ('Industry', 'industry'),
            ('Market Cap', 'marketCap'),
            ('P/E Ratio', 'trailingPE'),
            ('52-Week High', 'fiftyTwoWeekHigh'),
            ('52-Week Low', 'fiftyTwoWeekLow')
        ]
        
        for label, key in fields:
            if key in info and info[key] is not None:
                value = info[key]
                if key == 'marketCap' and isinstance(value, (int, float)):
                    print(f"{label:15}: ${value:,.0f}")
                elif isinstance(value, (int, float)):
                    print(f"{label:15}: {value:.2f}")
                else:
                    print(f"{label:15}: {value}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 55)
    print("[DONE] TERMINAL DEMO COMPLETE!")
    print("")
    print("[TIP] To use the interactive terminal, run:")
    print("   python simple_openbb_terminal.py")
    print("")
    print("Available commands in interactive mode:")
    print("  quote, chart, info, news, ma, summary, help, quit")
    print("  set <symbol> - Change current symbol")
    print("")


if __name__ == "__main__":
    demo_terminal()