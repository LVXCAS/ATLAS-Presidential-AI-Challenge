#!/usr/bin/env python3
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
from datetime import datetime
import requests

print("=" * 60)
print("HIVE TRADE - COMPLETE SYSTEM STATUS")
print("=" * 60)

# Check API connection
load_dotenv()
api = tradeapi.REST(
    os.getenv('ALPACA_API_KEY'),
    os.getenv('ALPACA_SECRET_KEY'),
    os.getenv('ALPACA_BASE_URL'),
    api_version='v2'
)

print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Current positions
print(">> LIVE CRYPTO POSITIONS:")
try:
    positions = api.list_positions()
    crypto_positions = [p for p in positions if 'USD' in p.symbol and len(p.symbol) > 5]
    
    total_pnl = 0
    for pos in crypto_positions:
        pnl = float(pos.unrealized_pl)
        value = float(pos.market_value)
        total_pnl += pnl
        pnl_pct = (pnl / value) * 100 if value != 0 else 0
        print(f"   {pos.symbol}: {pos.qty} | ${value:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
    
    print(f"   TOTAL P&L: ${total_pnl:+.2f}")
except Exception as e:
    print(f"   Error: {e}")

print()

# Trade log
print(">> RECENT LIVE TRADES:")
try:
    if os.path.exists('live_crypto_trades.log'):
        with open('live_crypto_trades.log', 'r') as f:
            trades = f.read().strip().split('\n')
            for trade in trades[-5:]:  # Last 5 trades
                if trade:
                    parts = trade.split(',')
                    if len(parts) >= 4:
                        timestamp = parts[0]
                        symbol = parts[1] 
                        side = parts[2]
                        amount = parts[3]
                        print(f"   {timestamp[11:19]} - {symbol} {side} ${amount}")
    else:
        print("   No trade log found")
except Exception as e:
    print(f"   Error: {e}")

print()

# Backend status
print(">> BACKEND API STATUS:")
try:
    response = requests.get('http://localhost:8001/', timeout=5)
    if response.status_code == 200:
        print("   Backend API: RUNNING ✓")
        
        # Try crypto endpoint
        crypto_response = requests.get('http://localhost:8001/api/crypto/status', timeout=5)
        if crypto_response.status_code == 200:
            print("   Crypto API: CONNECTED ✓")
        else:
            print("   Crypto API: ERROR")
    else:
        print("   Backend API: ERROR")
except Exception as e:
    print(f"   Backend API: OFFLINE ({e})")

print()

# Account status
print(">> ACCOUNT STATUS:")
try:
    account = api.get_account()
    print(f"   Portfolio: ${float(account.portfolio_value):,.2f}")
    print(f"   Buying Power: ${float(account.buying_power):,.2f}")
    print(f"   Day Trade Count: {account.daytrade_count}")
except Exception as e:
    print(f"   Error: {e}")

print()
print("=" * 60)
print("SYSTEM STATUS COMPLETE")
print("=" * 60)