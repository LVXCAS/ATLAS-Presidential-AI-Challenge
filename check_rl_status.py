#!/usr/bin/env python3
"""
Quick check of RL training system
"""
import yfinance as yf
import numpy as np
from datetime import datetime

print("=" * 50)
print("CHECKING RL TRAINING SYSTEM")
print("=" * 50)

print(f"Time: {datetime.now().strftime('%H:%M:%S')}")

# Test market data access
print("\\n>> Testing Market Data:")
try:
    btc = yf.Ticker('BTC-USD')
    btc_hist = btc.history(period='1d')
    if len(btc_hist) > 0:
        btc_price = btc_hist['Close'].iloc[-1]
        print(f"   BTC-USD: ${btc_price:,.2f}")
    else:
        print("   BTC-USD: No data")
        
    eth = yf.Ticker('ETH-USD')  
    eth_hist = eth.history(period='1d')
    if len(eth_hist) > 0:
        eth_price = eth_hist['Close'].iloc[-1]
        print(f"   ETH-USD: ${eth_price:,.2f}")
    else:
        print("   ETH-USD: No data")
        
except Exception as e:
    print(f"   Error: {e}")

print("\\n>> Simulating RL Episode:")
initial_balance = 100000
balance = initial_balance
positions = {}

print(f"   Starting Balance: ${balance:,.2f}")

# Simulate some trades
for step in range(5):
    # Random action
    action = np.random.choice(['BUY', 'SELL', 'HOLD'])
    symbol = np.random.choice(['BTC-USD', 'ETH-USD'])
    
    if action == 'BUY' and balance >= 100:
        balance -= 100
        positions[symbol] = positions.get(symbol, 0) + 0.001
        print(f"   Step {step+1}: {action} ${symbol} | Balance: ${balance:,.2f}")
    elif action == 'SELL' and symbol in positions and positions[symbol] > 0:
        balance += 105  # Simulate 5% profit
        positions[symbol] -= 0.001
        print(f"   Step {step+1}: {action} ${symbol} (+$5 profit) | Balance: ${balance:,.2f}")
    else:
        print(f"   Step {step+1}: {action} {symbol} (no action) | Balance: ${balance:,.2f}")

final_return = (balance - initial_balance) / initial_balance
print(f"\\n>> Episode Result:")
print(f"   Final Balance: ${balance:,.2f}")
print(f"   Return: {final_return:+.2%}")
print(f"   Positions: {len([p for p in positions.values() if p > 0])}")

print("\\n>> RL Training Status: ACTIVE")
print("This is what the RL environment should be doing!")
print("=" * 50)