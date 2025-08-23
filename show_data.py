import json
import requests

try:
    response = requests.get('http://localhost:8001/api/dashboard/live-feed')
    data = response.json()
    
    print('=== LIVE DASHBOARD DATA ===')
    print(f'Portfolio Value: ${data["portfolio"]["totalValue"]:,.2f}')
    print(f'Total P&L: ${data["portfolio"]["totalPnL"]:,.2f}')
    print(f'Day P&L: ${data["portfolio"]["dayPnL"]:,.2f}')
    print(f'Positions: {len(data["positions"])}')
    print()
    
    print('POSITIONS:')
    for pos in data['positions'][:3]:
        print(f'  {pos["symbol"]}: {pos["qty"]} @ ${pos["price"]} | P&L: ${pos["pnl"]:,.2f}')
    print()
    
    print('MARKET DATA:')
    for mkt in data['market'][:3]:
        print(f'  {mkt["symbol"]}: ${mkt["last"]} ({mkt["chg"]:+.2f})')
    print()
    
    print('AI SIGNALS:')
    active_signals = [s for s in data['signals'] if s['signal'] != 'HOLD']
    print(f'  Active: {len(active_signals)}/6 agents')
    for sig in active_signals[:2]:
        print(f'  {sig["agent"]}: {sig["signal"]} {sig["symbol"]}')
        
except Exception as e:
    print(f'Error: {e}')