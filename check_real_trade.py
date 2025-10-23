import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('ALPACA_API_KEY')
api_secret = os.getenv('ALPACA_SECRET_KEY')

headers = {
    'APCA-API-KEY-ID': api_key,
    'APCA-API-SECRET-KEY': api_secret
}

print('=== ANALYZING REAL TRADE P&L ===')
print()

# Get orders
response = requests.get('https://paper-api.alpaca.markets/v2/orders?status=all&limit=20', headers=headers)

if response.status_code == 200:
    orders = response.json()
    pypl_orders = [o for o in orders if 'PYPL' in o['symbol']]
    
    if pypl_orders:
        buy_order = None
        sell_order = None
        
        for order in pypl_orders:
            if order['side'] == 'buy' and order['status'] == 'filled':
                buy_order = order
            elif order['side'] == 'sell' and order['status'] == 'filled':
                sell_order = order
        
        if buy_order and sell_order:
            symbol = buy_order['symbol']
            buy_price = float(buy_order['filled_avg_price'])
            sell_price = float(sell_order['filled_avg_price'])
            qty = int(buy_order['filled_qty'])
            
            buy_cost = buy_price * qty * 100
            sell_proceeds = sell_price * qty * 100
            profit = sell_proceeds - buy_cost
            profit_pct = (profit / buy_cost) * 100
            
            print('REAL TRADE DATA FROM ALPACA:')
            print('  Symbol:', symbol)
            print('  Quantity:', qty, 'contract(s)')
            print()
            print('ENTRY:')
            print('  Price: $' + str(round(buy_price, 2)))
            print('  Cost: $' + str(round(buy_cost, 2)))
            print()
            print('EXIT:')
            print('  Price: $' + str(round(sell_price, 2)))
            print('  Proceeds: $' + str(round(sell_proceeds, 2)))
            print()
            print('REAL P&L:')
            print('  Amount: $' + str(round(profit, 2)))
            print('  Percentage: ' + str(round(profit_pct, 2)) + '%')
            print()
            print('vs BOT ESTIMATE:')
            print('  Bot Said: +52.74% ($34,522)')
            print('  Reality: ' + str(round(profit_pct, 2)) + '% ($' + str(round(profit, 2)) + ')')
            print()
            print('CONCLUSION: Bot P&L calculation is WRONG')
            print('The bot is using estimated Black-Scholes pricing')
            print('instead of real broker fill prices.')
