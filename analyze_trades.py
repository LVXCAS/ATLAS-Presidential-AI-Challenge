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

# Get all filled orders
response = requests.get('https://paper-api.alpaca.markets/v2/orders?status=filled&limit=100', headers=headers)

if response.status_code == 200:
    orders = response.json()

    # Group orders by symbol to find pairs
    symbols = {}
    for order in orders:
        sym = order['symbol']
        if sym not in symbols:
            symbols[sym] = {'buys': [], 'sells': []}

        if order['side'] == 'buy':
            symbols[sym]['buys'].append(order)
        else:
            symbols[sym]['sells'].append(order)

    print('=== COMPLETED TRADES (BUY + SELL PAIRS) ===\n')

    trades = []
    for sym, data in symbols.items():
        # Match buys with sells
        for buy in data['buys']:
            for sell in data['sells']:
                buy_price = float(buy['filled_avg_price'])
                sell_price = float(sell['filled_avg_price'])
                qty = int(buy['filled_qty'])

                pnl = (sell_price - buy_price) * qty * 100
                pnl_pct = ((sell_price - buy_price) / buy_price) * 100

                trades.append({
                    'symbol': sym,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'qty': qty,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'buy_date': buy['created_at'][:10]
                })

    # Sort by pnl_pct
    trades.sort(key=lambda x: x['pnl_pct'], reverse=True)

    if trades:
        print('=== HIGHEST PERCENTAGE GAIN ===')
        best = trades[0]
        print(f"  Symbol: {best['symbol']}")
        print(f"  Entry: ${best['buy_price']:.2f}")
        print(f"  Exit: ${best['sell_price']:.2f}")
        print(f"  Quantity: {best['qty']} contract(s)")
        print(f"  P&L: ${best['pnl']:.2f}")
        print(f"  Percentage: {best['pnl_pct']:+.2f}%")
        print(f"  Date: {best['buy_date']}")
        print()

        print('=== WORST TRADE (BIGGEST LOSS) ===')
        worst = trades[-1]
        print(f"  Symbol: {worst['symbol']}")
        print(f"  Entry: ${worst['buy_price']:.2f}")
        print(f"  Exit: ${worst['sell_price']:.2f}")
        print(f"  Quantity: {worst['qty']} contract(s)")
        print(f"  P&L: ${worst['pnl']:.2f}")
        print(f"  Percentage: {worst['pnl_pct']:+.2f}%")
        print(f"  Date: {worst['buy_date']}")
        print()

        print('=== ALL TRADES SORTED BY PERCENTAGE ===')
        for i, t in enumerate(trades, 1):
            print(f"{i}. {t['symbol']:20s} | {t['pnl_pct']:+7.2f}% | ${t['pnl']:+8.2f} | {t['buy_date']}")

        print()
        print('=== SUMMARY STATISTICS ===')
        total_pnl = sum(t['pnl'] for t in trades)
        avg_pnl_pct = sum(t['pnl_pct'] for t in trades) / len(trades)
        winners = [t for t in trades if t['pnl'] > 0]
        losers = [t for t in trades if t['pnl'] < 0]

        print(f"  Total Trades: {len(trades)}")
        print(f"  Winners: {len(winners)} ({len(winners)/len(trades)*100:.1f}%)")
        print(f"  Losers: {len(losers)} ({len(losers)/len(trades)*100:.1f}%)")
        print(f"  Total P&L: ${total_pnl:.2f}")
        print(f"  Average P&L%: {avg_pnl_pct:+.2f}%")

    else:
        print('No completed trade pairs found')
else:
    print(f'Error: {response.status_code}')
