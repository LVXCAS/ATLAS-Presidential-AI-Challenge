"""Quick script to get E8 trade IDs for marketing"""
from tradelocker import TLAPI
import os
from dotenv import load_dotenv

load_dotenv()

email = os.getenv("E8_EMAIL") or os.getenv("TRADELOCKER_EMAIL")
password = os.getenv("E8_PASSWORD") or os.getenv("TRADELOCKER_PASSWORD")

if not email or not password:
    print("ERROR: Missing E8_EMAIL or E8_PASSWORD in .env file")
    exit(1)

tl = TLAPI(
    environment="https://demo.tradelocker.com",
    username=email,
    password=password,
    server="E8-Live",
    log_level='error'  # Suppress debug output
)

positions = tl.get_all_positions()

print('\n' + '='*70)
print('LIVE TRADES - E8 $200K CHALLENGE')
print('='*70)

symbol_map = {
    6119: 'EUR/USD',
    6116: 'GBP/USD',
    6120: 'USD/JPY'
}

for idx, row in positions.iterrows():
    inst_id = row['tradableInstrumentId']
    symbol = symbol_map.get(inst_id, f'Instrument {inst_id}')
    side = row['side'].upper()
    qty = row['qty']
    entry = row['avgPrice']
    pl = row['unrealizedPl']

    print(f'\nTrade #{idx+1}:')
    print(f'  Symbol: {symbol}')
    print(f'  Position ID: {row["id"]}')
    print(f'  Direction: {side}')
    print(f'  Size: {qty} lots ({int(qty * 100000):,} units)')
    print(f'  Entry Price: {entry:.5f}')
    print(f'  Unrealized P/L: ${pl:,.2f}')

    # Check for TP/SL IDs
    if row.get('takeProfitId'):
        print(f'  TP Order ID: {row["takeProfitId"]}')
    if row.get('stopLossId'):
        print(f'  SL Order ID: {row["stopLossId"]}')

print('\n' + '='*70)
print(f'Total Unrealized: ${positions["unrealizedPl"].sum():,.2f}')
print('Status: ACTIVE [OK]')
print('='*70)
