import requests
import os

# Get current prices
OANDA_API = 'https://api-fxpractice.oanda.com'
OANDA_TOKEN = os.getenv('OANDA_API_KEY')

headers = {'Authorization': f'Bearer {OANDA_TOKEN}'}
resp = requests.get(f'{OANDA_API}/v3/accounts/101-001-28787896-001/pricing?instruments=EUR_USD,GBP_USD', headers=headers)
data = resp.json()

print('CURRENT MARKET PRICES:')
print('=' * 60)
for price in data['prices']:
    symbol = price['instrument']
    bid = float(price['bids'][0]['price'])
    ask = float(price['asks'][0]['price'])
    print(f'{symbol}: Bid {bid:.5f} | Ask {ask:.5f}')

print('\n')
print('YOUR POSITIONS:')
print('=' * 60)

# EUR/USD SHORT at 1.16449
eur_entry = 1.16449
eur_tp = 1.16449 * 0.98  # 2% profit target
eur_sl = 1.16449 * 1.01  # 1% stop loss
eur_current_bid = [p for p in data['prices'] if p['instrument'] == 'EUR_USD'][0]['bids'][0]['price']
eur_current = float(eur_current_bid)

eur_pips_to_tp = (eur_entry - eur_tp) * 10000
eur_pips_to_sl = (eur_sl - eur_entry) * 10000
eur_pips_from_entry = (eur_entry - eur_current) * 10000

print(f'EUR/USD SHORT @ 1.16449:')
print(f'  Current: {eur_current:.5f}')
print(f'  Movement: +{eur_pips_from_entry:.1f} pips in profit')
print(f'  Distance to TP (1.14120): {eur_pips_to_tp:.1f} pips')
print(f'  Distance to SL (1.17613): {eur_pips_to_sl:.1f} pips')
print(f'  TP/SL Ratio: {eur_pips_to_tp/eur_pips_to_sl:.1f}x farther to TP')

# GBP/USD SHORT at 1.32015
gbp_entry = 1.32015
gbp_tp = 1.32015 * 0.98
gbp_sl = 1.32015 * 1.01
gbp_current_bid = [p for p in data['prices'] if p['instrument'] == 'GBP_USD'][0]['bids'][0]['price']
gbp_current = float(gbp_current_bid)

gbp_pips_to_tp = (gbp_entry - gbp_tp) * 10000
gbp_pips_to_sl = (gbp_sl - gbp_entry) * 10000
gbp_pips_from_entry = (gbp_entry - gbp_current) * 10000

print(f'\nGBP/USD SHORT @ 1.32015:')
print(f'  Current: {gbp_current:.5f}')
print(f'  Movement: +{gbp_pips_from_entry:.1f} pips in profit')
print(f'  Distance to TP (1.29375): {gbp_pips_to_tp:.1f} pips')
print(f'  Distance to SL (1.33335): {gbp_pips_to_sl:.1f} pips')
print(f'  TP/SL Ratio: {gbp_pips_to_tp/gbp_pips_to_sl:.1f}x farther to TP')

print('\n')
print('WEEKEND GAP RISK:')
print('=' * 60)
print(f'Total exposure: 20 lots (2,000,000 units)')
print(f'50-pip gap against you = ${50 * 20 * 10:,} loss')
print(f'100-pip gap against you = ${100 * 20 * 10:,} loss')
print(f'Current profit: $4,510')
print(f'E8 max DD: $11,940 (6%)')
print(f'\nMarkets close in ~4 hours. Sunday gap risk unmanageable.')
