#!/usr/bin/env python3
"""
Check trade history to find peak P/L
"""
import os
from dotenv import load_dotenv
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.pricing as pricing

load_dotenv()

oanda_token = os.getenv('OANDA_API_KEY')
oanda_account_id = '101-001-37330890-001'

client = API(access_token=oanda_token, environment='practice')

# Get current trade
r = trades.TradesList(accountID=oanda_account_id)
response = client.request(r)
trades_list = response.get('trades', [])

if trades_list:
    trade = trades_list[0]
    instrument = trade['instrument']
    trade_id = trade['id']
    units = int(trade['currentUnits'])
    entry_price = float(trade['price'])
    current_unrealized = float(trade.get('unrealizedPL', 0))

    # Get current price
    params = {"instruments": instrument}
    r = pricing.PricingInfo(accountID=oanda_account_id, params=params)
    price_data = client.request(r)
    current_price = float(price_data['prices'][0]['closeoutBid'])

    print("="*70)
    print("USD/JPY SHORT TRADE ANALYSIS")
    print("="*70)
    print(f"\nTrade ID: {trade_id}")
    print(f"Instrument: {instrument}")
    print(f"Direction: SHORT")
    print(f"Units: {abs(units):,}")
    print(f"\nEntry Price: {entry_price:.5f}")
    print(f"Current Price: {current_price:.5f}")
    print(f"Current P/L: ${current_unrealized:,.2f}")

    # Calculate what the P/L would have been at different price levels
    print(f"\n{'='*70}")
    print("POTENTIAL PEAK ANALYSIS")
    print("="*70)

    # For SHORT position: profit when price goes DOWN
    # P/L = units × (entry_price - current_price)

    # If you hit $3,000 profit, work backwards to find the price
    # 3000 = 1,256,249 × (153.069 - X)
    # X = 153.069 - (3000 / 1,256,249)

    target_pls = [1000, 1500, 2000, 2500, 3000, 3500, 4000]

    print(f"\nTo reach various P/L levels, USD/JPY would need to be at:")
    print(f"{'Target P/L':<15} {'Price Needed':<15} {'Pips from Entry':<15}")
    print("-"*70)

    for target_pl in target_pls:
        # For SHORT: target_pl = units × (entry - price) / conversion_rate
        # Simplify: price = entry - (target_pl × conversion_rate / units)

        # Calculate price needed for this P/L
        # JPY pairs: 1 pip = 0.01
        pip_value = abs(units) / 100  # Value per pip for JPY pair
        pips_needed = target_pl / pip_value * 100
        price_needed = entry_price - (pips_needed * 0.01)

        status = "[REACHED]" if current_unrealized >= target_pl else ""
        print(f"${target_pl:>6,.0f}       {price_needed:>8.5f}       {pips_needed:>6.1f} pips    {status}")

    # Show current position
    current_pips = (entry_price - current_price) * 100
    print(f"\n{'='*70}")
    print(f"CURRENT POSITION")
    print(f"{'='*70}")
    print(f"Current Price: {current_price:.5f}")
    print(f"Pips from Entry: {current_pips:.1f} pips")
    print(f"Current P/L: ${current_unrealized:,.2f}")

    # If you hit $3k, estimate where that was
    if current_unrealized < 3000:
        print(f"\n{'='*70}")
        print(f"TO HIT $3,000 AGAIN:")
        print(f"{'='*70}")
        pip_value_usd = abs(units) / 100
        pips_needed_for_3k = (3000 / pip_value_usd) * 100
        price_for_3k = entry_price - (pips_needed_for_3k * 0.01)
        pips_to_go = pips_needed_for_3k - current_pips

        print(f"Need USD/JPY to fall to: {price_for_3k:.5f}")
        print(f"Need {pips_to_go:.1f} more pips down from current price")
        print(f"That would be {pips_needed_for_3k:.1f} pips total from entry")

else:
    print("No active trades found")
