#!/usr/bin/env python3
"""
Test Crypto Trading with Alpaca
"""

import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

def test_crypto():
    load_dotenv()
    
    api = tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        os.getenv('ALPACA_BASE_URL'),
        api_version='v2'
    )
    
    print("TESTING CRYPTO TRADING CAPABILITIES")
    print("=" * 50)
    
    # Test crypto symbols
    crypto_symbols = ['BTCUSD', 'ETHUSD', 'LTCUSD', 'BCHUSD', 'ADAUSD', 'SOLUSD']
    
    print("\n[CRYPTO ASSET AVAILABILITY]")
    available_crypto = []
    
    for symbol in crypto_symbols:
        try:
            asset = api.get_asset(symbol)
            print(f"  {symbol}: Available - {asset.tradable}")
            if asset.tradable:
                available_crypto.append(symbol)
        except Exception as e:
            print(f"  {symbol}: Not available - {e}")
    
    print(f"\nAvailable crypto assets: {available_crypto}")
    
    # Test getting crypto quotes
    print("\n[CRYPTO QUOTES]")
    for symbol in available_crypto[:3]:  # Test first 3
        try:
            # Get latest trade for crypto
            trade = api.get_latest_crypto_trade(symbol, 'CBSE')  # Coinbase
            print(f"  {symbol}: ${trade.price} (Size: {trade.size}) - {trade.timestamp}")
        except Exception as e:
            print(f"  {symbol}: Quote error - {e}")
    
    # Test crypto order (small amount)
    print("\n[TEST CRYPTO ORDER]")
    if 'BTCUSD' in available_crypto:
        try:
            # Place a very small test order for $10 worth of BTC
            test_order = api.submit_order(
                symbol='BTCUSD',
                notional=10,  # $10 worth of BTC
                side='buy',
                type='market',
                time_in_force='day'
            )
            
            print(f"  BTC Test Order: {test_order.id} - Status: {test_order.status}")
            
            # Cancel immediately 
            api.cancel_order(test_order.id)
            print("  Test order cancelled successfully!")
            
        except Exception as e:
            print(f"  BTC Order Error: {e}")
    
    # Check if we can trade 24/7
    print("\n[24/7 TRADING STATUS]")
    clock = api.get_clock()
    print(f"  Market Open: {clock.is_open}")
    print(f"  Current Time: {clock.timestamp}")
    print("  Crypto trades 24/7 - no market hours restrictions!")
    
    return available_crypto

if __name__ == "__main__":
    available = test_crypto()
    
    print("\n" + "=" * 50)
    print("CRYPTO TRADING READY!")
    print(f"Available: {', '.join(available)}")
    print("Commands to trade crypto:")
    for symbol in available[:3]:
        print(f"  python trade_now.py buy {symbol} 50  # Buy $50 worth")
    print("=" * 50)