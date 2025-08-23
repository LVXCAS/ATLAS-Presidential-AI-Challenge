#!/usr/bin/env python3
"""
Simple Crypto Trade Test
"""
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

print(">> Loading crypto test...")

load_dotenv()
api = tradeapi.REST(
    os.getenv('ALPACA_API_KEY'),
    os.getenv('ALPACA_SECRET_KEY'),
    os.getenv('ALPACA_BASE_URL'),
    api_version='v2'
)

print(">> API loaded, checking account...")

try:
    account = api.get_account()
    print(f">> Account OK: ${float(account.portfolio_value):,.2f}")
    
    print(">> Checking positions...")
    positions = api.list_positions()
    crypto_positions = [p for p in positions if 'USD' in p.symbol and len(p.symbol) > 5]
    
    print(f">> Current crypto positions: {len(crypto_positions)}")
    for pos in crypto_positions:
        print(f"   {pos.symbol}: {pos.qty} @ ${float(pos.market_value):,.2f}")
    
    print(">> Testing crypto quote...")
    try:
        # Just test getting a quote, not placing a trade
        latest = api.get_latest_trade('BTCUSD')
        print(f">> BTC Latest: ${latest.price}")
    except Exception as e:
        print(f">> Quote error: {e}")
    
    print(">> Test complete - ready for live trading!")
    
except Exception as e:
    print(f">> Error: {e}")