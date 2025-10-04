#!/usr/bin/env python3
"""Check if Alpaca paper account can trade options"""

import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest

load_dotenv('.env.paper')

try:
    api = TradingClient(
        api_key=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        paper=True
    )

    print("ALPACA PAPER ACCOUNT - OPTIONS CAPABILITY CHECK")
    print("=" * 60)

    # Check account info
    account = api.get_account()
    print(f"\nAccount Status: {account.status}")
    print(f"Trading Blocked: {account.trading_blocked}")
    print(f"Account Blocked: {account.account_blocked}")
    print(f"Pattern Day Trader: {account.pattern_day_trader}")

    # Check if options are enabled
    print(f"\nOptions Trading:")
    print(f"  Enabled: {hasattr(account, 'options_trading_level')}")
    if hasattr(account, 'options_trading_level'):
        print(f"  Level: {account.options_trading_level}")

    # Try to fetch option contracts for AAPL
    print(f"\nTrying to fetch AAPL option contracts...")
    try:
        request = GetOptionContractsRequest(
            underlying_symbols=["AAPL"],
            status="active",
            limit=5
        )
        contracts = api.get_option_contracts(request)
        print(f"  Success! Found {len(contracts)} contracts")
        if contracts:
            print(f"\n  Example contract:")
            c = contracts[0]
            print(f"    Symbol: {c.symbol}")
            print(f"    Strike: ${c.strike_price}")
            print(f"    Expiry: {c.expiration_date}")
            print(f"    Type: {c.type}")
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"\n  This likely means options trading is NOT enabled on this paper account")

    # Check positions
    print(f"\nCurrent Positions:")
    positions = api.get_all_positions()
    if positions:
        for pos in positions:
            print(f"  {pos.symbol}: {pos.qty} shares @ ${pos.avg_entry_price}")
    else:
        print("  No positions (paper trading logs trades but may not show positions)")

    # Check orders
    print(f"\nRecent Orders:")
    orders = api.get_orders(limit=5)
    if orders:
        for order in orders:
            print(f"  {order.symbol}: {order.side} {order.qty} @ {order.type} - Status: {order.status}")
    else:
        print("  No orders found")

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("=" * 60)

    # The key issue
    print("\nAlpaca Paper Trading Limitation:")
    print("  - Paper accounts can LOG options trades (what we're doing)")
    print("  - Paper accounts CANNOT execute real options orders")
    print("  - You need a LIVE account with options approval for real execution")
    print("\nWhat we're doing:")
    print("  - Simulating options trades with calculations")
    print("  - Logging all trade details to JSON files")
    print("  - Building track record for prop firm applications")
    print("\nTo actually execute options:")
    print("  - Apply for live options trading approval at Alpaca")
    print("  - Or continue paper trading to build 30-day track record")
    print("  - Then apply to prop firms with your documented performance")

except Exception as e:
    print(f"ERROR connecting to Alpaca: {e}")
    print("\nThis might mean:")
    print("  1. API keys are incorrect")
    print("  2. Network connectivity issue")
    print("  3. Alpaca API is down")
