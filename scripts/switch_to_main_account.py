#!/usr/bin/env python3
"""
SWITCH TO MAIN ALPACA ACCOUNT
Quick script to verify and switch Alpaca accounts

Main Account (TARGET):
  - Account ID: PA3MS5F52RNL
  - Equity: $956k
  - Safe for trading

Secondary Account (CURRENT - WRONG):
  - Account ID: PA3RRV5YYKAS
  - Equity: $91k
  - Negative cash: -$84k (DANGER!)
"""

import os
from dotenv import load_dotenv

print("\n" + "="*70)
print("ALPACA ACCOUNT CHECKER")
print("="*70)

# Load current credentials
load_dotenv()

current_key = os.getenv('ALPACA_API_KEY', 'NOT_FOUND')
current_secret = os.getenv('ALPACA_SECRET_KEY', 'NOT_FOUND')
current_url = os.getenv('ALPACA_BASE_URL', 'NOT_FOUND')

print("\n[CURRENT .ENV CONFIGURATION]")
print(f"  API Key: {current_key[:20]}...")
print(f"  Secret:  {current_secret[:20]}...")
print(f"  URL:     {current_url}")

# Test connection
print("\n[TESTING CONNECTION]")
try:
    from alpaca.trading.client import TradingClient

    client = TradingClient(current_key, current_secret, paper=True)
    account = client.get_account()

    print(f"[OK] Connected to Alpaca")
    print(f"\n[ACCOUNT DETAILS]")
    print(f"  Account ID: {account.account_number}")
    print(f"  Equity: ${float(account.equity):,.2f}")
    print(f"  Cash: ${float(account.cash):,.2f}")
    print(f"  Buying Power: ${float(account.buying_power):,.2f}")
    print(f"  Open Positions: {len(client.get_all_positions())}")

    # Check which account we're connected to
    if account.account_number == "PA3MS5F52RNL":
        print(f"\n[OK] Connected to MAIN account - Ready for trading!")
        print("="*70)
    elif account.account_number == "PA3RRV5YYKAS":
        print(f"\n[WARNING] Connected to SECONDARY account!")
        print(f"  This account has NEGATIVE CASH: ${float(account.cash):,.2f}")
        print(f"  DO NOT USE for Monday trading!")
        print("\n[ACTION REQUIRED]")
        print("  1. Go to: https://alpaca.markets/")
        print("  2. Log in to your account")
        print("  3. Switch to Paper Trading account: PA3MS5F52RNL")
        print("  4. Generate new API keys (or find existing ones)")
        print("  5. Update .env file with:")
        print("     ALPACA_API_KEY=<main_account_key>")
        print("     ALPACA_SECRET_KEY=<main_account_secret>")
        print("  6. Run this script again to verify")
        print("="*70)
    else:
        print(f"\n[UNKNOWN] Connected to account: {account.account_number}")
        print("="*70)

except Exception as e:
    print(f"[ERROR] Could not connect: {e}")
    print("\n[ACTION REQUIRED]")
    print("  Check your API credentials in .env file")
    print("="*70)

print("\n")
