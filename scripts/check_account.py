#!/usr/bin/env python3
"""Quick script to check which Alpaca account we're connected to"""
from alpaca.trading.client import TradingClient
import os
from dotenv import load_dotenv

load_dotenv()

client = TradingClient(
    os.getenv('ALPACA_API_KEY'),
    os.getenv('ALPACA_SECRET_KEY'),
    paper=True
)

account = client.get_account()
print(f"\n{'='*60}")
print(f"ALPACA ACCOUNT CONNECTION TEST")
print(f"{'='*60}")
print(f"Account Number: {account.account_number}")
print(f"Equity: ${float(account.equity):,.2f}")
print(f"Cash: ${float(account.cash):,.2f}")
print(f"Buying Power: ${float(account.buying_power):,.2f}")
print(f"Options Buying Power: ${float(account.options_buying_power):,.2f}")
print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
print(f"{'='*60}\n")
