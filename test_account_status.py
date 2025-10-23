"""Quick test to verify Alpaca account status"""

from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')

client = TradingClient(api_key, secret_key, paper=True)
account = client.get_account()

print("=" * 60)
print("ALPACA ACCOUNT STATUS")
print("=" * 60)
print(f"Account Status: {account.status}")
print(f"Trading Blocked: {account.trading_blocked}")
print(f"Account Blocked: {account.account_blocked}")
print(f"Cash: ${float(account.cash):,.2f}")
print(f"Buying Power: ${float(account.buying_power):,.2f}")
print(f"Pattern Day Trader: {account.pattern_day_trader}")
print(f"Currency: {account.currency}")
print("=" * 60)

if account.status == 'ACTIVE' and not account.trading_blocked and not account.account_blocked:
    print("[OK] Alpaca account is READY for trading!")
else:
    print("[WARNING] Account may have issues - check status above")
