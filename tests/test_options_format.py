import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
load_dotenv()
api = tradeapi.REST(key_id=os.getenv('ALPACA_API_KEY'), secret_key=os.getenv('ALPACA_SECRET_KEY'), base_url=os.getenv('ALPACA_BASE_URL'), api_version='v2')

# Test different SPY options symbol formats
test_symbols = [
    'SPY251003C00660000',  # Our current format
    'SPY  251003C00660000',  # With spaces (OCC standard)
    'SPY251003C660',       # Short format
    'SPY251003C660000',    # No leading zeros on strike
    'SPY251004C00660000',  # Next day
    'SPY251010C00660000',  # Next week
]

print("Testing SPY options symbol formats:")
for symbol in test_symbols:
    try:
        quote = api.get_latest_quote(symbol)
        print(f"SUCCESS: {symbol} - Bid: ${quote.bid_price}, Ask: ${quote.ask_price}")
    except Exception as e:
        print(f"FAILED: {symbol} - {str(e)}")

# Try to list any available options
try:
    print("\nChecking if any options are available...")
    assets = api.list_assets(status='active', asset_class='us_option')
    print(f"Total options found: {len(assets)}")
    if assets:
        print("First 5 options:")
        for asset in assets[:5]:
            print(f"  {asset.symbol}")
except Exception as e:
    print(f"Error listing options: {e}")

# Check if we're in paper trading mode
try:
    account = api.get_account()
    print(f"\nAccount type: {'Paper' if 'paper' in str(api.base_url) else 'Live'}")
    print(f"Trading blocked: {account.trading_blocked}")
    print(f"Pattern day trader: {account.pattern_day_trader}")
except Exception as e:
    print(f"Error checking account: {e}")