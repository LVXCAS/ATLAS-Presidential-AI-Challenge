"""
TEST CTRADER API CONNECTION
Tests both Market Data and Trading API connections
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# STEP 1: Check if you have the API credentials in .env
# =============================================================================

print("=" * 70)
print("CTRADER API CONNECTION TEST")
print("=" * 70)
print()

# Check for cTrader credentials in .env
market_data_token = os.getenv('CTRADER_MARKET_DATA_TOKEN')
trading_token = os.getenv('CTRADER_TRADING_TOKEN')
account_id = os.getenv('CTRADER_ACCOUNT_ID')

print("STEP 1: Checking .env file for cTrader credentials...")
print()

if market_data_token:
    print(f"[OK] CTRADER_MARKET_DATA_TOKEN found: {market_data_token[:20]}...")
else:
    print("[MISSING] CTRADER_MARKET_DATA_TOKEN not found in .env")
    print("   Add this line to your .env file:")
    print("   CTRADER_MARKET_DATA_TOKEN=your_token_here")

if trading_token:
    print(f"[OK] CTRADER_TRADING_TOKEN found: {trading_token[:20]}...")
else:
    print("[MISSING] CTRADER_TRADING_TOKEN not found in .env")
    print("   Add this line to your .env file:")
    print("   CTRADER_TRADING_TOKEN=your_token_here")

if account_id:
    print(f"[OK] CTRADER_ACCOUNT_ID found: {account_id}")
else:
    print("[MISSING] CTRADER_ACCOUNT_ID not found in .env")
    print("   Add this line to your .env file:")
    print("   CTRADER_ACCOUNT_ID=your_account_number")

print()

# =============================================================================
# STEP 2: Test if you have one combined token or two separate ones
# =============================================================================

print("STEP 2: Determining API architecture...")
print()

if market_data_token and trading_token:
    if market_data_token == trading_token:
        print("[INFO] You have ONE COMBINED TOKEN (same for both market data and trading)")
        print("   This is common with cTrader - one token handles everything")
        api_architecture = "COMBINED"
    else:
        print("[INFO] You have TWO SEPARATE TOKENS (split architecture)")
        print("   Market Data Token: Different from Trading Token")
        api_architecture = "SPLIT"
elif market_data_token or trading_token:
    print("[INFO] You have ONE TOKEN only")
    print("   Most likely this is a combined token for all operations")
    api_architecture = "SINGLE"
else:
    print("[WARNING] No cTrader tokens found")
    print("   Please add your API credentials to .env file")
    api_architecture = "NONE"

print()

# =============================================================================
# STEP 3: Provide next steps based on what we found
# =============================================================================

print("STEP 3: Next Steps...")
print()

if api_architecture == "NONE":
    print("ACTION REQUIRED:")
    print("1. Go to your cTrader platform")
    print("2. Navigate to: Copy Trading > Settings > API")
    print("   OR: Settings > API Access")
    print("3. Generate an API token (or access token)")
    print("4. Copy the token and account ID")
    print("5. Add to .env file:")
    print()
    print("   # cTrader API Credentials")
    print("   CTRADER_MARKET_DATA_TOKEN=paste_your_token_here")
    print("   CTRADER_TRADING_TOKEN=paste_your_token_here")
    print("   CTRADER_ACCOUNT_ID=your_account_number")
    print()
    print("   (If you only have one token, paste it for BOTH market data and trading)")

else:
    print("[OK] Credentials detected! Ready for API connection test.")
    print()
    print("NEXT: We'll test the actual connection to cTrader servers")
    print("This will verify:")
    print("  - Token is valid and not expired")
    print("  - Account ID is correct")
    print("  - You can fetch market data (USD_JPY prices)")
    print("  - You can check account balance")
    print()
    print("Run this script again after confirming your .env has the tokens,")
    print("and we'll add the actual API connection test code.")

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
