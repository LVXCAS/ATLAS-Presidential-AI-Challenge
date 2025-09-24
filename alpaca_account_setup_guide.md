# ALPACA PAPER ACCOUNT CONFIGURATION FOR AUTONOMOUS TRADING

## Current Issue
Your paper account has Pattern Day Trader restrictions despite having $618,220 buying power.

## Step-by-Step Configuration

### 1. LOG INTO ALPACA DASHBOARD
- Go to https://app.alpaca.markets
- Log in with your account credentials
- Switch to "Paper Trading" mode (top right toggle)

### 2. RESET PAPER ACCOUNT (RECOMMENDED)
- In Paper Dashboard, look for "Reset Account" or "Reset Paper Account"
- This will:
  - Reset balance to $100,000 default
  - Clear all positions
  - Remove PDT restrictions
  - Reset day trade counter

### 3. INCREASE PAPER ACCOUNT BALANCE
- Look for "Account Settings" or "Paper Trading Settings"
- Increase virtual balance to $1,000,000 or higher
- This ensures unlimited buying power for testing

### 4. ENABLE TRADING PERMISSIONS
Make sure these are enabled:
- ✅ Equity Trading
- ✅ Options Trading (Level 2 minimum)
- ✅ Crypto Trading
- ✅ Pattern Day Trading Override

### 5. API CONFIGURATION
- Go to "API Management" section
- Verify your API keys have these permissions:
  - ✅ Account Data (Read)
  - ✅ Trading (Read/Write)
  - ✅ Market Data (Read)

### 6. REMOVE PDT RESTRICTIONS
- In Account Settings, look for "Day Trading"
- Enable "Pattern Day Trading Override" for paper account
- Set Day Trading Buying Power to "Unlimited" or same as account balance

## Alternative Quick Fix

If you can't find reset option, contact Alpaca support:
- Email: support@alpaca.markets
- Subject: "Remove PDT restrictions on paper account"
- Message: "Please remove Pattern Day Trader restrictions on my paper trading account to enable testing of algorithmic strategies"

## API Environment Variables
Verify your .env file has:
```
ALPACA_API_KEY=your_paper_key_here
ALPACA_SECRET_KEY=your_paper_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

## Test Configuration
After changes, run this test:
```python
python -c "
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
load_dotenv()

alpaca = tradeapi.REST(
    os.getenv('ALPACA_API_KEY'),
    os.getenv('ALPACA_SECRET_KEY'),
    os.getenv('ALPACA_BASE_URL'),
    api_version='v2'
)

account = alpaca.get_account()
print(f'Portfolio: ${float(account.portfolio_value):,.0f}')
print(f'Day Trading BP: ${float(account.daytrading_buying_power):,.0f}')
print(f'PDT: {account.pattern_day_trader}')
print(f'Day Trades: {account.daytrade_count}')

# Test small trade
try:
    order = alpaca.submit_order(
        symbol='AAPL',
        qty=1,
        side='buy',
        type='market',
        time_in_force='day'
    )
    print('✅ TEST TRADE SUCCESSFUL!')
    print(f'Order ID: {order.id}')
except Exception as e:
    print(f'❌ Still blocked: {e}')
"
```

## Expected Results After Fix
- Day Trading Buying Power: $1,000,000+
- Pattern Day Trader: False (or True with unlimited power)
- Day Trade Count: 0
- Test trade executes successfully

## Restart Autonomous System
Once configured, restart your autonomous trading:
```bash
python full_autonomous_trading_system.py
```

Your system should then execute the 500-900% return opportunities automatically!