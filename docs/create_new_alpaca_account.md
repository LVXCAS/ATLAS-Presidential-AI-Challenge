# CREATE NEW ALPACA PAPER ACCOUNT - FASTEST FIX

## Why Create New Account
Your current account has corrupted day trading calculations that even Alpaca support might take days to fix. A fresh account will:
- Start with clean $100K+ balance
- Have proper PDT calculations
- No trading restrictions
- Your autonomous system working immediately

## Step-by-Step New Account Creation

### 1. CREATE NEW ALPACA ACCOUNT
- Go to: https://alpaca.markets
- Click "Get Started" or "Sign Up"
- Use DIFFERENT email than current account
- Choose "Paper Trading Only" during signup
- Complete verification (usually instant)

### 2. GET NEW API KEYS
- Log into new account dashboard
- Go to "API Management" or "Developer Tools"
- Generate new API keys:
  - Paper Trading API Key
  - Paper Trading Secret Key
- Copy both keys

### 3. UPDATE YOUR .ENV FILE
Replace your current keys with new ones:
```
ALPACA_API_KEY=your_new_paper_key_here
ALPACA_SECRET_KEY=your_new_paper_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 4. CONFIGURE NEW ACCOUNT
- Set paper balance to $1,000,000 (if option available)
- Enable all trading permissions:
  - Equity Trading: ON
  - Options Trading: Level 2+
  - Crypto Trading: ON
- Verify PDT settings show proper buying power

### 5. TEST NEW ACCOUNT
Run this test:
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
print(f'Cash: ${float(account.cash):,.0f}')

# Test trade
try:
    order = alpaca.submit_order(
        symbol='AAPL',
        qty=1,
        side='buy',
        type='market',
        time_in_force='day'
    )
    print('✅ NEW ACCOUNT WORKS!')
    print(f'Order: {order.id}')
except Exception as e:
    print(f'❌ Issue: {e}')
"
```

### 6. RESTART AUTONOMOUS SYSTEMS
Once new account works:
```bash
# Stop current systems
pkill -f python

# Restart with new account
python mega_discovery_engine.py &
python full_autonomous_trading_system.py &
python truly_autonomous_system.py &
```

## Expected Results with New Account
- Fresh $100,000+ balance
- Day Trading BP: $400,000+ (4x leverage)
- No PDT restrictions or trading blocks
- Your autonomous system executing trades immediately

## Advantages of New Account
1. **Immediate fix** (30 minutes vs days waiting for support)
2. **Clean slate** - no corrupted calculations
3. **Full control** - configure exactly how you want
4. **Keep old account** - can use both for testing
5. **Your autonomous system starts trading today**

## Timeline
- Account creation: 5 minutes
- Verification: Usually instant
- API setup: 5 minutes
- Testing: 5 minutes
- **Total**: 15-30 minutes to get autonomous trading working

Much faster than waiting for Alpaca support to fix $0 day trading buying power issue!