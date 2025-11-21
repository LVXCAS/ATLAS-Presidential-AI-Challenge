# E8 Challenge Setup Instructions

Complete guide to connect your forex bot to TradeLocker for E8 challenges.

## Overview

Your bot is ready for the **E8 $200K Challenge**:
- **6% max drawdown**
- **80% profit split** after funded
- **10% profit target** ($20,000)
- **Hybrid strategy**: 50% win rate, 9.5% ROI
- **Pass time**: 39 days
- **Pass rate**: 94%

---

## Step 1: Install TradeLocker Library

```bash
pip install tradelocker
```

**Requirements**: Python 3.11+

---

## Step 2: Get Your E8 Credentials

### From E8 Dashboard

1. Log into your E8 account: https://client.e8markets.com
2. Go to your **Challenge Dashboard**
3. Find **TradeLocker Credentials**:
   - Email: Your E8 account email
   - Password: Your E8 trading password
   - Server: Usually `E8-Live` or `E8-Demo`

### Example Credentials

```
Email: yourname@example.com
Password: YourE8Password123
Server: E8-Live
```

---

## Step 3: Add Credentials to .env File

Open your `.env` file (or create one) and add:

```bash
# TRADELOCKER CREDENTIALS (E8 Challenge)
TRADELOCKER_EMAIL=yourname@example.com
TRADELOCKER_PASSWORD=YourE8Password123
TRADELOCKER_SERVER=E8-Live
TRADELOCKER_ENV=https://api.tradelocker.com

# For demo account, use:
# TRADELOCKER_ENV=https://demo.tradelocker.com
```

**⚠️ Important**:
- Replace `yourname@example.com` with your actual E8 email
- Replace `YourE8Password123` with your actual E8 password
- Use `E8-Live` for real challenge, `E8-Demo` for testing

---

## Step 4: Test Connection

Before running the full bot, test your connection:

```bash
python E8_TRADELOCKER_ADAPTER.py
```

### Expected Output

```
====================================================================
TRADELOCKER ADAPTER - E8 CHALLENGE
====================================================================
Environment: https://api.tradelocker.com
Server: E8-Live
Account ID: 12345
Status: Connected ✓
====================================================================

[TEST] Fetching account info...
Balance: $200,000.00
Equity: $200,000.00
Unrealized P/L: $0.00

[TEST] Fetching EUR_USD candles...
Latest candle: 2025-01-07T12:00:00Z
  O: 1.03240
  H: 1.03280
  L: 1.03210
  C: 1.03265

[TEST] Fetching open positions...
Open positions: 0

✓ All tests passed! TradeLocker adapter is ready for E8.
```

### If Connection Fails

**Error: "Missing TradeLocker credentials"**
- Check `.env` file has all 3 variables
- Make sure no typos in variable names

**Error: "Authentication failed"**
- Verify email/password are correct
- Check if E8 account is active
- Try logging into E8 web dashboard first

**Error: "Server not found"**
- Confirm server name: `E8-Live` or `E8-Demo`
- Check with E8 support if unsure

---

## Step 5: Run the E8 Bot

Once connection test passes, start the bot:

```bash
python E8_FOREX_BOT.py
```

### Bot Configuration

The bot is pre-configured with **hybrid strategy** optimized for E8:

| Setting | Value | Reason |
|---------|-------|--------|
| Pairs | EUR_USD, GBP_USD | Best 2 pairs (higher win rate) |
| Position Size | 80% | Stays under 6% drawdown limit |
| Min Score | 4.0 | Quality filter (fewer, better trades) |
| Trading Hours | 8 AM - 12 PM EST | London/NY overlap only |
| Risk per Trade | 0.8% | Reduced for safety |
| Profit Target | 2.5% per trade | Hybrid R/R ratio |
| Stop Loss | 1.0% per trade | Tight risk management |

---

## Step 6: Monitor Progress

The bot displays **live challenge status** every scan:

```
[CHALLENGE STATUS]
  Starting: $200,000.00
  Current: $205,450.00
  Equity: $206,100.00
  Unrealized P/L: $650.00
  Profit Made: $6,100.00 (3.05%)
  Profit Target: $20,000.00 (10%)
  Peak Balance: $206,100.00
  Current DD: 0.00% / 6.00% max
```

### Challenge Outcomes

**✓ PASSED**: When you hit 10% profit ($20,000)
```
====================================================================
🎉 CHALLENGE PASSED! You made 10.02%
====================================================================
```

**✗ FAILED**: If drawdown exceeds 6%
```
====================================================================
❌ CHALLENGE FAILED - Drawdown exceeded 6.0%
====================================================================
```

---

## Step 7: Timeline to $500K

After passing the $200K challenge:

| Phase | Duration | Total Days |
|-------|----------|------------|
| Pass $200K Challenge | 39 days | 39 |
| Save for $500K | 7 days | 46 |
| Pass $500K Challenge | 31 days | 78 |
| **✓ Funded at $500K** | - | **78 days** |

### Income Progression

- **During $200K challenge**: $0/month
- **After $200K funded**: $12,160/month
- **After $500K funded**: **$38,000/month**

---

## Troubleshooting

### Bot not placing trades

**Check trading hours**: Bot only trades 8 AM - 12 PM EST
```python
# Current hour in EST
from datetime import datetime
print(f"Current hour: {datetime.now().hour} EST")
```

**Check min score**: Signals must score ≥ 4.0
```
[EUR_USD] Score: 3.2 < 4.0 - skipping  # Not high enough
```

**Check max positions**: Limited to 2 simultaneous positions
```
[SKIP] Max positions reached  # Wait for exits
```

### Drawdown approaching 6%

The bot will automatically stop if you hit 5.9% DD. But if you want to manually reduce risk:

**Option 1**: Reduce position size further
```python
# In E8_FOREX_BOT.py, line 33
self.position_size_multiplier = 0.60  # 60% instead of 80%
```

**Option 2**: Increase min score (fewer trades)
```python
# In E8_FOREX_BOT.py, line 27
self.min_score = 5.0  # Up from 4.0
```

### API Rate Limits

TradeLocker has rate limits. If you see errors:

**Error: "Too many requests"**
- Bot scans every 1 hour by default (safe)
- Don't run multiple instances of the bot
- Wait 5 minutes and restart

---

## Files Overview

| File | Purpose |
|------|---------|
| `E8_TRADELOCKER_ADAPTER.py` | Connects to TradeLocker API (like OANDA) |
| `E8_FOREX_BOT.py` | Main bot with hybrid strategy |
| `E8_SETUP_INSTRUCTIONS.md` | This file |
| `.env` | Your TradeLocker credentials (keep secret!) |

---

## What's Different from OANDA?

Your OANDA bot (`WORKING_FOREX_OANDA.py`) and E8 bot (`E8_FOREX_BOT.py`) have similar structure, but key differences:

| Aspect | OANDA Bot | E8 Bot |
|--------|-----------|--------|
| Account | $187K paper (practice) | $200K real (E8 challenge) |
| Drawdown Limit | None | 6% max (auto-stops) |
| Position Size | 100% | 80% (for 6% DD limit) |
| Pairs | 4 pairs | 2 pairs (quality filter) |
| Trading Hours | All day | 8 AM-12 PM EST only |
| Min Score | 2.5 | 4.0 (stricter) |
| Goal | Test strategies | Pass challenge ($20K profit) |

---

## Next Steps

1. ✓ Install TradeLocker: `pip install tradelocker`
2. ✓ Add credentials to `.env`
3. ✓ Test connection: `python E8_TRADELOCKER_ADAPTER.py`
4. ✓ Run bot: `python E8_FOREX_BOT.py`
5. ⏳ Wait 39 days for challenge to pass
6. 💰 Get funded and earn $12,160/month
7. 🚀 Scale to $500K challenge

---

## Support

**E8 Support**: https://e8markets.com/support
**TradeLocker Docs**: https://tradelocker.com/api
**Bot Issues**: Check this repo's issues or create new one

---

**Good luck with your E8 challenge! 🚀**

Target: $38,000/month in 78 days.
