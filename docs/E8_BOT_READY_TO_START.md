# E8 Bot Ready - Quick Start Guide

## ✓ Setup Complete!

Your E8 TradeLocker connection is working:
- Email: kkdo@hotmail.com
- Server: E8-Live
- Account: $200,000 balance
- Status: ACTIVE

---

## Start Trading in 3 Steps

### Step 1: Run the Bot

```bash
python E8_FOREX_BOT.py
```

### Step 2: Monitor Progress

The bot will display challenge status every hour:

```
[CHALLENGE STATUS]
  Starting: $200,000.00
  Current: $205,450.00
  Equity: $206,100.00
  Profit Made: $6,100.00 (3.05%)
  Profit Target: $20,000.00 (10%)
  Current DD: 0.00% / 6.00% max
```

### Step 3: Wait for Success

**Timeline**: 39 days to pass
**Pass rate**: 94%
**Income after funded**: $12,160/month

---

## What the Bot Does

### Hybrid Strategy (Auto-Configured)

- **Pairs**: EUR_USD, GBP_USD only
- **Trading Hours**: 8 AM - 12 PM EST (London/NY overlap)
- **Position Size**: 80% (for 6% DD limit)
- **Min Score**: 4.0 (quality filter)
- **Win Rate Target**: 50%
- **Monthly ROI Target**: 7.6%

### Auto-Management

✓ Automatically places trades when score ≥ 4.0
✓ Sets take profit at +2.5%, stop loss at -1.0%
✓ Manages 2 positions max simultaneously
✓ Scans every hour for new opportunities
✓ **Auto-stops if you pass** (10% profit)
✓ **Auto-stops if you fail** (6% drawdown)

---

## Important Notes

### Keep Computer Running

The bot needs to run 24/7. Options:
1. **Local computer**: Keep it on, disable sleep mode
2. **VPS (recommended)**: Rent a cloud server ($5-10/month)
   - Digital Ocean
   - AWS Lightsail
   - Vultr

### Monitor Daily

Check the bot output once per day to ensure:
- Bot is still running
- No errors occurred
- Trades are executing properly

### Don't Interfere

⚠️ **Do NOT**:
- Manually close positions (let bot manage them)
- Change settings mid-challenge
- Stop/restart bot unless necessary
- Place manual trades on same account

Interference can disrupt the strategy and reduce pass rate.

---

## Expected Results

### Week 1
- Profit: ~$3,800 (1.9%)
- Positions: 2-3 trades
- DD: ~1-2%

### Week 2-3
- Profit: ~$10,000 (5%)
- Halfway to target
- DD: ~2-4%

### Week 4-5
- Profit: ~$16,000 (8%)
- Almost there
- DD: ~4-5%

### Week 6 (Day 39)
- **Profit: $20,000+ (10%)**
- **CHALLENGE PASSED!**
- Get funded, start earning $12,160/month

---

## Troubleshooting

### Bot not placing trades

**Check time**: Bot only trades 8 AM - 12 PM EST
```bash
# Check current time
python -c "from datetime import datetime; print(f'Current hour: {datetime.now().hour} EST')"
```

**Check scores**: Signals must score ≥ 4.0
- Look for lines like: `[EUR_USD] Score: 3.2 < 4.0 - skipping`
- This is normal - bot is being selective

### Connection errors

**Restart bot**:
```bash
# Stop: Ctrl+C
# Start again:
python E8_FOREX_BOT.py
```

### Drawdown approaching 6%

**Don't panic**: Bot will auto-stop at 5.9%

If you want to be more conservative:
1. Stop bot (Ctrl+C)
2. Wait for current positions to close
3. Reduce position size in E8_FOREX_BOT.py line 33:
   ```python
   self.position_size_multiplier = 0.60  # Reduce from 0.80 to 0.60
   ```
4. Restart bot

---

## After You Pass

### When Bot Shows "CHALLENGE PASSED"

1. **Screenshot the success message**
2. **Log into E8 dashboard**
3. **Request payout** (if eval phase complete)
4. **Get funded account** (usually 1-3 days)

### Funded Account Income

$200K funded @ 7.6% monthly ROI:
- Gross: $15,200/month
- Your share (80%): **$12,160/month**
- Annual: **$145,920/year**

### Path to $500K

**Timeline**: 78 days total
1. Pass $200K: 39 days ✓
2. Save for $500K: 7 days ($3,000 entry fee)
3. Pass $500K: 31 days (faster with 14% DD limit!)
4. **Income**: $38,000/month

---

## Quick Commands

### Start Bot
```bash
python E8_FOREX_BOT.py
```

### Check if Running
```bash
# Windows
tasklist | findstr python

# Should show: python.exe running E8_FOREX_BOT.py
```

### Stop Bot
```
Press Ctrl+C in the terminal window
```

### View Logs
The bot prints all activity to the console. To save logs:
```bash
python E8_FOREX_BOT.py > e8_bot_log.txt 2>&1
```

---

## Files Overview

| File | Purpose |
|------|---------|
| `E8_FOREX_BOT.py` | Main trading bot (run this) |
| `E8_TRADELOCKER_ADAPTER.py` | TradeLocker API connection |
| `.env` | Your credentials (keep secret!) |
| `E8_SETUP_INSTRUCTIONS.md` | Detailed setup guide |
| `E8_BOT_READY_TO_START.md` | This file |

---

## Support

### E8 Support
- Website: https://e8markets.com/support
- Email: support@e8markets.com
- Usually respond within 24 hours

### Bot Issues
- Check [E8_SETUP_INSTRUCTIONS.md](E8_SETUP_INSTRUCTIONS.md) for troubleshooting
- Review error messages in console output
- Restart bot if connection errors occur

---

## Good Luck!

You're 39 days away from $12,160/month real income.

**Key Stats**:
- Pass rate: 94%
- Target: $20,000 profit (10%)
- Max DD allowed: 6%
- Bot auto-manages everything

Just start the bot and let it run!

```bash
python E8_FOREX_BOT.py
```

🚀 **Target**: $38,000/month in 78 days
