# E8 Demo Account - Connection Guide for ATLAS

**Your Credentials:**
```
Email: kkdo@hotmail.com
Password: 56H2K*kd
Platform: MatchTrader
Server: MatchTrader-Demo
Webtrader: https://mtr.e8markets.com
```

---

## Problem: Cloudflare Bot Protection

The E8 MatchTrader REST API is behind Cloudflare bot protection, which blocks automated Python requests.

**Error:** `403 Forbidden - Enable JavaScript and cookies to continue`

This means we can't use the standard REST API directly.

---

## Solution Options

### Option 1: Use MatchTrader WebSocket API (Recommended)

MatchTrader supports WebSocket connections with STOMP protocol for real-time data. This **may bypass** Cloudflare if accessed differently.

**Pros:**
- Real-time market data
- Faster than REST
- Less likely to be blocked

**Cons:**
- Need to implement STOMP protocol
- More complex setup

**Status:** Need to test if WebSocket bypasses Cloudflare

---

### Option 2: Browser Automation (Selenium/Playwright)

Use Selenium to control a real browser and interact with the webtrader.

**Pros:**
- Definitely works (uses real browser)
- Can access all features
- Bypasses Cloudflare automatically

**Cons:**
- Slower (browser overhead)
- Less stable (browser crashes)
- Requires Chrome/Firefox installed

**Implementation Time:** 2-3 hours

---

### Option 3: Switch to MT5 or cTrader (If Available)

Check if your E8 account supports MT5 or cTrader instead.

**MT5 Advantages:**
- Standard MetaTrader API (well-documented)
- Python library available: MetaTrader5
- No Cloudflare issues
- Widely supported

**cTrader Advantages:**
- FIX protocol support (already researched)
- Python cTrader-fix library exists
- Institutional-grade

**Check:** Log into E8 dashboard and see if you can switch platforms

---

### Option 4: Manual Trading with ATLAS Signals (Simplest)

Run ATLAS in "signal mode" - it generates trade signals, you execute manually.

**How it works:**
1. ATLAS runs paper trading with simulated data
2. When it wants to trade, it sends you a Telegram notification
3. You manually execute the trade in MatchTrader webtrader
4. You report back the result (win/loss, P/L)
5. ATLAS learns from the results

**Pros:**
- No API needed
- Works 100%
- You verify each trade before execution

**Cons:**
- Requires manual intervention
- Not fully autonomous

**Best for:** Paper training phase (60 days)

---

## Recommended Approach

**For Now (Paper Training):**

Use **Option 4 (Manual Trading with Signals)** because:
- You're testing the strategy anyway (60-day validation)
- Manual execution ensures you understand each trade
- No need to fight Cloudflare during learning phase
- You can monitor and intervene if needed

**After Paper Training (E8 Deployment):**

Implement **Option 2 (Browser Automation)** or **Option 3 (Switch to MT5)** because:
- Proven strategy (passed 60-day validation)
- Automation required for 24/5 operation
- Worth the 2-3 hour setup time

---

## Next Steps

### Immediate: Check Platform Options

1. Log into E8 account at https://e8markets.com
2. Go to your account/challenge dashboard
3. Check if you can create an MT5 or cTrader demo account instead
4. If yes → Use MT5 (easiest API integration)
5. If no → Proceed with manual signals approach

### Manual Signal Mode Setup (2 hours)

If we go with manual signals:

1. **Configure ATLAS for signal mode**
   ```python
   config = {
       "mode": "signal_only",
       "execution": "manual",
       "notifications": "telegram"
   }
   ```

2. **Set up Telegram bot** (for signals)
   - ATLAS sends: "BUY EUR/USD 3 lots, SL: 1.08200, TP: 1.08500"
   - You execute in webtrader
   - You reply: "EXECUTED" or "CLOSED +$1800"

3. **Run paper training**
   - ATLAS generates signals based on real market data
   - You manually execute if you agree
   - System learns from results
   - After 60 days: Review performance

### Browser Automation Setup (3 hours)

If we go with Selenium automation:

1. Install Selenium
   ```bash
   pip install selenium webdriver-manager
   ```

2. Create MatchTrader browser adapter
   - Auto-login to webtrader
   - Parse account balance from page
   - Click buy/sell buttons
   - Set SL/TP via UI

3. Integrate with ATLAS
   - Replace REST API calls with browser actions
   - Slower but works 100%

---

## My Recommendation

**Let's check if E8 offers MT5 first.** If they do, we'll use that instead (1 hour setup vs 3 hours for Selenium).

**If MatchTrader only:**
- Week 1-8 (Paper Training): Manual signals via Telegram
- Week 9+ (E8 Live): Selenium automation or manual (if comfortable)

**Want me to:**
1. Help you check E8 platform options?
2. Set up manual signal mode with Telegram?
3. Build Selenium automation for MatchTrader?
4. Test if WebSocket API bypasses Cloudflare?

Let me know what you prefer!
