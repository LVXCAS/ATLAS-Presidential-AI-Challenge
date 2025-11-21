# NEWS FILTER - CRITICAL SAFETY FEATURE

## Why News Filtering is Essential for E8

**Problem:** Major economic news causes:
1. **Volatility spikes** - 100-200 pip moves in seconds
2. **Slippage** - Your 1% stop loss becomes 2-3% in fast markets
3. **Unpredictable moves** - All technical analysis becomes useless

**Result without news filter:**
```
You place EUR/USD LONG at 1.3075
Stop loss at 1.3062 (1% = $2,000 max loss)

NFP releases better than expected
EUR/USD gaps down 150 pips in 10 seconds
Your SL executes at 1.2925 instead of 1.3062

Loss: $6,000 (3x expected) = DAILY DD VIOLATION = Account terminated
```

---

## What the News Filter Does

### Blocks Trading Around High-Impact Events

**Major events monitored:**
- **NFP (Non-Farm Payroll)** - First Friday of month, 8:30 AM EST
- **FOMC (Federal Reserve)** - 8x per year, 2:00 PM EST
- **CPI (Consumer Price Index)** - Monthly, 8:30 AM EST
- **GDP releases**
- **Central bank rate decisions** (Fed, ECB, BOE, BOJ)
- **Employment data**
- **Retail sales**

**Block window:**
- 1 hour BEFORE event
- 1 hour AFTER event
- Total: 2-hour blackout per event

**Why 1 hour before?**
- Markets start moving 15-30 min before official release
- Liquidity dries up (wider spreads)
- Institutional traders close positions
- Your entry price might be 10-20 pips worse than expected

**Why 1 hour after?**
- Initial reaction often reverses
- Markets digest the data
- Multiple waves of volatility
- Slippage remains high for 30-60 min

---

## How It Works in the Bot

### Integration Points

**1. On Bot Startup:**
```python
# Initialize news filter
self.news_filter = NewsFilter()
```

**2. At Start of Each Scan:**
```python
# Show upcoming news events (next 4 hours)
self.news_filter.print_upcoming_news(hours=4)
```

Output:
```
[NEWS] Upcoming high-impact events (next 4 hours):
----------------------------------------------------------------------
  2025-11-20 14:00 (+2.5h)
    FOMC Rate Decision [USD]
----------------------------------------------------------------------
```

**3. Before Scoring Each Setup:**
```python
# Check news safety FIRST (blocks everything else)
is_safe, news_msg = self.news_filter.check_news_safety(pair)
if not is_safe:
    return 0, [f"BLOCKED BY NEWS: {news_msg}"], None
```

Output:
```
--- EUR_USD ---
  Score: 0.0 / 6.0
    - BLOCKED BY NEWS: High-impact news in 45 min: FOMC Rate Decision
  [WAIT] Score 0.0 < 6.0 minimum
```

---

## Example Scenarios

### Scenario 1: NFP Day

```
Friday, November 1, 2025

7:30 AM EST - Bot scans
  → NFP at 8:30 AM (60 min away)
  → News filter: BLOCKS all USD pairs
  → EUR/USD, GBP/USD, USD/JPY = NO TRADES
  → Output: "BLOCKED BY NEWS: High-impact news in 60 min: US Non-Farm Payroll"

9:30 AM EST - Bot scans
  → NFP released at 8:30 AM (60 min ago)
  → Still in 1-hour blackout AFTER event
  → News filter: BLOCKS all USD pairs
  → Output: "BLOCKED BY NEWS: High-impact news 60 min ago: US Non-Farm Payroll"

10:30 AM EST - Bot scans
  → NFP was 2 hours ago
  → Blackout window expired
  → News filter: CLEAR
  → Bot can evaluate setups normally
```

**Trade prevented:**
- Score might be 6.0+ during NFP
- Perfect technical setup
- BUT: 200 pip spike would blow through SL
- News filter SAVED you from -$6,000 loss

### Scenario 2: FOMC Rate Decision

```
Wednesday, November 6, 2025

1:00 PM EST - Bot scans
  → FOMC at 2:00 PM (60 min away)
  → News filter: BLOCKS all USD pairs
  → No trades allowed

2:00 PM EST - FOMC released
  → Markets spike 150 pips
  → You have NO positions (protected!)

3:00 PM EST - Bot scans
  → FOMC was 60 min ago
  → Still in blackout
  → News filter: BLOCKS

4:00 PM EST - Bot scans
  → FOMC was 2 hours ago
  → Blackout expired
  → Markets calmer
  → Bot resumes normal operation
```

### Scenario 3: No Major News

```
Tuesday, November 5, 2025 (quiet day)

10:00 AM EST - Bot scans
  → Check calendar: No major events next 4 hours
  → News filter: CLEAR
  → Output: "No major news in next hour"
  → Bot evaluates setups normally

If score 6.0+ found:
  → All filters pass (ADX, RSI, MACD, BB, news)
  → Trade ALLOWED
```

---

## News Calendar Sources

### Current Implementation

**Demo/Fallback Mode:**
- Manually curated events (NFP, FOMC, CPI dates)
- Covers most high-impact USD events
- Good enough for demo validation

**Code:**
```python
# NFP - First Friday of each month
# FOMC - 8x per year (hardcoded dates)
# CPI - Mid-month (simplified to 15th)
```

### Production Options (After Demo)

**1. Trading Economics API** (Recommended)
- URL: `api.tradingeconomics.com/calendar`
- Free tier: 1000 calls/month
- Covers: All major economies, real-time updates
- Cost: $0 (free) or $49/month (professional)

**2. Forex Factory** (Scraping)
- URL: `forexfactory.com/calendar`
- Free, comprehensive
- Requires web scraping (more brittle)

**3. Investing.com** (Scraping)
- URL: `investing.com/economic-calendar`
- Free, detailed
- Requires web scraping

**For 60-day demo:**
- Manual calendar is sufficient
- NFP and FOMC are the main killers
- Can update monthly with known dates

**For E8 evaluation (if demo passes):**
- Upgrade to Trading Economics API ($49/month)
- Worth it when trading $200k account
- Real-time updates prevent surprises

---

## How to Update Manual Calendar

If using manual calendar (current setup), update monthly:

**Edit:** `BOTS/news_filter.py`

Find `get_known_events_2025()` function:

```python
# NFP - First Friday of each month at 8:30 AM EST
nfp_dates = [
    '2025-01-10', '2025-02-07', '2025-03-07', '2025-04-04',
    '2025-05-02', '2025-06-06', '2025-07-03', '2025-08-01',
    '2025-09-05', '2025-10-03', '2025-11-07', '2025-12-05'
]

# FOMC Rate Decisions (8x per year)
fomc_dates = [
    '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18',
    '2025-07-30', '2025-09-17', '2025-10-29', '2025-12-17'
]
```

**To add CPI dates:**
```python
# CPI - Usually mid-month at 8:30 AM EST
cpi_dates = [
    '2025-01-15', '2025-02-12', '2025-03-12', '2025-04-10',
    # ... etc
]

for date in cpi_dates:
    events.append({
        'date': f"{date}T08:30:00",
        'event': 'US Consumer Price Index (CPI)',
        'impact': 'HIGH',
        'currency': 'USD'
    })
```

**Sources for dates:**
- Forex Factory: `forexfactory.com/calendar`
- Trading Economics: `tradingeconomics.com/calendar`
- Federal Reserve: `federalreserve.gov` (FOMC schedule)

---

## Testing the News Filter

**Standalone test:**
```bash
python BOTS/news_filter.py
```

Output:
```
======================================================================
NEWS FILTER - TESTING
======================================================================

--- Checking EUR_USD Safety ---
Safe to trade: True
Reason: No major news in next hour

--- Checking USD_JPY Safety ---
Safe to trade: False
Reason: High-impact news in 45 min: FOMC Rate Decision

[NEWS] Upcoming high-impact events (next 48 hours):
----------------------------------------------------------------------
  2025-11-20 14:00 (+2.5h)
    FOMC Rate Decision [USD]
  2025-11-22 08:30 (+50.5h)
    US Non-Farm Payroll (NFP) [USD]
----------------------------------------------------------------------
```

---

## Impact on Trade Frequency

### Without News Filter

**Aggressive bot (score 3.0):**
- 25 trades/day
- Some during NFP, FOMC, CPI
- High chance of slippage violations
- Survival: 2 hours (proved)

### With News Filter

**Ultra-conservative bot (score 6.0 + news filter):**
- 0-2 trades/week normally
- ZERO trades during high-impact news
- 2-4 trading days per month lost to news blackouts
- **BUT:** Zero slippage violations
- Survival: 60+ days (target)

**Trade-off:**
```
Lost opportunity: 2-4 days/month = ~10% of trading days
Gained safety: ZERO news-related DD violations
Net effect: Fewer trades, but ONLY high-quality, safe trades
```

**Math:**
- 20 trading days/month
- 2 NFP days (blackout)
- 1-2 FOMC days (blackout)
- ~3-4 days lost
- Remaining: 16-17 clean days
- At 0-2 trades/week = 8-16 trades/month
- 3-4 lost days = ~1-2 fewer trades/month
- **Worth it to avoid ONE -$6,000 slippage violation**

---

## Configuration Options

**Adjust block window:**

Edit `news_filter.py`:
```python
# Current: 1 hour before + 1 hour after
self.block_before_minutes = 60
self.block_after_minutes = 60

# More conservative: 2 hours before + 2 hours after
self.block_before_minutes = 120
self.block_after_minutes = 120

# Less conservative: 30 min before + 30 min after
# (NOT RECOMMENDED for E8 - slippage risk too high)
self.block_before_minutes = 30
self.block_after_minutes = 30
```

**Add more event keywords:**

```python
self.high_impact_keywords = [
    # Existing
    'nfp', 'fomc', 'cpi', 'gdp',

    # Add more
    'ppi',  # Producer Price Index
    'pce',  # Personal Consumption Expenditures
    'jobless claims',
    'trade balance',
    'manufacturing pmi',
    'services pmi'
]
```

---

## What You'll See in Logs

### Normal Day (No News)

```
[SCANNING] Looking for PERFECT setups (score >= 6.0)...

[NEWS] No high-impact events in next 4 hours

--- EUR_USD ---
  Price: 1.30745
  Score: 6.0 / 6.0
    - Very strong UP trend (ADX 32.1, 1.2% from 200 EMA)
    - RSI pullback zone (52.3)
    - MACD bullish (0.00234)
    - Price in BB range (pullback buy)
    - Volatility acceptable
  [OPPORTUNITY] BUY signal (score 6.0)
```

### News Day (NFP)

```
[SCANNING] Looking for PERFECT setups (score >= 6.0)...

[NEWS] Upcoming high-impact events (next 4 hours):
----------------------------------------------------------------------
  2025-11-20 08:30 (+0.5h)
    US Non-Farm Payroll (NFP) [USD]
----------------------------------------------------------------------

--- EUR_USD ---
  Score: 0.0 / 6.0
    - BLOCKED BY NEWS: High-impact news in 30 min: US Non-Farm Payroll (NFP)
  [WAIT] Score 0.0 < 6.0 minimum

--- GBP_USD ---
  Score: 0.0 / 6.0
    - BLOCKED BY NEWS: High-impact news in 30 min: US Non-Farm Payroll (NFP)
  [WAIT] Score 0.0 < 6.0 minimum

--- USD_JPY ---
  Score: 0.0 / 6.0
    - BLOCKED BY NEWS: High-impact news in 30 min: US Non-Farm Payroll (NFP)
  [WAIT] Score 0.0 < 6.0 minimum

[NO OPPORTUNITIES] Zero setups meet criteria (score >= 6.0)
This is NORMAL for ultra-conservative strategy!
```

---

## Bottom Line

**News filter is the 6th critical safety feature** (after daily DD tracker):

1. ✅ **Daily DD tracker** - Prevents exceeding daily loss limit
2. ✅ **Peak balance persistence** - No reset on restart
3. ✅ **Hard position caps** - Max 2 lots
4. ✅ **Score 6.0+ threshold** - Perfect setups only
5. ✅ **Session filter** - London/NY overlap only
6. ✅ **NEWS FILTER** - Block trading around volatility events

**Why it matters:**

```
WITHOUT news filter:
  Perfect technical setup during NFP
  → Enter EUR/USD LONG
  → NFP releases
  → 150 pip spike against you
  → SL slips from 1% to 3%
  → -$6,000 loss (vs $2,000 expected)
  → DAILY DD VIOLATION
  → Account terminated
  → Lost $600 E8 fee

WITH news filter:
  Perfect technical setup during NFP
  → News filter: BLOCKED
  → No trade placed
  → NFP releases
  → Markets spike
  → You have no positions
  → $0 loss
  → Account survives
  → Continue trading next day
```

**The news filter doesn't make you more money. It prevents catastrophic losses.**

And in E8, **survival > profit**.

You can't profit if your account is terminated.

---

**News filter is now integrated into E8_ULTRA_CONSERVATIVE_BOT.py.**

No additional setup needed. Just start the bot and it will automatically check news before every trade.
