# OpenBB Platform v4.5.0 - Quick Reference

**Status:** ✅ Installed and Running

**Web Interface:** http://127.0.0.1:6900/docs

---

## 🚀 Quick Start

### Launch the Web UI

**Option 1:** Double-click `launch_openbb_api.bat`

**Option 2:** Command line
```bash
openbb-api --host 127.0.0.1 --port 6900
```

**Option 3:** Already running now (background process)

---

## 🌐 Access Points

| Interface | URL | Purpose |
|-----------|-----|---------|
| **Swagger UI** | http://127.0.0.1:6900/docs | Interactive API explorer |
| **ReDoc** | http://127.0.0.1:6900/redoc | Alternative docs |
| **OpenAPI JSON** | http://127.0.0.1:6900/openapi.json | API specification |
| **Workspace** | https://my.openbb.co/app/platform | Cloud dashboard (optional) |

---

## 📊 Key Features for Your Trading System

### 1. **Options Data** (`/derivatives/options/chains`)
**Why it matters:** Get real-time implied volatility for strike selection

**Example:**
```python
from openbb import obb

chains = obb.derivatives.options.chains(symbol='TSLA', provider='yfinance')
df = chains.to_dataframe()

# Get average IV
current_iv = df['implied_volatility'].mean()
print(f"TSLA IV: {current_iv:.1%}")
```

**Use case:**
- If IV > 75th percentile → Use Iron Condors (sell premium)
- If IV < 25th percentile → Use Dual Options (buy options)

---

### 2. **Equity Price Data** (`/equity/price/historical`)
**Why it matters:** Backup data source if Alpaca goes down

**Providers available:**
- yfinance (free)
- polygon (requires API key)
- alpha_vantage (free tier)
- intrinio (paid)

**Example:**
```python
hist = obb.equity.price.historical(
    symbol='AAPL',
    start_date='2025-10-01',
    end_date='2025-10-08',
    provider='yfinance'
)
```

---

### 3. **Economic Calendar** (`/economy/calendar`)
**Why it matters:** Avoid trading before FOMC, CPI, NFP

**Providers:**
- FMP (requires free API key)
- nasdaq (free, limited)

**Example:**
```python
from datetime import datetime, timedelta

calendar = obb.economy.calendar(
    provider='fmp',
    start_date=datetime.now(),
    end_date=datetime.now() + timedelta(days=1)
)

for event in calendar.results:
    print(f"{event.date}: {event.event}")
```

**Integration:**
- Pause trading 30 minutes before major events
- Resume after market digests the news

---

### 4. **News Feed** (`/news/world`, `/news/company`)
**Why it matters:** Filter out stocks with negative headlines

**Providers:**
- benzinga (requires API key)
- fmp (requires API key)
- intrinio (requires API key)

**Example:**
```python
news = obb.news.company(symbol='TSLA', provider='benzinga', limit=5)

for article in news.results:
    print(f"{article.date}: {article.title}")
    print(f"Sentiment: {article.sentiment}")  # if available
```

**Integration:**
- Skip trades if negative news in last 2 hours
- Boost confidence if positive earnings surprise

---

### 5. **Unusual Options Activity** (`/derivatives/options/unusual`)
**Why it matters:** Detect institutional smart money

**Example:**
```python
# Note: This endpoint may vary by provider
unusual = obb.derivatives.options.unusual(provider='intrinio')

for activity in unusual.results:
    print(f"{activity.symbol}: {activity.type} ${activity.strike}")
    print(f"Volume/OI: {activity.volume / activity.open_interest:.1f}x")
```

**Signal:**
- Volume/OI > 3.0 = New large positions
- Follow the smart money direction

---

### 6. **Congress Trading** (`/regulators/congress/trading`) **NEW in v4.5.0**
**Why it matters:** Politicians often trade before market-moving legislation

**Example:**
```python
congress = obb.regulators.congress.trading(provider='congress_gov')

for trade in congress.results:
    print(f"{trade.representative}: {trade.ticker} {trade.transaction_type}")
    print(f"Amount: {trade.amount}")
```

**Integration:**
- If senator buys defense stocks → Check for defense bill vote
- If rep sells tech → Check for regulation news

---

## 🔌 Integration with Your Scanner

### Week 3 Priority: Add IV Rank to Scanner

**File:** `week2_sp500_scanner.py`

**Add before trade execution:**

```python
def get_iv_rank(symbol):
    """Get IV rank to decide strategy"""
    try:
        from openbb import obb

        chains = obb.derivatives.options.chains(symbol=symbol, provider='yfinance')
        df = chains.to_dataframe()

        iv_values = df['implied_volatility'].dropna()
        current_iv = iv_values.mean()
        iv_min = iv_values.min()
        iv_max = iv_values.max()

        iv_rank = ((current_iv - iv_min) / (iv_max - iv_min)) * 100

        return {
            'current_iv': current_iv,
            'iv_rank': iv_rank,
            'strategy': 'iron_condor' if iv_rank > 75 else 'dual_options'
        }
    except:
        return {'strategy': 'dual_options'}  # Fallback
```

**Use in scanner:**
```python
# In execute_top_opportunities()
for opp in to_execute:
    iv_data = get_iv_rank(opp['symbol'])

    if iv_data['iv_rank'] > 75:
        # High IV → Sell premium
        self.iron_condor_engine.execute_iron_condor(...)
    else:
        # Low IV → Buy options
        self.options_engine.execute_dual_strategy(...)
```

---

## 🔑 API Keys (Optional)

Some providers require free API keys:

**Free tiers available:**
- **FMP:** https://financialmodelingprep.com (250 calls/day free)
- **Alpha Vantage:** https://www.alphavantage.co (25 calls/day free)
- **Polygon:** https://polygon.io (5 calls/minute free)

**Set keys in:** `~/.openbb/user_settings.json`

```json
{
  "credentials": {
    "fmp_api_key": "your_key_here",
    "benzinga_api_key": "your_key_here"
  }
}
```

**Note:** yfinance provider works without any API keys (already working for you)

---

## 📱 OpenBB Workspace (Optional Cloud UI)

**What it is:** Browser-based dashboard that connects to your local API

**Access:** https://my.openbb.co/app/platform

**Features:**
- Drag-and-drop widgets
- Create custom dashboards
- Share with team (if you scale to 80 accounts)
- Mobile access to your data

**Connect it:**
1. Visit https://my.openbb.co/app/platform
2. Point to http://127.0.0.1:6900
3. Start building custom dashboards

**Use case for you:**
- Build a live P&L dashboard showing all 80 accounts
- Create alerts for unusual options activity
- Monitor economic calendar visually

---

## 🛠️ Advanced: Custom Endpoints

You can add custom endpoints to the API:

**Example:** Add your own scanner endpoint

**File:** `custom_scanner_api.py`
```python
from openbb_platform_api.main import app

@app.get("/api/v1/scanner/momentum")
async def momentum_scan(min_momentum: float = 0.05):
    """Custom momentum scanner endpoint"""
    # Your scanner logic here
    return {"opportunities": [...]}
```

**Launch:**
```bash
openbb-api --app custom_scanner_api.py --port 6900
```

**Access:** http://127.0.0.1:6900/docs

---

## 🔍 Exploring Available Endpoints

**Swagger UI:** http://127.0.0.1:6900/docs

**Key sections:**
1. **Equity** - Stock prices, fundamentals, estimates
2. **Derivatives** - Options chains, unusual activity
3. **Economy** - GDP, inflation, interest rates, calendar
4. **News** - Company news, world news, sentiment
5. **Regulators** - SEC filings, congress trades, insider trades
6. **Crypto** - Cryptocurrency data (if you expand later)

**Try it:**
1. Open Swagger UI
2. Expand `/derivatives/options/chains`
3. Click "Try it out"
4. Enter symbol: TSLA
5. Select provider: yfinance
6. Click "Execute"
7. See live TSLA options data with IV

---

## 💡 Pro Tips

1. **Use yfinance for free data** - It works great for your needs
2. **Cache API responses** - Don't spam the same request
3. **Combine with Alpaca** - Use OpenBB as backup/validation
4. **Test in Swagger first** - Then copy the working code
5. **Monitor rate limits** - Free tiers have limits, pace your requests

---

## 🚦 Status Check

**Is it running?**
```bash
# Check if port 6900 is listening
netstat -an | findstr "6900"
```

**Stop the server:**
```bash
# Find the process
tasklist | findstr "openbb-api"

# Kill it
taskkill /F /PID <process_id>
```

**Restart:**
```bash
openbb-api --host 127.0.0.1 --port 6900
```

---

## 📚 Official Documentation

- **OpenBB Docs:** https://docs.openbb.co/platform
- **API Reference:** http://127.0.0.1:6900/redoc (when running)
- **GitHub:** https://github.com/OpenBB-finance/OpenBB
- **Discord:** https://discord.gg/openbb

---

## 🎯 Week 3 Integration Roadmap

### Day 6-7: Add IV Rank
- Integrate `get_iv_rank()` into scanner
- Use IV rank to choose strategy
- Test on 5-10 trades

### Day 8-9: Add Economic Calendar Filter
- Fetch calendar at start of day
- Skip trading 30min before major events
- Track performance difference

### Day 10: Add News Filter (Optional)
- Get API key from FMP (free)
- Filter out stocks with negative news in last 2 hours
- Compare results

### Week 4+: Advanced Features
- Unusual options activity detection
- Congress trading signals
- Multi-provider fallback (if Alpaca down, use OpenBB)

---

**Current Status:**
✅ OpenBB Platform v4.5.0 installed
✅ API server running on port 6900
✅ Web UI accessible at http://127.0.0.1:6900/docs
✅ yfinance provider working (no API keys needed)
✅ Integration examples created

**Next Step:** Open http://127.0.0.1:6900/docs in your browser and explore! 🚀
