# How The Bot Gets Market Data and Analyzes Stocks

**Date:** October 16, 2025

---

## QUICK ANSWER

The bot uses **Yahoo Finance + 6 other APIs** to download historical price data (60 days), then calculates 50+ technical indicators across multiple timeframes (daily, weekly, monthly) before making any trade.

---

## DATA SOURCES (7 APIs with Automatic Fallback)

1. **Yahoo Finance** - Primary free source
2. **Alpaca API** - Broker real-time data  
3. **Polygon.io** - Professional market data
4. **Alpha Vantage** - Technical analysis data
5. **Finnhub** - News and fundamentals
6. **TwelveData** - Backup technical data
7. **OpenBB Platform** - Comprehensive financial data

If one fails, automatically tries the next!

---

## TIMEFRAMES ANALYZED

The bot looks at **5 different timeframes** simultaneously:

- **1 day (1d)** - Intraday price action
- **5 days (5d)** - Short-term trend
- **1 week (1wk)** - Weekly trend  
- **1 month (1mo)** - Monthly trend
- **3 months (3mo)** - Quarterly trend

**Multi-Timeframe Bonus:**
If daily and weekly trends both agree (both uptrend or both downtrend), the bot adds +20% to confidence score!

---

## WHAT DATA IS FETCHED

### Historical Price Data (60 days):
- Open, High, Low, Close prices
- Volume
- Date/Time for each candle

### Example for AAPL:
```
Date        Open    High    Low     Close   Volume
2025-08-01  175.20  178.50  174.80  177.30  45.2M
2025-08-02  177.50  180.20  177.00  179.80  52.1M
2025-08-03  179.90  182.30  179.50  181.50  48.7M
... (60 days total)
```

---

## TECHNICAL INDICATORS CALCULATED

From the 60-day price history, bot calculates:

### Trend Indicators:
- SMA 20 (20-day moving average)
- SMA 50 (50-day moving average)
- EMA 12 & 26 (exponential averages)
- MACD (trend momentum)

### Momentum:
- RSI (14-period, 0-100 scale)
- Price momentum (5, 10, 20-day)
- Momentum acceleration

### Volatility:
- Bollinger Bands (upper/lower)
- Realized volatility (annualized)
- ATR (average true range)

### Support/Resistance:
- Recent highs (resistance levels)
- Recent lows (support levels)
- Nearest support/resistance

### Volume:
- Average volume
- Volume ratio (current vs average)
- Volume surge detection

---

## HOW IT WORKS

### Step 1: Fetch Raw Data
```python
# Get 60 days of price bars
ticker = yf.Ticker('AAPL')
hist = ticker.history(period='60d')

Result:
  Date | Open | High | Low | Close | Volume
  (60 rows of data)
```

### Step 2: Calculate Indicators
```python
# RSI calculation
delta = Close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rsi = 100 - (100 / (1 + gain/loss))

# MACD
ema_12 = Close.ewm(span=12).mean()
ema_26 = Close.ewm(span=26).mean()
macd = ema_12 - ema_26
```

### Step 3: Identify Signals
```python
# Check for trading signals
if rsi < 30:  # Oversold
    confidence += 8%
if current_price > sma_20:  # Above average
    confidence += 5%
if volume > avg_volume * 2:  # High volume
    confidence += 10%
```

### Step 4: Multi-Timeframe Check
```python
# Daily trend
daily_trend = 'UPTREND' if price > sma_20 else 'DOWNTREND'

# Weekly trend
weekly_trend = calculate_trend(weekly_data)

# Alignment bonus
if daily_trend == weekly_trend:
    confidence += 20%
```

---

## EXAMPLE OUTPUT

After fetching and analyzing, bot gets:

```python
market_data = {
    'symbol': 'AAPL',
    'current_price': 180.50,
    'timestamp': 2025-10-16 13:00:00,
    
    'rsi': 58.2,                # Momentum indicator
    'macd': 2.45,               # Trend strength
    'sma_20': 178.30,           # 20-day average
    'bollinger_upper': 185.00,  # Upper band
    'bollinger_lower': 172.00,  # Lower band
    'realized_vol': 25.8,       # Volatility %
    
    'nearest_support': 175.00,
    'nearest_resistance': 185.00,
    
    'overall_signal': 'BULLISH',
    'signal_confidence': 0.68,
    'bullish_factors': [
        'above_sma20',
        'macd_bullish',
        'volume_surge'
    ]
}
```

---

## WHERE THIS DATA IS USED

### 1. Opportunity Scanning
Bot checks each watchlist symbol to find trades

### 2. Confidence Scoring
All indicators feed into confidence calculation (must be ≥80%)

### 3. Position Monitoring  
Check open positions every cycle

### 4. Exit Decisions
Decide when to exit based on current indicators

---

## CACHING

To avoid rate limits:
- Data cached for 5 minutes
- Reused for multiple analyses
- Refreshed automatically when stale

---

## KEY FILES

1. **OPTIONS_BOT.py:979** - Main data fetching function
2. **enhanced_technical_analysis_multiapi.py** - Indicator calculations
3. **analysis/multitimeframe_analyzer.py** - Multi-timeframe analysis

---

## SUMMARY

**The bot sees the market through:**

✓ 60 days of historical price data
✓ 50+ technical indicators
✓ 5 different timeframes (daily to quarterly)
✓ Support/resistance levels
✓ Volume analysis
✓ Trend detection
✓ Momentum calculations

**Before every trade decision!**

This is how your bot "reads the graph" and understands market conditions.
