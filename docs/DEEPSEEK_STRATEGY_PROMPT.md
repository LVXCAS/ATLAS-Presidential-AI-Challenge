# DeepSeek Query: High Win-Rate Forex Strategy for E8 Challenge

## Context
I'm trading a $200K E8 prop firm challenge with these constraints:
- **Max Drawdown**: 6% ($12,000)
- **Profit Target**: 10% ($20,000)
- **Pairs**: EUR/USD, GBP/USD, USD/JPY
- **Timeframe**: Must pass challenge in 30-90 days
- **Tools**: Backtrader for backtesting, OANDA for data, Python for execution

## Problem
I just backtested 3 "market microstructure" strategies over 90 days:
- **London Fakeout**: Low volume breakouts of Asian range
- **NY Absorption**: Rejections at previous day high/low
- **Tokyo Gap Fill**: Gap trading during Asian session

**Results**: -$255,810 loss (-128%), 36.7% win rate, 54% max drawdown

These strategies FAILED because:
1. Low win rates (27-43%)
2. Too many false signals
3. Targets rarely hit, stops always hit
4. No real statistical edge

## Question for DeepSeek

**What is a PROVEN, HIGH WIN-RATE forex strategy that:**

1. **Works on EUR/USD, GBP/USD, USD/JPY** (liquid major pairs)
2. **Has 50%+ win rate** (ideally 60-70%)
3. **Can be backtested with historical data** (no discretionary/visual analysis)
4. **Stays under 6% drawdown** (strict risk management)
5. **Uses common indicators** (RSI, MACD, Moving Averages, Bollinger Bands, ATR, etc.)
6. **Has published research or proven track record** (not theoretical)

## Specific Requirements

**Entry Criteria**: Clear, rule-based entry conditions (no "wait for confirmation" - need exact rules)

**Exit Criteria**: Defined stop-loss and take-profit levels (in pips or ATR multiples)

**Position Sizing**: How to calculate lot size based on account risk (1% per trade for E8)

**Timeframe**: Which chart timeframe works best (5m, 15m, 1H, 4H, Daily?)

**Session**: Does it work best during specific trading sessions (London, NY, Tokyo, overlap)?

## Examples of Strategy Types I'm Looking For

- **Mean Reversion**: RSI oversold/overbought with Bollinger Bands
- **Trend Following**: Moving average crossovers with ADX confirmation
- **Breakout**: Support/resistance breaks with volume confirmation
- **Carry Trade**: Interest rate differential strategies
- **Statistical Arbitrage**: Pairs trading or correlation strategies

## What I DON'T Want

- ❌ Martingale or grid strategies
- ❌ High-frequency scalping (need 50+ trades/day)
- ❌ Discretionary "read the market" approaches
- ❌ Strategies requiring exotic indicators or paid tools
- ❌ Anything that risks more than 2% per trade

## Output Format Requested

Please provide:

### 1. Strategy Name & Overview
Brief description of the strategy and why it works

### 2. Entry Rules
Exact conditions (e.g., "Enter LONG when: RSI(14) < 30 AND price touches lower Bollinger Band AND MACD crosses above signal line")

### 3. Exit Rules
- Stop-loss placement
- Take-profit target(s)
- Trailing stop rules (if any)

### 4. Position Sizing Formula
How to calculate lot size for 1% account risk

### 5. Optimal Parameters
- Timeframe (e.g., 1H)
- Trading hours (e.g., London + NY overlap: 8 AM - 12 PM EST)
- Indicator settings (e.g., RSI period = 14)

### 6. Expected Performance
- Win rate (%)
- Average risk:reward ratio
- Expected drawdown
- Monthly return estimate

### 7. Risk Management Rules
- Max trades per day
- Daily loss limit
- Pair correlation rules

### 8. Research/Proof
- Link to published research, books, or verified backtests
- OR provide sample Backtrader pseudocode for validation

---

## Additional Notes

- I have access to TA-Lib (200+ indicators)
- I can implement any mathematical formula
- Backtrader handles all execution logic
- I need something that will actually work, not blow my account

**Goal**: Pass E8 challenge in 30-60 days with consistent, low-drawdown returns.

Please recommend the SINGLE BEST strategy you know that meets these criteria.
