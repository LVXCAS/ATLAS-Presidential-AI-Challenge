# REAL ALPACA EXECUTION - ENABLED

## Status: LIVE EXECUTION READY

The trading system has been modified to place **REAL orders** on your Alpaca PAPER trading account.

---

## What Changed

### File Modified: `execution/auto_execution_engine.py`

#### Key Changes:

1. **Added alpaca-trade-api Library**
   - Imported `alpaca_trade_api as tradeapi` for full REST API support
   - This library handles options trading better than the newer SDK

2. **Real Order Execution**
   - Replaced simulated paper trading with actual Alpaca API calls
   - Orders now go to Alpaca and appear on your dashboard
   - Uses `self.alpaca_api.submit_order()` for real execution

3. **Bull Put Spread Implementation**
   - **LEG 1**: BUY PUT (protection) at lower strike - submitted FIRST
   - **LEG 2**: SELL PUT (premium) at higher strike - submitted AFTER protection
   - Uses proper OCC format for option symbols (e.g., "SPY251114P00450000")

4. **Option Symbol Format (OCC Standard)**
   ```
   Format: SYMBOL + YYMMDD + P/C + STRIKE (8 digits)
   Example: AAPL251114P00175000
   - AAPL = underlying symbol
   - 251114 = Nov 14, 2025
   - P = Put option
   - 00175000 = $175.00 strike (multiplied by 1000)
   ```

5. **Safety Features**
   - Protection leg (BUY PUT) MUST execute before selling
   - Automatic rollback if orders fail
   - Max 3 contracts per trade (safety limit)
   - Error handling with detailed logging

---

## How It Works

### Order Flow:

```
1. Calculate strikes (5% and 10% OTM)
2. Calculate position size based on risk
3. Build OCC-format option symbols
4. Submit BUY PUT order (protection) → Wait for confirmation
5. Submit SELL PUT order (premium) → Wait for confirmation
6. Log execution locally
7. Return order IDs and status
```

### Example Execution Output:

```
[AUTO-EXECUTE] AAPL Bull Put Spread - REAL ORDERS
  Price: $175.50
  Score: 8.50
  Confidence: 72%
  Mode: PAPER TRADING

  Sell Put Strike: $167 (collect premium)
  Buy Put Strike:  $158 (protection)
  Contracts: 1
  Expected Credit: $270.00
  Max Risk: $630.00
  Expiration: 2025-11-14

  [ORDER SYMBOLS]
    Buy Put:  AAPL251114P00158000
    Sell Put: AAPL251114P00167000

  [LEG 1/2] Buying 1 protective put @ $158...
    [OK] Buy order submitted: abc123-def456
    Status: accepted

  [LEG 2/2] Selling 1 put @ $167 for credit...
    [OK] Sell order submitted: xyz789-uvw012
    Status: accepted

  [SUCCESS] Bull Put Spread EXECUTED on Alpaca!
  Position ID: 1
  Orders placed: 2
  These orders will appear on your Alpaca dashboard
  Dashboard: https://app.alpaca.markets/paper/dashboard/overview
```

---

## What You'll See on Alpaca Dashboard

Your Alpaca paper account will show:

1. **Open Orders**: Both legs (BUY PUT and SELL PUT)
2. **Positions**: When filled, you'll see both option positions
3. **Order History**: Complete record of all submitted orders
4. **Account Impact**: Credit/debit reflected in buying power

### Dashboard URL:
https://app.alpaca.markets/paper/dashboard/overview

---

## Code Changes Summary

### Before (Simulated):
```python
# PAPER TRADING SIMULATION: Track as virtual position
execution = {
    'symbol': symbol,
    'status': 'OPEN',
    'paper_trade': self.paper_trading,
}
self.open_positions.append(execution)
```

### After (Real Execution):
```python
# LEG 1: BUY PUT (protection)
buy_order = self.alpaca_api.submit_order(
    symbol=buy_put_symbol,
    qty=num_contracts,
    side='buy',
    type='market',
    time_in_force='day'
)

# LEG 2: SELL PUT (premium)
sell_order = self.alpaca_api.submit_order(
    symbol=sell_put_symbol,
    qty=num_contracts,
    side='sell',
    type='market',
    time_in_force='day'
)

# Track with real order IDs
execution = {
    'alpaca_order_ids': [buy_order.id, sell_order.id],
    'real_execution': True,
}
```

---

## Risk Management

### Built-in Safety Features:

1. **Max Risk Per Trade**: $500 (configurable)
2. **Max Contracts**: 3 per spread (safety limit)
3. **Protection First**: Always buy protection before selling
4. **Max Positions**: 5 concurrent positions
5. **Score Threshold**: 8.0+ for options execution
6. **Rollback**: Automatic cancellation if legs fail

### Strike Selection:
- **Sell Strike**: 5% below current price (95% of stock price)
- **Buy Strike**: 10% below current price (90% of stock price)
- **Spread Width**: Typically $5-10 depending on stock price

### Position Sizing:
```
Max Risk Per Contract = (Spread Width - Expected Credit) × 100
Number of Contracts = Max Risk Per Trade / Max Risk Per Contract
Capped at 3 contracts maximum
```

---

## Testing Recommendations

### Before Going Live:

1. **Test with Small Position**
   - Start with 1 contract
   - Use liquid options (SPY, QQQ, AAPL)
   - Check that orders appear on dashboard

2. **Verify Order Fills**
   - Watch Alpaca dashboard for order status
   - Confirm both legs execute
   - Check that positions are correct

3. **Monitor First Trade**
   - Watch for 30-60 minutes after execution
   - Verify no unexpected behavior
   - Check execution logs

4. **Check Execution Logs**
   - Location: `executions/execution_log_YYYYMMDD.json`
   - Contains all order details and IDs
   - Use for tracking and debugging

---

## Usage Example

```python
from execution.auto_execution_engine import AutoExecutionEngine

# Initialize engine (PAPER TRADING)
engine = AutoExecutionEngine(
    paper_trading=True,  # Uses paper account
    max_risk_per_trade=500  # Max $500 risk per trade
)

# Create opportunity
opportunity = {
    'symbol': 'AAPL',
    'asset_type': 'OPTIONS',
    'strategy': 'BULL_PUT_SPREAD',
    'final_score': 8.5,
    'confidence': 0.72,
    'price': 175.50
}

# Execute REAL order on Alpaca
result = engine.execute_opportunity(opportunity)

if result:
    print(f"SUCCESS! Order IDs: {result['alpaca_order_ids']}")
    print(f"Check dashboard for live positions")
else:
    print("Execution failed - check logs")
```

---

## Important Notes

### This System Will:
- Place REAL orders on Alpaca (paper account)
- Orders will FILL at market prices
- Positions will appear on your dashboard
- Use real buying power

### This System Will NOT:
- Use live money (paper trading only)
- Execute without proper score threshold
- Place more than 3 contracts per trade
- Execute without protection leg first

---

## Troubleshooting

### If Orders Don't Appear:

1. **Check Credentials**
   - Verify `.env` has correct Alpaca keys
   - Ensure using paper API keys (start with "PK")

2. **Check Market Hours**
   - Options only trade during market hours (9:30 AM - 4:00 PM ET)
   - Orders outside hours will be rejected

3. **Check Option Symbols**
   - Verify symbol format is correct (OCC standard)
   - Ensure expiration date is valid (future Friday)
   - Check that strikes exist for the underlying

4. **Check Account Status**
   - Verify account has sufficient buying power
   - Check account is approved for options trading
   - Ensure no trading restrictions

### Common Errors:

1. **"Cannot proceed without protection leg"**
   - Protection (BUY PUT) failed to execute
   - Check option symbol format
   - Verify option contract exists

2. **"Failed to sell put"**
   - Sell leg failed after protection executed
   - Check logs for specific error
   - May need to manually close protection leg

3. **"Alpaca API not available"**
   - Missing `alpaca-trade-api` library
   - Run: `pip install alpaca-trade-api`

---

## Next Steps

1. **Test with Paper Account**
   - Run small test with 1 contract
   - Verify orders appear on dashboard
   - Monitor execution and fills

2. **Monitor Performance**
   - Track win rate and P&L
   - Review execution logs daily
   - Adjust parameters as needed

3. **Scale Gradually**
   - Start with 1 contract per trade
   - Increase to 2-3 after successful tests
   - Never exceed risk limits

4. **Stay Informed**
   - Check dashboard regularly
   - Monitor open positions
   - Close positions before expiration

---

## Summary

**Status**: READY FOR REAL EXECUTION (Paper Trading)

**File Modified**: `C:\Users\lucas\PC-HIVE-TRADING\execution\auto_execution_engine.py`

**Key Feature**: Orders now placed on Alpaca via REST API

**Safety**: Protection leg always executed first

**Dashboard**: https://app.alpaca.markets/paper/dashboard/overview

**Execution Logs**: `executions/execution_log_YYYYMMDD.json`

---

## CRITICAL: This Is Real Execution

Even though it's paper trading, the system now places REAL orders that will EXECUTE at market prices. Orders will appear on your dashboard and will affect your paper account's positions and buying power.

**Test carefully before scaling up!**
