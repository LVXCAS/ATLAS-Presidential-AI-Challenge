# Backtesting Data Directory

This directory contains historical OHLCV data for backtesting forex strategies.

## Files (Git Ignored)

The CSV files are **not tracked in git** because:
- They're large (~3,076 lines each = ~9,228 total lines)
- They can be regenerated from OANDA API
- They bloat repository size

**Expected files:**
- `EUR_USD_H1_6M.csv` - 6 months of hourly EUR/USD data
- `GBP_USD_H1_6M.csv` - 6 months of hourly GBP/USD data
- `USD_JPY_H1_6M.csv` - 6 months of hourly USD/JPY data

## How to Regenerate Data

Run the data fetch script:

```bash
python backtesting/fetch_oanda_data_for_backtest.py
```

**Requirements:**
- OANDA API credentials in `.env` file:
  ```
  OANDA_API_KEY=your_api_key_here
  OANDA_ACCOUNT_ID=your_account_id_here
  ```
- `oandapyV20` library installed: `pip install oandapyV20`
- `pandas` library installed: `pip install pandas`

**What it downloads:**
- 3 currency pairs (EUR/USD, GBP/USD, USD/JPY)
- H1 timeframe (hourly candles)
- Last 6 months of data (~4,320 candles per pair)
- Formatted for Backtrader compatibility

**Output format:**
```csv
datetime,open,high,low,close,volume
2025-05-21 00:00:00,1.08450,1.08520,1.08400,1.08490,1234
2025-05-21 01:00:00,1.08490,1.08550,1.08470,1.08530,1456
...
```

## Usage in Backtests

```python
import pandas as pd
from backtrader.feeds import PandasData

# Load data
df = pd.read_csv('backtesting/data/EUR_USD_H1_6M.csv', parse_dates=['datetime'])
data = PandasData(dataname=df)

# Add to Backtrader cerebro
cerebro.adddata(data, name='EUR_USD')
```

## Data Freshness

The data is a snapshot from when you run the fetch script. For most backtests, 6 months of H1 data provides enough statistical significance.

If you need more recent data or a different timeframe:

1. Edit `fetch_oanda_data_for_backtest.py`
2. Modify the `start_date` / `end_date` range
3. Change `granularity` parameter (H1, H4, D, etc.)
4. Re-run the script

---

**Last fetched:** Run the script to download fresh data
**Total expected candles per pair:** ~4,320 (180 days × 24 hours)
**Total file size:** ~500KB per CSV
