# Cached Historical Data (Read-Only)

This folder stores historical market data used for offline, reproducible demos.
Live APIs are intentionally disabled in this repository.

## Folder layout
```
data/
  equities/
  fx/
```

## CSV schema (required)
- date
- open
- high
- low
- close
- volume

Timestamps should be ISO 8601 (UTC recommended). The loader normalizes and
sorts timestamps so outputs are deterministic.

## Naming
Lowercase filenames with underscores are recommended:
- `data/fx/eur_usd.csv`
- `data/equities/spy.csv`

Uppercase or compact names are also accepted by the loader:
- `data/fx/EURUSD.csv`
- `data/fx/GBPUSD.csv`
- `data/fx/USDJPY.csv`
- `data/equities/AAPL.csv`
- `data/equities/SPY.csv`
- `data/equities/MSFT.csv`
- `data/equities/TSLA.csv`

If a cached file is missing (or pandas is unavailable), ATLAS falls back to
synthetic data with a warning so the demo still runs offline.

Note: sample CSVs in this repo are synthetic placeholders for format only.
Replace them with real historical data from public sources if desired.
