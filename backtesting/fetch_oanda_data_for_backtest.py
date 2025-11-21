"""
FETCH OANDA DATA FOR BACKTESTING

Download 6 months of H1 forex data from OANDA for backtest comparison.

Pairs: EUR/USD, GBP/USD, USD/JPY
Timeframe: H1 (hourly)
Period: Last 6 months
Format: CSV compatible with Backtrader
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import OANDA client directly
try:
    import oandapyV20
    from oandapyV20 import API
    from oandapyV20.endpoints.instruments import InstrumentsCandles
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False
    print("[ERROR] oandapyV20 not installed: pip install oandapyV20")


def fetch_forex_data(pair, start_date, end_date, granularity='H1'):
    """
    Fetch historical forex data from OANDA

    Args:
        pair: Currency pair (EUR_USD, GBP_USD, USD_JPY)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        granularity: Timeframe (H1 = hourly)

    Returns:
        DataFrame with OHLCV data
    """

    if not OANDA_AVAILABLE:
        print(f"[ERROR] OANDA client not available")
        return None

    print(f"\n[FETCHING] {pair} from {start_date} to {end_date}")

    try:
        # Get OANDA credentials
        account_id = os.getenv('OANDA_ACCOUNT_ID')
        api_key = os.getenv('OANDA_API_KEY')

        if not account_id or not api_key:
            print(f"  [ERROR] OANDA credentials not found in .env")
            return None

        # Initialize OANDA API client
        client = API(access_token=api_key)

        # Format pair for OANDA (EUR_USD format)
        instrument = pair

        # Prepare request parameters
        params = {
            "granularity": granularity,
            "count": 5000,  # Max candles per request
        }

        print(f"  Requesting candles from OANDA...")

        # Create request
        request = InstrumentsCandles(instrument=instrument, params=params)

        # Execute request
        response = client.request(request)

        if 'candles' not in response:
            print(f"  [ERROR] No candles in response")
            return None

        candles = response['candles']
        print(f"  Retrieved {len(candles)} candles")

        # Parse candles into list of dicts
        data = []
        for candle in candles:
            if candle['complete']:  # Only use complete candles
                data.append({
                    'time': candle['time'],
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume'])
                })

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'])

        # Filter to date range
        df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]

        # Sort by time
        df = df.sort_values('time')

        # Reset index
        df = df.reset_index(drop=True)

        print(f"  Filtered to {len(df)} candles in date range")

        return df

    except Exception as e:
        print(f"  [ERROR] Failed to fetch data: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_to_csv(df, pair, output_dir='backtesting/data'):
    """Save DataFrame to CSV format compatible with Backtrader"""

    if df is None or df.empty:
        print(f"[ERROR] Cannot save empty DataFrame")
        return False

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare data for Backtrader
    # Required columns: datetime, open, high, low, close, volume
    bt_df = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
    bt_df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

    # Save to CSV
    filename = f"{pair}_H1_6M.csv"
    filepath = output_path / filename

    bt_df.to_csv(filepath, index=False)

    print(f"  [SAVED] {filepath} ({len(bt_df)} candles)")

    return True


def fetch_all_pairs():
    """Fetch data for all forex pairs needed for backtest"""

    print("=" * 80)
    print("FETCHING OANDA DATA FOR BACKTEST")
    print("=" * 80)

    # Define date range (last 6 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # ~6 months

    print(f"\nDate range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Timeframe: H1 (hourly)")
    print(f"Expected candles per pair: ~4,320")

    # Pairs to fetch
    pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']

    success_count = 0

    for pair in pairs:
        df = fetch_forex_data(
            pair,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            granularity='H1'
        )

        if df is not None and not df.empty:
            if save_to_csv(df, pair):
                success_count += 1

    print(f"\n{'='*80}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*80}")
    print(f"  Successfully downloaded: {success_count}/{len(pairs)} pairs")

    if success_count == len(pairs):
        print(f"\n  [SUCCESS] All data ready for backtest")
        print(f"\n  Next step: Run backtest comparison")
        print(f"    python backtesting/BACKTEST_SAFE_VS_AGGRESSIVE.py")
    else:
        print(f"\n  [WARNING] Some downloads failed - check OANDA credentials")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    print("=" * 80)
    print("OANDA DATA FETCH FOR BACKTESTING")
    print("=" * 80)
    print()
    print("Downloading:")
    print("  - EUR/USD, GBP/USD, USD/JPY")
    print("  - H1 timeframe (hourly candles)")
    print("  - Last 6 months")
    print("  - ~4,320 candles per pair")
    print()
    print("Output: backtesting/data/[PAIR]_H1_6M.csv")
    print("=" * 80)

    fetch_all_pairs()
