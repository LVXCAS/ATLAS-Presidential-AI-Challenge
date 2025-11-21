"""Test what date range E8 actually has data for"""
from datetime import datetime, timedelta
from E8_TRADELOCKER_ADAPTER import E8TradeLockerAdapter

adapter = E8TradeLockerAdapter()

# Try different time ranges to find where data exists
test_ranges = [
    ("Recent - Last 7 days", datetime(2024, 11, 3), datetime(2024, 11, 10)),
    ("1 Month Ago", datetime(2024, 10, 10), datetime(2024, 10, 17)),
    ("3 Months Ago", datetime(2024, 8, 10), datetime(2024, 8, 17)),
    ("6 Months Ago", datetime(2024, 5, 10), datetime(2024, 5, 17)),
    ("1 Year Ago", datetime(2023, 11, 10), datetime(2023, 11, 17)),
]

print("\n" + "="*70)
print("TESTING E8 DATA AVAILABILITY")
print("="*70)

inst_id = adapter._get_instrument_id('EUR_USD')
print(f"EUR_USD instrument ID: {inst_id}")

for label, start, end in test_ranges:
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())

    print(f"\n[{label}]")
    print(f"  Range: {start} to {end}")
    print(f"  Timestamps: {start_ts} to {end_ts}")

    try:
        df = adapter.tl.get_price_history(
            instrument_id=inst_id,
            resolution='1H',
            start_timestamp=start_ts,
            end_timestamp=end_ts
        )

        if not df.empty:
            print(f"  [OK] GOT DATA! {len(df)} candles")
            print(f"     First: {datetime.fromtimestamp(df.iloc[0]['t'])}")
            print(f"     Last: {datetime.fromtimestamp(df.iloc[-1]['t'])}")
            break
        else:
            print(f"  [X] No data")
    except Exception as e:
        print(f"  [X] Error: {e}")

print("\n" + "="*70)
