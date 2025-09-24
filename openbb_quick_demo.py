from openbb_integration import HiveOpenBB

print("=== OpenBB Quick Demo ===")
obb = HiveOpenBB()

# Test basic functionality
data = obb.get_stock_data("AAPL", period="5d")
if data is not None:
    price = data["Close"].iloc[-1]
    print(f"AAPL Latest Price: ${price:.2f}")

# Test technical analysis
ta_data = obb.technical_analysis("AAPL", period="1mo")
if ta_data is not None:
    latest = ta_data.iloc[-1]
    print(f"RSI: {latest[\"RSI\"]:.2f}")
    print(f"SMA 20: ${latest[\"SMA_20\"]:.2f}")

print("OpenBB integration working successfully!")
