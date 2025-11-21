"""
DIAGNOSTIC TEST - Find out why NO trades are executing
Tests: Data fetching, Scoring, Execution
"""
import os
from dotenv import load_dotenv
load_dotenv()

print("="*70)
print("DIAGNOSTIC TEST - Why are we not trading?")
print("="*70)

# Test 1: API Keys
print("\n[TEST 1] API Keys")
api_key = os.getenv('ALPACA_API_KEY')
api_secret = os.getenv('ALPACA_SECRET_KEY')
print(f"  API Key: {api_key[:10]}... (length: {len(api_key)})")
print(f"  Secret: {api_secret[:10]}... (length: {len(api_secret)})")

# Test 2: Data Fetching
print("\n[TEST 2] Data Fetching for SPY")
try:
    import alpaca_trade_api as tradeapi
    api = tradeapi.REST(
        key_id=api_key,
        secret_key=api_secret,
        base_url=os.getenv('ALPACA_BASE_URL'),
        api_version='v2'
    )

    # Get account
    account = api.get_account()
    print(f"  Account Status: {account.status}")
    print(f"  Equity: ${float(account.equity):,.2f}")
    print(f"  Buying Power: ${float(account.buying_power):,.2f}")

    # Get SPY data
    from datetime import datetime, timedelta
    end = datetime.now()
    start = end - timedelta(days=5)

    bars = api.get_bars('SPY', '1Day', start=start.isoformat(), end=end.isoformat()).df
    print(f"\n  SPY Data Retrieved: {len(bars)} bars")
    print(f"  Latest Close: ${bars['close'].iloc[-1]:.2f}")
    print(f"  Latest Volume: {bars['volume'].iloc[-1]:,.0f}")

    # Calculate simple momentum
    close_prices = bars['close'].values
    momentum = (close_prices[-1] - close_prices[0]) / close_prices[0] * 100
    print(f"  5-Day Momentum: {momentum:.2f}%")

    # Simple scoring
    score = 0
    if momentum > 0:
        score += 5
    if bars['volume'].iloc[-1] > bars['volume'].mean():
        score += 3

    print(f"\n  CALCULATED SCORE: {score}/10")

except Exception as e:
    print(f"  ERROR: {e}")

# Test 3: Order Placement (dry run)
print("\n[TEST 3] Order Placement Test")
try:
    # Check if we can place orders
    print(f"  Account can trade: {account.trading_blocked == False}")
    print(f"  Pattern day trader: {account.pattern_day_trader}")
    print(f"  Day trade count: {account.daytrade_count}")

    # Try to get current position
    try:
        position = api.get_position('SPY')
        print(f"  Current SPY Position: {position.qty} shares @ ${position.avg_entry_price}")
    except:
        print(f"  No existing SPY position")

    # Simulate order
    print(f"\n  SIMULATION: If score > 5, would place order for SPY")
    if score > 5:
        print(f"  WOULD EXECUTE: BUY 1 SPY @ market")
        print(f"  [PAPER MODE - Not actually placing order]")
    else:
        print(f"  SKIP: Score {score} too low (need > 5)")

except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
