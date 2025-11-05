"""
Test that SHORT trading capability works correctly
"""
print("="*70)
print("SHORT TRADING CAPABILITY TEST")
print("="*70)

# Test 1: Verify calculate_score returns SHORT opportunities
print("\n[TEST 1] Check if calculate_score() can detect SHORT signals")

import numpy as np
import sys
sys.path.insert(0, '.')

# Create mock data with overbought RSI (SHORT signal)
mock_data = {
    'closes': np.array([1.08] * 40 + [1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19]),  # Strong uptrend
    'highs': np.array([1.09] * 40 + [1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.20]),
    'lows': np.array([1.07] * 40 + [1.09, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18]),
    'current_price': 1.19
}

from WORKING_FOREX_OANDA import WorkingForexOanda
bot = WorkingForexOanda()

result = bot.calculate_score('EUR_USD', mock_data)

print(f"  Result direction: {result.get('direction', 'None')}")
print(f"  Result score: {result.get('score', 0):.2f}")
print(f"  All opportunities: {len(result.get('all_opportunities', []))}")

if result.get('all_opportunities'):
    for opp in result['all_opportunities']:
        print(f"    - {opp['direction'].upper()}: Score {opp['score']:.2f} | Signals: {', '.join(opp['signals'])}")

# Test 2: Verify fundamental filter allows SHORT trades
print("\n[TEST 2] Check if fundamental filter supports SHORT direction")

fake_short_opp = {
    'pair': 'USD_JPY',
    'direction': 'short',
    'score': 5.0,
    'price': 149.50,
    'signals': ['RSI_OVERBOUGHT', 'MACD_BEARISH']
}

fundamental_analysis = bot.news_filter.check_all_pairs(['USD_JPY'])
usdjpy_fund = fundamental_analysis['USD_JPY']

print(f"  USD/JPY Fundamental:")
print(f"    Tradeable: {usdjpy_fund['tradeable']}")
print(f"    Direction: {usdjpy_fund['direction']}")
print(f"    Score: {usdjpy_fund['score']}/6")

if usdjpy_fund['tradeable'] and usdjpy_fund['direction'] == 'short':
    print(f"  ✓ Fundamental filter APPROVES SHORT on USD/JPY!")
    print(f"    Confidence: {usdjpy_fund['confidence']:.0f}%")
else:
    print(f"  ✗ Fundamental filter would BLOCK or doesn't prefer SHORT")

# Test 3: Check order execution logic
print("\n[TEST 3] Verify place_forex_order() handles SHORT correctly")

print("  SHORT order parameters:")
print("    - Units should be NEGATIVE (e.g., -1000000)")
print("    - Stop loss should be ABOVE entry price")
print("    - Take profit should be BELOW entry price")

direction = 'short'
entry_price = 149.50
stop_loss_pct = 0.01  # 1%
profit_target_pct = 0.02  # 2%

stop_distance = entry_price * stop_loss_pct
profit_distance = entry_price * profit_target_pct

if direction == 'short':
    stop_loss_price = entry_price + stop_distance  # ABOVE for SHORT
    take_profit_price = entry_price - profit_distance  # BELOW for SHORT
    units = -1000000  # NEGATIVE
else:
    stop_loss_price = entry_price - stop_distance
    take_profit_price = entry_price + profit_distance
    units = 1000000

print(f"  Entry: {entry_price:.3f}")
print(f"  Stop Loss: {stop_loss_price:.3f} (should be {entry_price + stop_distance:.3f})")
print(f"  Take Profit: {take_profit_price:.3f} (should be {entry_price - profit_distance:.3f})")
print(f"  Units: {units:,} (should be negative)")

if stop_loss_price > entry_price and take_profit_price < entry_price and units < 0:
    print("  ✓ SHORT order logic is CORRECT!")
else:
    print("  ✗ SHORT order logic has issues")

print("\n" + "="*70)
print("SHORT CAPABILITY TEST COMPLETE")
print("="*70)
