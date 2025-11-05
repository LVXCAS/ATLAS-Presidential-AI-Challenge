"""
Force an immediate scan to see what bot would do RIGHT NOW
This simulates exactly what the running bot should be doing
"""
import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.getcwd())

# Import the actual trading bot
from WORKING_FOREX_OANDA import WorkingForexOanda

print("="*70)
print("FORCING IMMEDIATE SCAN - EXACT BOT BEHAVIOR")
print("="*70)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Create bot instance
bot = WorkingForexOanda()

print(f"Bot Configuration:")
print(f"  Pairs: {bot.forex_pairs}")
print(f"  Min Score: {bot.min_score}/10")
print(f"  Max Positions: {bot.max_positions}")
print(f"  10x Leverage: {bot.use_10x_leverage}")
print()

# Check current positions
current_positions = bot.get_current_positions()
print(f"Current Positions: {len(current_positions)}")

# Scan all pairs
print("\n" + "="*70)
print("SCANNING ALL PAIRS NOW...")
print("="*70)

all_opportunities = []

for pair in bot.forex_pairs:
    print(f"\n[SCANNING {pair}]")

    # Get data
    data = bot.get_forex_data(pair)
    if not data:
        print(f"  [ERROR] Could not get data")
        continue

    # Calculate score
    result = bot.calculate_score(pair, data)

    if result:
        # Check if it's a list (multiple opportunities) or single dict
        opportunities = result if isinstance(result, list) else [result]

        for opp in opportunities:
            print(f"  Direction: {opp.get('direction', 'N/A').upper()}")
            print(f"  Score: {opp.get('score', 0):.1f}/10")
            print(f"  Signals: {', '.join(opp.get('signals', []))}")

            if opp.get('score', 0) >= bot.min_score:
                print(f"  [PASS] Above {bot.min_score}/10 threshold")
                all_opportunities.append(opp)
            else:
                print(f"  [SKIP] Below {bot.min_score}/10 threshold")
    else:
        print(f"  No opportunities found")

print("\n" + "="*70)
print(f"OPPORTUNITIES FOUND: {len(all_opportunities)}")
print("="*70)

if all_opportunities:
    for opp in all_opportunities:
        print(f"\n{opp['pair']} {opp['direction'].upper()}: {opp['score']:.1f}/10")
        print(f"  Signals: {', '.join(opp['signals'])}")

# Now check fundamentals
if all_opportunities and bot.news_filter:
    print("\n" + "="*70)
    print("CHECKING FUNDAMENTAL FILTER...")
    print("="*70)

    fund_results = bot.news_filter.check_all_pairs(bot.forex_pairs)

    for opp in all_opportunities:
        pair = opp['pair']
        direction = opp['direction']

        print(f"\n{pair} {direction.upper()}:")

        if pair in fund_results:
            fund = fund_results[pair]
            print(f"  Fundamental Score: {fund.get('score', 0)}/6")
            print(f"  Tradeable: {fund.get('tradeable', False)}")
            print(f"  Fund Direction: {fund.get('direction', 'N/A')}")
            print(f"  Confidence: {fund.get('confidence', 0):.0f}%")

            if not fund['tradeable']:
                print(f"  [BLOCKED] Fundamentals unclear")
            elif fund['direction'] and fund['direction'] != direction:
                print(f"  [BLOCKED] Direction mismatch (want {fund['direction'].upper()}, got {direction.upper()})")
            else:
                print(f"  [APPROVED] Fundamentals align or neutral")
                print(f"  >>> THIS SHOULD TRADE! <<<")

# Check safety
if bot.news_filter:
    print("\n" + "="*70)
    print("CHECKING SAFETY/NEWS FILTER...")
    print("="*70)

    safety = bot.news_filter.is_safe_to_trade_v2()
    print(f"Safe to trade: {safety['safe']}")
    print(f"Reason: {safety['reason']}")

print("\n" + "="*70)
print("SCAN COMPLETE")
print("="*70)
print()

if all_opportunities:
    print(f"Found {len(all_opportunities)} technical opportunities")
    print(f"If none traded, check fundamental alignment above")
else:
    print("No technical opportunities met the threshold")
    print("This is normal for Asian session (low liquidity)")
