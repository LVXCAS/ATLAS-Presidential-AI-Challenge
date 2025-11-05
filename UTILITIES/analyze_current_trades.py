"""
Analyze current open trades to decide what to do
"""
import sys
sys.path.insert(0, '.')
from WORKING_FOREX_OANDA import WorkingForexOanda

bot = WorkingForexOanda()

print('='*70)
print('REAL-TIME TRADE ANALYSIS')
print('='*70)

# Get current technical signals
opportunities = bot.scan_forex()

print('\nCURRENT TECHNICAL SIGNALS:')
for opp in opportunities:
    if opp['pair'] in ['EUR_USD', 'GBP_USD']:
        print(f"\n{opp['pair']}:")
        print(f"  Score: {opp['score']:.1f}/10")
        print(f"  Signals: {', '.join(opp['signals'])}")

# Check positions
positions = bot.get_current_positions()
print('\n' + '='*70)
print('CURRENT POSITIONS & P/L:')
print('='*70)

total_pl = 0
for pos in positions:
    inst = pos['instrument']
    if inst in ['EUR_USD', 'GBP_USD']:
        long_units = int(pos['long']['units']) if float(pos['long']['units']) > 0 else 0
        pl = float(pos['long']['unrealizedPL']) if long_units > 0 else float(pos['short']['unrealizedPL'])
        total_pl += pl

        print(f'\n{inst}:')
        print(f"  Units: {long_units:,}")
        print(f"  P/L: ${pl:.2f}")
        print(f"  Pips down: {abs(pl)/100:.1f} pips (each pip = $100 with 1M units)")

print(f"\nTOTAL UNREALIZED P/L: ${total_pl:.2f}")

# Get fundamental scores
print('\n' + '='*70)
print('FUNDAMENTAL ANALYSIS SUMMARY:')
print('='*70)

fund_analysis = bot.news_filter.check_all_pairs(['EUR_USD', 'GBP_USD'])

for pair in ['EUR_USD', 'GBP_USD']:
    data = fund_analysis[pair]
    print(f"\n{pair}:")
    print(f"  Fundamental Score: {data['score']}/6")
    print(f"  Tradeable: {data['tradeable']} (need score >= 3 or <= -3)")
    print(f"  Reasons: {', '.join(data['reasons'])}")

# Final recommendation
print('\n' + '='*70)
print('VERDICT & OPTIONS:')
print('='*70)

print('\nEUR/USD:')
print('  Technical: 3.5/10 (WEAK - barely cleared 2.5 threshold)')
print('  Fundamental: 2/6 (BELOW threshold - should NOT trade)')
print('  Status: WEAK TRADE - cut losses recommended')

print('\nGBP/USD:')
print('  Technical: 6.0/10 (STRONG - RSI oversold, MACD bullish, ADX 55)')
print('  Fundamental: 2/6 (BELOW threshold - should NOT trade)')
print('  Status: MIXED - strong technical BUT weak fundamental')

print('\n' + '='*70)
print('THREE OPTIONS:')
print('='*70)

print('\n[1] CLOSE BOTH NOW (Recommended)')
print('    Loss: ~$920')
print('    Why: Neither passes updated fundamental filter')
print('    Benefit: Clean slate, bot restarts with proper filters')
print('    Risk: None (cut losses now)')

print('\n[2] CLOSE EUR/USD ONLY, KEEP GBP/USD')
print('    Loss: ~$350 (EUR)')
print('    Risk: Up to $13,200 if GBP stops out')
print('    Why: GBP has strong technicals (6/10 score)')
print('    Benefit: Keep best technical setup')

print('\n[3] LET BOTH RUN TO STOPS/TARGETS')
print(f'    Current loss: ${total_pl:.2f}')
print('    Max loss: $24,800 (both stop out)')
print('    Max gain: $49,600 (both hit targets)')
print('    Why: RSI oversold can mean bounce coming')
print('    Risk: HIGH - trading against fundamentals')

print('\n' + '='*70)
print('MOST CONSERVATIVE: Option 1 (close both)')
print('BALANCED: Option 2 (close EUR, keep GBP)')
print('AGGRESSIVE: Option 3 (hold both)')
print('='*70)
