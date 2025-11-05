"""
Test why the bot isn't executing trades
"""
import sys
sys.path.insert(0, '.')
from WORKING_FOREX_OANDA import WorkingForexOanda

bot = WorkingForexOanda()
print('\n' + '='*70)
print('SIMULATING FULL TRADE CYCLE')
print('='*70)

# Step 1: Get balance
balance = bot.get_account_balance()
print(f'\n[1] Balance: ${balance:,.2f}')

# Step 2: Scan
print(f'\n[2] Scanning for opportunities...')
opportunities = bot.scan_forex()
print(f'    Found {len(opportunities)} opportunities above min_score {bot.min_score}')

if opportunities:
    for opp in opportunities:
        print(f"      - {opp['pair']}: Score {opp['score']:.2f}")

    # Step 3: Check which pairs we already hold
    print(f'\n[3] Checking existing positions...')
    current_positions = bot.get_current_positions()
    current_pairs = [p['instrument'] for p in current_positions]
    print(f'    Current pairs: {current_pairs}')

    # Step 4: Execute trades (this will show news filter in action)
    print(f'\n[4] Attempting to execute trades with news filter...')
    trades_executed = bot.execute_trades(opportunities)
    print(f'\n[5] RESULT: {trades_executed} trades executed')

    if trades_executed == 0:
        print('\n>>> ALL TRADES WERE BLOCKED <<<')
        print('Most likely reason: NEWS FILTER blocking trades')
        print('Check above for [BLOCKED] messages from news filter')
else:
    print('    No opportunities found (all below min_score threshold)')

print('\n' + '='*70)
