#!/usr/bin/env python3
"""
TEST ACTUAL FUNCTIONALITY
Not just "does it return a response" but "does it actually work"
"""

from telegram_remote_control import TelegramRemoteControl
import traceback

def test_command_functionality(cmd, expected_keywords):
    """Test if command returns expected content"""
    bot = TelegramRemoteControl()

    try:
        print(f"\nTesting: {cmd}")
        response = bot.handle_command(cmd)

        # Check for error indicators
        has_error = any(err in response.lower() for err in ['error', 'failed', 'exception', 'traceback'])
        has_expected = any(kw.lower() in response.lower() for kw in expected_keywords)

        print(f"  Response length: {len(response)} chars")
        print(f"  Has errors: {has_error}")
        print(f"  Has expected content: {has_expected}")

        if has_error:
            print(f"  ERROR CONTENT:")
            print(f"  {response[:500]}")
            return False

        if not has_expected:
            print(f"  MISSING EXPECTED KEYWORDS: {expected_keywords}")
            print(f"  RESPONSE: {response[:300]}")
            return False

        print(f"  [WORKS]")
        return True

    except Exception as e:
        print(f"  [EXCEPTION] {str(e)}")
        traceback.print_exc()
        return False

def main():
    print("="*70)
    print("TESTING ACTUAL COMMAND FUNCTIONALITY")
    print("="*70)

    tests = {
        '/status': ['SYSTEM STATUS', 'Forex', 'Options', 'Futures'],
        '/positions': ['POSITIONS', 'forex', 'positions'],
        '/regime': ['MARKET REGIME', 'Fear', 'Greed'],
        '/pnl': ['UNIFIED P&L', 'Balance', 'Total'],
        '/risk': ['RISK', 'limit', 'switch'],
        '/pipeline': ['PIPELINE', 'Discovered', 'Paper', 'Live'],
        '/rebalance': ['ALLOCATION', 'Forex', 'Futures', 'Options', 'Target'],
        '/earnings': ['EARNINGS', 'found', 'Criteria'],
        '/confluence': ['CONFLUENCE', 'found', 'align'],
        '/viral': ['VIRAL', 'found', 'stocks'],
        '/regime status': ['REGIME', 'switching', 'regime'],
        '/regime manual': ['REGIME', 'DISABLED', 'manual'],
        '/regime auto': ['REGIME', 'ENABLED', 'auto'],
        '/help': ['COMMANDS', 'STATUS', 'SCANNERS', 'EMERGENCY']
    }

    results = {}

    for cmd, keywords in tests.items():
        results[cmd] = test_command_functionality(cmd, keywords)

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    working = [cmd for cmd, result in results.items() if result]
    broken = [cmd for cmd, result in results.items() if not result]

    print(f"\nWORKING: {len(working)}/{len(tests)}")
    for cmd in working:
        print(f"  [OK] {cmd}")

    print(f"\nBROKEN: {len(broken)}/{len(tests)}")
    for cmd in broken:
        print(f"  [FAIL] {cmd}")

    return broken

if __name__ == '__main__':
    broken = main()

    if broken:
        print("\n" + "="*70)
        print("BROKEN COMMANDS NEED FIXING:")
        print("="*70)
        for cmd in broken:
            print(f"  - {cmd}")
    else:
        print("\n" + "="*70)
        print("ALL COMMANDS WORKING!")
        print("="*70)
