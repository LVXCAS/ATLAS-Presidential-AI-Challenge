#!/usr/bin/env python3
"""
COMPREHENSIVE TELEGRAM COMMAND TEST
Test ALL commands to ensure complete system functionality
"""

from telegram_remote_control import TelegramRemoteControl
import time

def test_all_commands():
    """Test every single Telegram command"""
    bot = TelegramRemoteControl()

    print("\n" + "="*70)
    print("COMPREHENSIVE TELEGRAM COMMAND TEST")
    print("="*70)

    results = {}

    # Category 1: STATUS COMMANDS
    print("\n[CATEGORY 1] STATUS COMMANDS")
    print("-" * 70)

    commands_status = [
        '/status',
        '/positions',
        '/regime',
        '/pnl',
        '/risk',
        '/pipeline',
        '/rebalance'
    ]

    for cmd in commands_status:
        try:
            print(f"\nTesting: {cmd}")
            response = bot.handle_command(cmd)
            success = len(response) > 0 and 'Error' not in response[:100]
            results[cmd] = 'PASS' if success else 'FAIL'
            print(f"  Result: {results[cmd]}")
            print(f"  Response length: {len(response)} chars")
            if not success and 'Error' in response:
                print(f"  Error preview: {response[:200]}")
            time.sleep(0.5)
        except Exception as e:
            results[cmd] = 'FAIL'
            print(f"  Result: FAIL")
            print(f"  Exception: {str(e)[:100]}")

    # Category 2: SCANNER COMMANDS
    print("\n[CATEGORY 2] SCANNER COMMANDS")
    print("-" * 70)

    commands_scanners = [
        '/earnings',
        '/confluence',
        '/viral'
    ]

    for cmd in commands_scanners:
        try:
            print(f"\nTesting: {cmd}")
            response = bot.handle_command(cmd)
            success = len(response) > 0
            results[cmd] = 'PASS' if success else 'FAIL'
            print(f"  Result: {results[cmd]}")
            print(f"  Response length: {len(response)} chars")
            time.sleep(0.5)
        except Exception as e:
            results[cmd] = 'FAIL'
            print(f"  Result: FAIL")
            print(f"  Exception: {str(e)[:100]}")

    # Category 3: REGIME AUTO-SWITCHER
    print("\n[CATEGORY 3] REGIME AUTO-SWITCHER COMMANDS")
    print("-" * 70)

    commands_regime = [
        '/regime status',
        '/regime manual',  # Disable first
        '/regime auto'     # Then enable
    ]

    for cmd in commands_regime:
        try:
            print(f"\nTesting: {cmd}")
            response = bot.handle_command(cmd)
            success = len(response) > 0
            results[cmd] = 'PASS' if success else 'FAIL'
            print(f"  Result: {results[cmd]}")
            print(f"  Response length: {len(response)} chars")
            time.sleep(0.5)
        except Exception as e:
            results[cmd] = 'FAIL'
            print(f"  Result: FAIL")
            print(f"  Exception: {str(e)[:100]}")

    # Category 4: STRATEGY DEPLOYMENT (non-destructive tests only)
    print("\n[CATEGORY 4] STRATEGY DEPLOYMENT COMMANDS")
    print("-" * 70)

    # Only test /pipeline (read-only), skip /run_pipeline and /deploy
    try:
        cmd = '/pipeline'
        print(f"\nTesting: {cmd}")
        response = bot.handle_command(cmd)
        success = len(response) > 0
        results[cmd] = 'PASS' if success else 'FAIL'
        print(f"  Result: {results[cmd]}")
        print(f"  Response length: {len(response)} chars")
    except Exception as e:
        results[cmd] = 'FAIL'
        print(f"  Result: FAIL")
        print(f"  Exception: {str(e)[:100]}")

    print(f"\n  Skipping /run_pipeline (launches background process)")
    print(f"  Skipping /deploy (requires strategy name)")
    results['/run_pipeline'] = 'SKIP'
    results['/deploy'] = 'SKIP'

    # Category 5: REMOTE START (skip - would actually start systems)
    print("\n[CATEGORY 5] REMOTE START COMMANDS")
    print("-" * 70)

    start_commands = ['/start_forex', '/start_futures', '/start_options', '/restart_all']
    for cmd in start_commands:
        print(f"  Skipping {cmd} (would start live systems)")
        results[cmd] = 'SKIP'

    # Category 6: RISK MANAGEMENT
    print("\n[CATEGORY 6] RISK MANAGEMENT COMMANDS")
    print("-" * 70)

    # Test /risk (already tested in status)
    # Skip /risk override (would reset kill-switch)
    print(f"  /risk already tested in STATUS")
    print(f"  Skipping /risk override (would reset kill-switch)")
    results['/risk override'] = 'SKIP'

    # Category 7: EMERGENCY COMMANDS
    print("\n[CATEGORY 7] EMERGENCY COMMANDS")
    print("-" * 70)

    print(f"  Skipping /stop (would stop all trading)")
    print(f"  Skipping /kill_all (NUCLEAR OPTION)")
    results['/stop'] = 'SKIP'
    results['/kill_all'] = 'SKIP'

    # Category 8: HELP
    print("\n[CATEGORY 8] HELP COMMAND")
    print("-" * 70)

    try:
        cmd = '/help'
        print(f"\nTesting: {cmd}")
        response = bot.handle_command(cmd)

        # Verify help includes all sections
        has_status = 'STATUS:' in response
        has_scanners = 'SCANNERS:' in response
        has_deployment = 'DEPLOYMENT' in response
        has_regime = 'REGIME' in response
        has_emergency = 'EMERGENCY:' in response

        success = all([has_status, has_scanners, has_deployment, has_regime, has_emergency])
        results[cmd] = 'PASS' if success else 'FAIL'

        print(f"  Result: {results[cmd]}")
        print(f"  Response length: {len(response)} chars")
        print(f"  Sections found:")
        print(f"    - STATUS: {has_status}")
        print(f"    - SCANNERS: {has_scanners}")
        print(f"    - DEPLOYMENT: {has_deployment}")
        print(f"    - REGIME: {has_regime}")
        print(f"    - EMERGENCY: {has_emergency}")
    except Exception as e:
        results[cmd] = 'FAIL'
        print(f"  Result: FAIL")
        print(f"  Exception: {str(e)[:100]}")

    # Test unknown command
    print("\n[CATEGORY 9] UNKNOWN COMMAND HANDLING")
    print("-" * 70)

    try:
        cmd = '/invalid_command'
        print(f"\nTesting: {cmd}")
        response = bot.handle_command(cmd)
        success = 'Unknown' in response and '/help' in response
        results['unknown_cmd'] = 'PASS' if success else 'FAIL'
        print(f"  Result: {results['unknown_cmd']}")
        print(f"  Response: {response[:100]}")
    except Exception as e:
        results['unknown_cmd'] = 'FAIL'
        print(f"  Result: FAIL")
        print(f"  Exception: {str(e)[:100]}")

    # SUMMARY
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v == 'PASS')
    failed = sum(1 for v in results.values() if v == 'FAIL')
    skipped = sum(1 for v in results.values() if v == 'SKIP')
    total = len(results)

    print(f"\nTotal Commands: {total}")
    print(f"  PASSED: {passed}")
    print(f"  FAILED: {failed}")
    print(f"  SKIPPED: {skipped} (destructive/live operations)")

    print("\n" + "-"*70)
    print("DETAILED RESULTS:")
    print("-"*70)

    for cmd, result in sorted(results.items()):
        status_symbol = {
            'PASS': '[PASS]',
            'FAIL': '[FAIL]',
            'SKIP': '[SKIP]'
        }[result]
        print(f"  {status_symbol} {cmd}")

    print("\n" + "="*70)

    if failed == 0:
        print("ALL TESTED COMMANDS WORKING!")
        print("="*70)
        return True
    else:
        print(f"WARNING: {failed} COMMANDS FAILED")
        print("="*70)
        return False

if __name__ == '__main__':
    success = test_all_commands()

    if success:
        print("\nSYSTEM READY FOR PRODUCTION")
        print("Start bot: python telegram_remote_control.py")
    else:
        print("\nFIX FAILURES BEFORE PRODUCTION USE")
