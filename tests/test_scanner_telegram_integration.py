#!/usr/bin/env python3
"""
TEST SCANNER TELEGRAM INTEGRATION
Verify all 4 new scanners work via Telegram commands
"""

from telegram_remote_control import TelegramRemoteControl

def test_all_scanners():
    """Test all scanner integrations"""
    bot = TelegramRemoteControl()

    print("\n" + "="*70)
    print("TESTING SCANNER TELEGRAM INTEGRATION")
    print("="*70)

    # Test 1: Earnings Scanner
    print("\n1. Testing /earnings command...")
    try:
        response = bot.handle_command('/earnings')
        print(f"[OK] Earnings scanner working")
        print(f"     Response length: {len(response)} chars")
    except Exception as e:
        print(f"[FAIL] Earnings scanner failed: {e}")

    # Test 2: Confluence Scanner
    print("\n2. Testing /confluence command...")
    try:
        response = bot.handle_command('/confluence')
        print(f"[OK] Confluence scanner working")
        print(f"     Response length: {len(response)} chars")
    except Exception as e:
        print(f"[FAIL] Confluence scanner failed: {e}")

    # Test 3: Viral Scanner
    print("\n3. Testing /viral command...")
    try:
        response = bot.handle_command('/viral')
        print(f"[OK] Viral scanner working")
        print(f"     Response length: {len(response)} chars")
    except Exception as e:
        print(f"[FAIL] Viral scanner failed: {e}")

    # Test 4: Rebalancer
    print("\n4. Testing /rebalance command...")
    try:
        response = bot.handle_command('/rebalance')
        print(f"[OK] Rebalancer working")
        # Don't print response due to Unicode issue
        print(f"     Response length: {len(response)} chars")
    except Exception as e:
        print(f"[FAIL] Rebalancer failed: {e}")

    # Test 5: Help command (should include new commands)
    print("\n5. Testing /help command...")
    try:
        response = bot.handle_command('/help')

        # Check if new commands are in help
        has_earnings = '/earnings' in response
        has_confluence = '/confluence' in response
        has_viral = '/viral' in response
        has_rebalance = '/rebalance' in response

        if all([has_earnings, has_confluence, has_viral, has_rebalance]):
            print(f"[OK] Help includes all 4 new commands")
        else:
            print(f"[FAIL] Help missing commands:")
            if not has_earnings: print("       - /earnings")
            if not has_confluence: print("       - /confluence")
            if not has_viral: print("       - /viral")
            if not has_rebalance: print("       - /rebalance")
    except Exception as e:
        print(f"[FAIL] Help command failed: {e}")

    print("\n" + "="*70)
    print("INTEGRATION TEST COMPLETE")
    print("="*70)
    print("\nNEXT STEPS:")
    print("1. Start Telegram bot: python telegram_remote_control.py")
    print("2. From your phone, send these commands:")
    print("   /earnings - Upcoming earnings plays")
    print("   /confluence - Multi-timeframe setups")
    print("   /viral - Trending stocks on Reddit")
    print("   /rebalance - Portfolio allocation status")
    print("\n" + "="*70)

if __name__ == '__main__':
    test_all_scanners()
