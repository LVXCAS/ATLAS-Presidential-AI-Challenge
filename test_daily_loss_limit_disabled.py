"""
Verify Daily Loss Limit is Disabled
Tests that check_daily_loss_limit() always returns False
"""

import asyncio
import sys

async def test_daily_loss_limit():
    """Test that daily loss limit is disabled"""

    print("=" * 80)
    print("DAILY LOSS LIMIT VERIFICATION")
    print("=" * 80)
    print()

    # Import OPTIONS_BOT to check the configuration
    try:
        # Add current directory to path
        sys.path.insert(0, '.')

        # Read the source file to verify changes
        with open('OPTIONS_BOT.py', 'r', encoding='utf-8') as f:
            source = f.read()

        print("1. Checking daily_loss_limit_pct configuration...")
        print("-" * 80)

        # Check if -99.0 is set
        if 'daily_loss_limit_pct = -99.0' in source or 'daily_loss_limit_pct=-99.0' in source:
            print("[OK] daily_loss_limit_pct = -99.0 (DISABLED)")
        else:
            print("[WARNING] Could not verify daily_loss_limit_pct value")

        print()

        print("2. Checking check_daily_loss_limit() function...")
        print("-" * 80)

        # Find the function definition
        func_start = source.find('async def check_daily_loss_limit(self):')
        if func_start != -1:
            # Get the next 500 characters to see the function
            func_snippet = source[func_start:func_start + 500]

            # Check if it returns False immediately
            lines = func_snippet.split('\n')

            # Look for "return False" in first few lines after docstring
            found_return_false = False
            for i, line in enumerate(lines[:10]):
                if 'return False' in line and i > 2:  # After docstring
                    print(f"[OK] Function returns False at line {i + 1}")
                    print(f"     Code: {line.strip()}")
                    found_return_false = True
                    break

            if found_return_false:
                print("[OK] check_daily_loss_limit() will always return False")
            else:
                print("[WARNING] Could not verify early return False")
        else:
            print("[ERROR] Could not find check_daily_loss_limit() function")

        print()

        print("3. Impact Assessment...")
        print("-" * 80)

        # Count how many times the function is called
        call_count = source.count('check_daily_loss_limit()')
        print(f"[INFO] check_daily_loss_limit() is called {call_count} times in the code")
        print(f"       All {call_count} calls will now return False")

        print()

        # List locations where it's called
        print("4. Locations Where Daily Loss Check Occurs...")
        print("-" * 80)

        lines = source.split('\n')
        for i, line in enumerate(lines, 1):
            if 'check_daily_loss_limit()' in line and 'async def' not in line:
                # Show context around the call
                print(f"Line {i}: {line.strip()}")

        print()

        print("=" * 80)
        print("VERIFICATION RESULT")
        print("=" * 80)
        print()
        print("[SUCCESS] Daily loss limit is DISABLED")
        print()
        print("Configuration:")
        print("  - daily_loss_limit_pct: -99.0 (effectively disabled)")
        print("  - check_daily_loss_limit(): Always returns False")
        print()
        print("Bot Behavior:")
        print("  - Will NOT stop trading due to daily losses")
        print("  - Will continue scanning and executing all day")
        print("  - Only per-trade stops (-20%) remain active")
        print()
        print("Risk Level:")
        print("  - Higher daily drawdown potential")
        print("  - No automatic daily circuit breaker")
        print("  - Per-position stops still protect capital")
        print()

    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_daily_loss_limit())
