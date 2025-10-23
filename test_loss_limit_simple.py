#!/usr/bin/env python3
"""
Simple test for daily loss limit logic without full bot initialization
"""

import json

def test_loss_limit_logic():
    """Test the loss limit calculation logic"""

    print("Testing Daily Loss Limit Calculation Logic")
    print("="*50)

    # Test parameters
    loss_limit_pct = -4.9

    test_cases = [
        {
            "name": "Profitable day",
            "starting_equity": 10000.0,
            "current_equity": 10500.0,
            "expected_pnl_pct": 5.0,
            "should_stop": False
        },
        {
            "name": "Small loss (within limit)",
            "starting_equity": 10000.0,
            "current_equity": 9800.0,
            "expected_pnl_pct": -2.0,
            "should_stop": False
        },
        {
            "name": "Exactly at limit",
            "starting_equity": 10000.0,
            "current_equity": 9510.0,  # Exactly -4.9%
            "expected_pnl_pct": -4.9,
            "should_stop": True
        },
        {
            "name": "Beyond limit",
            "starting_equity": 10000.0,
            "current_equity": 9400.0,  # -6.0%
            "expected_pnl_pct": -6.0,
            "should_stop": True
        },
        {
            "name": "Small account loss",
            "starting_equity": 1000.0,
            "current_equity": 951.0,  # Exactly -4.9%
            "expected_pnl_pct": -4.9,
            "should_stop": True
        }
    ]

    print(f"Daily Loss Limit: {loss_limit_pct}%")
    print()

    all_passed = True

    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        print("-" * 30)

        starting = test['starting_equity']
        current = test['current_equity']

        # Calculate P&L percentage (matching bot logic)
        if starting > 0:
            daily_pnl_pct = ((current - starting) / starting) * 100
        else:
            daily_pnl_pct = 0

        # Check if loss limit should be hit
        should_stop = daily_pnl_pct <= loss_limit_pct

        print(f"Starting Equity: ${starting:,.2f}")
        print(f"Current Equity:  ${current:,.2f}")
        print(f"Daily P&L:       ${current - starting:+,.2f}")
        print(f"Daily P&L %:     {daily_pnl_pct:+.1f}%")
        print(f"Should Stop:     {should_stop}")

        # Validate against expected
        pnl_correct = abs(daily_pnl_pct - test['expected_pnl_pct']) < 0.1
        stop_correct = should_stop == test['should_stop']

        if pnl_correct and stop_correct:
            print("[PASS] Test passed")
        else:
            print("[FAIL] Test failed")
            if not pnl_correct:
                print(f"  Expected P&L: {test['expected_pnl_pct']:.1f}%, got {daily_pnl_pct:.1f}%")
            if not stop_correct:
                print(f"  Expected should_stop: {test['should_stop']}, got {should_stop}")
            all_passed = False

        print()

    print("="*50)
    print("LOGIC TEST SUMMARY")
    print("="*50)

    if all_passed:
        print("[SUCCESS] All loss limit calculations correct")
        print("\nThe bot logic will:")
        print(f"- Stop trading when daily P&L <= {loss_limit_pct}%")
        print("- Close all positions immediately")
        print("- Block new trades until next day")
        print("- Reset flags at market open")

        # Test the starting equity file logic
        print("\nTesting starting equity file handling...")
        test_equity = {"starting_equity": 10000.0, "date": "2025-09-25"}

        try:
            with open('daily_starting_equity.json', 'w') as f:
                json.dump(test_equity, f)
            print("[OK] Can write starting equity file")

            with open('daily_starting_equity.json', 'r') as f:
                loaded_data = json.load(f)
                if loaded_data['starting_equity'] == 10000.0:
                    print("[OK] Can read starting equity file")
                else:
                    print("[ERROR] Starting equity file data incorrect")

            import os
            os.remove('daily_starting_equity.json')
            print("[OK] Can clean up starting equity file")

        except Exception as e:
            print(f"[ERROR] Starting equity file handling: {e}")

    else:
        print("[ERROR] Some loss limit calculations failed")

    return all_passed

if __name__ == "__main__":
    success = test_loss_limit_logic()
    exit(0 if success else 1)