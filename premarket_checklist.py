"""
Pre-Market Trading Checklist

Interactive checklist to ensure readiness for trading
"""

import sys
from datetime import datetime

def print_header():
    print("=" * 70)
    print("HIVE TRADING - PRE-MARKET CHECKLIST")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def ask_confirmation(question):
    """Ask for user confirmation"""
    while True:
        response = input(f"{question} (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no")

def run_checklist():
    """Run the pre-market checklist"""

    print_header()

    checklist_items = [
        "System validation has passed",
        "Broker API connections are working",
        "Account balances have been verified",
        "Risk limits are properly configured",
        "Market data feeds are operational",
        "Strategy parameters are set correctly",
        "Paper trading has been tested",
        "Monitoring systems are active",
        "Emergency procedures are understood",
        "Position sizing rules are in place"
    ]

    completed_items = 0
    failed_items = []

    print("Please confirm each item has been completed:\n")

    for i, item in enumerate(checklist_items, 1):
        if ask_confirmation(f"{i:2d}. {item}"):
            print("    [OK] Confirmed\n")
            completed_items += 1
        else:
            print("    [PENDING] Not completed\n")
            failed_items.append(item)

    # Summary
    print("=" * 70)
    print("CHECKLIST SUMMARY")
    print("=" * 70)

    completion_rate = (completed_items / len(checklist_items)) * 100
    print(f"Completed: {completed_items}/{len(checklist_items)} ({completion_rate:.1f}%)")

    if completion_rate == 100:
        print("\nSTATUS: [READY] All items completed - Ready for trading!")
        print("\nYou may proceed with live trading operations.")
    elif completion_rate >= 80:
        print("\nSTATUS: [MOSTLY READY] Most items completed - Review pending items")
        print("\nPending items:")
        for item in failed_items:
            print(f"  - {item}")
        print("\nConsider addressing these before live trading.")
    else:
        print("\nSTATUS: [NOT READY] Several items need attention")
        print("\nPending items:")
        for item in failed_items:
            print(f"  - {item}")
        print("\nComplete these items before proceeding with live trading.")

    print("\n" + "=" * 70)

    return completion_rate == 100

if __name__ == "__main__":
    ready = run_checklist()
    sys.exit(0 if ready else 1)
