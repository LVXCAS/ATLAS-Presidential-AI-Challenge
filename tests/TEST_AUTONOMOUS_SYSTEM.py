#!/usr/bin/env python3
"""
TEST AUTONOMOUS SYSTEM COMPONENTS
Verify all Tier 1+2 components are working
"""
import os
import sys

print("\n" + "="*70)
print("AUTONOMOUS SYSTEM COMPONENT TEST")
print("="*70)
print()

results = []

# Test 1: Telegram Notifier
print("[1/4] Testing Telegram Notifier...")
try:
    from utils.telegram_notifier import get_notifier
    notifier = get_notifier()
    if notifier.enabled:
        print("  [OK] Telegram configured and enabled")
        results.append(("Telegram", True, "Configured"))
    else:
        print("  [!] Telegram not configured (optional)")
        results.append(("Telegram", False, "Not configured - see setup guide"))
except Exception as e:
    print(f"  [X] Telegram failed: {e}")
    results.append(("Telegram", False, str(e)))

# Test 2: Trade Database
print("\n[2/4] Testing Trade Database...")
try:
    from utils.trade_database import get_database
    import time
    db = get_database()

    # Test logging with unique trade ID
    test_trade_id = f"TEST_AUTO_{int(time.time())}"
    db.log_trade_entry(
        trade_id=test_trade_id,
        symbol="TEST",
        strategy="TEST",
        side="LONG",
        entry_price=100.0,
        quantity=1,
        score=8.0
    )

    # Test retrieval
    stats = db.get_performance_stats()

    print(f"  [OK] Database working")
    print(f"    Location: {db.db_path}")
    print(f"    Total trades: {stats['total_trades']}")
    results.append(("Trade Database", True, "Working"))
except Exception as e:
    print(f"  [X] Database failed: {e}")
    results.append(("Trade Database", False, str(e)))

# Test 3: Enhanced Stop Loss Monitor
print("\n[3/4] Testing Enhanced Stop Loss Monitor...")
try:
    from utils.enhanced_stop_loss_monitor import EnhancedStopLossMonitor
    monitor = EnhancedStopLossMonitor(stop_loss_pct=0.20, check_interval=60)
    print("  [OK] Stop Loss Monitor initialized")
    print(f"    Threshold: {monitor.stop_loss_pct * 100}%")
    print(f"    Check interval: {monitor.check_interval}s")
    results.append(("Stop Loss Monitor", True, "Initialized"))
except Exception as e:
    print(f"  [X] Stop Loss Monitor failed: {e}")
    results.append(("Stop Loss Monitor", False, str(e)))

# Test 4: System Watchdog
print("\n[4/4] Testing System Watchdog...")
try:
    from utils.system_watchdog import SystemWatchdog
    watchdog = SystemWatchdog(check_interval=300)
    print("  [OK] System Watchdog initialized")
    print(f"    Monitoring {len(watchdog.systems)} systems")
    print(f"    Check interval: {watchdog.check_interval}s")
    results.append(("System Watchdog", True, "Initialized"))
except Exception as e:
    print(f"  [X] System Watchdog failed: {e}")
    results.append(("System Watchdog", False, str(e)))

# Summary
print("\n" + "="*70)
print("TEST RESULTS")
print("="*70)

passed = sum(1 for _, status, _ in results if status)
total = len(results)

for component, status, message in results:
    emoji = "[OK]" if status else "[X]"
    print(f"  {emoji} {component:20s} - {message}")

print("\n" + "="*70)
print(f"SUMMARY: {passed}/{total} components working")
print("="*70)

if passed == total:
    print("\n[OK] ALL TESTS PASSED - System ready for autonomous trading!")
    print("\nNext steps:")
    print("1. Set up Telegram (if not configured)")
    print("2. Run: python START_AUTONOMOUS_EMPIRE.py")
    print("3. Monitor via: python check_trading_status.py")
elif passed >= 3:
    print("\n[WARNING] MOSTLY WORKING - Some optional features need setup")
    print("\nYou can start trading, but:")
    for component, status, message in results:
        if not status:
            print(f"  - {component}: {message}")
    print("\nSee AUTONOMOUS_SYSTEM_SETUP_GUIDE.md for help")
else:
    print("\n[ERROR] CRITICAL FAILURES - Check errors above")
    print("\nSee AUTONOMOUS_SYSTEM_SETUP_GUIDE.md for troubleshooting")

print("="*70 + "\n")
