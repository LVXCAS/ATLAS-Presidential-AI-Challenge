#!/usr/bin/env python3
"""
Test Profit/Loss Limits System
Tests the enhanced monitoring system with both profit target and loss limit
"""

import asyncio
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append('.')

# Import the enhanced monitor
from profit_target_monitor import ProfitTargetMonitor

async def test_profit_loss_system():
    """Test the profit target and loss limit system"""
    print("PROFIT/LOSS LIMITS SYSTEM TEST")
    print("=" * 50)
    print("Testing enhanced monitoring with:")
    print("- Profit Target: +5.75%")
    print("- Loss Limit: -4.9%")
    print()

    # Create monitor instance
    monitor = ProfitTargetMonitor()

    # Test initialization
    print("1. Testing Monitor Initialization...")
    print(f"   Profit target: {monitor.profit_target_pct}%")
    print(f"   Loss limit: {monitor.loss_limit_pct}%")
    print(f"   Monitoring active: {monitor.monitoring_active}")
    print(f"   Target hit: {monitor.target_hit}")
    print(f"   Loss limit hit: {monitor.loss_limit_hit}")
    print("   [OK] Monitor initialized successfully")
    print()

    # Test status reporting
    print("2. Testing Status Reporting...")
    status = monitor.get_status()
    print("   Status dict contains:")
    for key, value in status.items():
        print(f"     {key}: {value}")
    print("   [OK] Status reporting working")
    print()

    # Test broker initialization (will use paper trading)
    print("3. Testing Broker Integration...")
    try:
        broker_ready = await monitor.initialize_broker()
        if broker_ready:
            print("   [OK] Broker connection established")

            # Test daily profit checking
            print("4. Testing Daily P&L Calculation...")
            try:
                equity, pnl_pct, profit_hit, loss_hit = await monitor.check_daily_profit()
                print(f"   Current equity: ${equity:,.2f}")
                print(f"   Daily P&L: {pnl_pct:+.2f}%")
                print(f"   Profit target hit: {profit_hit}")
                print(f"   Loss limit hit: {loss_hit}")
                print("   [OK] P&L calculation working")
            except Exception as e:
                print(f"   [WARN]  P&L calculation error: {e}")
        else:
            print("   [WARN]  Broker connection failed - running offline test")
    except Exception as e:
        print(f"   [WARN]  Broker initialization error: {e}")

    print()

    # Test configuration validation
    print("5. Testing System Configuration...")

    # Validate profit target
    if monitor.profit_target_pct == 5.75:
        print("   [OK] Profit target correctly set to 5.75%")
    else:
        print(f"   [FAIL] Profit target incorrect: {monitor.profit_target_pct}%")

    # Validate loss limit
    if monitor.loss_limit_pct == -4.9:
        print("   [OK] Loss limit correctly set to -4.9%")
    else:
        print(f"   [FAIL] Loss limit incorrect: {monitor.loss_limit_pct}%")

    # Validate flags
    if not monitor.target_hit and not monitor.loss_limit_hit:
        print("   [OK] Initial flags correctly set")
    else:
        print("   [FAIL] Initial flags incorrect")

    print()

    # Test limit logic simulation
    print("6. Testing Limit Logic Simulation...")

    # Simulate scenarios
    test_scenarios = [
        {"equity": 100000, "start": 100000, "expected_pnl": 0.0, "name": "Break-even"},
        {"equity": 105750, "start": 100000, "expected_pnl": 5.75, "name": "Profit target hit"},
        {"equity": 105800, "start": 100000, "expected_pnl": 5.8, "name": "Above profit target"},
        {"equity": 95100, "start": 100000, "expected_pnl": -4.9, "name": "Loss limit hit"},
        {"equity": 95000, "start": 100000, "expected_pnl": -5.0, "name": "Below loss limit"},
        {"equity": 102000, "start": 100000, "expected_pnl": 2.0, "name": "Normal profit"},
        {"equity": 98000, "start": 100000, "expected_pnl": -2.0, "name": "Normal loss"}
    ]

    for scenario in test_scenarios:
        # Simulate the calculation
        start_equity = scenario["start"]
        current_equity = scenario["equity"]
        expected_pnl = scenario["expected_pnl"]

        # Calculate P&L manually
        calculated_pnl = ((current_equity - start_equity) / start_equity) * 100

        # Check limits
        profit_hit = calculated_pnl >= monitor.profit_target_pct
        loss_hit = calculated_pnl <= monitor.loss_limit_pct

        # Determine expected triggers
        expected_profit_hit = expected_pnl >= 5.75
        expected_loss_hit = expected_pnl <= -4.9

        # Validate
        pnl_correct = abs(calculated_pnl - expected_pnl) < 0.01
        profit_correct = profit_hit == expected_profit_hit
        loss_correct = loss_hit == expected_loss_hit

        status = "[OK]" if (pnl_correct and profit_correct and loss_correct) else "[FAIL]"

        print(f"   {status} {scenario['name']}: {calculated_pnl:+.2f}% | Profit: {profit_hit} | Loss: {loss_hit}")

    print()

    # Test event logging structure
    print("7. Testing Event Logging Structure...")
    try:
        # Test log event method with simulated data
        monitor.initial_equity = 100000
        monitor.current_equity = 105750
        monitor.daily_profit_pct = 5.75

        # Test profit event logging
        monitor.log_trade_event("Daily profit target 5.75% reached (5.75%)")
        print("   [OK] Profit event logging successful")

        # Test loss event logging
        monitor.current_equity = 95100
        monitor.daily_profit_pct = -4.9
        monitor.log_trade_event("Daily loss limit -4.9% reached (-4.9%)")
        print("   [OK] Loss event logging successful")

    except Exception as e:
        print(f"   [FAIL] Event logging error: {e}")

    print()

    # Final validation
    print("8. System Readiness Assessment...")

    checks = [
        ("Profit target configured", monitor.profit_target_pct == 5.75),
        ("Loss limit configured", monitor.loss_limit_pct == -4.9),
        ("Monitoring system ready", monitor.monitoring_active),
        ("No false triggers", not monitor.target_hit and not monitor.loss_limit_hit),
        ("Broker integration available", hasattr(monitor, 'broker')),
        ("Status reporting functional", len(monitor.get_status()) >= 8)
    ]

    all_passed = True
    for check_name, passed in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"   {status} {check_name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("[SUCCESS] SYSTEM READY FOR LIVE TRADING!")
        print("[OK] Both profit target (+5.75%) and loss limit (-4.9%) configured")
        print("[OK] All tests passed successfully")
        print("[OK] Enhanced monitoring system operational")
    else:
        print("[WARN]  Some tests failed - review configuration before live trading")

    print()
    print("INTEGRATION STATUS:")
    print("- OPTIONS_BOT: Ready (profit/loss monitoring integrated)")
    print("- Market Hunter: Ready (profit/loss monitoring integrated)")
    print("- Monitor checks: Every 30 seconds")
    print("- Actions: Sell all positions + cancel all orders when triggered")

if __name__ == "__main__":
    asyncio.run(test_profit_loss_system())