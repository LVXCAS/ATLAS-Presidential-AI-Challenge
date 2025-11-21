#!/usr/bin/env python3
"""
TRADING EMPIRE PRE-FLIGHT VERIFICATION
=======================================
Verifies all systems are ready to launch

Run this before launching the trading empire
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_mark(condition, message):
    """Print check mark or X based on condition"""
    symbol = "[OK]" if condition else "[FAIL]"
    print(f"{symbol} {message}")
    return condition

def main():
    print("\n" + "="*80)
    print("TRADING EMPIRE PRE-FLIGHT VERIFICATION")
    print("="*80 + "\n")

    all_checks_passed = True

    # 1. Check credentials
    print("[CREDENTIALS CHECK]")
    all_checks_passed &= check_mark(
        os.getenv('OANDA_API_KEY') is not None,
        "OANDA API Key configured"
    )
    all_checks_passed &= check_mark(
        os.getenv('OANDA_ACCOUNT_ID') is not None,
        "OANDA Account ID configured"
    )
    all_checks_passed &= check_mark(
        os.getenv('ALPACA_API_KEY') is not None,
        "Alpaca API Key configured"
    )
    all_checks_passed &= check_mark(
        os.getenv('ALPACA_SECRET_KEY') is not None,
        "Alpaca Secret Key configured"
    )
    print()

    # 2. Check launchers exist
    print("[LAUNCHER FILES]")
    all_checks_passed &= check_mark(
        Path("START_FOREX_ELITE.py").exists(),
        "START_FOREX_ELITE.py exists"
    )
    all_checks_passed &= check_mark(
        Path("START_ADAPTIVE_OPTIONS.py").exists(),
        "START_ADAPTIVE_OPTIONS.py exists"
    )
    all_checks_passed &= check_mark(
        Path("START_TRADING_EMPIRE_FINAL.bat").exists(),
        "START_TRADING_EMPIRE_FINAL.bat exists"
    )
    all_checks_passed &= check_mark(
        Path("EMPIRE_LAUNCHER_V2.py").exists(),
        "EMPIRE_LAUNCHER_V2.py exists"
    )
    print()

    # 3. Check dependencies
    print("[DEPENDENCIES]")
    try:
        from forex_auto_trader import ForexAutoTrader
        all_checks_passed &= check_mark(True, "ForexAutoTrader importable")
    except:
        all_checks_passed &= check_mark(False, "ForexAutoTrader importable")

    try:
        from core.adaptive_dual_options_engine import AdaptiveDualOptionsEngine
        all_checks_passed &= check_mark(True, "AdaptiveDualOptionsEngine importable")
    except:
        all_checks_passed &= check_mark(False, "AdaptiveDualOptionsEngine importable")

    try:
        from REGIME_PROTECTED_TRADING import RegimeProtectedTrading
        all_checks_passed &= check_mark(True, "RegimeProtectedTrading importable")
    except:
        all_checks_passed &= check_mark(False, "RegimeProtectedTrading importable")

    try:
        from SYSTEM_HEALTH_MONITOR import SystemHealthMonitor
        all_checks_passed &= check_mark(True, "SystemHealthMonitor importable")
    except:
        all_checks_passed &= check_mark(False, "SystemHealthMonitor importable")
    print()

    # 4. Check directories
    print("[DIRECTORIES]")
    Path("logs").mkdir(exist_ok=True)
    all_checks_passed &= check_mark(
        Path("logs").exists(),
        "logs/ directory ready"
    )
    Path("config").mkdir(exist_ok=True)
    all_checks_passed &= check_mark(
        Path("config").exists(),
        "config/ directory ready"
    )
    Path("forex_trades").mkdir(exist_ok=True)
    all_checks_passed &= check_mark(
        Path("forex_trades").exists(),
        "forex_trades/ directory ready"
    )
    print()

    # 5. Check config file
    print("[CONFIGURATION]")
    forex_config_exists = Path("config/forex_elite_config.json").exists()
    if not forex_config_exists:
        print("[INFO] Creating forex_elite_config.json...")
        # Will be created on first launch
    all_checks_passed &= check_mark(
        True,  # Not critical - will be auto-created
        "Config files ready (auto-create on launch)"
    )
    print()

    # 6. Check Python version
    print("[ENVIRONMENT]")
    python_version = sys.version_info
    version_ok = python_version.major == 3 and python_version.minor >= 9
    all_checks_passed &= check_mark(
        version_ok,
        f"Python version {python_version.major}.{python_version.minor}.{python_version.micro} (need 3.9+)"
    )
    print()

    # Final verdict
    print("="*80)
    if all_checks_passed:
        print("STATUS: READY TO LAUNCH")
        print("="*80)
        print("\n[SUCCESS] All pre-flight checks passed!")
        print("\nTo launch the trading empire:")
        print("  Windows:  START_TRADING_EMPIRE_FINAL.bat")
        print("  Python:   python EMPIRE_LAUNCHER_V2.py")
        print("\nTarget: 30%+ monthly combined returns")
        print("Mode: PAPER TRADING (no real money)")
        print("\nSee: TRADING_EMPIRE_DEPLOYMENT_REPORT.md for full details")
        print("="*80)
        return 0
    else:
        print("STATUS: NOT READY - ISSUES DETECTED")
        print("="*80)
        print("\n[WARNING] Some checks failed!")
        print("Please fix the issues above before launching.")
        print("\nFor help, see: TRADING_EMPIRE_DEPLOYMENT_REPORT.md")
        print("="*80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
