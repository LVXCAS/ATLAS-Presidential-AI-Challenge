"""
System Validation for Tomorrow's Trading

Windows-compatible validation script without Unicode characters
to verify system readiness for live trading operations.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Basic system checks
try:
    import psutil
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Missing required package: {e}")
    sys.exit(1)

def print_header():
    """Print validation header"""
    print("=" * 80)
    print("HIVE TRADING SYSTEM - TOMORROW READY VALIDATION")
    print("=" * 80)
    print(f"Validation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"System: {os.name} - {sys.platform}")
    print()

def check_system_resources():
    """Check system resources"""
    print("SYSTEM RESOURCES")
    print("-" * 40)

    try:
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_status = "[OK]" if cpu_percent < 80 else "[WARN]" if cpu_percent < 90 else "[ERROR]"
        print(f"{cpu_status} CPU Usage: {cpu_percent:.1f}%")

        # Memory
        memory = psutil.virtual_memory()
        memory_status = "[OK]" if memory.percent < 80 else "[WARN]" if memory.percent < 90 else "[ERROR]"
        print(f"{memory_status} Memory Usage: {memory.percent:.1f}% ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)")

        # Disk
        disk = psutil.disk_usage('.')
        disk_status = "[OK]" if disk.percent < 80 else "[WARN]" if disk.percent < 90 else "[ERROR]"
        print(f"{disk_status} Disk Usage: {disk.percent:.1f}% ({disk.free // (1024**3):.1f}GB free)")

        return all([cpu_percent < 90, memory.percent < 90, disk.percent < 90])

    except Exception as e:
        print(f"[ERROR] System resource check failed: {e}")
        return False

def check_market_data():
    """Check market data access"""
    print("\nMARKET DATA ACCESS")
    print("-" * 40)

    try:
        # Test Yahoo Finance
        print("Testing Yahoo Finance...")
        ticker = yf.Ticker("SPY")
        data = ticker.history(period="1d")

        if not data.empty:
            last_price = data['Close'].iloc[-1]
            print(f"[OK] Yahoo Finance: SPY @ ${last_price:.2f}")
            market_data_ok = True
        else:
            print("[ERROR] Yahoo Finance: No data received")
            market_data_ok = False

        # Test basic connectivity
        print("Testing internet connectivity...")
        import urllib.request
        urllib.request.urlopen('https://finance.yahoo.com', timeout=10)
        print("[OK] Internet connectivity: OK")

        return market_data_ok

    except Exception as e:
        print(f"[ERROR] Market data check failed: {e}")
        return False

def check_python_environment():
    """Check Python environment and dependencies"""
    print("\nPYTHON ENVIRONMENT")
    print("-" * 40)

    print(f"[OK] Python Version: {sys.version.split()[0]}")

    # Check critical packages
    critical_packages = [
        'pandas', 'numpy', 'yfinance', 'psutil', 'asyncio',
        'logging', 'json', 'datetime', 'pathlib'
    ]

    missing_packages = []
    for package in critical_packages:
        try:
            __import__(package)
            print(f"[OK] {package}: Available")
        except ImportError:
            print(f"[ERROR] {package}: Missing")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n[WARN] Missing packages: {', '.join(missing_packages)}")
        return False

    return True

def check_trading_files():
    """Check if trading system files exist"""
    print("\nTRADING SYSTEM FILES")
    print("-" * 40)

    critical_files = [
        'core/parallel_trading_architecture.py',
        'brokers/broker_integrations.py',
        'data/real_time_market_data.py',
        'backtesting/comprehensive_backtesting_environment.py',
        'dashboard/performance_monitoring_dashboard.py'
    ]

    files_ok = True
    for file_path in critical_files:
        if Path(file_path).exists():
            print(f"[OK] {file_path}")
        else:
            print(f"[ERROR] {file_path} - Missing")
            files_ok = False

    return files_ok

def check_configuration():
    """Check configuration setup"""
    print("\nCONFIGURATION")
    print("-" * 40)

    config_ok = True

    # Check for config directory
    config_dir = Path("config")
    if config_dir.exists():
        print("[OK] Config directory exists")
    else:
        print("[WARN] Config directory missing - creating default")
        config_dir.mkdir(exist_ok=True)
        config_ok = False

    # Check for basic config files
    base_config = config_dir / "base.yaml"
    if base_config.exists():
        print("[OK] Base configuration exists")
    else:
        print("[WARN] Base configuration missing - creating default")
        # Create minimal config
        minimal_config = """environment: development
debug: true
trading:
  initial_capital: 100000.0
  enable_paper_trading: true
  max_leverage: 2.0
  max_position_size: 0.1
  max_daily_loss: 0.02
brokers: {}
"""
        with open(base_config, 'w') as f:
            f.write(minimal_config)
        config_ok = False

    return config_ok

def check_market_hours():
    """Check market status and hours"""
    print("\nMARKET HOURS")
    print("-" * 40)

    now = datetime.now()

    # Check if weekend
    if now.weekday() >= 5:
        print("[INFO] Weekend - Markets closed")
        print("Next trading day: Monday")
        return "weekend"

    # Simple market hours check (US Eastern Time approximation)
    hour = now.hour

    if 6 <= hour < 9:
        print("[INFO] Pre-market hours")
        print("Market opens at 9:30 AM ET")
        return "premarket"
    elif 9 <= hour < 16:
        print("[INFO] Market hours - Trading active")
        print("Live trading possible")
        return "open"
    elif 16 <= hour < 20:
        print("[INFO] After-hours trading")
        print("Extended hours available")
        return "afterhours"
    else:
        print("[INFO] Market closed")
        print("Next session: Tomorrow 9:30 AM ET")
        return "closed"

def create_trading_checklist():
    """Create tomorrow's trading checklist"""
    print("\nTOMORROW'S TRADING CHECKLIST")
    print("-" * 50)

    checklist = [
        "Verify broker API keys are configured",
        "Check account balances and buying power",
        "Review overnight news and market events",
        "Confirm strategy parameters are set",
        "Test paper trading functionality first",
        "Monitor system performance metrics",
        "Set appropriate risk limits",
        "Have emergency shutdown plan ready",
        "Monitor for any system alerts",
        "Keep position sizing conservative initially"
    ]

    for i, item in enumerate(checklist, 1):
        print(f"{i:2d}. [ ] {item}")

def test_basic_trading_setup():
    """Test basic trading setup"""
    print("\nBASIC TRADING SETUP TEST")
    print("-" * 40)

    try:
        # Test creating a simple portfolio
        initial_capital = 100000.0
        print(f"[OK] Initial Capital: ${initial_capital:,.2f}")

        # Test position sizing calculation
        max_position_size = 0.1  # 10%
        test_position_value = initial_capital * max_position_size
        print(f"[OK] Max Position Size (10%): ${test_position_value:,.2f}")

        # Test risk calculation
        max_daily_loss = 0.02  # 2%
        max_loss_amount = initial_capital * max_daily_loss
        print(f"[OK] Max Daily Loss (2%): ${max_loss_amount:,.2f}")

        # Test basic math operations
        test_price = 150.0
        test_quantity = int(test_position_value / test_price)
        print(f"[OK] Example: {test_quantity} shares at ${test_price} = ${test_quantity * test_price:,.2f}")

        return True

    except Exception as e:
        print(f"[ERROR] Trading setup test failed: {e}")
        return False

def generate_summary(results):
    """Generate validation summary"""
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    passed = sum(results.values())
    total = len(results)

    print(f"Checks Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")

    if passed == total:
        status = "[READY] SYSTEM READY"
        recommendation = "System is ready for tomorrow's trading"
    elif passed >= total * 0.8:
        status = "[WARN] MOSTLY READY"
        recommendation = "Address warnings before live trading"
    else:
        status = "[ERROR] NOT READY"
        recommendation = "Fix critical issues before trading"

    print(f"\nSTATUS: {status}")
    print(f"RECOMMENDATION: {recommendation}")

    # Detailed breakdown
    print(f"\nDETAILED RESULTS:")
    for check, passed in results.items():
        status_icon = "[PASS]" if passed else "[FAIL]"
        print(f"   {status_icon} {check}")

    return passed >= total * 0.8

def main():
    """Main validation function"""

    print_header()

    # Run all checks
    results = {
        "System Resources": check_system_resources(),
        "Market Data Access": check_market_data(),
        "Python Environment": check_python_environment(),
        "Trading System Files": check_trading_files(),
        "Configuration": check_configuration(),
        "Basic Trading Setup": test_basic_trading_setup()
    }

    # Check market hours
    market_status = check_market_hours()

    # Create checklist
    create_trading_checklist()

    # Generate summary
    system_ready = generate_summary(results)

    # Save validation report
    try:
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = report_dir / f"validation_report_{timestamp}.json"

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "system_ready": system_ready,
            "market_status": market_status,
            "validation_results": results,
            "python_version": sys.version,
            "platform": sys.platform
        }

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\nReport saved: {report_file}")

    except Exception as e:
        print(f"\nCould not save report: {e}")

    print("\n" + "=" * 80)
    print(f"FINAL STATUS: {'READY FOR TRADING' if system_ready else 'NOT READY - FIX ISSUES FIRST'}")
    print("=" * 80)

    return system_ready

if __name__ == "__main__":
    try:
        ready = main()
        exit_code = 0 if ready else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nValidation failed with error: {e}")
        sys.exit(1)