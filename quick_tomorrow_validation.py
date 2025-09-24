"""
Quick Tomorrow-Ready Validation Script

Simplified validation that runs immediately without complex imports
to verify system readiness for tomorrow's trading.
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Basic system checks
import psutil
import yfinance as yf
import pandas as pd

def print_header():
    """Print validation header"""
    print("=" * 80)
    print("HIVE TRADING TOMORROW-READY VALIDATION")
    print("=" * 80)
    print(f"Validation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"System: {os.name} - {sys.platform}")
    print()

def check_system_resources():
    """Check system resources"""
    print("📊 SYSTEM RESOURCES")
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
        disk = psutil.disk_usage('/')
        disk_status = "[OK]" if disk.percent < 80 else "[WARN]" if disk.percent < 90 else "[ERROR]"
        print(f"{disk_status} Disk Usage: {disk.percent:.1f}% ({disk.free // (1024**3):.1f}GB free)")

        return all([cpu_percent < 90, memory.percent < 90, disk.percent < 90])

    except Exception as e:
        print(f"❌ System resource check failed: {e}")
        return False

def check_market_data():
    """Check market data access"""
    print("\n📈 MARKET DATA ACCESS")
    print("-" * 40)

    try:
        # Test Yahoo Finance
        print("🔍 Testing Yahoo Finance...")
        ticker = yf.Ticker("SPY")
        data = ticker.history(period="1d")

        if not data.empty:
            last_price = data['Close'].iloc[-1]
            print(f"✅ Yahoo Finance: SPY @ ${last_price:.2f}")
            market_data_ok = True
        else:
            print("❌ Yahoo Finance: No data received")
            market_data_ok = False

        # Test basic connectivity
        print("🔍 Testing internet connectivity...")
        import urllib.request
        urllib.request.urlopen('https://finance.yahoo.com', timeout=10)
        print("✅ Internet connectivity: OK")

        return market_data_ok

    except Exception as e:
        print(f"❌ Market data check failed: {e}")
        return False

def check_python_environment():
    """Check Python environment and dependencies"""
    print("\n🐍 PYTHON ENVIRONMENT")
    print("-" * 40)

    print(f"✅ Python Version: {sys.version.split()[0]}")

    # Check critical packages
    critical_packages = [
        'pandas', 'numpy', 'yfinance', 'psutil', 'asyncio',
        'logging', 'json', 'datetime', 'pathlib'
    ]

    missing_packages = []
    for package in critical_packages:
        try:
            __import__(package)
            print(f"✅ {package}: Available")
        except ImportError:
            print(f"❌ {package}: Missing")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        return False

    return True

def check_trading_files():
    """Check if trading system files exist"""
    print("\n📁 TRADING SYSTEM FILES")
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
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            files_ok = False

    return files_ok

def check_configuration():
    """Check configuration setup"""
    print("\n⚙️ CONFIGURATION")
    print("-" * 40)

    config_ok = True

    # Check for config directory
    config_dir = Path("config")
    if config_dir.exists():
        print("✅ Config directory exists")
    else:
        print("⚠️ Config directory missing - will create default")
        config_dir.mkdir(exist_ok=True)
        config_ok = False

    # Check for basic config files
    base_config = config_dir / "base.yaml"
    if base_config.exists():
        print("✅ Base configuration exists")
    else:
        print("⚠️ Base configuration missing - will create default")
        # Create minimal config
        minimal_config = """
environment: development
debug: true
trading:
  initial_capital: 100000.0
  enable_paper_trading: true
  max_leverage: 2.0
  max_position_size: 0.1
"""
        with open(base_config, 'w') as f:
            f.write(minimal_config)
        config_ok = False

    return config_ok

def check_market_hours():
    """Check market status and hours"""
    print("\n🕐 MARKET HOURS")
    print("-" * 40)

    now = datetime.now()

    # Check if weekend
    if now.weekday() >= 5:
        print("⚠️ Weekend - Markets closed")
        print("📅 Next trading day: Monday")
        return "weekend"

    # Simple market hours check (US Eastern Time approximation)
    hour = now.hour

    if 6 <= hour < 9:
        print("🌅 Pre-market hours")
        print("📈 Market opens at 9:30 AM ET")
        return "premarket"
    elif 9 <= hour < 16:
        print("🔔 Market hours - Trading active")
        print("📊 Live trading possible")
        return "open"
    elif 16 <= hour < 20:
        print("🌆 After-hours trading")
        print("📉 Extended hours available")
        return "afterhours"
    else:
        print("🌙 Market closed")
        print("📅 Next session: Tomorrow 9:30 AM ET")
        return "closed"

def create_trading_checklist():
    """Create tomorrow's trading checklist"""
    print("\n📋 TOMORROW'S TRADING CHECKLIST")
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
        print(f"{i:2d}. ☐ {item}")

def generate_summary(results):
    """Generate validation summary"""
    print("\n" + "=" * 80)
    print("📊 VALIDATION SUMMARY")
    print("=" * 80)

    passed = sum(results.values())
    total = len(results)

    print(f"📈 Checks Passed: {passed}/{total}")
    print(f"📊 Success Rate: {(passed/total)*100:.1f}%")

    if passed == total:
        status = "🟢 SYSTEM READY"
        recommendation = "✅ System is ready for tomorrow's trading"
    elif passed >= total * 0.8:
        status = "🟡 MOSTLY READY"
        recommendation = "⚠️ Address warnings before live trading"
    else:
        status = "🔴 NOT READY"
        recommendation = "❌ Fix critical issues before trading"

    print(f"\n🎯 STATUS: {status}")
    print(f"💡 RECOMMENDATION: {recommendation}")

    # Detailed breakdown
    print(f"\n📋 DETAILED RESULTS:")
    for check, passed in results.items():
        status_icon = "✅" if passed else "❌"
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
        "Configuration": check_configuration()
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

        print(f"\n💾 Report saved: {report_file}")

    except Exception as e:
        print(f"\n⚠️ Could not save report: {e}")

    print("\n" + "=" * 80)

    return system_ready

if __name__ == "__main__":
    try:
        ready = main()
        exit_code = 0 if ready else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)