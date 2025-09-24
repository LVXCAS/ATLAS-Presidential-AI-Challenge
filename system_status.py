"""
System Status Dashboard

Quick status check for all trading system components
"""

import json
import psutil
from datetime import datetime
from pathlib import Path

def check_system_status():
    """Check overall system status"""

    print("=" * 60)
    print("HIVE TRADING SYSTEM STATUS")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # System resources
    print("SYSTEM RESOURCES:")
    print(f"  CPU Usage: {psutil.cpu_percent(interval=1):.1f}%")
    print(f"  Memory Usage: {psutil.virtual_memory().percent:.1f}%")
    print(f"  Disk Usage: {psutil.disk_usage('.').percent:.1f}%")
    print()

    # File system
    print("FILE SYSTEM:")
    critical_files = [
        'config/trading_config.yaml',
        'test_paper_trading.py',
        'premarket_checklist.py',
        '.env'
    ]

    for file_path in critical_files:
        status = "[OK]" if Path(file_path).exists() else "[MISSING]"
        print(f"  {status} {file_path}")
    print()

    # Configuration
    print("CONFIGURATION:")
    try:
        with open('config/trading_config.yaml', 'r') as f:
            import yaml
            config = yaml.safe_load(f)

        print(f"  Environment: {config.get('environment', 'unknown')}")
        print(f"  Paper Trading: {config.get('trading', {}).get('enable_paper_trading', 'unknown')}")
        print(f"  Initial Capital: ${config.get('trading', {}).get('initial_capital', 0):,.2f}")
        print("  [OK] Configuration loaded")
    except Exception as e:
        print(f"  [ERROR] Configuration: {e}")
    print()

    # Recent reports
    print("RECENT REPORTS:")
    reports_dir = Path('reports')
    if reports_dir.exists():
        report_files = list(reports_dir.glob('validation_report_*.json'))
        if report_files:
            latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
            print(f"  Latest validation: {latest_report.name}")

            try:
                with open(latest_report, 'r') as f:
                    report_data = json.load(f)
                print(f"  System ready: {report_data.get('system_ready', 'unknown')}")
                print(f"  Success rate: {sum(report_data.get('validation_results', {}).values()) / len(report_data.get('validation_results', {})) * 100:.1f}%")
            except:
                print("  [ERROR] Could not read report")
        else:
            print("  No validation reports found")
    else:
        print("  Reports directory not found")

    print()
    print("=" * 60)
    print("For detailed validation, run: python system_validation.py")
    print("For pre-market checklist, run: python premarket_checklist.py")
    print("For paper trading test, run: python test_paper_trading.py")
    print("=" * 60)

if __name__ == "__main__":
    check_system_status()
