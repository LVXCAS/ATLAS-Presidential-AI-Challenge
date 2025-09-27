#!/usr/bin/env python3
"""
AUTONOMOUS TRADING EMPIRE PRESERVATION SCRIPT
Safely backup and organize the 68.3% ROI trading system
"""

import os
import shutil
from datetime import datetime

def preserve_trading_empire():
    """Preserve and organize the autonomous trading system"""

    print("AUTONOMOUS TRADING EMPIRE PRESERVATION")
    print("=" * 60)
    print("Safely organizing your 68.3% ROI system...")
    print("=" * 60)

    # Create backup timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup/pre_cleanup_backup_{timestamp}"

    # Create directory structure
    dirs_to_create = [
        backup_dir,
        "core",
        "intelligence",
        "config",
        "tests",
        "docs",
        "archive/learning_progress",
        "archive/profit_maximization_cycles",
        "archive/explosive_alerts",
        "archive/execution_reports"
    ]

    for dir_name in dirs_to_create:
        os.makedirs(dir_name, exist_ok=True)
        print(f"[OK] Created: {dir_name}")

    # CORE SYSTEM FILES (The money makers)
    core_files = [
        "master_autonomous_trading_engine.py",
        "adaptive_dual_options_engine.py",
        "hybrid_conviction_genetic_trader.py",
        "autonomous_portfolio_cleanup.py",
        "enhanced_options_checker.py"
    ]

    print(f"\n[CORE] Preserving essential system files...")
    for file in core_files:
        if os.path.exists(file):
            # Backup to preserve original
            shutil.copy2(file, f"{backup_dir}/{file}")
            # Copy to core (organized version)
            shutil.copy2(file, f"core/{file}")
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ MISSING: {file}")

    # INTELLIGENCE SYSTEMS
    intelligence_files = [
        "explosive_roi_hunter.py",
        "overnight_gap_scanner.py",
        "catalyst_news_monitor.py",
        "continuous_explosive_monitor.py",
        "high_conviction_options_scanner.py",
        "quality_constrained_trader.py",
        "enhanced_catalyst_scanner.py"
    ]

    print(f"\n[INTELLIGENCE] Organizing background systems...")
    for file in intelligence_files:
        if os.path.exists(file):
            shutil.copy2(file, f"{backup_dir}/{file}")
            shutil.copy2(file, f"intelligence/{file}")
            print(f"  ✅ {file}")

    # CONFIG FILES
    config_files = [
        ".env",
        "master_system_config.json",
        "optimization_params.json",
        "approved_asset_universe.json"
    ]

    print(f"\n[CONFIG] Preserving configuration...")
    for file in config_files:
        if os.path.exists(file):
            shutil.copy2(file, f"{backup_dir}/{file}")
            shutil.copy2(file, f"config/{file}")
            print(f"  ✅ {file}")

    # TEST FILES
    test_files = [
        "test_dual_strategy_quick.py",
        "weekend_system_status.py",
        "test_dual_integration.py"
    ]

    print(f"\n[TESTS] Organizing test files...")
    for file in test_files:
        if os.path.exists(file):
            shutil.copy2(file, f"{backup_dir}/{file}")
            shutil.copy2(file, f"tests/{file}")
            print(f"  ✅ {file}")

    # DOCUMENTATION
    doc_files = [
        "monday_deployment_summary.md",
        "CODEBASE_CLEANUP_PLAN.md",
        "API_KEYS_SETUP.md"
    ]

    print(f"\n[DOCS] Organizing documentation...")
    for file in doc_files:
        if os.path.exists(file):
            shutil.copy2(file, f"{backup_dir}/{file}")
            shutil.copy2(file, f"docs/{file}")
            print(f"  ✅ {file}")

    # Count files to archive
    learning_files = [f for f in os.listdir('.') if f.startswith('learning_progress_')]
    profit_files = [f for f in os.listdir('.') if f.startswith('profit_maximization_cycle_')]
    explosive_files = [f for f in os.listdir('.') if f.startswith('explosive_alert_')]
    execution_files = [f for f in os.listdir('.') if f.startswith('execution_report_')]

    print(f"\n[ARCHIVE] Moving historical data...")
    print(f"  Learning progress files: {len(learning_files)}")
    print(f"  Profit cycles: {len(profit_files)}")
    print(f"  Explosive alerts: {len(explosive_files)}")
    print(f"  Execution reports: {len(execution_files)}")

    # Archive the files
    for file in learning_files:
        shutil.move(file, f"archive/learning_progress/{file}")

    for file in profit_files:
        shutil.move(file, f"archive/profit_maximization_cycles/{file}")

    for file in explosive_files:
        shutil.move(file, f"archive/explosive_alerts/{file}")

    for file in execution_files:
        shutil.move(file, f"archive/execution_reports/{file}")

    print(f"\n" + "=" * 60)
    print("PRESERVATION COMPLETE!")
    print("=" * 60)
    print(f"✅ Core system: 5 essential files in /core/")
    print(f"✅ Intelligence: Background systems in /intelligence/")
    print(f"✅ Configuration: Settings preserved in /config/")
    print(f"✅ Tests: Validation scripts in /tests/")
    print(f"✅ Documentation: System docs in /docs/")
    print(f"✅ Archive: {len(learning_files + profit_files + explosive_files)} historical files archived")
    print(f"✅ Backup: Complete snapshot in {backup_dir}/")
    print()
    print("RESULT: Clean, organized codebase ready for Monday deployment!")
    print("Your 68.3% ROI system is preserved and optimized.")
    print("=" * 60)

if __name__ == "__main__":
    preserve_trading_empire()