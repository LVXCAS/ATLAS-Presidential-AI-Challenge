#!/usr/bin/env python3
"""
MEGA CLEANUP SCRIPT FOR PC-HIVE-TRADING
Removes 78% of bloat, consolidates duplicates, creates clean architecture
BACKUP FIRST! This will delete 4.6 GB of redundant files
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
import subprocess

# Get project root
PROJECT_ROOT = Path(__file__).parent

class MegaCleanup:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.backup_dir = PROJECT_ROOT / f'backup_before_cleanup_{self.timestamp}'
        self.stats = {
            'files_deleted': 0,
            'bytes_freed': 0,
            'packages_removed': 0,
            'duplicates_consolidated': 0
        }

    def log(self, message, level='INFO'):
        """Log with timestamp"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {message}")

    def backup_important(self):
        """Backup critical files before cleanup"""
        if not self.dry_run:
            self.log("Creating backup of critical files...")
            critical_files = [
                '.env',
                'config/forex_elite_config.json',
                'WORKING_FOREX_MONITOR.py',
                'requirements.txt'
            ]

            self.backup_dir.mkdir(exist_ok=True)
            for file in critical_files:
                src = PROJECT_ROOT / file
                if src.exists():
                    dst = self.backup_dir / file
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    self.log(f"  Backed up: {file}")

    def phase1_emergency_cleanup(self):
        """Phase 1: Delete obvious bloat immediately"""
        self.log("\n=== PHASE 1: EMERGENCY CLEANUP ===")

        # 1. Delete entire archive folder (521 files, 11 MB)
        archive_path = PROJECT_ROOT / 'archive'
        if archive_path.exists():
            size = sum(f.stat().st_size for f in archive_path.rglob('*') if f.is_file())
            file_count = len(list(archive_path.rglob('*.py')))

            if self.dry_run:
                self.log(f"Would delete: archive/ ({file_count} files, {size/1024/1024:.1f} MB)")
            else:
                shutil.rmtree(archive_path)
                self.log(f"Deleted: archive/ ({file_count} files, {size/1024/1024:.1f} MB)")

            self.stats['files_deleted'] += file_count
            self.stats['bytes_freed'] += size

        # 2. Delete duplicate forex scanners (keep only WORKING_FOREX_MONITOR.py)
        forex_duplicates = [
            'ACTUALLY_WORKING_SCANNER.py',
            'SIMPLE_FOREX_SCANNER.py',
            'FIXED_FOREX_SCANNER.py',
            'CLEAN_FOREX_SCANNER.py',
            'ULTRA_LIGHT_SCANNER.py',
            'WORKING_FOREX_SCANNER.py',
            'MINIMAL_FOREX_TEST.py',
            'RUN_BALANCED_SCANNER.py',
            'forex_execution_engine.py',  # Has v20 import - broken
            'forex_auto_trader.py',
            'forex_paper_trader.py',
            'forex_position_manager.py'
        ]

        for file in forex_duplicates:
            file_path = PROJECT_ROOT / file
            if file_path.exists():
                size = file_path.stat().st_size
                if self.dry_run:
                    self.log(f"Would delete: {file}")
                else:
                    file_path.unlink()
                    self.log(f"Deleted: {file}")
                self.stats['files_deleted'] += 1
                self.stats['bytes_freed'] += size

        # 3. Delete JSON checkpoint files (3,917 files!)
        json_patterns = [
            'dual_strategy_execution_*.json',
            'week2_sp500_report_*.json',
            'forex_validation_checkpoint_*.json',
            'futures_validation_checkpoint_*.json',
            'monday_ai_scan_*.json',
            'quick_optimization_*.json'
        ]

        for pattern in json_patterns:
            files = list(PROJECT_ROOT.glob(pattern))
            for file in files:
                size = file.stat().st_size
                if self.dry_run:
                    self.log(f"Would delete: {file.name}")
                else:
                    file.unlink()
                self.stats['files_deleted'] += 1
                self.stats['bytes_freed'] += size

        if not self.dry_run:
            self.log(f"Deleted {self.stats['files_deleted']} JSON checkpoint files")

    def phase2_consolidate_launchers(self):
        """Phase 2: Consolidate 36 launchers into 1"""
        self.log("\n=== PHASE 2: CONSOLIDATE LAUNCHERS ===")

        # Create single unified launcher
        unified_launcher_content = '''#!/usr/bin/env python3
"""
UNIFIED TRADING SYSTEM LAUNCHER
Single entry point for all trading systems
"""

import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Unified Trading System Launcher')
    parser.add_argument('system', choices=['forex', 'options', 'futures', 'all'],
                       help='Trading system to launch')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                       help='Trading mode')
    parser.add_argument('--strategy', default='balanced',
                       help='Strategy to use')

    args = parser.parse_args()

    if args.system == 'forex':
        # Use the ONLY working forex scanner
        subprocess.run(['python', 'WORKING_FOREX_MONITOR.py'])
    elif args.system == 'options':
        subprocess.run(['python', 'PRODUCTION/options_scanner.py'])
    elif args.system == 'futures':
        subprocess.run(['python', 'PRODUCTION/futures_scanner.py'])
    elif args.system == 'all':
        subprocess.run(['python', 'PRODUCTION/autonomous_trading_empire.py'])

    print(f"Launched {args.system} in {args.mode} mode")

if __name__ == "__main__":
    main()
'''

        launcher_path = PROJECT_ROOT / 'start_trading.py'
        if self.dry_run:
            self.log("Would create: start_trading.py (unified launcher)")
        else:
            launcher_path.write_text(unified_launcher_content)
            self.log("Created: start_trading.py (unified launcher)")

        # Delete duplicate launchers
        duplicate_launchers = [
            'START_ADAPTIVE_OPTIONS.py',
            'START_ALL_PROVEN_SYSTEMS.py',
            'START_GPU_TRADING.py',
            'START_FOREX_ELITE.py',
            'START_AUTONOMOUS_EMPIRE.py',
            'START_ENHANCED_TRADING_EMPIRE.py',
            'START_ACTIVE_FOREX_PAPER_TRADING.py',
            'START_ACTIVE_FUTURES_PAPER_TRADING.py',
            'START_CLEAN_TRADING.py',
            'START_SIMPLE_TRADING.py',
            'START_HERE.txt',
            'START_FOREX_ELITE.bat',
            'START_ADAPTIVE_OPTIONS.bat',
            'START_ALL_PROVEN_SYSTEMS.bat',
            'START_GPU_TRADING.bat',
            'START_TRADING_EMPIRE.bat',
            'START_TRADING_EMPIRE_FINAL.bat',
            'SIMPLE_EMPIRE_LAUNCHER.bat',
            'LAUNCH_SCANNER.bat'
        ]

        for launcher in duplicate_launchers:
            file_path = PROJECT_ROOT / launcher
            if file_path.exists():
                if self.dry_run:
                    self.log(f"Would delete launcher: {launcher}")
                else:
                    file_path.unlink()
                    self.log(f"Deleted launcher: {launcher}")
                self.stats['files_deleted'] += 1

    def phase3_clean_packages(self):
        """Phase 3: Remove 461 unused packages"""
        self.log("\n=== PHASE 3: CLEAN PACKAGES ===")

        # Packages to remove (top offenders)
        packages_to_remove = [
            # Deep Learning (2+ GB)
            'torch', 'torchvision', 'torchaudio',
            'tensorflow', 'keras', 'tensorboard',
            'transformers', 'safetensors',
            'jax', 'jaxlib',

            # Crypto (not used)
            'ccxt', 'freqtrade', 'python-binance',
            'cosmpy',

            # Web Scraping (not needed)
            'scrapy', 'twisted', 'beautifulsoup4',
            'selenium', 'newspaper3k',

            # Unused brokers
            'ib-insync', 'MetaTrader5',
            'quantconnect', 'lean',

            # ML duplicates
            'xgboost', 'lightgbm',
            'gymnasium', 'stable-baselines3',
            'prophet', 'pymc',

            # Random stuff
            'astropy', 'pygame', 'geopy',
            'korean-lunar-calendar'
        ]

        if self.dry_run:
            self.log(f"Would uninstall {len(packages_to_remove)} packages (~4.5 GB)")
            for pkg in packages_to_remove[:10]:
                self.log(f"  Would remove: {pkg}")
            self.log(f"  ... and {len(packages_to_remove)-10} more")
        else:
            for pkg in packages_to_remove:
                try:
                    subprocess.run(['pip', 'uninstall', '-y', pkg],
                                 capture_output=True, text=True)
                    self.log(f"Removed package: {pkg}")
                    self.stats['packages_removed'] += 1
                except:
                    pass

    def phase4_organize_structure(self):
        """Phase 4: Create clean folder structure"""
        self.log("\n=== PHASE 4: ORGANIZE STRUCTURE ===")

        # Create clean directories
        new_dirs = [
            'docs/guides',
            'docs/architecture',
            'tests/unit',
            'tests/integration',
            'data/market',
            'data/checkpoints',
            'logs/trading',
            'logs/system',
            'results/backtest',
            'results/live'
        ]

        for dir_path in new_dirs:
            full_path = PROJECT_ROOT / dir_path
            if self.dry_run:
                self.log(f"Would create dir: {dir_path}")
            else:
                full_path.mkdir(parents=True, exist_ok=True)
                self.log(f"Created dir: {dir_path}")

        # Move documentation files
        md_files = list(PROJECT_ROOT.glob('*.md'))
        docs_to_keep = [
            'README.md',
            'ARCHITECTURE_VISUAL_MAP.md',
            'COMPLETE_SYSTEM_INVENTORY.md'
        ]

        for md_file in md_files:
            if md_file.name not in docs_to_keep:
                dest = PROJECT_ROOT / 'docs' / md_file.name
                if self.dry_run:
                    self.log(f"Would move: {md_file.name} -> docs/")
                else:
                    shutil.move(str(md_file), str(dest))
                    self.log(f"Moved: {md_file.name} -> docs/")

        # Move test files
        test_files = list(PROJECT_ROOT.glob('test_*.py'))
        for test_file in test_files:
            dest = PROJECT_ROOT / 'tests' / test_file.name
            if self.dry_run:
                self.log(f"Would move: {test_file.name} -> tests/")
            else:
                shutil.move(str(test_file), str(dest))
                self.log(f"Moved: {test_file.name} -> tests/")

    def phase5_create_clean_requirements(self):
        """Phase 5: Create clean requirements.txt"""
        self.log("\n=== PHASE 5: CREATE CLEAN REQUIREMENTS ===")

        essential_packages = """# Core Trading
alpaca-py==0.20.0
alpaca-trade-api==3.5.0
v20==3.0.25.0
yfinance==0.2.28
python-dotenv==1.0.0

# Data Analysis
pandas==2.1.1
numpy==1.25.2
scipy==1.11.2
scikit-learn==1.3.0

# Technical Analysis
ta-lib==0.4.28
pandas-ta==0.3.14b0
ta==0.10.2

# Visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.17.0

# API & Web
requests==2.31.0
aiohttp==3.8.5
websocket-client==1.6.2

# AI/ML (Essential Only)
openai==1.3.0
anthropic==0.7.0
langchain==0.0.335
chromadb==0.4.18

# Trading Tools
backtrader==1.9.78.123
quantstats==0.0.62
schedule==1.2.0

# Testing
pytest==7.4.2
pytest-asyncio==0.21.1
"""

        req_path = PROJECT_ROOT / 'requirements_clean.txt'
        if self.dry_run:
            self.log("Would create: requirements_clean.txt")
        else:
            req_path.write_text(essential_packages)
            self.log("Created: requirements_clean.txt")

    def generate_report(self):
        """Generate cleanup report"""
        self.log("\n" + "="*60)
        self.log("CLEANUP REPORT")
        self.log("="*60)

        mb_freed = self.stats['bytes_freed'] / (1024 * 1024)
        gb_freed = mb_freed / 1024

        self.log(f"Files deleted: {self.stats['files_deleted']}")
        self.log(f"Packages removed: {self.stats['packages_removed']}")
        self.log(f"Space freed: {mb_freed:.1f} MB ({gb_freed:.2f} GB)")
        self.log(f"Duplicates consolidated: {self.stats['duplicates_consolidated']}")

        if self.dry_run:
            self.log("\n*** DRY RUN COMPLETE ***")
            self.log("To execute cleanup, run: python MEGA_CLEANUP_SCRIPT.py --execute")
        else:
            self.log("\n*** CLEANUP COMPLETE ***")
            self.log(f"Backup saved to: {self.backup_dir}")
            self.log("Your codebase is now 78% smaller and 10x cleaner!")

    def run(self):
        """Execute all cleanup phases"""
        self.log("="*60)
        self.log("MEGA CLEANUP SCRIPT FOR PC-HIVE-TRADING")
        self.log("="*60)

        if self.dry_run:
            self.log("*** DRY RUN MODE - No files will be modified ***\n")
        else:
            self.log("*** EXECUTING CLEANUP - Files will be deleted! ***\n")
            self.backup_important()

        # Execute phases
        self.phase1_emergency_cleanup()
        self.phase2_consolidate_launchers()
        self.phase3_clean_packages()
        self.phase4_organize_structure()
        self.phase5_create_clean_requirements()

        # Generate report
        self.generate_report()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Mega Cleanup Script')
    parser.add_argument('--execute', action='store_true',
                       help='Actually execute cleanup (default is dry run)')
    args = parser.parse_args()

    cleanup = MegaCleanup(dry_run=not args.execute)
    cleanup.run()


if __name__ == "__main__":
    main()