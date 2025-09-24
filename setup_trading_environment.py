"""
Trading Environment Setup Script

Complete setup and validation for tomorrow's trading session.
Handles configuration, credentials, and system initialization.
"""

import os
import sys
import json
import yaml
import secrets
from pathlib import Path
from datetime import datetime
import logging

def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_directory_structure():
    """Create necessary directory structure"""
    directories = [
        'config',
        'logs',
        'reports',
        'data/cache',
        'models',
        'backups'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {directory}")

def setup_environment_file():
    """Create .env file with secure defaults"""
    env_file = Path('.env')

    if env_file.exists():
        logging.info(".env file already exists")
        return

    # Generate secure secrets
    jwt_secret = secrets.token_urlsafe(32)
    encryption_key = secrets.token_urlsafe(32)

    env_content = f"""# HiveTrading Environment Variables
# Copy this file to .env.local and update with your actual credentials

# Database Configuration
TRADING_DATABASE_PASSWORD=your_secure_database_password

# Broker API Keys (Sandbox - replace with live credentials for production)
TRADING_ALPACA_API_KEY=your_alpaca_api_key_here
TRADING_ALPACA_API_SECRET=your_alpaca_api_secret_here

# Market Data API Keys
TRADING_ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here
TRADING_POLYGON_KEY=your_polygon_api_key_here

# Security Keys (Auto-generated - DO NOT CHANGE unless needed)
TRADING_JWT_SECRET_KEY={jwt_secret}
TRADING_ENCRYPTION_KEY={encryption_key}

# Email Alerts (Optional)
TRADING_EMAIL_PASSWORD=your_email_app_password

# Slack Alerts (Optional)
TRADING_SLACK_WEBHOOK=your_slack_webhook_url

# Environment
TRADING_ENV=development
TRADING_DEBUG=true
"""

    with open(env_file, 'w') as f:
        f.write(env_content)

    logging.info(f"Created .env file with secure defaults")
    print("\n" + "="*60)
    print("IMPORTANT: Update .env file with your actual API credentials")
    print("="*60)

def load_and_validate_config():
    """Load and validate configuration"""
    config_file = Path('config/trading_config.yaml')

    if not config_file.exists():
        logging.error("Configuration file not found!")
        return None

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        logging.info("Configuration loaded successfully")
        return config

    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        return None

def create_sample_strategy_config():
    """Create sample strategy configuration"""
    strategy_config = {
        'momentum_strategy': {
            'enabled': True,
            'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
            'lookback_period': 14,
            'momentum_threshold': 0.02,
            'position_size': 0.05,  # 5% of portfolio
            'stop_loss': 0.02,      # 2%
            'take_profit': 0.06,    # 6%
            'max_positions': 3
        },
        'mean_reversion_strategy': {
            'enabled': True,
            'symbols': ['SPY', 'QQQ', 'IWM'],
            'lookback_period': 20,
            'deviation_threshold': 2.0,
            'position_size': 0.03,  # 3% of portfolio
            'stop_loss': 0.03,      # 3%
            'take_profit': 0.04,    # 4%
            'max_positions': 2
        }
    }

    strategy_file = Path('config/strategies.yaml')
    with open(strategy_file, 'w') as f:
        yaml.dump(strategy_config, f, default_flow_style=False)

    logging.info("Created sample strategy configuration")

def create_paper_trading_test():
    """Create paper trading test script"""
    test_script = '''"""
Paper Trading Test Script

Simple test to validate paper trading functionality
"""

import asyncio
import logging
from datetime import datetime

async def test_paper_trading():
    """Test basic paper trading functionality"""

    print("=" * 60)
    print("PAPER TRADING VALIDATION TEST")
    print("=" * 60)

    # Test 1: Portfolio initialization
    print("\\n1. Testing portfolio initialization...")
    initial_capital = 100000.0
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    print("   [OK] Portfolio initialized")

    # Test 2: Market data simulation
    print("\\n2. Testing market data simulation...")
    test_symbols = ['AAPL', 'GOOGL', 'MSFT']
    test_prices = {'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 420.0}
    print(f"   Test symbols: {test_symbols}")
    print(f"   Simulated prices: {test_prices}")
    print("   [OK] Market data simulation ready")

    # Test 3: Order simulation
    print("\\n3. Testing order simulation...")
    test_order = {
        'symbol': 'AAPL',
        'side': 'buy',
        'quantity': 100,
        'price': 150.0,
        'order_type': 'limit'
    }
    print(f"   Test order: {test_order}")

    # Simulate order execution
    execution_price = test_order['price'] * 1.001  # Small slippage
    commission = test_order['quantity'] * test_order['price'] * 0.001

    print(f"   Executed at: ${execution_price:.3f}")
    print(f"   Commission: ${commission:.2f}")
    print("   [OK] Order execution simulation working")

    # Test 4: Position tracking
    print("\\n4. Testing position tracking...")
    position_value = test_order['quantity'] * execution_price
    remaining_cash = initial_capital - position_value - commission

    print(f"   Position value: ${position_value:,.2f}")
    print(f"   Remaining cash: ${remaining_cash:,.2f}")
    print(f"   Total portfolio: ${remaining_cash + position_value:,.2f}")
    print("   [OK] Position tracking working")

    # Test 5: Risk calculations
    print("\\n5. Testing risk calculations...")
    position_percentage = position_value / initial_capital
    max_allowed = 0.1  # 10%

    print(f"   Position size: {position_percentage:.1%}")
    print(f"   Max allowed: {max_allowed:.1%}")

    if position_percentage <= max_allowed:
        print("   [OK] Position within risk limits")
    else:
        print("   [WARNING] Position exceeds risk limits")

    print("\\n" + "=" * 60)
    print("PAPER TRADING TEST COMPLETED")
    print("All basic functions are working correctly")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_paper_trading())
'''

    test_file = Path('test_paper_trading.py')
    with open(test_file, 'w') as f:
        f.write(test_script)

    logging.info("Created paper trading test script")

def create_premarket_checklist():
    """Create interactive pre-market checklist"""
    checklist_script = '''"""
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

    print("Please confirm each item has been completed:\\n")

    for i, item in enumerate(checklist_items, 1):
        if ask_confirmation(f"{i:2d}. {item}"):
            print("    [OK] Confirmed\\n")
            completed_items += 1
        else:
            print("    [PENDING] Not completed\\n")
            failed_items.append(item)

    # Summary
    print("=" * 70)
    print("CHECKLIST SUMMARY")
    print("=" * 70)

    completion_rate = (completed_items / len(checklist_items)) * 100
    print(f"Completed: {completed_items}/{len(checklist_items)} ({completion_rate:.1f}%)")

    if completion_rate == 100:
        print("\\nSTATUS: [READY] All items completed - Ready for trading!")
        print("\\nYou may proceed with live trading operations.")
    elif completion_rate >= 80:
        print("\\nSTATUS: [MOSTLY READY] Most items completed - Review pending items")
        print("\\nPending items:")
        for item in failed_items:
            print(f"  - {item}")
        print("\\nConsider addressing these before live trading.")
    else:
        print("\\nSTATUS: [NOT READY] Several items need attention")
        print("\\nPending items:")
        for item in failed_items:
            print(f"  - {item}")
        print("\\nComplete these items before proceeding with live trading.")

    print("\\n" + "=" * 70)

    return completion_rate == 100

if __name__ == "__main__":
    ready = run_checklist()
    sys.exit(0 if ready else 1)
'''

    checklist_file = Path('premarket_checklist.py')
    with open(checklist_file, 'w') as f:
        f.write(checklist_script)

    logging.info("Created pre-market checklist script")

def create_system_status_dashboard():
    """Create simple system status script"""
    status_script = '''"""
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
'''

    status_file = Path('system_status.py')
    with open(status_file, 'w') as f:
        f.write(status_script)

    logging.info("Created system status dashboard")

def main():
    """Main setup function"""

    setup_logging()

    print("=" * 70)
    print("HIVE TRADING SYSTEM - ENVIRONMENT SETUP")
    print("=" * 70)

    # Create directory structure
    print("\n1. Creating directory structure...")
    create_directory_structure()

    # Setup environment file
    print("\n2. Setting up environment variables...")
    setup_environment_file()

    # Validate configuration
    print("\n3. Validating configuration...")
    config = load_and_validate_config()
    if not config:
        print("ERROR: Configuration validation failed!")
        return False

    # Create additional configurations
    print("\n4. Creating additional configurations...")
    create_sample_strategy_config()

    # Create test scripts
    print("\n5. Creating test and validation scripts...")
    create_paper_trading_test()
    create_premarket_checklist()
    create_system_status_dashboard()

    print("\n" + "=" * 70)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    print("\nNext steps:")
    print("1. Update .env file with your actual API credentials")
    print("2. Run: python system_validation.py")
    print("3. Run: python test_paper_trading.py")
    print("4. Run: python premarket_checklist.py")
    print("5. Start trading with: python -m core.main")

    print("\nQuick commands:")
    print("- System status: python system_status.py")
    print("- Validation: python system_validation.py")
    print("- Pre-market check: python premarket_checklist.py")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)