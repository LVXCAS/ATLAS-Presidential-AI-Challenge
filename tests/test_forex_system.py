#!/usr/bin/env python3
"""
FOREX TRADING SYSTEM - COMPREHENSIVE TEST SUITE

Tests all components:
1. OANDA data fetcher
2. Execution engine
3. Position manager
4. Strategy signal detection
5. Auto-trader orchestration
6. Configuration loading
7. Safety limits
8. Logging

Run this before going live to ensure everything works!
"""

import os
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Test results tracker
class TestResults:
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0

    def add_test(self, name, passed, message=""):
        self.tests.append({
            'name': name,
            'passed': passed,
            'message': message
        })
        if passed:
            self.passed += 1
            print(f"  [PASS] {name}")
            if message:
                print(f"         {message}")
        else:
            self.failed += 1
            print(f"  [FAIL] {name}")
            if message:
                print(f"         {message}")

    def print_summary(self):
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total Tests: {len(self.tests)}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {self.passed/len(self.tests)*100:.1f}%")

        if self.failed > 0:
            print("\nFailed Tests:")
            for test in self.tests:
                if not test['passed']:
                    print(f"  - {test['name']}: {test['message']}")

        print("="*70)

        return self.failed == 0


results = TestResults()


def test_imports():
    """Test that all required modules can be imported"""

    print("\n[TEST 1] Import Required Modules")

    try:
        import v20
        results.add_test("Import v20", True, "OANDA API library available")
    except ImportError:
        results.add_test("Import v20", False, "Run: pip install v20")

    try:
        import pandas
        results.add_test("Import pandas", True)
    except ImportError:
        results.add_test("Import pandas", False, "Run: pip install pandas")

    try:
        import numpy
        results.add_test("Import numpy", True)
    except ImportError:
        results.add_test("Import numpy", False, "Run: pip install numpy")

    try:
        from dotenv import load_dotenv
        results.add_test("Import python-dotenv", True)
    except ImportError:
        results.add_test("Import python-dotenv", False, "Run: pip install python-dotenv")


def test_file_structure():
    """Test that all required files exist"""

    print("\n[TEST 2] File Structure")

    required_files = [
        'forex_auto_trader.py',
        'forex_execution_engine.py',
        'forex_position_manager.py',
        'forex_v4_optimized.py',
        'data/oanda_data_fetcher.py',
        'config/forex_config.json'
    ]

    for filepath in required_files:
        exists = os.path.exists(filepath)
        results.add_test(f"File exists: {filepath}", exists,
                        "" if exists else "File missing!")


def test_configuration():
    """Test configuration loading and validation"""

    print("\n[TEST 3] Configuration")

    config_path = 'config/forex_config.json'

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        results.add_test("Load config file", True)

        # Check required keys
        required_keys = ['account', 'trading', 'strategy', 'risk_management']
        for key in required_keys:
            has_key = key in config
            results.add_test(f"Config has '{key}' section", has_key)

        # Check critical values
        if 'trading' in config:
            has_pairs = 'pairs' in config['trading'] and len(config['trading']['pairs']) > 0
            results.add_test("Trading pairs configured", has_pairs,
                           f"Pairs: {config['trading'].get('pairs', [])}")

        if 'risk_management' in config:
            risk = config['risk_management'].get('risk_per_trade', 0)
            reasonable_risk = 0 < risk <= 0.05  # 0-5%
            results.add_test("Risk per trade reasonable", reasonable_risk,
                           f"Risk: {risk*100}% (should be 0-5%)")

    except Exception as e:
        results.add_test("Load config file", False, str(e))


def test_data_fetcher():
    """Test OANDA data fetcher"""

    print("\n[TEST 4] OANDA Data Fetcher")

    try:
        from data.oanda_data_fetcher import OandaDataFetcher

        # Initialize (will use practice mode)
        fetcher = OandaDataFetcher(practice=True)
        results.add_test("Initialize data fetcher", True)

        # Test fetching data (will work even without API key - returns None gracefully)
        df = fetcher.get_bars('EUR_USD', 'H1', limit=50)

        if df is not None and not df.empty:
            results.add_test("Fetch EUR/USD data", True,
                           f"Got {len(df)} candles")

            # Check data structure
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            has_cols = all(col in df.columns for col in required_cols)
            results.add_test("Data has OHLCV columns", has_cols)

            # Check data validity
            valid_data = (df['high'] >= df['low']).all() and (df['close'] > 0).all()
            results.add_test("Data is valid", valid_data,
                           "High >= Low, Close > 0")
        else:
            results.add_test("Fetch EUR/USD data", False,
                           "No API key configured (expected in testing)")

    except Exception as e:
        results.add_test("Initialize data fetcher", False, str(e))


def test_execution_engine():
    """Test forex execution engine"""

    print("\n[TEST 5] Forex Execution Engine")

    try:
        from forex_execution_engine import ForexExecutionEngine

        # Initialize in paper trading mode
        engine = ForexExecutionEngine(paper_trading=True)
        results.add_test("Initialize execution engine", True)

        # Test paper order placement
        result = engine.place_market_order(
            pair='EUR_USD',
            direction='LONG',
            units=1000,
            stop_loss=1.08000,
            take_profit=1.09000
        )

        if result and result['success']:
            results.add_test("Place paper order", True,
                           f"Trade ID: {result['trade_id']}")

            # Test position query
            positions = engine.get_open_positions()
            results.add_test("Query open positions", True,
                           f"Found {len(positions)} positions")

            # Test position closing
            if positions:
                closed = engine.close_position(positions[0]['trade_id'], "Test")
                results.add_test("Close position", closed)
        else:
            results.add_test("Place paper order", False, "Order failed")

        # Test position size calculation
        size = engine.calculate_position_size(
            balance=10000,
            risk_percent=0.01,
            stop_pips=30,
            pair='EUR_USD'
        )
        reasonable_size = 1000 <= size <= 10000
        results.add_test("Calculate position size", reasonable_size,
                       f"Size: {size} units (0.{size//1000:02d} lot)")

    except Exception as e:
        results.add_test("Initialize execution engine", False, str(e))


def test_position_manager():
    """Test forex position manager"""

    print("\n[TEST 6] Forex Position Manager")

    try:
        from forex_execution_engine import ForexExecutionEngine
        from forex_position_manager import ForexPositionManager
        from data.oanda_data_fetcher import OandaDataFetcher

        # Initialize components
        engine = ForexExecutionEngine(paper_trading=True)
        fetcher = OandaDataFetcher(practice=True)
        manager = ForexPositionManager(
            execution_engine=engine,
            data_fetcher=fetcher,
            check_interval=10
        )
        results.add_test("Initialize position manager", True)

        # Place a test trade
        trade = engine.place_market_order(
            pair='EUR_USD',
            direction='LONG',
            units=1000,
            stop_loss=1.08000,
            take_profit=1.09000
        )

        if trade:
            # Add to manager
            manager.add_position(trade)
            results.add_test("Add position to manager", True)

            # Check positions
            active = manager.get_active_positions()
            results.add_test("Query active positions", len(active) > 0,
                           f"{len(active)} active")

            # Test position check
            check_results = manager.check_all_positions()
            results.add_test("Check all positions", True)

        # Test logging
        log_path = 'forex_trades/test_positions.json'
        manager.save_positions_log(log_path)
        log_exists = os.path.exists(log_path)
        results.add_test("Save position log", log_exists)

        # Cleanup
        if log_exists:
            os.remove(log_path)

    except Exception as e:
        results.add_test("Initialize position manager", False, str(e))


def test_strategy():
    """Test forex strategy signal detection"""

    print("\n[TEST 7] Forex Strategy (V4 Optimized)")

    try:
        from forex_v4_optimized import ForexV4OptimizedStrategy

        # Initialize strategy
        strategy = ForexV4OptimizedStrategy()
        results.add_test("Initialize strategy", True,
                       f"EMA: {strategy.ema_fast}/{strategy.ema_slow}/{strategy.ema_trend}")

        # Create test data with clear bullish trend
        dates = pd.date_range(start='2025-01-01', periods=300, freq='H')
        np.random.seed(42)

        # Strong uptrend
        trend = np.linspace(1.0800, 1.0900, 300)
        noise = np.random.normal(0, 0.0003, 300)
        close_prices = trend + noise

        df = pd.DataFrame({
            'open': close_prices - np.random.uniform(0, 0.0002, 300),
            'high': close_prices + np.random.uniform(0, 0.0003, 300),
            'low': close_prices - np.random.uniform(0, 0.0003, 300),
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, 300)
        }, index=dates)

        # Test indicator calculation
        df_with_indicators = strategy.calculate_indicators(df)
        has_indicators = all(col in df_with_indicators.columns
                           for col in ['ema_fast', 'ema_slow', 'ema_trend', 'rsi', 'atr', 'adx'])
        results.add_test("Calculate indicators", has_indicators)

        # Test opportunity analysis
        opportunity = strategy.analyze_opportunity(df, 'EUR_USD')

        if opportunity:
            results.add_test("Generate signal", True,
                           f"{opportunity['direction']} (Score: {opportunity['score']:.1f})")

            # Validate opportunity structure
            required_fields = ['symbol', 'direction', 'entry_price', 'stop_loss',
                             'take_profit', 'risk_reward', 'score']
            has_fields = all(field in opportunity for field in required_fields)
            results.add_test("Opportunity has required fields", has_fields)

            # Validate rules
            valid = strategy.validate_rules(opportunity)
            results.add_test("Opportunity passes validation", valid)

        else:
            results.add_test("Generate signal", False,
                           "No signal (may be due to filters)")

    except Exception as e:
        results.add_test("Initialize strategy", False, str(e))


def test_auto_trader():
    """Test main auto-trader orchestration"""

    print("\n[TEST 8] Forex Auto-Trader")

    try:
        from forex_auto_trader import ForexAutoTrader

        # Initialize auto-trader
        trader = ForexAutoTrader(config_path='config/forex_config.json')
        results.add_test("Initialize auto-trader", True)

        # Test configuration
        has_config = trader.config is not None
        results.add_test("Load configuration", has_config)

        # Test safety limits
        stop_reason = trader.check_safety_limits()
        results.add_test("Check safety limits", stop_reason is None,
                       f"Reason: {stop_reason}" if stop_reason else "All limits OK")

        # Test position limits
        can_trade = trader.check_position_limits()
        results.add_test("Check position limits", can_trade)

        # Test single iteration (dry run)
        print("\n  Running single iteration (dry run)...")
        stats = trader.run_iteration()

        results.add_test("Run single iteration", True,
                       f"Signals: {stats['signals_found']}, Trades: {stats['trades_executed']}")

    except Exception as e:
        results.add_test("Initialize auto-trader", False, str(e))


def test_safety_features():
    """Test safety features and limits"""

    print("\n[TEST 9] Safety Features")

    # Test emergency stop file
    stop_file = 'STOP_FOREX_TRADING.txt'

    # Create stop file
    with open(stop_file, 'w') as f:
        f.write("TEST")

    stop_exists = os.path.exists(stop_file)
    results.add_test("Create emergency stop file", stop_exists)

    # Test detection
    try:
        from forex_auto_trader import ForexAutoTrader

        trader = ForexAutoTrader(config_path='config/forex_config.json')
        stop_reason = trader.check_safety_limits()

        results.add_test("Detect emergency stop file", stop_reason == "EMERGENCY_STOP_FILE")

    except Exception as e:
        results.add_test("Detect emergency stop file", False, str(e))

    # Cleanup
    if os.path.exists(stop_file):
        os.remove(stop_file)

    # Test risk limits
    try:
        trader = ForexAutoTrader(config_path='config/forex_config.json')

        # Simulate max trades
        trader.daily_trades = trader.config['trading']['max_daily_trades']
        stop_reason = trader.check_safety_limits()
        results.add_test("Detect max daily trades limit", stop_reason == "MAX_DAILY_TRADES")

    except Exception as e:
        results.add_test("Detect max daily trades limit", False, str(e))


def test_logging():
    """Test logging functionality"""

    print("\n[TEST 10] Logging")

    # Check directories exist or can be created
    log_dirs = ['forex_trades', 'logs']

    for log_dir in log_dirs:
        os.makedirs(log_dir, exist_ok=True)
        dir_exists = os.path.exists(log_dir)
        results.add_test(f"Directory exists: {log_dir}", dir_exists)

    # Test trade logging
    try:
        from forex_auto_trader import ForexAutoTrader

        trader = ForexAutoTrader(config_path='config/forex_config.json')

        # Create test opportunity and execution result
        opportunity = {
            'symbol': 'EUR_USD',
            'direction': 'LONG',
            'entry_price': 1.08500,
            'stop_loss': 1.08200,
            'take_profit': 1.09100,
            'stop_pips': 30.0,
            'target_pips': 60.0,
            'score': 9.5
        }

        execution_result = {
            'trade_id': 'TEST_001',
            'entry_price': 1.08500,
            'timestamp': datetime.now().isoformat(),
            'units': 1000,
            'mode': 'TEST'
        }

        # Log trade
        trader.log_trade(opportunity, execution_result)

        # Check log file exists
        date_str = datetime.now().strftime('%Y%m%d')
        log_file = f'forex_trades/execution_log_{date_str}.json'
        log_exists = os.path.exists(log_file)

        results.add_test("Create trade log", log_exists)

        if log_exists:
            # Verify log contents
            with open(log_file, 'r') as f:
                log_data = json.load(f)

            has_trades = 'trades' in log_data and len(log_data['trades']) > 0
            results.add_test("Log contains trades", has_trades)

    except Exception as e:
        results.add_test("Create trade log", False, str(e))


def test_batch_scripts():
    """Test that batch scripts exist"""

    print("\n[TEST 11] Automation Scripts")

    batch_files = [
        'START_FOREX_TRADER.bat',
        'START_FOREX_BACKGROUND.bat',
        'STOP_FOREX_TRADER.bat',
        'CHECK_FOREX_STATUS.bat',
        'SETUP_FOREX_AUTOMATION.bat'
    ]

    for batch_file in batch_files:
        exists = os.path.exists(batch_file)
        results.add_test(f"Batch script: {batch_file}", exists)


def test_documentation():
    """Test that documentation exists"""

    print("\n[TEST 12] Documentation")

    doc_files = [
        'FOREX_TRADING_GUIDE.md'
    ]

    for doc_file in doc_files:
        exists = os.path.exists(doc_file)
        results.add_test(f"Documentation: {doc_file}", exists)


def main():
    """Run all tests"""

    print("="*70)
    print("FOREX TRADING SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Run all tests
    test_imports()
    test_file_structure()
    test_configuration()
    test_data_fetcher()
    test_execution_engine()
    test_position_manager()
    test_strategy()
    test_auto_trader()
    test_safety_features()
    test_logging()
    test_batch_scripts()
    test_documentation()

    # Print summary
    all_passed = results.print_summary()

    if all_passed:
        print("\n" + "="*70)
        print("ALL TESTS PASSED!")
        print("="*70)
        print("\nSystem is ready for trading!")
        print("\nNext steps:")
        print("1. Set up OANDA API credentials in config/forex_config.json")
        print("2. Run in paper trading mode for 1 week: python forex_auto_trader.py")
        print("3. Review results and adjust if needed")
        print("4. When confident, enable practice trading")
        print("5. After success in practice, consider live trading")
        print("\nStart trading: python forex_auto_trader.py")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("SOME TESTS FAILED")
        print("="*70)
        print("\nPlease fix the failed tests before trading.")
        print("Check error messages above for details.")
        print("="*70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
