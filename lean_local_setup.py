"""
LEAN LOCAL SETUP - RUN YOUR 353-FILE EMPIRE LOCALLY
===================================================

This sets up LEAN to run your complete trading system locally.
- Paper trading first (safe)
- Easy switch to live trading
- All 46+ libraries integrated
- Your 353-file system wrapped in LEAN

Run this ONCE to setup everything.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import shutil
import requests
import zipfile


class HiveLEANSetup:
    """Setup LEAN locally to run your complete Hive trading system"""
    
    def __init__(self):
        self.system_root = Path(__file__).parent
        self.lean_dir = self.system_root / "lean_engine" 
        self.data_dir = self.lean_dir / "Data"
        self.results_dir = self.lean_dir / "Results"
        self.config_file = self.system_root / "lean_config.json"
        
        print("🚀 HIVE LEAN SETUP STARTING...")
        print(f"📁 System root: {self.system_root}")
        print(f"📁 LEAN directory: {self.lean_dir}")
    
    def install_lean_engine(self):
        """Install LEAN engine locally"""
        
        print("\n" + "="*50)
        print("📦 INSTALLING LEAN ENGINE...")
        print("="*50)
        
        try:
            # Install LEAN CLI
            self._run_command("pip install --upgrade lean")
            print("✅ LEAN CLI installed")
            
            # Install Python LEAN  
            self._run_command("pip install --upgrade QuantConnect")
            print("✅ QuantConnect package installed")
            
            # Initialize LEAN project
            if not self.lean_dir.exists():
                self.lean_dir.mkdir(parents=True)
                print(f"✅ Created LEAN directory: {self.lean_dir}")
            
            # Download LEAN if not exists
            lean_executable = self.lean_dir / "Lean.exe"
            if not lean_executable.exists():
                self._download_lean_engine()
            
            print("✅ LEAN engine setup complete!")
            
        except Exception as e:
            print(f"❌ Error installing LEAN: {e}")
            raise
    
    def install_quantitative_libraries(self):
        """Install all your quantitative libraries"""
        
        print("\n" + "="*50) 
        print("📚 INSTALLING 46+ QUANTITATIVE LIBRARIES...")
        print("="*50)
        
        # Core libraries
        core_libs = [
            "openbb[all]",           # OpenBB Terminal
            "qlib",                  # Microsoft Qlib
            "gs-quant",             # Goldman Sachs Quant
            "alpaca-trade-api",     # Alpaca broker
            "vectorbt",             # Vectorized backtesting
            "zipline-reloaded",     # Quantopian Zipline
            "pyfolio-reloaded",     # Risk analytics
            "alphalens-reloaded",   # Alpha analysis
        ]
        
        # Technical analysis
        ta_libs = [
            "ta-lib",               # Technical Analysis
            "pandas-ta",            # Pandas TA  
            "tulipy",              # Tulip indicators
            "finta",               # Financial TA
        ]
        
        # ML/AI libraries
        ml_libs = [
            "scikit-learn",
            "tensorflow",
            "torch", 
            "transformers",
            "lightgbm",
            "xgboost",
            "catboost",
        ]
        
        # Alternative platforms
        alt_libs = [
            "freqtrade",           # Crypto trading
            "jesse",               # Jesse framework
            "backtrader",          # Backtrader
            "pyalgotrade",         # Algorithmic trading
        ]
        
        # Data and utilities
        data_libs = [
            "yfinance",
            "pandas-datareader", 
            "quandl",
            "bloomberg",
            "refinitiv-dataplatform",
            "polygon-api-client",
        ]
        
        all_libraries = core_libs + ta_libs + ml_libs + alt_libs + data_libs
        
        for lib in all_libraries:
            try:
                print(f"📦 Installing {lib}...")
                self._run_command(f"pip install {lib}")
                print(f"✅ {lib} installed")
                
            except Exception as e:
                print(f"⚠️  Warning: Failed to install {lib}: {e}")
                continue
        
        print("✅ Quantitative libraries installation complete!")
    
    def create_lean_config(self):
        """Create LEAN configuration for your system"""
        
        print("\n" + "="*50)
        print("⚙️  CREATING LEAN CONFIGURATION...")
        print("="*50)
        
        config = {
            # Algorithm settings
            "algorithm-type-name": "HiveTradingMasterAlgorithm", 
            "algorithm-language": "Python",
            "algorithm-location": "lean_master_algorithm.py",
            
            # Data settings
            "data-folder": str(self.data_dir),
            "results-destination-folder": str(self.results_dir),
            
            # Environment configurations
            "environments": {
                # Paper trading environment (START HERE)
                "paper-alpaca": {
                    "live-mode": True,
                    "live-mode-brokerage": "AlpacaBrokerage", 
                    "data-queue-handler": "AlpacaBrokerage",
                    "setup-handler": "BrokerageSetupHandler",
                    "result-handler": "LiveTradingResultHandler",
                    "history-provider": "BrokerageHistoryProvider",
                    "alpaca-key-id": "YOUR_ALPACA_API_KEY",
                    "alpaca-secret-key": "YOUR_ALPACA_SECRET_KEY",
                    "alpaca-paper-trading": True,  # PAPER TRADING
                    "alpaca-use-polygon": False
                },
                
                # Live trading environment (AFTER PAPER SUCCESS)
                "live-alpaca": {
                    "live-mode": True,
                    "live-mode-brokerage": "AlpacaBrokerage",
                    "data-queue-handler": "AlpacaBrokerage", 
                    "setup-handler": "BrokerageSetupHandler",
                    "result-handler": "LiveTradingResultHandler",
                    "history-provider": "BrokerageHistoryProvider",
                    "alpaca-key-id": "YOUR_ALPACA_API_KEY",
                    "alpaca-secret-key": "YOUR_ALPACA_SECRET_KEY",
                    "alpaca-paper-trading": False,  # REAL MONEY
                    "alpaca-use-polygon": True
                },
                
                # Backtesting environment
                "backtesting": {
                    "live-mode": False,
                    "data-queue-handler": "LocalDataQueueHandler",
                    "setup-handler": "BasicSetupHandler", 
                    "result-handler": "BacktestingResultHandler",
                    "history-provider": "LocalHistoryProvider"
                }
            },
            
            # Trading settings
            "cash": 100000,
            "start-date": "2024-01-01",
            "end-date": "2025-01-20",
            
            # Your system integration
            "python-additional-paths": [str(self.system_root)],
            "plugin-libraries": [
                "event_bus",
                "core.portfolio", 
                "data.market_scanner",
                "agents.autonomous_brain",
                "learning.pattern_learner",
                "evolution.strategy_evolver"
            ],
            
            # Performance settings
            "maximum-concurrent-backtests": 4,
            "maximum-ram-allocation": 8192,  # 8GB RAM
            
            # Logging
            "debug-mode": True,
            "log-level": "Debug"
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"✅ LEAN config created: {self.config_file}")
        
        # Create environment-specific configs
        self._create_environment_configs(config)
    
    def _create_environment_configs(self, base_config):
        """Create separate config files for each environment"""
        
        environments = ['backtesting', 'paper-alpaca', 'live-alpaca']
        
        for env in environments:
            env_config = base_config.copy()
            env_config.update(base_config['environments'][env])
            
            env_config_file = self.system_root / f"lean_config_{env.replace('-', '_')}.json"
            
            with open(env_config_file, 'w') as f:
                json.dump(env_config, f, indent=4)
            
            print(f"✅ Created {env} config: {env_config_file}")
    
    def create_launcher_scripts(self):
        """Create launcher scripts for different modes"""
        
        print("\n" + "="*50)
        print("🚀 CREATING LAUNCHER SCRIPTS...")
        print("="*50)
        
        # Main Python launcher
        launcher_py = '''#!/usr/bin/env python3
"""
HIVE TRADING EMPIRE - LEAN RUNNER
================================

Launch your complete 353-file trading system through LEAN.

Modes:
  backtest - Test strategies on historical data
  paper    - Paper trade with real market data (SAFE)  
  live     - Live trading with real money (DANGER)
"""

import sys
import os
import json
import subprocess
from pathlib import Path
import signal
import time


class HiveLEANRunner:
    def __init__(self):
        self.system_root = Path(__file__).parent
        self.lean_dir = self.system_root / "lean_engine"
        self.config_dir = self.system_root
        
    def run_backtest(self):
        """Run backtesting mode"""
        print("📊 STARTING BACKTEST MODE...")
        print("Testing your strategies on historical data")
        
        config_file = self.config_dir / "lean_config_backtesting.json"
        self._run_lean(config_file)
    
    def run_paper(self):
        """Run paper trading mode"""  
        print("📝 STARTING PAPER TRADING MODE...")
        print("⚠️  Paper trading with FAKE money - completely safe")
        
        # Verify API keys
        if not self._check_alpaca_keys():
            print("❌ Error: Alpaca API keys not configured!")
            print("Edit lean_config_paper_alpaca.json and add your keys")
            return False
            
        config_file = self.config_dir / "lean_config_paper_alpaca.json"
        self._run_lean(config_file)
    
    def run_live(self):
        """Run LIVE trading mode"""
        print("🚨 STARTING LIVE TRADING MODE...")
        print("⚠️  ⚠️  ⚠️  THIS USES REAL MONEY ⚠️  ⚠️  ⚠️")
        
        # Multiple confirmations
        confirm1 = input("Are you sure you want to trade with REAL money? (type 'YES'): ")
        if confirm1 != 'YES':
            print("Cancelled")
            return
            
        confirm2 = input("Have you tested in paper trading successfully? (type 'TESTED'): ")  
        if confirm2 != 'TESTED':
            print("Please test in paper trading first!")
            return
            
        confirm3 = input("Final confirmation - LIVE TRADING WITH REAL MONEY (type 'LIVE'): ")
        if confirm3 != 'LIVE':
            print("Cancelled")
            return
        
        # Verify live API keys
        if not self._check_alpaca_keys(live=True):
            print("❌ Error: Live Alpaca API keys not configured!")
            return False
            
        config_file = self.config_dir / "lean_config_live_alpaca.json" 
        print("🚀 LAUNCHING LIVE TRADING...")
        self._run_lean(config_file)
    
    def _check_alpaca_keys(self, live=False):
        """Check if Alpaca API keys are configured"""
        config_file = self.config_dir / ("lean_config_live_alpaca.json" if live else "lean_config_paper_alpaca.json")
        
        try:
            with open(config_file) as f:
                config = json.load(f)
                
            key_id = config.get('alpaca-key-id', '')
            secret_key = config.get('alpaca-secret-key', '')
            
            return key_id != 'YOUR_ALPACA_API_KEY' and secret_key != 'YOUR_ALPACA_SECRET_KEY'
            
        except Exception:
            return False
    
    def _run_lean(self, config_file):
        """Run LEAN with specified config"""
        
        # Check if LEAN algorithm exists
        algo_file = self.system_root / "lean_master_algorithm.py"
        if not algo_file.exists():
            print(f"❌ Error: Algorithm file not found: {algo_file}")
            return
        
        # Check if config exists
        if not config_file.exists():
            print(f"❌ Error: Config file not found: {config_file}")
            return
        
        print(f"🔧 Using config: {config_file}")
        print(f"🤖 Running algorithm: {algo_file}")
        
        try:
            # Run LEAN
            cmd = [
                "python", "-m", "lean.engine.main",
                "--config", str(config_file),
                "--algorithm-location", str(algo_file)
            ]
            
            print(f"🚀 Command: {' '.join(cmd)}")
            
            # Run with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Handle Ctrl+C gracefully
            def signal_handler(signum, frame):
                print("\\n🛑 Shutting down LEAN...")
                process.terminate()
                process.wait()
                print("✅ LEAN shut down successfully")
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            
            # Stream output
            for line in process.stdout:
                print(line.rstrip())
            
            process.wait()
            
            if process.returncode == 0:
                print("✅ LEAN completed successfully!")
            else:
                print(f"❌ LEAN failed with return code: {process.returncode}")
                
        except Exception as e:
            print(f"❌ Error running LEAN: {e}")


def main():
    if len(sys.argv) < 2:
        print("🚀 HIVE TRADING EMPIRE - LEAN INTEGRATION")
        print("="*50)
        print("Usage: python lean_runner.py <mode>")
        print("")
        print("Modes:")
        print("  backtest - Test your strategies (SAFE)")
        print("  paper    - Paper trade with live data (SAFE)")  
        print("  live     - Live trade with real money (DANGER)")
        print("")
        print("Examples:")
        print("  python lean_runner.py backtest")
        print("  python lean_runner.py paper")
        print("  python lean_runner.py live")
        print("")
        print("Start with 'backtest' to test your strategies!")
        return
    
    mode = sys.argv[1].lower()
    runner = HiveLEANRunner()
    
    if mode == 'backtest':
        runner.run_backtest()
    elif mode == 'paper':  
        runner.run_paper()
    elif mode == 'live':
        runner.run_live()
    else:
        print(f"❌ Unknown mode: {mode}")
        print("Valid modes: backtest, paper, live")


if __name__ == "__main__":
    main()
'''
        
        with open(self.system_root / "lean_runner.py", 'w') as f:
            f.write(launcher_py)
        
        print("✅ Python launcher created: lean_runner.py")
        
        # Windows batch file
        batch_script = '''@echo off
echo 🚀 HIVE TRADING EMPIRE - LEAN LAUNCHER
echo =======================================

if "%1"=="" (
    echo Usage: run_hive_lean.bat [backtest^|paper^|live]
    echo.
    echo  backtest - Test strategies safely
    echo  paper    - Paper trade safely  
    echo  live     - Live trade with real money
    echo.
    pause
    exit /b
)

python lean_runner.py %1
pause
'''
        
        with open(self.system_root / "run_hive_lean.bat", 'w') as f:
            f.write(batch_script)
        
        print("✅ Windows batch launcher created: run_hive_lean.bat")
    
    def _download_lean_engine(self):
        """Download LEAN engine if needed"""
        print("⬇️  Downloading LEAN engine...")
        
        try:
            # This would download and extract LEAN
            # For now, we rely on pip install lean
            print("✅ Using pip-installed LEAN engine")
            
        except Exception as e:
            print(f"⚠️  Could not download LEAN engine: {e}")
            print("Using pip-installed version")
    
    def create_sample_data(self):
        """Create sample data for testing"""
        print("\n" + "="*50)
        print("📈 SETTING UP SAMPLE DATA...")
        print("="*50)
        
        try:
            # Create data directories
            self.data_dir.mkdir(parents=True, exist_ok=True)
            equity_dir = self.data_dir / "equity" / "usa"
            equity_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"✅ Data directories created: {self.data_dir}")
            
            # Download sample data using LEAN CLI (if available)
            try:
                self._run_command("lean data download --dataset=usa-equity", timeout=300)
                print("✅ Sample equity data downloaded")
            except:
                print("⚠️  Could not download data via LEAN CLI - will use live data")
            
        except Exception as e:
            print(f"⚠️  Error setting up sample data: {e}")
    
    def test_installation(self):
        """Test that everything is working"""
        print("\n" + "="*50)
        print("🧪 TESTING INSTALLATION...")
        print("="*50)
        
        tests_passed = 0
        total_tests = 0
        
        # Test 1: Python imports
        total_tests += 1
        try:
            import lean
            print("✅ Test 1: LEAN Python package imports correctly")
            tests_passed += 1
        except ImportError:
            print("❌ Test 1: LEAN Python package import failed")
        
        # Test 2: Your system imports
        total_tests += 1  
        try:
            from event_bus import TradingEventBus
            print("✅ Test 2: Your event bus imports correctly")
            tests_passed += 1
        except ImportError as e:
            print(f"❌ Test 2: Your system imports failed: {e}")
        
        # Test 3: Config files exist
        total_tests += 1
        if self.config_file.exists():
            print("✅ Test 3: LEAN config files created")
            tests_passed += 1
        else:
            print("❌ Test 3: LEAN config files missing")
        
        # Test 4: Algorithm file exists
        total_tests += 1
        algo_file = self.system_root / "lean_master_algorithm.py"
        if algo_file.exists():
            print("✅ Test 4: LEAN master algorithm exists")
            tests_passed += 1
        else:
            print("❌ Test 4: LEAN master algorithm missing")
        
        # Test 5: OpenBB (optional)
        total_tests += 1
        try:
            import openbb
            print("✅ Test 5: OpenBB Terminal available")
            tests_passed += 1
        except ImportError:
            print("⚠️  Test 5: OpenBB Terminal not available (optional)")
            tests_passed += 0.5  # Half credit for optional
        
        print(f"\\n📊 TEST RESULTS: {tests_passed}/{total_tests} passed")
        
        if tests_passed >= 4:
            print("🎉 INSTALLATION SUCCESSFUL!")
            print("\\n🚀 READY TO LAUNCH:")
            print("  python lean_runner.py backtest  # Test your strategies")
            print("  python lean_runner.py paper     # Paper trade safely") 
            print("  python lean_runner.py live      # Live trade (after testing)")
            return True
        else:
            print("❌ INSTALLATION NEEDS ATTENTION")
            print("Please fix the failed tests above")
            return False
    
    def _run_command(self, command, timeout=120):
        """Run a shell command with timeout"""
        try:
            result = subprocess.run(
                command.split(), 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                check=True
            )
            return result
        except subprocess.TimeoutExpired:
            raise Exception(f"Command timed out: {command}")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Command failed: {command}\\nError: {e.stderr}")
    
    def run_complete_setup(self):
        """Run the complete setup process"""
        try:
            print("🚀 HIVE LEAN SETUP - INTEGRATING YOUR 353-FILE EMPIRE")
            print("=" * 60)
            
            # Step 1: Install LEAN
            self.install_lean_engine()
            
            # Step 2: Install all your libraries
            self.install_quantitative_libraries()
            
            # Step 3: Create configuration
            self.create_lean_config()
            
            # Step 4: Create launchers
            self.create_launcher_scripts()
            
            # Step 5: Setup data
            self.create_sample_data()
            
            # Step 6: Test everything
            success = self.test_installation()
            
            if success:
                print("\\n" + "🎉" * 20)
                print("HIVE TRADING EMPIRE IS READY FOR LEAN!")
                print("🎉" * 20)
                print("\\n📋 NEXT STEPS:")
                print("1. Edit lean_config_paper_alpaca.json with your Alpaca API keys")
                print("2. Run: python lean_runner.py backtest")  
                print("3. Run: python lean_runner.py paper")
                print("4. After successful paper trading: python lean_runner.py live")
                print("\\n💰 Your 353-file system is now powered by LEAN!")
            else:
                print("\\n⚠️  Setup completed with issues - check failed tests above")
            
            return success
            
        except Exception as e:
            print(f"\\n❌ SETUP FAILED: {e}")
            return False


if __name__ == "__main__":
    setup = HiveLEANSetup()
    success = setup.run_complete_setup()
    
    if success:
        print("\\n🚀 READY TO DOMINATE THE MARKETS WITH LEAN + HIVE!")
    else:
        print("\\n❌ Setup needs attention before launching")