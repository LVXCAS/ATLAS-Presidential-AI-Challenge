#!/usr/bin/env python3
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
        print("[BACKTEST] STARTING BACKTEST MODE...")
        print("Testing your strategies on historical data")
        
        config_file = self.config_dir / "lean_config_backtesting.json"
        self._run_lean(config_file)
    
    def run_paper(self):
        """Run paper trading mode"""  
        print("[PAPER] STARTING PAPER TRADING MODE...")
        print("WARNING: Paper trading with FAKE money - completely safe")
        
        # Verify API keys
        if not self._check_alpaca_keys():
            print("ERROR: Alpaca API keys not configured!")
            print("Edit lean_config_paper_alpaca.json and add your keys")
            return False
            
        config_file = self.config_dir / "lean_config_paper_alpaca.json"
        self._run_lean(config_file)
    
    def run_live(self):
        """Run LIVE trading mode"""
        print("[LIVE] STARTING LIVE TRADING MODE...")
        print("WARNING: THIS USES REAL MONEY - EXTREME CAUTION REQUIRED")
        
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
            print("ERROR: Live Alpaca API keys not configured!")
            return False
            
        config_file = self.config_dir / "lean_config_live_alpaca.json" 
        print("[LIVE] LAUNCHING LIVE TRADING...")
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
        algo_file = self.system_root / "lean_simple_test_algorithm.py"
        if not algo_file.exists():
            print(f"ERROR: Algorithm file not found: {algo_file}")
            return
        
        # Check if config exists
        if not config_file.exists():
            print(f"ERROR: Config file not found: {config_file}")
            return
        
        print(f"[CONFIG] Using config: {config_file}")
        print(f"[ALGO] Running algorithm: {algo_file}")
        
        try:
            # Copy algorithm to lean_engine directory
            import shutil
            lean_algo_file = self.lean_dir / "lean_simple_test_algorithm.py"
            shutil.copy2(algo_file, lean_algo_file)
            
            # Change to lean_engine directory and run backtest
            os.chdir(self.lean_dir)
            
            # Run LEAN backtest from the engine directory
            cmd = [
                "lean", "backtest", ".",
                "--config", str(config_file.resolve())
            ]
            
            print(f"[CMD] Command: {' '.join(cmd)}")
            
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
                print("\n[STOP] Shutting down LEAN...")
                process.terminate()
                process.wait()
                print("[SUCCESS] LEAN shut down successfully")
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            
            # Stream output
            for line in process.stdout:
                print(line.rstrip())
            
            process.wait()
            
            if process.returncode == 0:
                print("[SUCCESS] LEAN completed successfully!")
            else:
                print(f"[ERROR] LEAN failed with return code: {process.returncode}")
                
        except Exception as e:
            print(f"[ERROR] Error running LEAN: {e}")


def main():
    if len(sys.argv) < 2:
        print("[HIVE] HIVE TRADING EMPIRE - LEAN INTEGRATION")
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
        print(f"[ERROR] Unknown mode: {mode}")
        print("Valid modes: backtest, paper, live")


if __name__ == "__main__":
    main()