#!/usr/bin/env python3
"""
FRIDAY MORNING SYSTEM CHECK
===========================
Quick verification that all systems are ready for Day 3 trading
"""

import sys
from datetime import datetime
from colorama import init, Fore, Style

init(autoreset=True)

def check_imports():
    """Verify all critical imports work"""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}FRIDAY SYSTEM CHECK - Day 3 Pre-Market Verification")
    print(f"{Fore.CYAN}{'='*70}\n")

    checks_passed = 0
    checks_total = 0

    # 1. Core dependencies
    print(f"{Fore.YELLOW}[1/10] Checking core dependencies...")
    checks_total += 1
    try:
        import numpy as np
        import pandas as pd
        from dotenv import load_dotenv
        import os
        print(f"{Fore.GREEN}  [OK] Core dependencies OK")
        checks_passed += 1
    except Exception as e:
        print(f"{Fore.RED}  [FAIL] Core dependencies FAILED: {e}")

    # 2. ML/DL libraries
    print(f"\n{Fore.YELLOW}[2/10] Checking ML/DL systems...")
    checks_total += 1
    try:
        import xgboost as xgb
        import lightgbm as lgb
        import torch
        from sklearn.ensemble import RandomForestClassifier
        print(f"{Fore.GREEN}  ✓ XGBoost v{xgb.__version__}")
        print(f"{Fore.GREEN}  ✓ LightGBM v{lgb.__version__}")
        print(f"{Fore.GREEN}  ✓ PyTorch v{torch.__version__}")

        # Check GPU
        if torch.cuda.is_available():
            print(f"{Fore.GREEN}  ✓ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"{Fore.YELLOW}  ⚠ GPU not available (CPU mode)")

        checks_passed += 1
    except Exception as e:
        print(f"{Fore.RED}  ✗ ML/DL systems FAILED: {e}")

    # 3. RL libraries
    print(f"\n{Fore.YELLOW}[3/10] Checking RL systems...")
    checks_total += 1
    try:
        from stable_baselines3 import PPO, A2C, DQN
        print(f"{Fore.GREEN}  ✓ Stable-Baselines3 (PPO, A2C, DQN)")
        checks_passed += 1
    except Exception as e:
        print(f"{Fore.RED}  ✗ RL systems FAILED: {e}")

    # 4. Technical indicators
    print(f"\n{Fore.YELLOW}[4/10] Checking technical indicators...")
    checks_total += 1
    try:
        import pandas_ta as pta
        import ta
        print(f"{Fore.GREEN}  ✓ pandas-ta (150+ indicators)")
        print(f"{Fore.GREEN}  ✓ ta library")
        checks_passed += 1
    except Exception as e:
        print(f"{Fore.RED}  ✗ Technical indicators FAILED: {e}")

    # 5. Alpaca API
    print(f"\n{Fore.YELLOW}[5/10] Checking Alpaca connection...")
    checks_total += 1
    try:
        load_dotenv('.env.paper')
        from alpaca.trading.client import TradingClient

        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')

        if not api_key or not secret_key:
            print(f"{Fore.RED}  ✗ API keys not found in .env.paper")
        else:
            client = TradingClient(api_key, secret_key, paper=True)
            account = client.get_account()

            print(f"{Fore.GREEN}  ✓ Alpaca API connected")
            print(f"{Fore.GREEN}  ✓ Account value: ${float(account.portfolio_value):,.2f}")
            print(f"{Fore.GREEN}  ✓ Cash available: ${float(account.cash):,.2f}")
            checks_passed += 1
    except Exception as e:
        print(f"{Fore.RED}  ✗ Alpaca connection FAILED: {e}")

    # 6. Time series momentum
    print(f"\n{Fore.YELLOW}[6/10] Checking time series momentum...")
    checks_total += 1
    try:
        from time_series_momentum_strategy import TimeSeriesMomentumStrategy
        momentum = TimeSeriesMomentumStrategy()
        print(f"{Fore.GREEN}  ✓ Time series momentum strategy loaded")
        checks_passed += 1
    except Exception as e:
        print(f"{Fore.RED}  ✗ Momentum strategy FAILED: {e}")

    # 7. Options validator
    print(f"\n{Fore.YELLOW}[7/10] Checking options validator...")
    checks_total += 1
    try:
        from enhanced_options_validator import EnhancedOptionsValidator
        validator = EnhancedOptionsValidator()
        print(f"{Fore.GREEN}  ✓ Black-Scholes options validator loaded")
        checks_passed += 1
    except Exception as e:
        print(f"{Fore.RED}  ✗ Options validator FAILED: {e}")

    # 8. Portfolio manager
    print(f"\n{Fore.YELLOW}[8/10] Checking portfolio manager...")
    checks_total += 1
    try:
        from enhanced_portfolio_manager import EnhancedPortfolioManager
        portfolio = EnhancedPortfolioManager()
        print(f"{Fore.GREEN}  ✓ Portfolio manager loaded")
        checks_passed += 1
    except Exception as e:
        print(f"{Fore.RED}  ✗ Portfolio manager FAILED: {e}")

    # 9. Continuous scanner
    print(f"\n{Fore.YELLOW}[9/10] Checking continuous scanner...")
    checks_total += 1
    try:
        # Don't import to avoid running it, just check file exists
        import os
        if os.path.exists('continuous_week1_scanner.py'):
            print(f"{Fore.GREEN}  ✓ continuous_week1_scanner.py exists")
            checks_passed += 1
        else:
            print(f"{Fore.RED}  ✗ continuous_week1_scanner.py not found")
    except Exception as e:
        print(f"{Fore.RED}  ✗ Scanner check FAILED: {e}")

    # 10. Mission control
    print(f"\n{Fore.YELLOW}[10/10] Checking mission control logger...")
    checks_total += 1
    try:
        from mission_control_logger import MissionControlLogger
        print(f"{Fore.GREEN}  ✓ Mission control logger loaded")
        checks_passed += 1
    except Exception as e:
        print(f"{Fore.RED}  ✗ Mission control FAILED: {e}")

    # Summary
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}SYSTEM CHECK SUMMARY")
    print(f"{Fore.CYAN}{'='*70}\n")

    if checks_passed == checks_total:
        print(f"{Fore.GREEN}{Style.BRIGHT}✓ ALL SYSTEMS READY ({checks_passed}/{checks_total})")
        print(f"{Fore.GREEN}{'='*70}\n")
        print(f"{Fore.WHITE}Launch command: {Fore.YELLOW}FRIDAY_LAUNCH.bat")
        print(f"{Fore.WHITE}Or manually: {Fore.YELLOW}python continuous_week1_scanner.py\n")
        return True
    else:
        print(f"{Fore.RED}{Style.BRIGHT}✗ SOME SYSTEMS FAILED ({checks_passed}/{checks_total})")
        print(f"{Fore.RED}{'='*70}\n")
        print(f"{Fore.YELLOW}Fix errors above before launching scanner\n")
        return False


def check_positions():
    """Quick check of current positions"""
    print(f"{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}CURRENT POSITIONS (Thursday Close)")
    print(f"{Fore.CYAN}{'='*70}\n")

    try:
        from dotenv import load_dotenv
        import os
        from alpaca.trading.client import TradingClient

        load_dotenv('.env.paper')
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')

        client = TradingClient(api_key, secret_key, paper=True)
        positions = client.get_all_positions()

        if not positions:
            print(f"{Fore.YELLOW}No open positions")
            return

        winning = 0
        losing = 0
        total_pl = 0

        for pos in positions:
            pl = float(pos.unrealized_pl)
            pl_pct = float(pos.unrealized_plpc) * 100
            total_pl += pl

            if pl >= 0:
                winning += 1
                color = Fore.GREEN
                icon = "WIN"
            else:
                losing += 1
                color = Fore.RED
                icon = "LOSS"

            print(f"{color}[{icon}] {pos.symbol}")
            print(f"{color}  P&L: ${pl:+,.2f} ({pl_pct:+.1f}%)")
            print(f"{color}  Qty: {pos.qty} @ ${float(pos.avg_entry_price):.2f}\n")

        print(f"{Fore.CYAN}{'='*70}")
        print(f"{Fore.WHITE}Total Positions: {len(positions)} ({winning} winning, {losing} losing)")
        print(f"{Fore.WHITE}Total Unrealized P&L: {Fore.GREEN if total_pl >= 0 else Fore.RED}${total_pl:+,.2f}")
        print(f"{Fore.CYAN}{'='*70}\n")

    except Exception as e:
        print(f"{Fore.RED}Error checking positions: {e}\n")


def main():
    """Run complete system check"""

    # Header
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}")
    print("="*70)
    print("                    FRIDAY DAY 3 SYSTEM CHECK                       ")
    print("                    Momentum-Enhanced Scanner                       ")
    print("="*70)
    print(f"{Style.RESET_ALL}")

    print(f"{Fore.WHITE}Date: {datetime.now().strftime('%A, %B %d, %Y')}")
    print(f"{Fore.WHITE}Time: {datetime.now().strftime('%I:%M:%S %p PDT')}")
    print(f"{Fore.WHITE}Week: 1/4 | Day: 3/5\n")

    # Run checks
    all_good = check_imports()

    # Check positions
    check_positions()

    # Final status
    if all_good:
        print(f"{Fore.GREEN}{Style.BRIGHT}>>> READY FOR FRIDAY TRADING <<<\n")
        print(f"{Fore.WHITE}Next steps:")
        print(f"{Fore.YELLOW}  1. Review FRIDAY_PREMARKET_CHECKLIST.md")
        print(f"{Fore.YELLOW}  2. Launch scanner at 6:30 AM PDT with FRIDAY_LAUNCH.bat")
        print(f"{Fore.YELLOW}  3. Monitor for 4.0+ momentum-enhanced opportunities\n")
    else:
        print(f"{Fore.RED}{Style.BRIGHT}>>> FIX ERRORS BEFORE TRADING <<<\n")

    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
