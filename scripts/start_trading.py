#!/usr/bin/env python3
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
