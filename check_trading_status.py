#!/usr/bin/env python3
"""Check if trading systems are running"""
import os
import psutil
import json

print("\n" + "="*70)
print("TRADING EMPIRE STATUS CHECK")
print("="*70)

# Check PIDs
forex_running = False
scanner_running = False

try:
    with open("forex_elite.pid", "r") as f:
        forex_pid = int(f.read().strip())
    if psutil.pid_exists(forex_pid):
        proc = psutil.Process(forex_pid)
        if "python" in proc.name().lower():
            forex_running = True
            print(f"\n[FOREX ELITE]")
            print(f"  Status: [OK] RUNNING")
            print(f"  PID: {forex_pid}")
            print(f"  CPU: {proc.cpu_percent(interval=0.5):.1f}%")
            print(f"  Memory: {proc.memory_info().rss / 1024 / 1024:.1f} MB")
except:
    pass

if not forex_running:
    print(f"\n[FOREX ELITE]")
    print(f"  Status: [X] NOT RUNNING")

try:
    with open("scanner.pid", "r") as f:
        scanner_pid = int(f.read().strip())
    if psutil.pid_exists(scanner_pid):
        proc = psutil.Process(scanner_pid)
        if "python" in proc.name().lower():
            scanner_running = True
            print(f"\n[OPTIONS SCANNER]")
            print(f"  Status: [OK] RUNNING")
            print(f"  PID: {scanner_pid}")
            print(f"  CPU: {proc.cpu_percent(interval=0.5):.1f}%")
            print(f"  Memory: {proc.memory_info().rss / 1024 / 1024:.1f} MB")
except:
    pass

if not scanner_running:
    print(f"\n[OPTIONS SCANNER]")
    print(f"  Status: [X] NOT RUNNING")

# Check accounts
print(f"\n[ACCOUNTS]")
try:
    from alpaca.trading.client import TradingClient
    from dotenv import load_dotenv
    load_dotenv()

    client = TradingClient(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        paper=True
    )
    account = client.get_account()
    print(f"  Alpaca (Options): PA3MS5F52RNL")
    print(f"    Equity: ${float(account.equity):,.2f}")
    print(f"    Options BP: ${float(account.options_buying_power):,.2f}")
except Exception as e:
    print(f"  Alpaca: [X] {e}")

try:
    import v20
    api = v20.Context(
        'api-fxpractice.oanda.com',
        443,
        token=os.getenv('OANDA_API_KEY')
    )
    response = api.account.get(os.getenv('OANDA_ACCOUNT_ID'))
    if response.status == 200:
        account = response.body['account']
        print(f"  OANDA (Forex): {account.id}")
        print(f"    Balance: ${float(account.balance):,.2f}")
        print(f"    Open Positions: {account.openPositionCount}")
except Exception as e:
    print(f"  OANDA: [X] {e}")

print(f"\n" + "="*70)
print(f"SUMMARY")
print(f"="*70)
print(f"  Forex Elite: {'[OK] RUNNING' if forex_running else '[X] STOPPED'}")
print(f"  Options Scanner: {'[OK] RUNNING' if scanner_running else '[X] STOPPED'}")
print(f"\n  Both systems: {'[OK] OPERATIONAL' if (forex_running and scanner_running) else '[!] CHECK LOGS'}")
print("="*70 + "\n")
