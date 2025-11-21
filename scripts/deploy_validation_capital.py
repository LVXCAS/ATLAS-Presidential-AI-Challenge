#!/usr/bin/env python3
"""
VALIDATION CAPITAL DEPLOYMENT - PHASE 1
Deploy $2K for 30-day validation before full $200K deployment

Per WHAT_IT_WILL_TAKE.md roadmap:
- Forex: $1,000
- Futures: $1,000
- Target: +10-30% over 30 days
- Accept: -10% as learning cost
- Minimum: 20 trades for statistical validation
"""

import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

print("\n" + "="*70)
print("VALIDATION CAPITAL DEPLOYMENT - PHASE 1")
print("="*70)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nPer WHAT_IT_WILL_TAKE.md execution roadmap")
print("="*70 + "\n")

# Capital allocation
TOTAL_CAPITAL = 2000
FOREX_ALLOCATION = 1000
FUTURES_ALLOCATION = 1000

print("CAPITAL ALLOCATION:")
print(f"  Total: ${TOTAL_CAPITAL:,}")
print(f"  Forex (OANDA): ${FOREX_ALLOCATION:,}")
print(f"  Futures (Alpaca): ${FUTURES_ALLOCATION:,}")
print()

# Success criteria
print("SUCCESS CRITERIA (30 days):")
print("  Target Return: +10% to +30%")
print("  Acceptable Loss: -10% (learning cost)")
print("  Minimum Trades: 20 (statistical validation)")
print("  Win Rate Target: 60%+")
print()

# Risk parameters
print("RISK MANAGEMENT:")
print("  Risk Per Trade: 1% ($10 forex, $10 futures)")
print("  Max Daily Loss: 3% ($30)")
print("  Position Size: Dynamic based on ATR")
print("  Stop Loss: 2x ATR")
print("  Take Profit: 1.5-2.0 R:R")
print()

# Check current mode
print("="*70)
print("CURRENT DEPLOYMENT STATUS")
print("="*70)

# Check if we're in paper or live mode
forex_mode = "PAPER TRADING"
futures_mode = "PAPER TRADING"

print(f"\nForex Mode: {forex_mode}")
print(f"Futures Mode: {futures_mode}")

if forex_mode == "PAPER TRADING" or futures_mode == "PAPER TRADING":
    print("\n" + "!"*70)
    print("WARNING: Currently in PAPER TRADING mode")
    print("!"*70)
    print("\nTo deploy real capital:")
    print("1. Open OANDA live account (min $100)")
    print("2. Open Alpaca live account (min $100)")
    print("3. Update .env with LIVE API credentials:")
    print("   - OANDA_API_KEY=<live_key>")
    print("   - OANDA_ACCOUNT_ID=<live_account>")
    print("   - ALPACA_API_KEY=<live_key>")
    print("   - ALPACA_SECRET_KEY=<live_secret>")
    print("   - ALPACA_BASE_URL=https://api.alpaca.markets")
    print("4. Set PAPER_TRADING=false in config files")
    print()

print("="*70)
print("PRE-DEPLOYMENT CHECKLIST")
print("="*70)

checklist = [
    ("OANDA account funded with $1,000+", False),
    ("Alpaca account funded with $1,000+", False),
    ("Live API credentials in .env", False),
    ("Risk limits configured (1% per trade)", False),
    ("Emergency stop procedures in place", False),
    ("Monitoring dashboard accessible", False),
    ("Telegram alerts configured", False),
    ("First 5 trades manual review enabled", False)
]

for i, (item, status) in enumerate(checklist, 1):
    symbol = "[OK]" if status else "[ ]"
    print(f"{symbol} {i}. {item}")

print("\n" + "="*70)
print("DEPLOYMENT PHASES")
print("="*70)

print("\nPHASE 1: VALIDATION ($2K - Days 1-30)")
print("  - Deploy $1K forex + $1K futures")
print("  - Target: +10-30% OR -10% max loss")
print("  - Validate: 20+ trades, 60%+ win rate")
print("  - Manual review: First 5 trades each system")

print("\nPHASE 2: SCALE ($200K - Days 31-90)")
print("  - ONLY IF Phase 1 successful")
print("  - Deploy $100K forex + $100K futures")
print("  - Add options layer (30-50% of profits)")
print("  - Target: +20-50% (2 months)")

print("\nPHASE 3: PROP FIRMS (Days 120-365)")
print("  - Use options profits for challenges")
print("  - Target: 10-30 funded accounts")
print("  - Control: $1M-$3M firm capital")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)

print("\nIf ready to deploy REAL capital:")
print("  1. Fund accounts (OANDA + Alpaca)")
print("  2. Update .env with live credentials")
print("  3. Run: python START_FOREX_ELITE.py --strategy strict --mode live")
print("  4. Run: python START_ACTIVE_FUTURES_PAPER_TRADING.py --mode live")
print("  5. Monitor: http://localhost:8501 (dashboard)")
print("  6. Track: Telegram notifications")

print("\nIf continuing paper trading:")
print("  - Systems already running in background")
print("  - Monitor for signal generation")
print("  - Validate strategy logic")
print("  - Build confidence before capital deployment")

print("\n" + "="*70)
print("RISK WARNING")
print("="*70)

print("""
Trading carries substantial risk of loss. Only trade with capital you can afford to lose.

Past performance (backtests) does not guarantee future results.

The $2K validation phase is designed to:
1. Test systems in live market conditions
2. Validate strategy logic with real execution
3. Accept -10% as acceptable learning cost
4. Build confidence before larger deployment

Never deploy more capital than you're willing to lose.
""")

print("="*70)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Reference: WHAT_IT_WILL_TAKE.md")
print("="*70 + "\n")
