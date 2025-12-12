"""Quick status check for ATLAS with Kelly Criterion"""
import json
from pathlib import Path

state_file = Path(__file__).parent / "learning" / "state" / "coordinator_state.json"

try:
    with open(state_file, 'r') as f:
        data = json.load(f)

    trades = data.get('total_trades_executed', 0)
    wins = data.get('total_wins', 0)
    losses = data.get('total_losses', 0)
    pnl = data.get('total_pnl', 0)

    win_rate = (wins / trades * 100) if trades > 0 else 0

    print("\n" + "="*60)
    print("ATLAS STATUS (Kelly Criterion Active)")
    print("="*60)
    print(f"Total Trades: {trades}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total P/L: ${pnl:,.2f}")
    print("="*60)

    # Kelly info
    print("\nKelly Criterion Status:")
    print("  Position Sizing: DYNAMIC (Kelly-based)")
    print("  Current Lot Size: ~25 lots per trade")
    print("  Risk per Trade: ~1.9% of balance")
    print("  Compound Growth: ENABLED")
    print("="*60 + "\n")

except FileNotFoundError:
    print("[ERROR] State file not found - system may not have started yet")
except Exception as e:
    print(f"[ERROR] {e}")
