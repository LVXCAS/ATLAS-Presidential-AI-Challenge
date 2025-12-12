"""
Change ATLAS Risk-Reward Ratio

Quick script to update R:R targets in config
"""
import json
from pathlib import Path

config_file = Path(__file__).parent / "config" / "hybrid_optimized.json"

# Load config
with open(config_file, 'r') as f:
    config = json.load(f)

print("\n" + "="*60)
print("ATLAS RISK-REWARD RATIO CONFIGURATOR")
print("="*60)

print("\nCurrent Configuration:")
print(f"  Stop Loss: {config['trading_parameters']['stop_loss_pips_min']}-{config['trading_parameters']['stop_loss_pips_max']} pips")
print(f"  Take Profit Targets: {config['trading_parameters']['take_profit_r_targets']}")
print(f"  Partial Close: {config['trading_parameters']['partial_close_pct']*100}%")

print("\nCurrent R:R Ratios:")
sl_pips = 14  # Average
for r_target in config['trading_parameters']['take_profit_r_targets']:
    tp_pips = sl_pips * r_target
    print(f"  {r_target}:1 → {int(tp_pips)} pips take profit")

print("\n" + "-"*60)
print("Available Presets:")
print("  1. Conservative (1.5:1, 2:1) - Higher win rate")
print("  2. Balanced (2:1, 3:1) - Medium win rate")
print("  3. Aggressive (3:1, 4:1) - Lower win rate, bigger wins")
print("  4. Custom - Set your own targets")
print("  5. Cancel - Keep current settings")

choice = input("\nSelect preset (1-5): ")

if choice == "1":
    config['trading_parameters']['take_profit_r_targets'] = [1.5, 2.0]
    print("\n✓ Set to Conservative: 1.5:1 and 2:1")
elif choice == "2":
    config['trading_parameters']['take_profit_r_targets'] = [2.0, 3.0]
    print("\n✓ Set to Balanced: 2:1 and 3:1")
elif choice == "3":
    config['trading_parameters']['take_profit_r_targets'] = [3.0, 4.0]
    print("\n✓ Set to Aggressive: 3:1 and 4:1")
elif choice == "4":
    r1 = float(input("First R target (e.g., 1.5): "))
    r2 = float(input("Second R target (e.g., 3.0): "))
    config['trading_parameters']['take_profit_r_targets'] = [r1, r2]
    print(f"\n✓ Set to Custom: {r1}:1 and {r2}:1")
else:
    print("\nCancelled - no changes made")
    exit()

# Save config
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n✓ Config saved to {config_file}")
print("\nNew pip values (with 14-pip stop loss):")
for r_target in config['trading_parameters']['take_profit_r_targets']:
    tp_pips = 14 * r_target
    print(f"  {r_target}:1 → {int(tp_pips)} pips")

print("\n⚠️  Restart ATLAS for changes to take effect")
print("="*60 + "\n")
