#!/usr/bin/env python3
"""
OPTIONS_BOT Launcher
Quick start for your options trading bots
"""

import os
import sys

print("=" * 70)
print("OPTIONS_BOT LAUNCHER")
print("=" * 70)
print("\nAvailable Options Trading Bots:\n")

bots = {
    '1': {
        'name': 'Options Hunter Bot',
        'file': 'options_hunter_bot.py',
        'description': 'Monte Carlo optimized, 100% options trading',
        'strategies': 'Bull Call Spreads (71.7%), Bear Put Spreads (86.4%)'
    },
    '2': {
        'name': 'Autonomous Options Income Agent',
        'file': 'autonomous_options_income_agent.py',
        'description': 'Cash-secured puts + selective call buying',
        'strategies': 'Income-focused weekly options'
    },
    '3': {
        'name': 'Real World Options Bot',
        'file': 'real_world_options_bot.py',
        'description': 'Production-ready with full risk management',
        'strategies': 'Multiple strategies with Greeks integration'
    },
    '4': {
        'name': 'Tomorrow Ready Options Bot',
        'file': 'tomorrow_ready_options_bot.py',
        'description': 'Next-day trading preparation',
        'strategies': 'Overnight analysis and morning execution'
    },
    '5': {
        'name': 'Adaptive Dual Options Engine',
        'file': 'adaptive_dual_options_engine.py',
        'description': 'Dual strategy optimization',
        'strategies': 'Adaptive strategy selection'
    }
}

for key, bot in bots.items():
    print(f"{key}. {bot['name']}")
    print(f"   File: {bot['file']}")
    print(f"   Description: {bot['description']}")
    print(f"   Strategies: {bot['strategies']}")
    print()

print("-" * 70)
choice = input("\nSelect bot to run (1-5) or 'q' to quit: ").strip()

if choice == 'q':
    print("Exiting...")
    sys.exit(0)

if choice not in bots:
    print(f"Invalid choice: {choice}")
    sys.exit(1)

selected_bot = bots[choice]
print(f"\nStarting {selected_bot['name']}...")
print(f"File: {selected_bot['file']}\n")
print("=" * 70)

# Run the selected bot
os.system(f'python {selected_bot["file"]}')
