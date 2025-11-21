#!/usr/bin/env python3
"""
Manually trigger OPTIONS learning cycle
Run this to immediately optimize parameters based on collected trades
"""

import sys
import json
from datetime import datetime
from options_learning_integration import get_tracker

print("="*70)
print("OPTIONS CONTINUOUS LEARNING CYCLE")
print("="*70)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Get the learning tracker
tracker = get_tracker()

print("[STEP 1] CHECKING SYSTEM STATUS")
print("-"*70)
config = tracker.config
print(f"Learning Enabled: {config.get('learning_enabled', False)}")
print(f"Min Trades Required: {config.get('min_feedback_samples', 20)}")
print(f"Current Trades: {len(tracker.completed_trades)}")
print(f"Max Parameter Change: {config.get('max_parameter_change', 0.20)*100}%")
print()

if len(tracker.completed_trades) < config.get('min_feedback_samples', 20):
    print(f"[ERROR] Not enough trades yet!")
    print(f"Need {config.get('min_feedback_samples', 20)}, have {len(tracker.completed_trades)}")
    print("Continue trading to collect more data.")
    sys.exit(1)

print("[STEP 2] ANALYZING TRADE PERFORMANCE")
print("-"*70)
stats = tracker.get_strategy_statistics()

print(f"Total trades loaded: {stats['total_trades']}")
print(f"Overall win rate: {stats['overall_win_rate']*100:.1f}%")
print(f"Overall profit factor: {stats['overall_profit_factor']:.2f}")

print("\nStrategy Breakdown:")
for strategy, data in stats['strategy_stats'].items():
    print(f"\n{strategy}:")
    print(f"  Total Trades: {data['total_trades']}")
    print(f"  Win Rate: {data['win_rate']*100:.1f}%")
    print(f"  Profit Factor: {data['profit_factor']:.2f}")

print()

print("[STEP 3] RUNNING OPTIMIZATION")
print("-"*70)
print("Analyzing trades to find optimal parameters...")
print()

# Note: Full continuous learning system requires backend.events module
# For now, we'll use the baseline parameters from the tracked trades
print("⚠ NOTE: Full ML optimization requires complete continuous learning system")
print("        Using statistical analysis of trade performance instead...")
print()

# Run the learning cycle
try:
    import asyncio
    result = asyncio.run(tracker.run_learning_cycle())

    print("[STEP 4] OPTIMIZATION RESULTS")
    print("-"*70)

    if result.get('success'):
        print("✓ Optimization SUCCESSFUL!")
        print()

        print("Current Parameters:")
        current = tracker.current_parameters
        for param, value in current.items():
            print(f"  {param}: {value}")

        print()
        print("Recommended Parameters:")
        optimized = result.get('optimized_parameters', {})
        for param, value in optimized.items():
            old = current.get(param, 0)
            change = ((value - old) / old * 100) if old != 0 else 0
            print(f"  {param}: {value} ({change:+.1f}% change)")

        print()
        print(f"Confidence Score: {result.get('confidence', 0)*100:.1f}%")
        print(f"Expected Win Rate Improvement: {result.get('expected_improvement', 0)*100:+.1f}%")

        if result.get('applied'):
            print()
            print("✓ Parameters have been APPLIED and saved!")
            print("  Next scanner run will use optimized parameters")
        else:
            print()
            print("⚠ Parameters NOT applied (low confidence or insufficient data)")
            print("  Continue collecting data for higher confidence")
    else:
        print("✗ Optimization failed")
        print(f"Reason: {result.get('error', 'Unknown error')}")

except Exception as e:
    print(f"[ERROR] Learning cycle failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("="*70)
print("LEARNING CYCLE COMPLETE")
print("="*70)
