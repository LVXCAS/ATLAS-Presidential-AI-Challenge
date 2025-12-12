"""
THRESHOLD ADJUSTER

Quickly adjust score threshold to get more/fewer trades.

Usage:
    python diagnostics/adjust_threshold.py --mode exploration  # Lower to 3.5
    python diagnostics/adjust_threshold.py --mode validation   # Raise to 4.5
    python diagnostics/adjust_threshold.py --threshold 4.0     # Custom
"""

import json
import argparse
from pathlib import Path


def adjust_threshold(mode: str = None, threshold: float = None):
    """Adjust score threshold in config."""

    config_file = Path(__file__).parent.parent / "config" / "hybrid_optimized.json"

    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)

    old_threshold = config["trading_parameters"]["score_threshold"]

    # Determine new threshold
    if threshold is not None:
        new_threshold = threshold
    elif mode == "exploration":
        new_threshold = 3.5
    elif mode == "refinement":
        new_threshold = 4.0
    elif mode == "validation":
        new_threshold = 4.5
    elif mode == "ultra_conservative":
        new_threshold = 6.0
    else:
        print(f"[ERROR] Must specify --mode or --threshold")
        return

    # Update config
    config["trading_parameters"]["score_threshold"] = new_threshold

    # Save config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 80)
    print("THRESHOLD ADJUSTED")
    print("=" * 80)
    print(f"\nOld Threshold: {old_threshold}")
    print(f"New Threshold: {new_threshold}")

    print(f"\nEXPECTED BEHAVIOR:")

    if new_threshold <= 3.5:
        print(f"  Trades/Week: 15-25 (exploration mode)")
        print(f"  Win Rate: ~50-55% (high volume, lower quality)")
        print(f"  Purpose: Generate training data")
    elif new_threshold <= 4.0:
        print(f"  Trades/Week: 10-15 (refinement mode)")
        print(f"  Win Rate: ~55-58% (balanced)")
        print(f"  Purpose: Optimize patterns")
    elif new_threshold <= 4.5:
        print(f"  Trades/Week: 8-12 (validation mode)")
        print(f"  Win Rate: ~58-62% (high quality)")
        print(f"  Purpose: E8 challenge ready")
    elif new_threshold <= 5.0:
        print(f"  Trades/Week: 4-8 (conservative)")
        print(f"  Win Rate: ~62-65% (very selective)")
        print(f"  Purpose: Risk minimization")
    else:
        print(f"  Trades/Week: 0-2 (ultra-conservative)")
        print(f"  Win Rate: ~65-70% (extreme selectivity)")
        print(f"  Purpose: Match Trader demo validation")

    print(f"\n{'=' * 80}")
    print(f"Config updated: {config_file}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adjust ATLAS score threshold")
    parser.add_argument("--mode", choices=["exploration", "refinement", "validation", "ultra_conservative"],
                        help="Preset mode")
    parser.add_argument("--threshold", type=float, help="Custom threshold (3.0-6.0)")

    args = parser.parse_args()

    if not args.mode and not args.threshold:
        print("Usage:")
        print("  python adjust_threshold.py --mode exploration")
        print("  python adjust_threshold.py --threshold 4.0")
        print("\nModes:")
        print("  exploration        - 3.5 threshold (15-25 trades/week)")
        print("  refinement         - 4.0 threshold (10-15 trades/week)")
        print("  validation         - 4.5 threshold (8-12 trades/week)")
        print("  ultra_conservative - 6.0 threshold (0-2 trades/week)")
    else:
        adjust_threshold(mode=args.mode, threshold=args.threshold)
