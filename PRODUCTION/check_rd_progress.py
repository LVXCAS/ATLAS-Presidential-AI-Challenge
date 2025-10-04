"""
R&D PROGRESS CHECKER

Run this daily to see what your R&D systems have discovered.
"""

import os
import json
from datetime import datetime
from glob import glob

def check_rd_progress():
    """Check progress of all R&D systems"""

    print("="*70)
    print("R&D DISCOVERY PROGRESS - WEEK 1")
    print("="*70)
    print(f"Checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    total_discoveries = 0

    # Check VectorBT results
    print("[1] VECTORBT MASS BACKTESTING")
    print("-" * 70)
    vectorbt_files = glob("vectorbt_results_*.json")
    if vectorbt_files:
        latest = max(vectorbt_files, key=os.path.getmtime)
        mtime = datetime.fromtimestamp(os.path.getmtime(latest))
        with open(latest) as f:
            results = json.load(f)

        print(f"Status: COMPLETE (last updated {mtime.strftime('%H:%M:%S')})")
        print(f"Strategies tested: {len(results)}")

        # Filter successful tests
        successful = [r for r in results if r.get('sharpe_ratio', 0) > 0]
        print(f"Successful strategies: {len(successful)}")

        if successful:
            # Top 5 by Sharpe
            top_5 = sorted(successful, key=lambda x: x.get('sharpe_ratio', 0), reverse=True)[:5]
            print("\nTop 5 by Sharpe Ratio:")
            for i, s in enumerate(top_5, 1):
                symbol = s.get('symbol', 'N/A')
                fast = s.get('fast', 0)
                slow = s.get('slow', 0)
                sharpe = s.get('sharpe_ratio', 0)
                total_return = s.get('total_return', 0)
                print(f"  {i}. {symbol} MA_{fast}_{slow}")
                print(f"     Sharpe: {sharpe:.2f} | Return: {total_return:.1%}")

        total_discoveries += len(successful)
    else:
        print("Status: NOT STARTED or RUNNING")
        print("Launch with: LAUNCH_WEEK1_RD_SYSTEMS.bat")

    print()

    # Check Hybrid R&D results
    print("[2] HYBRID R&D SYSTEM")
    print("-" * 70)
    rd_files = glob("rd_results_*.json") + glob("hybrid_rd_results_*.json")
    if rd_files:
        latest = max(rd_files, key=os.path.getmtime)
        mtime = datetime.fromtimestamp(os.path.getmtime(latest))
        with open(latest) as f:
            results = json.load(f)

        print(f"Status: COMPLETE (last updated {mtime.strftime('%H:%M:%S')})")

        strategies = results.get('strategies', [])
        print(f"Strategies discovered: {len(strategies)}")

        if strategies:
            print("\nTop strategies:")
            for i, s in enumerate(strategies[:5], 1):
                symbol = s.get('symbol', 'N/A')
                strategy_type = s.get('type', 'N/A')
                score = s.get('momentum_score', s.get('realized_vol', 0))
                print(f"  {i}. {symbol} ({strategy_type}): Score {score:.2f}")

        total_discoveries += len(strategies)
    else:
        print("Status: NOT STARTED or RUNNING")
        print("Launch with: python hybrid_rd_system.py")

    print()

    # Check Qlib results
    print("[3] QLIB FACTOR MINING")
    print("-" * 70)
    qlib_files = glob("qlib_results_*.json")
    if qlib_files:
        latest = max(qlib_files, key=os.path.getmtime)
        mtime = datetime.fromtimestamp(os.path.getmtime(latest))
        with open(latest) as f:
            results = json.load(f)

        print(f"Status: COMPLETE (last updated {mtime.strftime('%H:%M:%S')})")
        print(f"Factors tested: {results.get('factors_tested', 0)}")
        print(f"Symbols analyzed: {results.get('symbols_analyzed', 0)}")

        qlib_results = results.get('results', {})
        total_factors = sum(len(v) for v in qlib_results.values())
        print(f"Total factor calculations: {total_factors}")

        total_discoveries += total_factors
    else:
        print("Status: NOT STARTED or RUNNING")
        print("Note: Qlib requires data initialization")
        print("      May show as 'not available' without setup")

    print()

    # Check GPU evolution results
    print("[4] GPU GENETIC EVOLUTION")
    print("-" * 70)
    gpu_files = glob("gpu_evolution_results_*.json") + glob("genetic_evolution_*.json")
    if gpu_files:
        latest = max(gpu_files, key=os.path.getmtime)
        mtime = datetime.fromtimestamp(os.path.getmtime(latest))
        with open(latest) as f:
            results = json.load(f)

        print(f"Status: COMPLETE (last updated {mtime.strftime('%H:%M:%S')})")
        generations = results.get('generations', 0)
        print(f"Generations evolved: {generations}")

        best_strategies = results.get('best_strategies', [])
        print(f"Elite strategies discovered: {len(best_strategies)}")

        total_discoveries += len(best_strategies)
    else:
        print("Status: NOT STARTED or may be RUNNING")
        print("Note: GPU evolution runs 6-12 hours in background")
        print("      Check overnight for results")

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total discoveries: {total_discoveries}+")
    print()

    if total_discoveries > 50:
        print("Status: EXCELLENT")
        print("You have enough strategies to deploy in Week 2")
        print()
        print("Next steps:")
        print("  1. Review top discoveries")
        print("  2. Validate in LEAN: cd IntelStrategyLean && lean backtest .")
        print("  3. Prepare Week 2 deployment")
    elif total_discoveries > 20:
        print("Status: GOOD")
        print("Solid foundation for Week 2 scaling")
        print()
        print("Consider running additional R&D cycles to discover more")
    elif total_discoveries > 0:
        print("Status: PROGRESSING")
        print("R&D systems are working, continue running")
        print()
        print("Give VectorBT and Qlib a few more hours to complete")
    else:
        print("Status: JUST STARTED")
        print()
        print("Launch R&D systems with: LAUNCH_WEEK1_RD_SYSTEMS.bat")

    print()
    print("="*70)
    print()
    print("Run this script daily to track R&D progress:")
    print("  python check_rd_progress.py")
    print()

if __name__ == "__main__":
    check_rd_progress()
