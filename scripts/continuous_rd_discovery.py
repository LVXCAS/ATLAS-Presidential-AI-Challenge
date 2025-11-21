#!/usr/bin/env python3
"""
CONTINUOUS R&D DISCOVERY ENGINE
Runs every 6 hours to continuously find new alpha
"""

import asyncio
import sys
import os
from datetime import datetime
from hybrid_rd_system import HybridRDOrchestrator

async def continuous_rd_loop():
    """Run R&D discovery continuously"""

    cycle = 0

    while True:
        cycle += 1

        print("\n" + "="*70)
        print(f"R&D DISCOVERY CYCLE #{cycle}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        try:
            # Run full R&D cycle
            orchestrator = HybridRDOrchestrator()
            deployment_package = await orchestrator.run_full_rd_cycle()

            print(f"\n[CYCLE {cycle}] COMPLETE")
            print(f"  Strategies discovered: {len(deployment_package.get('strategies', []))}")
            print(f"  Next cycle in 6 hours")

        except Exception as e:
            print(f"\n[CYCLE {cycle}] ERROR: {e}")
            print("  Will retry next cycle")

        # Wait 6 hours before next discovery cycle
        # During market hours: discovers fresh opportunities
        # Overnight: prepares strategies for next day
        await asyncio.sleep(6 * 60 * 60)  # 6 hours

async def main():
    """Launch continuous R&D discovery"""

    print("="*70)
    print("CONTINUOUS R&D DISCOVERY ENGINE - STARTING")
    print("="*70)
    print("Mode: Continuous (runs every 6 hours)")
    print("Purpose: Keep finding new alpha strategies")
    print()
    print("Strategy Discovery:")
    print("  - Every 6 hours: Full R&D cycle")
    print("  - Historical research (yfinance)")
    print("  - Live validation (Alpaca)")
    print("  - Deploy-ready strategies saved")
    print()
    print("Output:")
    print("  - rd_validated_strategies_TIMESTAMP.json")
    print("  - New discoveries accumulate over time")
    print("="*70)
    print()

    # Run continuous discovery
    await continuous_rd_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nR&D Discovery stopped by user")
        sys.exit(0)
