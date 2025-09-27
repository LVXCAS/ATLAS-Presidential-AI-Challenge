#!/usr/bin/env python3
"""
Test dual strategy integration in master engine
"""

import asyncio
from hybrid_conviction_genetic_trader import HybridConvictionGeneticTrader
from adaptive_dual_options_engine import AdaptiveDualOptionsEngine

async def test_dual_integration():
    """Test the complete dual strategy integration"""

    print("TESTING DUAL STRATEGY INTEGRATION")
    print("=" * 60)

    try:
        # Initialize components
        trader = HybridConvictionGeneticTrader()
        dual_engine = AdaptiveDualOptionsEngine()

        print("[1/3] Getting hybrid signals...")
        signals = await trader.run_hybrid_analysis()

        if signals:
            print(f"[OK] Found {len(signals)} signals")

            print("[2/3] Testing dual strategy execution...")
            # Test with small buying power to avoid large trades
            test_buying_power = 100000

            # This is what should be called instead of execute_options_trades
            executed_trades = dual_engine.execute_dual_strategy(signals, test_buying_power)

            if executed_trades:
                print(f"[OK] Dual strategy executed: {len(executed_trades)} trades")
                print("[3/3] Integration test: SUCCESS")

                # Show what was executed
                for trade in executed_trades:
                    print(f"  - {trade.get('symbol', 'N/A')}: {trade.get('strategy', 'N/A')}")

                return True
            else:
                print("[X] Dual strategy execution failed")
                return False
        else:
            print("[X] No signals found")
            return False

    except Exception as e:
        print(f"[X] Integration test failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_dual_integration())
    print("=" * 60)
    print("INTEGRATION TEST:", "PASSED" if result else "FAILED")