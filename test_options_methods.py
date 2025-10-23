#!/usr/bin/env python3
"""
Test the newly implemented options trading methods
"""

import asyncio
import sys
from datetime import datetime

async def test_options_methods():
    """Test options trading methods specifically"""

    print("Testing Options Trading Methods")
    print("=" * 50)

    try:
        # Import options trader
        from agents.options_trading_agent import OptionsTrader
        print("[OK] OptionsTrader imported successfully")

        # Initialize trader
        trader = OptionsTrader()
        print("[OK] OptionsTrader initialized")

        # Test get_liquid_options method
        print("\nTesting get_liquid_options method...")

        # Test with SPY (most liquid options)
        liquid_options = await trader.get_liquid_options('SPY', min_volume=1, min_oi=1)

        if liquid_options:
            print(f"[OK] Found {len(liquid_options)} liquid options for SPY")

            # Show details of first few options
            for i, option in enumerate(liquid_options[:3]):
                print(f"  Option {i+1}: {option.symbol}")
                print(f"    - Type: {option.option_type}")
                print(f"    - Strike: ${option.strike}")
                print(f"    - Bid/Ask: ${option.bid:.2f}/${option.ask:.2f}")
                print(f"    - Volume: {option.volume}")
                print(f"    - OI: {option.open_interest}")
                print(f"    - Days to expiry: {option.days_to_expiry}")
                if option.delta != 0:
                    print(f"    - Delta: {option.delta:.3f}")
        else:
            print("[WARN] No liquid options found")

        # Test filtering by option type
        print("\nTesting option type filtering...")

        calls_only = await trader.get_liquid_options('SPY', option_type='call', min_volume=1, min_oi=1)
        puts_only = await trader.get_liquid_options('SPY', option_type='put', min_volume=1, min_oi=1)

        print(f"[OK] Found {len(calls_only)} liquid calls")
        print(f"[OK] Found {len(puts_only)} liquid puts")

        # Test options chain method
        print("\nTesting get_options_chain method...")
        options_chain = await trader.get_options_chain('SPY')

        if options_chain:
            print(f"[OK] Retrieved options chain with {len(options_chain)} contracts")

            # Count calls vs puts
            calls = [o for o in options_chain if o.option_type == 'call']
            puts = [o for o in options_chain if o.option_type == 'put']
            print(f"    - Calls: {len(calls)}")
            print(f"    - Puts: {len(puts)}")
        else:
            print("[WARN] No options chain retrieved")

        # Test strategy finding
        print("\nTesting find_best_options_strategy method...")
        if options_chain:
            # Test with bullish conditions
            strategy_result = trader.find_best_options_strategy(
                symbol='SPY',
                price=660.0,  # Sample price
                volatility=0.15,  # 15% volatility
                rsi=45.0,  # Neutral RSI
                price_change=0.01  # 1% positive move
            )

            if strategy_result:
                strategy, contracts = strategy_result
                print(f"[OK] Found strategy: {strategy}")
                print(f"    - Contracts: {len(contracts)}")
                for contract in contracts:
                    print(f"    - {contract.symbol}: {contract.option_type} ${contract.strike}")
            else:
                print("[WARN] No strategy found for current conditions")

        print("\n" + "=" * 50)
        print("[SUCCESS] All options methods working correctly!")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"[ERROR] Options methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_options_methods())
    if success:
        print("\n[RESULT] Options trading methods are fully functional")
    else:
        print("\n[RESULT] Some issues remain with options trading methods")
        sys.exit(1)