#!/usr/bin/env python3
"""
Diagnose why bot has 0% win rate
Analyze the complete chain from market data -> signals -> strategy selection -> trades
"""

import asyncio
import yfinance as yf
from datetime import datetime, timedelta
from agents.options_trading_agent import OptionsTrader, OptionsStrategy
from agents.broker_integration import AlpacaBrokerIntegration

async def main():
    print("=" * 80)
    print("DIAGNOSING BOT LOSING STREAK - 0% WIN RATE")
    print("=" * 80)
    print()

    # Symbols that lost money
    losing_symbols = ['COP', 'DVN', 'NKE', 'QCOM', 'RTX', 'SLB', 'TXN', 'XOP']

    # Initialize options trader
    broker = AlpacaBrokerIntegration(paper_trading=True)
    trader = OptionsTrader(broker)

    print("Step 1: Simulating market conditions on Oct 10, 2025")
    print("-" * 80)

    for symbol in losing_symbols[:3]:  # Test first 3 symbols
        print(f"\n{symbol}:")

        # Get historical data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='2mo')

        if len(hist) < 20:
            print(f"  ERROR: Insufficient data")
            continue

        # Simulate Oct 10 conditions (5 trading days ago)
        oct10_idx = -5

        # Calculate RSI
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Get Oct 10 values
        price = hist['Close'].iloc[oct10_idx]
        rsi_val = rsi.iloc[oct10_idx]
        price_prev = hist['Close'].iloc[oct10_idx-1]
        price_change = (price - price_prev) / price_prev

        print(f"  Price: ${price:.2f}")
        print(f"  Price Change: {price_change*100:+.2f}%")
        print(f"  RSI: {rsi_val:.1f}")

        # Fetch options chain
        print(f"\n  Fetching options chain...")
        options_chain = await trader.get_options_chain(symbol)

        if not options_chain:
            print(f"  ERROR: No options available")
            continue

        print(f"  Found {len(options_chain)} liquid contracts")

        # Call strategy selection
        print(f"\n  Calling find_best_options_strategy...")
        print(f"    symbol={symbol}, price={price:.2f}, volatility=25.0, rsi={rsi_val:.1f}, price_change={price_change:.4f}")

        strategy_result = trader.find_best_options_strategy(
            symbol=symbol,
            price=price,
            volatility=25.0,
            rsi=rsi_val,
            price_change=price_change
        )

        if strategy_result:
            strategy_type, contracts = strategy_result
            contract = contracts[0]

            print(f"\n  RESULT:")
            print(f"    Strategy: {strategy_type}")
            print(f"    Contract: {contract.symbol}")
            print(f"    Option Type: {contract.option_type}")
            print(f"    Strike: ${contract.strike:.2f}")
            print(f"    Delta: {contract.delta:.3f}")

            # Determine if this matches market direction
            market_direction = "BEARISH" if price_change < -0.005 else "BULLISH" if price_change > 0.005 else "NEUTRAL"
            expected_strategy = "LONG_PUT" if price_change < -0.005 else "LONG_CALL" if price_change > 0.005 else "NONE"

            print(f"\n  ANALYSIS:")
            print(f"    Market Direction: {market_direction} (change: {price_change*100:+.2f}%)")
            print(f"    Expected Strategy: {expected_strategy}")
            print(f"    Actual Strategy: {strategy_type}")
            print(f"    MATCH: {'YES ✓' if strategy_type.value == expected_strategy else 'NO ✗ WRONG!'}")

            # Check what actually happened
            actual_change = (hist['Close'].iloc[-1] - price) / price
            print(f"\n    Actual stock movement: {actual_change*100:+.2f}%")

            if strategy_type == OptionsStrategy.LONG_CALL and actual_change > 0:
                print(f"    Trade outcome: SHOULD HAVE WON (calls + stock up)")
            elif strategy_type == OptionsStrategy.LONG_PUT and actual_change < 0:
                print(f"    Trade outcome: SHOULD HAVE WON (puts + stock down)")
            else:
                print(f"    Trade outcome: SHOULD HAVE LOST (wrong direction!)")
        else:
            print(f"  No strategy selected")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Based on the analysis:

1. Bot correctly identifies bearish signals (price_change < -0.5%)
2. Strategy logic SHOULD select LONG_PUT
3. But historical trades show bot bought CALLS instead

Possible causes:
- Bug in find_best_options_strategy implementation
- OPTIONS_BOT.py passing wrong parameters
- price_momentum calculation inverted
- Strategy selection reversed somewhere

Next step: Check actual trades to confirm if wrong contracts were executed.
    """)

if __name__ == "__main__":
    asyncio.run(main())
