#!/usr/bin/env python3
"""
Test that RSI exhaustion filter now BLOCKS trades at extremes
This would have prevented both EUR/USD (RSI 75) and GBP/USD (RSI 27) losses
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from agents.technical_agent import TechnicalAgent

def test_short_at_oversold():
    """Test that SHORT is BLOCKED when RSI < 30 (oversold = reversal UP expected)"""
    print("\n" + "=" * 70)
    print("TEST 1: SHORT at RSI 27 (GBP_USD failure scenario)")
    print("=" * 70)

    agent = TechnicalAgent(initial_weight=1.5)

    # Simulate GBP_USD conditions that caused the -$1,800 loss
    market_data = {
        "pair": "GBP_USD",
        "price": 1.33282,
        "direction": "short",  # ATLAS wanted to go SHORT
        "indicators": {
            "rsi": 27,  # OVERSOLD - reversal UP expected
            "macd": -0.0005,
            "macd_hist": -0.0002,
            "adx": 35,
            "ema50": 1.33500,
            "ema200": 1.34000
        }
    }

    vote, confidence, reasoning = agent.analyze(market_data)

    print(f"\nMarket Conditions:")
    print(f"  Pair: GBP_USD")
    print(f"  Direction: SHORT (betting price will fall)")
    print(f"  RSI: 27 (oversold - price likely to BOUNCE UP)")
    print(f"\nTechnicalAgent Response:")
    print(f"  Vote: {vote}")
    print(f"  Confidence: {confidence:.2f}")

    if vote == "BLOCK":
        print(f"\n[OK] Trade BLOCKED - RSI exhaustion filter working!")
        print(f"  Reason: {reasoning.get('reason', 'N/A')}")
        print(f"  Recommendation: {reasoning.get('recommendation', 'N/A')}")
        return True
    else:
        print(f"\n[FAIL] Trade NOT blocked - filter failed!")
        print(f"  This is the bug that caused -$1,800 loss")
        return False

def test_long_at_overbought():
    """Test that LONG is BLOCKED when RSI > 70 (overbought = reversal DOWN expected)"""
    print("\n" + "=" * 70)
    print("TEST 2: LONG at RSI 75 (EUR_USD failure scenario)")
    print("=" * 70)

    agent = TechnicalAgent(initial_weight=1.5)

    # Simulate EUR_USD conditions that caused the -$3,575 loss
    market_data = {
        "pair": "EUR_USD",
        "price": 1.16624,
        "direction": "long",  # ATLAS wanted to go LONG
        "indicators": {
            "rsi": 75.2,  # OVERBOUGHT - reversal DOWN expected
            "macd": 0.0012,
            "macd_hist": 0.0008,
            "adx": 52.9,
            "ema50": 1.16000,
            "ema200": 1.15500
        }
    }

    vote, confidence, reasoning = agent.analyze(market_data)

    print(f"\nMarket Conditions:")
    print(f"  Pair: EUR_USD")
    print(f"  Direction: LONG (betting price will rise)")
    print(f"  RSI: 75.2 (overbought - price likely to FALL)")
    print(f"\nTechnicalAgent Response:")
    print(f"  Vote: {vote}")
    print(f"  Confidence: {confidence:.2f}")

    if vote == "BLOCK":
        print(f"\n[OK] Trade BLOCKED - RSI exhaustion filter working!")
        print(f"  Reason: {reasoning.get('reason', 'N/A')}")
        print(f"  Recommendation: {reasoning.get('recommendation', 'N/A')}")
        return True
    else:
        print(f"\n[FAIL] Trade NOT blocked - filter failed!")
        print(f"  This is the bug that caused -$3,575 loss")
        return False

def test_normal_rsi():
    """Test that trades are ALLOWED at normal RSI levels"""
    print("\n" + "=" * 70)
    print("TEST 3: LONG at RSI 55 (normal conditions)")
    print("=" * 70)

    agent = TechnicalAgent(initial_weight=1.5)

    market_data = {
        "pair": "EUR_USD",
        "price": 1.16000,
        "direction": "long",
        "indicators": {
            "rsi": 55,  # Normal range
            "macd": 0.0005,
            "macd_hist": 0.0003,
            "adx": 25,
            "ema50": 1.15900,
            "ema200": 1.15500
        }
    }

    vote, confidence, reasoning = agent.analyze(market_data)

    print(f"\nMarket Conditions:")
    print(f"  Pair: EUR_USD")
    print(f"  Direction: LONG")
    print(f"  RSI: 55 (healthy mid-range)")
    print(f"\nTechnicalAgent Response:")
    print(f"  Vote: {vote}")
    print(f"  Confidence: {confidence:.2f}")

    if vote != "BLOCK":
        print(f"\n[OK] Trade allowed at normal RSI levels")
        return True
    else:
        print(f"\n[FAIL] Trade blocked incorrectly - filter too aggressive!")
        return False

def main():
    print("=" * 70)
    print("RSI EXHAUSTION FILTER TEST SUITE")
    print("Testing fix for EUR/USD and GBP/USD failures")
    print("=" * 70)

    results = []
    results.append(test_short_at_oversold())  # GBP_USD scenario
    results.append(test_long_at_overbought())  # EUR_USD scenario
    results.append(test_normal_rsi())  # Normal trading

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n[SUCCESS] All tests passed - RSI exhaustion filter working!")
        print("\nProtection Summary:")
        print("  - LONG entries blocked when RSI > 70 (overbought)")
        print("  - SHORT entries blocked when RSI < 30 (oversold)")
        print("  - Normal trades (RSI 30-70) allowed")
        print("\nThis prevents:")
        print("  - Entering LONG at market tops (EUR/USD -$3,575 loss)")
        print("  - Entering SHORT at market bottoms (GBP/USD -$1,800 loss)")
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed")
        print("RSI exhaustion filter needs debugging")

if __name__ == "__main__":
    main()
