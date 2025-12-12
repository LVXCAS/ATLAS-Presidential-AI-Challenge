#!/usr/bin/env python3
"""
Test RSI Exhaustion Filter

Simulates the failed EUR/USD trade scenario (RSI 75.2) to verify
that the new filter would have blocked the trade.
"""

from agents.technical_agent import TechnicalAgent

def test_rsi_filter():
    """Test that RSI > 70 blocks LONG entries"""

    agent = TechnicalAgent(initial_weight=1.5)

    # Simulate EUR/USD conditions from failed trade
    market_data = {
        "pair": "EUR_USD",
        "price": 1.16624,
        "direction": "long",  # Intended trade direction
        "indicators": {
            "rsi": 75.2,  # Extreme overbought (same as failed trade)
            "macd": 0.0012,
            "macd_signal": 0.0010,
            "macd_hist": 0.0002,
            "adx": 52.9,  # Strong trend
            "ema50": 1.165,
            "ema200": 1.160,
            "bb_upper": 1.168,
            "bb_lower": 1.164,
            "bb_middle": 1.166,
            "atr": 0.001041
        }
    }

    print("=" * 70)
    print("RSI EXHAUSTION FILTER TEST")
    print("=" * 70)
    print(f"\nScenario: EUR/USD LONG entry attempt")
    print(f"RSI: {market_data['indicators']['rsi']}")
    print(f"ADX: {market_data['indicators']['adx']} (strong trend)")
    print(f"Price: {market_data['price']}")
    print(f"\nThis is the EXACT scenario that caused the -$3,575 loss.\n")

    # Get agent vote
    vote, confidence, reasoning = agent.analyze(market_data)

    print("-" * 70)
    print(f"Agent Vote: {vote}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Reasoning: {reasoning.get('reason', 'N/A')}")
    print(f"Message: {reasoning.get('message', 'N/A')}")
    print("-" * 70)

    if vote == "BLOCK":
        print("\n[SUCCESS] Filter BLOCKED the trade (as expected)")
        print("   This trade would NOT have been executed.")
        print("   Account would be protected from -$3,575 loss.\n")
        return True
    else:
        print("\n[FAILURE] Filter DID NOT BLOCK the trade")
        print("   Trade would still execute and lose money.\n")
        return False

def test_rsi_filter_short():
    """Test that RSI < 30 blocks SHORT entries"""

    agent = TechnicalAgent(initial_weight=1.5)

    market_data = {
        "pair": "EUR_USD",
        "price": 1.16000,
        "direction": "short",
        "indicators": {
            "rsi": 25.0,  # Extreme oversold
            "macd": -0.0012,
            "macd_signal": -0.0010,
            "macd_hist": -0.0002,
            "adx": 45.0,
            "ema50": 1.165,
            "ema200": 1.170,
            "bb_upper": 1.168,
            "bb_lower": 1.156,
            "bb_middle": 1.162,
            "atr": 0.001
        }
    }

    print("\n" + "=" * 70)
    print("RSI OVERSOLD FILTER TEST")
    print("=" * 70)
    print(f"\nScenario: EUR/USD SHORT entry attempt")
    print(f"RSI: {market_data['indicators']['rsi']} (extreme oversold)")
    print(f"ADX: {market_data['indicators']['adx']} (strong trend)\n")

    vote, confidence, reasoning = agent.analyze(market_data)

    print("-" * 70)
    print(f"Agent Vote: {vote}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Reasoning: {reasoning.get('reason', 'N/A')}")
    print("-" * 70)

    if vote == "BLOCK":
        print("\n[SUCCESS] Filter BLOCKED the SHORT trade")
        print("   Prevents entries at oversold exhaustion.\n")
        return True
    else:
        print("\n[FAILURE] Filter did not block SHORT at RSI 25\n")
        return False

def test_normal_entry():
    """Test that normal RSI values (50-65) don't get blocked"""

    agent = TechnicalAgent(initial_weight=1.5)

    market_data = {
        "pair": "EUR_USD",
        "price": 1.16500,
        "direction": "long",
        "indicators": {
            "rsi": 58.5,  # Normal momentum (not extreme)
            "macd": 0.0005,
            "macd_signal": 0.0003,
            "macd_hist": 0.0002,
            "adx": 35.0,
            "ema50": 1.164,
            "ema200": 1.160,
            "bb_upper": 1.167,
            "bb_lower": 1.162,
            "bb_middle": 1.1645,
            "atr": 0.0008
        }
    }

    print("\n" + "=" * 70)
    print("NORMAL RSI TEST (Should NOT Block)")
    print("=" * 70)
    print(f"\nScenario: EUR/USD LONG entry with healthy RSI")
    print(f"RSI: {market_data['indicators']['rsi']} (normal range)\n")

    vote, confidence, reasoning = agent.analyze(market_data)

    print("-" * 70)
    print(f"Agent Vote: {vote}")
    print(f"Confidence: {confidence:.2f}")
    print("-" * 70)

    if vote != "BLOCK":
        print("\n[SUCCESS] Normal RSI allowed trade to proceed")
        print("   Filter only blocks extremes (RSI > 70 or < 30).\n")
        return True
    else:
        print("\n[FAILURE] Filter blocked a normal RSI value\n")
        return False

if __name__ == "__main__":
    results = []

    # Test 1: Block LONG at RSI > 70 (the EUR/USD failure scenario)
    results.append(("RSI > 70 LONG Block", test_rsi_filter()))

    # Test 2: Block SHORT at RSI < 30
    results.append(("RSI < 30 SHORT Block", test_rsi_filter_short()))

    # Test 3: Allow normal RSI trades
    results.append(("Normal RSI Allowed", test_normal_entry()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)
    print("\n" + ("=" * 70))
    if all_passed:
        print("[SUCCESS] ALL TESTS PASSED - RSI Filter is working correctly!")
        print("\nThe EUR/USD failure (-$3,575) would have been PREVENTED.")
    else:
        print("[WARNING] SOME TESTS FAILED - Review filter logic")
    print("=" * 70 + "\n")
