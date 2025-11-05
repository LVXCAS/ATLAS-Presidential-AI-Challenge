#!/usr/bin/env python3
"""
Test Kelly Criterion position sizing logic
"""

# Simulate the Kelly sizing calculation
def calculate_position_size(technical_score, fundamental_score):
    """Test Kelly Criterion position sizing"""

    base_units = 100000
    leverage_multiplier = 10
    base_leveraged = base_units * leverage_multiplier

    # Technical score: 0-10 scale
    tech_normalized = technical_score / 10.0  # 0.0 to 1.0

    # Fundamental score: ±6 scale -> convert to 0-1
    fund_abs = abs(fundamental_score)
    fund_normalized = min(fund_abs / 6.0, 1.0)  # 0.0 to 1.0

    # Combined confidence (average of both signals)
    combined_confidence = (tech_normalized + fund_normalized) / 2.0

    # Map confidence to win probability (conservative estimates)
    base_win_rate = 0.55
    max_win_rate = 0.75
    win_probability = base_win_rate + (combined_confidence * (max_win_rate - base_win_rate))

    # Profit/Loss ratio for forex (2:1 target = 2.0)
    profit_loss_ratio = 2.0

    # Kelly formula: f* = (p*b - q) / b
    p = win_probability
    q = 1 - p
    b = profit_loss_ratio
    kelly_fraction = (p * b - q) / b

    # Use QUARTER-KELLY (very conservative for prop firm)
    quarter_kelly = kelly_fraction / 4.0

    # Cap between 0.5x and 1.5x base leverage
    MIN_MULTIPLIER = 0.5
    MAX_MULTIPLIER = 1.5

    position_multiplier = max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, quarter_kelly * 10))

    # Calculate final position size
    final_units = int(base_leveraged * position_multiplier)

    # Print Kelly analysis
    print(f"\n[KELLY SIZING TEST]")
    print(f"  Technical: {technical_score:.2f}/10 ({tech_normalized*100:.0f}%)")
    print(f"  Fundamental: {fundamental_score}/6 ({fund_normalized*100:.0f}%)")
    print(f"  Combined Confidence: {combined_confidence*100:.0f}%")
    print(f"  Win Probability: {win_probability*100:.1f}%")
    print(f"  Kelly Fraction: {kelly_fraction*100:.1f}%")
    print(f"  Quarter-Kelly: {quarter_kelly*100:.1f}%")
    print(f"  Position Multiplier: {position_multiplier:.2f}x")
    print(f"  Final Units: {final_units:,} ({final_units/base_units:.1f} lots)")

    return final_units

# Test scenarios
print("="*70)
print("KELLY CRITERION POSITION SIZING - TEST SCENARIOS")
print("="*70)

print("\n\n[SCENARIO 1: HIGH CONFIDENCE]")
print("Technical: 8.0/10, Fundamental: 5/6 (strong alignment)")
calculate_position_size(8.0, 5)

print("\n\n[SCENARIO 2: MEDIUM CONFIDENCE]")
print("Technical: 5.0/10, Fundamental: 3/6 (moderate alignment)")
calculate_position_size(5.0, 3)

print("\n\n[SCENARIO 3: LOW CONFIDENCE]")
print("Technical: 3.0/10, Fundamental: 3/6 (minimum threshold)")
calculate_position_size(3.0, 3)

print("\n\n[SCENARIO 4: WEAK FUNDAMENTAL]")
print("Technical: 7.0/10, Fundamental: 0/6 (no fundamental data)")
calculate_position_size(7.0, 0)

print("\n\n[SCENARIO 5: SHORT TRADE]")
print("Technical: 6.5/10, Fundamental: -4/6 (negative = SHORT)")
calculate_position_size(6.5, -4)

print("\n" + "="*70)
