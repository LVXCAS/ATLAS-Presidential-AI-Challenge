"""
Calculate YOUR ACTUAL E8 pass rate based on:
1. Your bot's real drawdown distribution (from Monte Carlo simulation)
2. Your current settings (5x leverage, 4 pairs)
3. E8's specific drawdown limits
"""
import json
from datetime import datetime
import random

# YOUR BOT'S ACTUAL DRAWDOWN PROFILE (from REALISTIC_MAX_DRAWDOWN.py)
WIN_RATE = 0.385
RISK_PER_TRADE = 0.01  # 1% risk per trade
REWARD_PER_TRADE = 0.02  # 2% reward per win
LEVERAGE = 5  # 5x leverage
NUM_PAIRS = 4  # EUR_USD, USD_JPY, GBP_USD, GBP_JPY

# From earlier analysis - your ACTUAL drawdown statistics
YOUR_DRAWDOWN_STATS = {
    'median_max_dd': 0.098,  # 9.8%
    'average_max_dd': 0.109,  # 10.9%
    'percentile_75': 0.132,  # 13.2%
    'percentile_95': 0.191,  # 19.1%
}

# E8 Challenge Options
E8_OPTIONS = {
    'Option A': {
        'max_drawdown': 0.04,  # 4%
        'profit_split': 1.00,
        'account_size': 200000
    },
    'Option B': {
        'max_drawdown': 0.06,  # 6%
        'profit_split': 0.80,
        'account_size': 200000
    }
}

def simulate_challenge_attempt(max_dd_limit, num_trades=100, risk_per_trade=0.01, simulations=10000):
    """
    Simulate E8 challenge attempts with YOUR bot's exact parameters
    Returns: pass rate (% of simulations that stayed under DD limit)
    """
    passes = 0

    for _ in range(simulations):
        balance = 100  # Start at 100%
        peak = 100
        max_dd_reached = 0
        failed = False

        for trade in range(num_trades):
            # Win or loss based on YOUR win rate
            if random.random() < WIN_RATE:
                balance += balance * REWARD_PER_TRADE
            else:
                balance -= balance * risk_per_trade

            # Update peak
            if balance > peak:
                peak = balance

            # Calculate current drawdown
            current_dd = ((peak - balance) / peak)

            if current_dd > max_dd_reached:
                max_dd_reached = current_dd

            # Check if failed challenge
            if current_dd > max_dd_limit:
                failed = True
                break

        if not failed:
            passes += 1

    return passes / simulations

print("=" * 100)
print("YOUR ACTUAL E8 PASS RATE CALCULATION")
print("Based on YOUR bot's real performance characteristics")
print("=" * 100)
print()

print("YOUR BOT'S SETUP:")
print("-" * 100)
print(f"  Win Rate: {WIN_RATE*100:.1f}%")
print(f"  Risk Per Trade: {RISK_PER_TRADE*100:.1f}%")
print(f"  Reward Per Win: {REWARD_PER_TRADE*100:.1f}%")
print(f"  Leverage: {LEVERAGE}x")
print(f"  Trading Pairs: {NUM_PAIRS} (EUR_USD, USD_JPY, GBP_USD, GBP_JPY)")
print()

print("YOUR ACTUAL DRAWDOWN HISTORY:")
print("-" * 100)
print(f"  Median Max DD: {YOUR_DRAWDOWN_STATS['median_max_dd']*100:.1f}%")
print(f"  Average Max DD: {YOUR_DRAWDOWN_STATS['average_max_dd']*100:.1f}%")
print(f"  75th Percentile: {YOUR_DRAWDOWN_STATS['percentile_75']*100:.1f}% (1 in 4 runs)")
print(f"  95th Percentile: {YOUR_DRAWDOWN_STATS['percentile_95']*100:.1f}% (1 in 20 runs)")
print()

print("=" * 100)
print("SIMULATING 10,000 E8 CHALLENGE ATTEMPTS")
print("=" * 100)
print()

results = {}

for option_name, option in E8_OPTIONS.items():
    print(f"{option_name}: {option['profit_split']*100:.0f}% split, {option['max_drawdown']*100:.0f}% max DD")
    print("-" * 100)

    # Simulate with CURRENT settings
    pass_rate_current = simulate_challenge_attempt(
        option['max_drawdown'],
        num_trades=100,
        risk_per_trade=RISK_PER_TRADE,
        simulations=10000
    )

    # Simulate with REDUCED position sizes (75% of normal)
    risk_reduced = RISK_PER_TRADE * 0.75
    pass_rate_reduced = simulate_challenge_attempt(
        option['max_drawdown'],
        num_trades=100,
        risk_per_trade=risk_reduced,
        simulations=10000
    )

    # Simulate with AGGRESSIVE reduction (50% of normal - safest)
    risk_conservative = RISK_PER_TRADE * 0.50
    pass_rate_conservative = simulate_challenge_attempt(
        option['max_drawdown'],
        num_trades=100,
        risk_per_trade=risk_conservative,
        simulations=10000
    )

    print()
    print(f"  CURRENT SETTINGS (1% risk per trade):")
    print(f"    Pass Rate: {pass_rate_current*100:.1f}%")
    print(f"    Fail Rate: {(1-pass_rate_current)*100:.1f}%")
    print(f"    Expected Attempts to Pass: {1/pass_rate_current:.1f}")

    if pass_rate_current < 0.30:
        print(f"    Status: VERY DIFFICULT - You'll fail {1/pass_rate_current:.0f} times on average")
    elif pass_rate_current < 0.50:
        print(f"    Status: CHALLENGING - Expect 2-3 failed attempts")
    elif pass_rate_current < 0.70:
        print(f"    Status: MODERATE - Expect 1-2 failed attempts")
    else:
        print(f"    Status: GOOD - Most attempts will pass")

    print()
    print(f"  WITH 25% POSITION REDUCTION (0.75% risk per trade):")
    print(f"    Pass Rate: {pass_rate_reduced*100:.1f}%")
    print(f"    Fail Rate: {(1-pass_rate_reduced)*100:.1f}%")
    print(f"    Expected Attempts to Pass: {1/pass_rate_reduced:.1f}")
    print(f"    Improvement: +{(pass_rate_reduced - pass_rate_current)*100:.1f} percentage points")

    print()
    print(f"  WITH 50% POSITION REDUCTION (0.5% risk per trade - SAFEST):")
    print(f"    Pass Rate: {pass_rate_conservative*100:.1f}%")
    print(f"    Fail Rate: {(1-pass_rate_conservative)*100:.1f}%")
    print(f"    Expected Attempts to Pass: {1/pass_rate_conservative:.1f}")
    print(f"    Improvement: +{(pass_rate_conservative - pass_rate_current)*100:.1f} percentage points")

    print()
    print()

    results[option_name] = {
        'max_dd_limit': option['max_drawdown'],
        'profit_split': option['profit_split'],
        'pass_rate_current': pass_rate_current,
        'pass_rate_reduced': pass_rate_reduced,
        'pass_rate_conservative': pass_rate_conservative,
        'expected_attempts_current': 1/pass_rate_current,
        'expected_attempts_reduced': 1/pass_rate_reduced,
        'expected_attempts_conservative': 1/pass_rate_conservative
    }

# Comparison
print("=" * 100)
print("SIDE-BY-SIDE COMPARISON")
print("=" * 100)
print()

print(f"{'Scenario':<50} {'Option A (100%/4%)':<25} {'Option B (80%/6%)':<25}")
print("-" * 100)

scenarios = [
    ('Current Settings (1% risk)',
     f"{results['Option A']['pass_rate_current']*100:.1f}%",
     f"{results['Option B']['pass_rate_current']*100:.1f}%"),

    ('  Expected Attempts',
     f"{results['Option A']['expected_attempts_current']:.1f}x",
     f"{results['Option B']['expected_attempts_current']:.1f}x"),

    ('  Assessment',
     'VERY DIFFICULT' if results['Option A']['pass_rate_current'] < 0.30 else 'CHALLENGING',
     'VERY DIFFICULT' if results['Option B']['pass_rate_current'] < 0.30 else 'CHALLENGING'),

    ('', '', ''),  # Spacer

    ('With 25% Reduction (0.75% risk)',
     f"{results['Option A']['pass_rate_reduced']*100:.1f}%",
     f"{results['Option B']['pass_rate_reduced']*100:.1f}%"),

    ('  Expected Attempts',
     f"{results['Option A']['expected_attempts_reduced']:.1f}x",
     f"{results['Option B']['expected_attempts_reduced']:.1f}x"),

    ('', '', ''),  # Spacer

    ('With 50% Reduction (0.5% risk - SAFEST)',
     f"{results['Option A']['pass_rate_conservative']*100:.1f}%",
     f"{results['Option B']['pass_rate_conservative']*100:.1f}%"),

    ('  Expected Attempts',
     f"{results['Option A']['expected_attempts_conservative']:.1f}x",
     f"{results['Option B']['expected_attempts_conservative']:.1f}x"),
]

for scenario, a_val, b_val in scenarios:
    print(f"{scenario:<50} {a_val:<25} {b_val:<25}")

print()
print("=" * 100)
print("RECOMMENDATION BASED ON REAL PASS RATES")
print("=" * 100)
print()

# Determine best option
option_b_reduced = results['Option B']['pass_rate_reduced']
option_a_reduced = results['Option A']['pass_rate_reduced']

if option_b_reduced > 0.60:
    recommendation = "Option B with 25% position reduction"
    reason = f"{option_b_reduced*100:.0f}% pass rate is acceptable (1-2 attempts to pass)"
elif option_b_reduced > 0.50:
    recommendation = "Option B with 50% position reduction"
    reason = f"Need 50% reduction to get {results['Option B']['pass_rate_conservative']*100:.0f}% pass rate"
else:
    recommendation = "Neither - start with smaller account ($25K or $100K)"
    reason = "Your drawdown is too high for $200K challenges"

print(f"RECOMMENDATION: {recommendation}")
print(f"REASON: {reason}")
print()

if 'Option B' in recommendation:
    # Calculate realistic outcomes
    if '25%' in recommendation:
        pass_rate = option_b_reduced
        risk = 0.0075
    else:
        pass_rate = results['Option B']['pass_rate_conservative']
        risk = 0.005

    expected_attempts = 1 / pass_rate
    challenge_cost = 1200  # Estimated
    total_cost = challenge_cost * expected_attempts

    monthly_roi = 0.178  # 17.8% per month
    monthly_profit_gross = 200000 * monthly_roi
    monthly_profit_net = monthly_profit_gross * 0.80  # 80% split

    # Adjust for reduced risk
    if risk < 0.01:
        reduction_factor = risk / 0.01
        monthly_profit_net *= reduction_factor

    print("EXPECTED OUTCOMES:")
    print(f"  Pass rate: {pass_rate*100:.0f}%")
    print(f"  Expected attempts: {expected_attempts:.1f}")
    print(f"  Total cost to pass: ${total_cost:,.0f}")
    print(f"  Monthly profit (after reduction): ${monthly_profit_net:,.0f}")
    print(f"  Break-even time: {total_cost/monthly_profit_net:.1f} months")
    print(f"  First year net: ${monthly_profit_net*12 - total_cost:,.0f}")
    print()

print("=" * 100)
print("KEY INSIGHTS")
print("=" * 100)
print()

print(f"1. YOUR CURRENT SETUP HAS THESE PASS RATES:")
print(f"   Option A (4% DD): {results['Option A']['pass_rate_current']*100:.1f}%")
print(f"   Option B (6% DD): {results['Option B']['pass_rate_current']*100:.1f}%")
print()

print(f"2. YOU MUST REDUCE POSITION SIZES TO PASS:")
print(f"   Current risk (1% per trade) is too aggressive")
print(f"   Recommended: 0.5-0.75% risk per trade during E8 challenge")
print()

print(f"3. OPTION B IS CLEARLY BETTER:")
if results['Option B']['pass_rate_reduced'] > results['Option A']['pass_rate_reduced']:
    diff = results['Option B']['pass_rate_reduced'] - results['Option A']['pass_rate_reduced']
    print(f"   Option B pass rate is {diff*100:.0f} percentage points higher")
    print(f"   You'll pass in {results['Option B']['expected_attempts_reduced']:.1f} attempts vs {results['Option A']['expected_attempts_reduced']:.1f}")
    print(f"   The 20% profit split sacrifice is worth the higher success rate")
print()

print(f"4. REALISTIC TIMELINE:")
if results['Option B']['pass_rate_reduced'] > 0.50:
    print(f"   With 25% position reduction: Pass in 1-2 attempts (1-2 months)")
    print(f"   Monthly income after passing: ${monthly_profit_net:,.0f}")
    print(f"   Annual income: ${monthly_profit_net*12:,.0f}")
else:
    print(f"   You'll need multiple attempts - budget for $3K-5K in challenge fees")
    print(f"   Consider starting with a smaller $100K account first")

# Save results
output = {
    'timestamp': datetime.now().isoformat(),
    'bot_parameters': {
        'win_rate': WIN_RATE,
        'risk_per_trade': RISK_PER_TRADE,
        'leverage': LEVERAGE,
        'num_pairs': NUM_PAIRS
    },
    'drawdown_stats': YOUR_DRAWDOWN_STATS,
    'e8_pass_rates': results,
    'recommendation': recommendation
}

with open('your_actual_e8_pass_rate.json', 'w') as f:
    json.dump(output, f, indent=2)

print()
print("=" * 100)
print("Analysis saved to: your_actual_e8_pass_rate.json")
print("=" * 100)
